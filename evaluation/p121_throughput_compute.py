#!/usr/local/bin/python

from functools import partial
import json
import math
import os
import argparse
from pathlib import Path
import queue
from typing import Callable, Generator, TextIO, Tuple
import multiprocessing as mp

import pandas as pd

from polyis.io import cache
from polyis.utilities import ProgressBar, get_config


config = get_config()
DATASETS = config['EXEC']['DATASETS']


def load_data_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the index construction and query execution data tables."""
    index_data = pd.read_csv(data_dir / 'index_construction.csv')
    query_data = pd.read_csv(data_dir / 'query_execution.csv')
    
    # print(f"Loaded {len(index_data)} index construction entries")
    # print(f"Loaded {len(query_data)} query execution entries")
    
    return index_data, query_data


def jsonl_loader(f: TextIO) -> Generator[dict, None, None]:
    """
    Load a JSONL file and return a generator of dictionaries.

    Args:
        f: The file object to load.

    Returns:
        A generator of dictionaries.
    """
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        yield json.loads(line)


def parse_runtime_file(file_path: str, stage: str, accessor: Callable[[dict], list[dict]] | None = None) -> pd.DataFrame:
    """
    Parse a runtime file and extract timing data.

    Args:
        file_path: The path to the runtime file.
        stage: The stage of the runtime file.
        accessor: A function to access the data in the runtime file.

    Returns:
        A list of dictionaries containing the timing data for the given stage.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    timings: list[dict] = []
    ignored_ops = ['total_frame_time', 'read_frame', 'save_canvas', 'save_mapping_files']
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            data = jsonl_loader(f)
        else:
            assert file_path.endswith('.json'), \
                f"Expected a JSONL file, got {file_path}"
            # JSON format - array of objects
            data = json.load(f)
            assert isinstance(data, list), \
                f"Expected a list of objects, got {type(data)}, {stage}, {stage}"

        for entry in data:
            if accessor is not None:
                entry = accessor(entry)
            if isinstance(entry, dict):
                entry = [entry]

            assert isinstance(entry, list), \
                f"Expected a list of dictionaries, got {type(entry)}, {stage}, {file_path}"
            for item in entry:
                if 'time' in item and isinstance(item['time'], (int, float)):
                    # Convert from milliseconds to seconds
                    item_copy = item.copy()
                    item_copy['time'] = item['time'] / 1000.0
                    if item_copy['op'] in ignored_ops:
                        continue
                    assert isinstance(item_copy['time'], (int, float)), f"Time value is not a number for {item_copy}"
                    assert item_copy['time'] >= 0 or ('Perfect_' in file_path and item_copy['op'] == 'transform'), \
                        f"Time value is not positive for {item_copy}, {file_path}, {stage}, {item}"
                    assert isinstance(item_copy['op'], str), f"Operation is not a string for {item_copy}"
                    timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
    
    # assert len(timings) > 0, f"No timings found for {file_path}, {stage}"
    return pd.DataFrame.from_records(timings, columns=['stage', 'time', 'op'])


def aggregate_per_op(file_timings: pd.DataFrame) -> pd.DataFrame:
    op_aggregates = file_timings.groupby('op').agg(
        time=pd.NamedAgg(column='time', aggfunc='sum'),
        count=pd.NamedAgg(column='time', aggfunc='count'),
    ).reset_index()
    assert isinstance(op_aggregates, pd.DataFrame), \
        f"Op aggregates is not a pandas DataFrame for {file_timings}"
    return op_aggregates


INDEX_DATA_ACCESSORS = {
    '011_tune_detect': lambda row: row[-1],
    '012_tune_create_training_data': lambda row: row,
    '013_tune_train_classifier': lambda row: row
}

QUERY_DATA_ACCESSORS = {
    '001_preprocess_groundtruth_detection': lambda row: row,
    '002_preprocess_groundtruth_tracking': lambda row: row['runtime'],
    '020_exec_classify': lambda row: row,
    '022_exec_prune_polyominoes': lambda row: row['runtime'],
    '030_exec_compress': lambda row: row['runtime'],
    '040_exec_detect': lambda row: row,
    '050_exec_uncompress': lambda row: row,
    '060_exec_track': lambda row: row['runtime']
}

# Map throughput stages to their originating script paths so exclusion rules stay readable.
STAGE_TO_SCRIPT_PATH = {
    '001_preprocess_groundtruth_detection': 'preprocess/p003_preprocess_groundtruth_detection.py',
    '002_preprocess_groundtruth_tracking': 'preprocess/p003_preprocess_groundtruth_tracking.py',
    '011_tune_detect': 'scripts/p011_tune_detect.py',
    '012_tune_create_training_data': 'scripts/p012_tune_create_training_data.py',
    '013_tune_train_classifier': 'scripts/p014_tune_train_classifier.py',
    '020_exec_classify': 'scripts/p020_exec_classify.py',
    '022_exec_prune_polyominoes': 'scripts/p022_exec_prune_polyominoes.py',
    '030_exec_compress': 'scripts/p030_exec_compress.py',
    '040_exec_detect': 'scripts/p040_exec_detect.py',
    '050_exec_uncompress': 'scripts/p050_exec_uncompress.py',
    '060_exec_track': 'scripts/p060_exec_track.py',
}

# Keep global runtime exclusions here so the policy stays centralized and easy to extend.
EXCLUDED_RUNTIME_OPS_BY_SCRIPT = {
    'scripts/p022_exec_prune_polyominoes.py': {'solve_ilp'},
    'scripts/p030_exec_compress.py': {'save_collage', 'pack_all_total'},
}


def excluded_runtime_ops(stage: str) -> set[str]:
    # Resolve the originating script path for the current throughput stage.
    script_path = STAGE_TO_SCRIPT_PATH.get(stage)
    # Return an empty set when the stage has no explicit exclusion policy.
    if script_path is None:
        return set()

    # Return the configured excluded ops for this script path.
    return EXCLUDED_RUNTIME_OPS_BY_SCRIPT.get(script_path, set())


def filter_excluded_runtime_ops(file_timings: pd.DataFrame, stage: str) -> pd.DataFrame:
    # Resolve the stage-local exclusion policy once before filtering.
    excluded_ops = excluded_runtime_ops(stage)
    # Return the original timings unchanged when no exclusions are configured.
    if not excluded_ops:
        return file_timings

    # Drop the excluded operations while preserving all other timing rows.
    return file_timings[~file_timings['op'].isin(excluded_ops)].reset_index(drop=True)


def _valid_scalar(val: object) -> bool:
    """Return False if value is missing or NaN (for use in row dicts from DataFrame.to_dict())."""
    if val is None:
        return False
    if isinstance(val, float):
        return not math.isnan(val)
    return True


def _process_runtime_row(args: tuple[dict, str, int]) -> tuple[int, pd.DataFrame | None, dict]:
    """
    Process a single runtime row (for use with multiprocessing.Pool).
    Returns (row_index, per_op_df, overall_dict) so the caller can preserve order.
    Uses module-level INDEX_DATA_ACCESSORS or QUERY_DATA_ACCESSORS by name to avoid pickling lambdas.
    """
    row_dict, accessors_key, row_idx = args
    accessors = INDEX_DATA_ACCESSORS if accessors_key == 'index' else QUERY_DATA_ACCESSORS
    _sr = row_dict.get('sample_rate')
    sample_rate = _sr if ('sample_rate' in row_dict and _valid_scalar(_sr)) else None
    _tr = row_dict.get('tracker')
    tracker = _tr if ('tracker' in row_dict and _valid_scalar(_tr)) else None
    _cs = row_dict.get('canvas_scale')
    canvas_scale = _cs if ('canvas_scale' in row_dict and _valid_scalar(_cs)) else None
    _th = row_dict.get('tracking_accuracy_threshold')
    tracking_accuracy_threshold = _th if ('tracking_accuracy_threshold' in row_dict and _valid_scalar(_th)) else None
    _classifier = row_dict.get('classifier')
    classifier = _classifier if ('classifier' in row_dict and _valid_scalar(_classifier)) else None
    _tilesize = row_dict.get('tilesize')
    tilesize = _tilesize if ('tilesize' in row_dict and _valid_scalar(_tilesize)) else None
    _tilepadding = row_dict.get('tilepadding')
    tilepadding = _tilepadding if ('tilepadding' in row_dict and _valid_scalar(_tilepadding)) else None
    videoset = row_dict.get('videoset')
    variant = row_dict.get('variant')
    variant_id = row_dict.get('variant_id')
    dataset = row_dict['dataset']
    video = row_dict['video']
    stage = row_dict['stage']
    runtime_file = row_dict['runtime_file']
    file_timings = parse_runtime_file(runtime_file, stage, accessors[stage])
    assert file_timings is not None, f"File timings are None for {stage}, {runtime_file}, {video}"
    file_timings = filter_excluded_runtime_ops(file_timings, stage)
    assert isinstance(file_timings, pd.DataFrame), (
        f"File timings are not a pandas DataFrame for {stage}, {runtime_file}, {video}"
    )
    per_op = aggregate_per_op(file_timings)
    per_op['stage'] = stage
    per_op['dataset'] = dataset
    per_op['videoset'] = videoset
    per_op['video'] = video
    per_op['variant'] = variant
    per_op['variant_id'] = variant_id
    per_op['classifier'] = classifier
    per_op['tilesize'] = tilesize
    per_op['tilepadding'] = tilepadding
    per_op['sample_rate'] = sample_rate
    per_op['tracking_accuracy_threshold'] = tracking_accuracy_threshold
    per_op['canvas_scale'] = canvas_scale
    per_op['tracker'] = tracker
    total_time = float(per_op['time'].sum())
    overall = {
        'stage': stage,
        'dataset': dataset,
        'videoset': videoset,
        'video': video,
        'variant': variant,
        'variant_id': variant_id,
        'classifier': classifier,
        'tilesize': tilesize,
        'tilepadding': tilepadding,
        'sample_rate': sample_rate,
        'tracking_accuracy_threshold': tracking_accuracy_threshold,
        'canvas_scale': canvas_scale,
        'tracker': tracker,
        'time': total_time,
    }
    return (row_idx, per_op if len(per_op) > 0 else None, overall)


def parse_runtime(accessors_key: str, df: pd.DataFrame, accessors: dict[str, Callable[[dict], list[dict]]],
                  worker_id: int, command_queue: "mp.Queue") -> tuple[pd.DataFrame, pd.DataFrame]:
    all_per_op: list[pd.DataFrame] = []
    overall: list[dict] = []
    accessors_key = 'index' if accessors is INDEX_DATA_ACCESSORS else 'query'
    rows_with_key = [(row.to_dict(), accessors_key, idx) for idx, (_, row) in enumerate(df.iterrows())]
    total = len(rows_with_key)
    kwargs = {'completed': 0, 'total': total, 'description': f"{df['dataset'].unique()[0]} {accessors_key}"}
    device = f'cuda:{worker_id}'
    command_queue.put((device, kwargs))
    n_workers = max(1, min(mp.cpu_count() - 1, total))
    results_by_idx: dict[int, tuple[pd.DataFrame | None, dict]] = {}
    with mp.Pool(processes=n_workers) as pool:
        completed = 0
        for row_idx, per_op, overall_item in pool.imap_unordered(_process_runtime_row, rows_with_key, chunksize=1):
            results_by_idx[row_idx] = (per_op, overall_item)
            completed += 1
            command_queue.put((device, {'completed': completed}))
    for idx in range(total):
        per_op, overall_item = results_by_idx[idx]
        if per_op is not None:
            all_per_op.append(per_op)
        overall.append(overall_item)
    return pd.concat(all_per_op, ignore_index=True), pd.DataFrame.from_records(overall)


def save_measurements(index_per_op: pd.DataFrame, index_overall: pd.DataFrame,
                      query_per_op: pd.DataFrame, query_overall: pd.DataFrame,
                      output_dir: str, dataset: str):
    """Save processed timing measurements to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    index_per_op_file = os.path.join(output_dir, 'index_construction_per_op.csv')
    index_per_op.to_csv(index_per_op_file, index=False)
    # print(f"Saved index construction per operation measurements to: {index_per_op_file}")

    index_overall_file = os.path.join(output_dir, 'index_construction_overall.csv')
    index_overall.to_csv(index_overall_file, index=False)
    # print(f"Saved index construction overall measurements to: {index_overall_file}")

    query_per_op_file = os.path.join(output_dir, 'query_execution_per_op.csv')
    query_per_op.to_csv(query_per_op_file, index=False)
    # print(f"Saved query execution per operation measurements to: {query_per_op_file}")

    query_overall_file = os.path.join(output_dir, 'query_execution_overall.csv')
    query_overall.to_csv(query_overall_file, index=False)
    # print(f"Saved query execution overall measurements to: {query_overall_file}")
    
    # Save metadata
    videos = sorted(query_overall['video'].unique())
    # Extract the non-null split labels written by the manifest stage.
    videosets = sorted([videoset for videoset in query_overall['videoset'].dropna().unique()]) \
        if 'videoset' in query_overall.columns else []
    # Extract the non-null variant labels written by the manifest stage.
    variants = sorted([variant for variant in query_overall['variant'].dropna().unique()]) \
        if 'variant' in query_overall.columns else []
    # Extract trackers, filtering out null values so the naive baseline stays unparameterized.
    trackers = []
    if 'tracker' in query_overall.columns:
        trackers = sorted([tracker for tracker in query_overall['tracker'].unique() if tracker is not None and not pd.isna(tracker)])
    # Extract pruning thresholds while preserving the explicit no-threshold state.
    tracking_accuracy_thresholds: list[float | None] = []
    if 'tracking_accuracy_threshold' in query_overall.columns:
        threshold_values = query_overall['tracking_accuracy_threshold'].unique()
        has_none = any((threshold is None) or (isinstance(threshold, float) and pd.isna(threshold)) for threshold in threshold_values)
        parsed_thresholds = sorted(
            [float(threshold) for threshold in threshold_values if threshold is not None and not (isinstance(threshold, float) and pd.isna(threshold))]
        )
        if has_none:
            tracking_accuracy_thresholds.append(None)
        tracking_accuracy_thresholds.extend(parsed_thresholds)
    # Extract canvas scales, filtering out null values so the naive baseline stays unparameterized.
    canvas_scales = []
    if 'canvas_scale' in query_overall.columns:
        canvas_scales = sorted([float(canvas_scale) for canvas_scale in query_overall['canvas_scale'].unique() if canvas_scale is not None and not pd.isna(canvas_scale)])
    # Extract sample rates, filtering out null values so the naive baseline stays unparameterized.
    sample_rates = []
    if 'sample_rate' in query_overall.columns:
        sample_rates = sorted([int(sample_rate) for sample_rate in query_overall['sample_rate'].unique() if sample_rate is not None and not pd.isna(sample_rate)])
    # Extract classifiers, filtering out null values for the naive baseline and index stages.
    classifiers = []
    if 'classifier' in query_overall.columns:
        classifiers = sorted([classifier for classifier in query_overall['classifier'].unique() if classifier is not None and not pd.isna(classifier)])
    # Extract tile sizes, filtering out null values for the naive baseline and stage-011 rows.
    tilesizes = []
    if 'tilesize' in query_overall.columns:
        tilesizes = sorted([int(tilesize) for tilesize in query_overall['tilesize'].unique() if tilesize is not None and not pd.isna(tilesize)])
    # Extract tile padding values, filtering out null values for the naive baseline.
    tilepadding_values = []
    if 'tilepadding' in query_overall.columns:
        tilepadding_values = sorted([tilepadding for tilepadding in query_overall['tilepadding'].unique() if tilepadding is not None and not pd.isna(tilepadding)])
    metadata = {
        'dataset': dataset,
        'videos': videos,
        'videosets': videosets,
        'variants': variants,
        'classifiers': classifiers,
        'tilesizes': tilesizes,
        'tilepadding_values': tilepadding_values,
        'sample_rates': sample_rates,
        'tracking_accuracy_thresholds': tracking_accuracy_thresholds,
        'canvas_scales': canvas_scales,
        'trackers': trackers,
        'index_stages': sorted(index_overall['stage'].unique()),
        'query_stages': sorted(query_overall['stage'].unique()),
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    # print(f"Saved metadata to: {metadata_file}")
    
    # print(f"Found {len(videos)} videos: {videos}")


def compute(dataset: str, worker_id: int, command_queue: "mp.Queue[dict]"):
    data_dir = cache.eval(dataset, 'tp')
    index_data, query_data = load_data_tables(data_dir)
    
    index_timings, index_summaries = parse_runtime('index', index_data, INDEX_DATA_ACCESSORS, worker_id, command_queue)
    
    query_timings, query_summaries = parse_runtime('query', query_data, QUERY_DATA_ACCESSORS, worker_id, command_queue)
    
    measurements_dir = cache.eval(dataset, 'tp', 'measurements')
    save_measurements(index_timings, index_summaries, query_timings, query_summaries, measurements_dir, dataset)


def main():
    """Main function to process runtime measurement data."""
    tasks: list[Callable[[int, "mp.Queue"], None]] = []
    for dataset in DATASETS:
        tasks.append(partial(compute, dataset))

    ProgressBar(num_tasks=len(tasks), num_workers=mp.cpu_count()).run_all(tasks)


if __name__ == '__main__':
    main()
