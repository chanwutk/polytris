#!/usr/local/bin/python

from functools import partial
import os
import json
import argparse
import queue
from typing import Callable, Generator, TextIO, Tuple
import multiprocessing as mp

import pandas as pd

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, ProgressBar


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process runtime measurement data for throughput analysis')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_data_tables(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the index construction and query execution data tables."""
    index_file = os.path.join(data_dir, 'index_construction.csv')
    query_file = os.path.join(data_dir, 'query_execution.csv')
    
    index_data = pd.read_csv(index_file)
    query_data = pd.read_csv(query_file)
    
    print(f"Loaded {len(index_data)} index construction entries")
    print(f"Loaded {len(query_data)} query execution entries")
    
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
                    assert item_copy['time'] > 0 or ('Perfect_' in file_path and item_copy['op'] == 'transform'), \
                        f"Time value is not positive for {item_copy}, {file_path}, {stage}"
                    assert isinstance(item_copy['op'], str), f"Operation is not a string for {item_copy}"
                    timings.append({'stage': stage, 'time': item_copy['time'], 'op': item_copy['op']})
    
    assert len(timings) > 0, f"No timings found for {file_path}, {stage}"
    return pd.DataFrame.from_records(timings)


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
    '001_preprocess_groundtruth_detection': lambda row: row['runtime'],
    '002_preprocess_groundtruth_tracking': lambda row: row['runtime'],
    '020_exec_classify': lambda row: row['runtime'],
    '030_exec_compress': lambda row: row['runtime'],
    '040_exec_detect': lambda row: row,
    '050_exec_uncompress': lambda row: row,
    '060_exec_track': lambda row: row['runtime']
}


def parse_runtime(df: pd.DataFrame, accessors: dict[str, Callable[[dict], list[dict]]],
                  worker_id: int, command_queue: "mp.Queue") -> tuple[pd.DataFrame, pd.DataFrame]:
    all_per_op: list[pd.DataFrame] = []
    overall: list[dict] = []

    kwargs = {'completed': 0,
            'total': len(df),
            'description': f"{df['dataset'].unique()[0]}",
            'completed': 0}
    device = f'cuda:{worker_id}'
    command_queue.put((device, kwargs))
    for idx, row in df.iterrows():
        dataset, video, classifier, tilesize, tilepadding, stage, runtime_file = row
        file_timings = parse_runtime_file(runtime_file, stage, accessors[stage])
        assert file_timings is not None, f"File timings are None for {stage}, {runtime_file}, {video}"

        per_op = aggregate_per_op(file_timings)
        per_op['stage'] = stage
        per_op['dataset'] = dataset
        per_op['video'] = video
        per_op['classifier'] = classifier
        per_op['tilesize'] = tilesize
        per_op['tilepadding'] = tilepadding
        
        all_per_op.append(per_op)

        total_time = int(per_op['time'].sum())
        overall.append({
            'stage': stage,
            'dataset': dataset,
            'video': video,
            'classifier': classifier,
            'tilesize': tilesize,
            'tilepadding': tilepadding,
            'time': total_time
        })
        command_queue.put((device, {'completed': idx}))

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
    metadata = {
        'dataset': dataset,
        'videos': sorted(query_overall['video'].unique()),
        'classifiers': sorted(query_overall['classifier'].unique()),
        # np.int64 cannot be serialized to JSON
        'tilesizes': sorted(int(ts) for ts in query_overall['tilesize'].unique()),
        'tilepadding_values': sorted(query_overall['tilepadding'].unique()),
        'index_stages': sorted(index_overall['stage'].unique()),
        'query_stages': sorted(query_overall['stage'].unique()),
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    # print(f"Saved metadata to: {metadata_file}")
    
    # print(f"Found {len(videos)} videos: {videos}")


def compute(dataset: str, worker_id: int, command_queue: "mp.Queue[dict]"):
    data_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput')
    index_data, query_data = load_data_tables(data_dir)
    
    index_timings, index_summaries = parse_runtime(index_data, INDEX_DATA_ACCESSORS, worker_id, command_queue)
    
    query_timings, query_summaries = parse_runtime(query_data, QUERY_DATA_ACCESSORS, worker_id, command_queue)
    
    measurements_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '080_throughput', 'measurements')
    save_measurements(index_timings, index_summaries, query_timings, query_summaries, measurements_dir, dataset)


def main():
    """Main function to process runtime measurement data."""
    args = parse_args()
    
    tasks: list[Callable[[int, "mp.Queue"], None]] = []
    for dataset in args.datasets:
        tasks.append(partial(compute, dataset))

    ProgressBar(num_tasks=len(tasks), num_workers=mp.cpu_count()).run_all(tasks)


if __name__ == '__main__':
    main()