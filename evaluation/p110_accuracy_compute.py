#!/usr/local/bin/python

import argparse
from functools import partial
import json
import multiprocessing as mp
import os
import shutil
import sys
from typing import Callable, override
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import CLEAR, Count, HOTA, Identity

from evaluation.manifests import build_query_task_manifest
from polyis.io import cache
from polyis.trackeval.dataset import Dataset
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
PARAM_COLUMNS = [
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
]


class NumpyEncoder(json.JSONEncoder):
    @override
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval')
    parser.add_argument('--metrics', type=str, default='HOTA',
                        help='Comma-separated list of metrics to evaluate')
    parser.add_argument('--no_parallel', action='store_true', default=False,
                        help='Whether to disable parallel processing')
    return parser.parse_args()


def resolve_groundtruth_tracking_dataset(dataset: str) -> str:
    # Map caldot1 detector variants to the shared caldot1 groundtruth tracks.
    if dataset.startswith('caldot1-y'):
        return 'caldot1'
    # Map caldot2 detector variants to the shared caldot2 groundtruth tracks.
    if dataset.startswith('caldot2-y'):
        return 'caldot2'
    # Map ams detector variants to the shared ams groundtruth tracks.
    if dataset.startswith('ams-y'):
        return 'ams'
    # Keep all other datasets on their own groundtruth tracks.
    return dataset


def resolve_reference_tracking_source(videoset: str,
                                      track_execution_dir: str | Path,
                                      groundtruth_execution_dir: str | Path) -> tuple[str, str]:
    # Use validation pseudo-groundtruth source for the validation split.
    if videoset == 'valid':
        return str(track_execution_dir), os.path.join('002_naive', 'tracking.jsonl')

    # Accept only non-validation splits that use canonical groundtruth tracking.
    assert videoset in {'test', 'train'}, f"Unknown videoset: {videoset}"
    return str(groundtruth_execution_dir), os.path.join('003_groundtruth', 'tracking.jsonl')


def normalize_param_value(value: object) -> object | None:
    # Convert pandas missing values into plain None for JSON serialization and path logic.
    if pd.isna(value):
        return None
    return value


def extract_task_metadata(task_row: dict) -> dict:
    # Materialize the shared metadata payload for accuracy result rows.
    return {
        'dataset': task_row['dataset'],
        'videoset': task_row['videoset'],
        'variant': task_row['variant'],
        'variant_id': task_row['variant_id'],
        'classifier': normalize_param_value(task_row['classifier']),
        'tilesize': normalize_param_value(task_row['tilesize']),
        'sample_rate': normalize_param_value(task_row['sample_rate']),
        'tracking_accuracy_threshold': normalize_param_value(task_row['tracking_accuracy_threshold']),
        'tilepadding': normalize_param_value(task_row['tilepadding']),
        'canvas_scale': normalize_param_value(task_row['canvas_scale']),
        'tracker': normalize_param_value(task_row['tracker']),
    }


def resolve_prediction_relative_path(task_row: dict) -> str:
    # Route the naive baseline to the preprocess tracking output without fake params.
    if task_row['variant'] == 'naive':
        return os.path.join('002_naive', 'tracking.jsonl')

    # Route Polytris configurations to their stage-060 tracking output.
    return os.path.join('060_uncompressed_tracks', task_row['variant_id'], 'tracking.jsonl')


def validate_task_inputs(task_row: dict, videos: list[str]) -> tuple[str, str, str]:
    # Resolve the prediction execution root for the evaluated dataset.
    track_execution_dir = cache.execution(task_row['dataset'])
    # Resolve the canonical groundtruth dataset for the evaluated detector variant.
    groundtruth_dataset = resolve_groundtruth_tracking_dataset(task_row['dataset'])
    # Resolve the canonical groundtruth execution root.
    groundtruth_execution_dir = cache.execution(groundtruth_dataset)

    # Validate that the prediction execution root already exists.
    assert os.path.exists(track_execution_dir), f"Tracking execution directory {track_execution_dir} does not exist"
    # Validate that the groundtruth execution root already exists.
    assert os.path.exists(groundtruth_execution_dir), (
        f"Groundtruth execution directory {groundtruth_execution_dir} does not exist"
    )

    # Resolve the GT base directory and the relative GT file path for this split.
    input_gt_dir, input_gt_rel = resolve_reference_tracking_source(
        task_row['videoset'],
        track_execution_dir,
        groundtruth_execution_dir,
    )
    # Resolve the relative prediction path for this variant.
    input_track_rel = resolve_prediction_relative_path(task_row)

    # Validate all required GT and prediction files before scheduling TrackEval.
    for video in videos:
        # Build the GT file path for the current video.
        gt_source_path = os.path.join(input_gt_dir, video, input_gt_rel)
        # Build the prediction file path for the current video.
        tracking_source_path = os.path.join(track_execution_dir, video, input_track_rel)

        # Fail fast when any expected GT file is missing.
        assert os.path.exists(gt_source_path), f"Reference tracking path {gt_source_path} does not exist"
        # Fail fast when any expected prediction file is missing.
        assert os.path.exists(tracking_source_path), f"Tracking source path {tracking_source_path} does not exist"

    return str(track_execution_dir), input_gt_dir, input_gt_rel


def build_metric_objects(metrics_list: list[str]) -> list[HOTA | CLEAR | Identity]:
    # Collect the requested TrackEval metric instances in a deterministic order.
    metrics: list[HOTA | CLEAR | Identity] = []

    # Expand each requested metric name into the corresponding TrackEval class.
    for metric_name in metrics_list:
        # Add the HOTA metric when requested.
        if metric_name == 'HOTA':
            metrics.append(HOTA())
        # Add the CLEAR metric when requested.
        elif metric_name == 'CLEAR':
            metrics.append(CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}))
        # Add the Identity metric when requested.
        elif metric_name == 'Identity':
            metrics.append(Identity({'THRESHOLD': 0.5}))

    return metrics


def evaluate_tracking_accuracy(task_row: dict,
                               videos: list[str],
                               metrics_list: list[str],
                               output_dir: str,
                               worker_id: int,
                               worker_id_queue: "mp.Queue"):
    # Normalize the metadata for this split-level task once up front.
    metadata = extract_task_metadata(task_row)
    # Validate the expected inputs and capture the resolved input directories.
    track_execution_dir, input_gt_dir, input_gt = validate_task_inputs(task_row, videos)
    # Resolve the relative prediction path for this variant.
    input_track = resolve_prediction_relative_path(task_row)

    # Build the TrackEval dataset configuration for this split-level task.
    dataset_config = {
        'output_fol': output_dir,
        'output_sub_fol': f"{task_row['videoset']}_{task_row['variant_id']}",
        'input_gt': input_gt,
        'input_track': input_track,
        'skip': 1,
        'tracker': task_row['variant_id'],
        'seq_list': sorted(videos),
        'input_dir': track_execution_dir,
        'input_gt_dir': input_gt_dir,
        'input_track_dir': track_execution_dir,
    }

    # Build the TrackEval evaluator configuration with simple per-task execution.
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': min(mp.cpu_count(), len(videos)),
        'BREAK_ON_ERROR': True,
        'LOG_ON_ERROR': os.path.join(output_dir, 'LOG.txt'),
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False,
    }

    # Ensure the destination directory exists before TrackEval writes results.
    os.makedirs(output_dir, exist_ok=True)

    # Materialize the requested metric objects once per task.
    metrics = build_metric_objects(metrics_list)
    # Construct the split-aware TrackEval dataset wrapper.
    eval_dataset = Dataset(dataset_config)
    # Construct the TrackEval evaluator instance.
    evaluator = trackeval.Evaluator(eval_config)

    # Suppress noisy TrackEval warnings so failures stay visible.
    with warnings.catch_warnings():
        # Silence TrackEval warning output for deterministic logs.
        warnings.filterwarnings("ignore")
        # Run TrackEval for the configured dataset and metrics.
        results = evaluator.evaluate([eval_dataset], metrics)

    # Validate the TrackEval response structure before extracting results.
    assert results and len(results) == 2, results
    # Validate the TrackEval execution status.
    assert results[1]['Dataset']['sort'] == 'Success', f"Evaluation failed: {results[1]}"

    # Extract the per-sequence tracker results from the nested TrackEval payload.
    tracker_results = results[0].get('Dataset', {}).get('sort', {})
    # Validate that TrackEval produced one combined row plus one row per video.
    expected_sequences = tuple(sorted(set(videos) | {'COMBINED_SEQ'}))
    actual_sequences = tuple(sorted(tracker_results.keys()))
    assert expected_sequences == actual_sequences, (
        f"Expected sequences {expected_sequences} do not match actual {actual_sequences}"
    )

    # Persist one JSON file per sequence so debugging retains per-video visibility.
    for seq, tracker_result in tracker_results.items():
        # Validate that the dataset wrapper produced the vehicle class payload.
        assert 'vehicle' in tracker_result, f"Vehicle results not found for {seq}"

        # Collect the requested metrics plus Count into a serializable payload.
        seq_metrics: dict[str, object] = {}
        vehicle_results = tracker_result['vehicle']
        for metric in metrics + [Count()]:
            # Resolve the canonical TrackEval metric name.
            metric_name = metric.get_name()
            # Validate that the metric is present in the TrackEval output.
            assert metric_name in vehicle_results, f"Metric {metric_name} not found in {vehicle_results}"
            # Store the metric payload for later aggregation and visualization.
            seq_metrics[metric_name] = vehicle_results[metric_name]

        # Assemble the canonical split-aware result row.
        result_data = {
            'video': None if seq == 'COMBINED_SEQ' else seq,
            'metrics': seq_metrics,
            **metadata,
        }

        # Name the combined split-level file `DATASET.json` and keep video filenames unchanged.
        output_file = 'DATASET' if seq == 'COMBINED_SEQ' else seq
        with open(os.path.join(output_dir, f'{output_file}.json'), 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)

    # Return the worker slot once the task is complete.
    worker_id_queue.put(worker_id)


def build_accuracy_task_rows(dataset: str) -> list[tuple[dict, list[str], str]]:
    # Materialize the dataset-local query manifest with both Polytris and naive variants.
    manifest_df = build_query_task_manifest(datasets=[dataset], include_naive=True)
    # Fail fast when the configured dataset has no manifest rows.
    assert not manifest_df.empty, f"No query task manifest rows found for dataset {dataset}"

    # Group the manifest into one split-level task per dataset/videoset/variant.
    task_groups = manifest_df.groupby(['dataset', 'videoset', 'variant', 'variant_id'], dropna=False, sort=True)
    task_rows: list[tuple[dict, list[str], str]] = []

    # Convert each grouped DataFrame into task metadata plus its concrete video list.
    for _, group_df in task_groups:
        # Resolve the unique videos for this split-level task.
        videos = sorted(group_df['video'].drop_duplicates().tolist())
        # Require at least one concrete video for every task.
        assert videos, f"No videos found for task group:\n{group_df}"

        # Materialize the first row as the shared split-level task metadata.
        task_row = group_df.iloc[0].to_dict()
        # Build the canonical raw output directory for this split-level task.
        output_dir = os.path.join(cache.eval(dataset, 'acc'), 'raw', task_row['videoset'], task_row['variant_id'])
        # Validate all required files before the task enters the worker queue.
        validate_task_inputs(task_row, videos)
        # Store the fully validated task tuple.
        task_rows.append((task_row, videos, output_dir))

    return task_rows


def main(args):
    # Log the configured datasets so the run remains easy to audit.
    print(f"Starting tracking accuracy evaluation for datasets: {DATASETS}")

    # Split the comma-separated metric string into a normalized list.
    metrics_list = [metric_name.strip() for metric_name in args.metrics.split(',') if metric_name.strip()]
    # Log the normalized metric selection.
    print(f"Evaluating metrics: {metrics_list}")

    # Collect the split-level evaluation tasks across all configured datasets.
    eval_tasks: list[Callable[[int, "mp.Queue"], None]] = []

    # Expand the validated task list for each configured dataset.
    for dataset in DATASETS:
        # Resolve the dataset-local evaluation directory.
        evaluation_dir = cache.eval(dataset, 'acc')
        # Remove stale accuracy outputs before scheduling the new run.
        if os.path.exists(evaluation_dir):
            shutil.rmtree(evaluation_dir)
        # Recreate the raw accuracy root eagerly so later writes are simple.
        os.makedirs(evaluation_dir / 'raw', exist_ok=True)

        # Schedule one worker task per validated split-level variant.
        for task_row, videos, output_dir in build_accuracy_task_rows(dataset):
            eval_tasks.append(partial(
                evaluate_tracking_accuracy,
                task_row,
                videos,
                metrics_list,
                output_dir,
            ))

    # Fail fast when no tasks were materialized from the configured manifest.
    assert eval_tasks, "No tracking accuracy tasks found. Please ensure the execution pipeline has been run first."
    # Log the total number of split-level TrackEval tasks.
    print(f"Found {len(eval_tasks)} split-level accuracy evaluation tasks")

    # Create the worker-id queue used to label concurrent TrackEval tasks.
    worker_id_queue = mp.Queue()
    # Seed the queue with a bounded set of logical worker identifiers.
    for worker_id in range(max(1, int(mp.cpu_count() * 0.8))):
        worker_id_queue.put(worker_id)

    # Run sequentially when the caller disables multiprocessing.
    if args.no_parallel:
        for eval_task in track(eval_tasks):
            # Acquire the next worker slot for the in-process task.
            worker_id = worker_id_queue.get()
            # Execute the TrackEval task in the current process.
            eval_task(worker_id, worker_id_queue)
        return

    # Track the spawned child processes for the parallel run.
    processes: list[mp.Process] = []

    # Start one process per validated TrackEval task.
    for eval_task in eval_tasks:
        # Acquire the next worker slot before spawning the task.
        worker_id = worker_id_queue.get()
        # Create the child process for the current TrackEval task.
        process = mp.Process(target=eval_task, args=(worker_id, worker_id_queue))
        # Start the child process immediately.
        process.start()
        # Record the child process for later cleanup.
        processes.append(process)

    # Wait for all child processes and terminate their resources.
    for process in track(processes):
        # Block until the child process completes.
        process.join()
        # Ensure the process object is cleaned up after completion.
        process.terminate()


if __name__ == '__main__':
    main(parse_args())
