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

import numpy as np
from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import Count, HOTA

from evaluation.manifests import build_split_video_manifest, load_sota_stat_manifest
from polyis.io import cache
from polyis.trackeval.dataset import Dataset
from polyis.utilities import dataset_root_name, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


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
    parser = argparse.ArgumentParser(description='Calculate HOTA scores for OTIF and LEAP tracking results')
    parser.add_argument('--no_parallel', action='store_true', default=False,
                        help='Whether to disable parallel processing')
    return parser.parse_args()


def validate_sota_task_inputs(dataset: str, system: str, param_id: int, videos: list[str]) -> tuple[str, str]:
    # Resolve the transformed SOTA tracking root for the current dataset/system pair.
    sota_dir = cache.sota(system, dataset)
    # Resolve the canonical groundtruth dataset used by this detector variant.
    gt_dataset = dataset_root_name(dataset)
    # Resolve the canonical groundtruth execution root.
    gt_execution_dir = cache.execution(gt_dataset)

    # Fail fast when the transformed SOTA root is missing.
    assert os.path.exists(sota_dir), f"SOTA directory {sota_dir} does not exist"
    # Fail fast when the canonical groundtruth root is missing.
    assert os.path.exists(gt_execution_dir), f"Groundtruth execution directory {gt_execution_dir} does not exist"

    # Validate the expected tracking and GT files for every configured test video.
    for video in videos:
        # Resolve the transformed SOTA tracking path for the current param/video pair.
        tracking_path = os.path.join(sota_dir, video, 'tracking_results', f'{param_id:03d}', 'tracking.jsonl')
        # Resolve the canonical GT tracking path for the current test video.
        groundtruth_path = os.path.join(gt_execution_dir, video, '003_groundtruth', 'tracking.jsonl')

        # Fail fast when the transformed tracking file is missing.
        assert os.path.exists(tracking_path), f"Tracking path {tracking_path} does not exist"
        # Fail fast when the canonical GT file is missing.
        assert os.path.exists(groundtruth_path), f"Groundtruth path {groundtruth_path} does not exist"

    return str(sota_dir), str(gt_execution_dir)


def evaluate_sota_tracking_accuracy(dataset: str,
                                    videos: list[str],
                                    param_id: int,
                                    system: str,
                                    output_dir: str,
                                    worker_id: int,
                                    worker_id_queue: "mp.Queue"):
    # Validate the expected split-aware tracking and GT inputs for this param_id.
    sota_dir, gt_execution_dir = validate_sota_task_inputs(dataset, system, param_id, videos)

    # Build the TrackEval dataset configuration using separate GT and tracker roots.
    dataset_config = {
        'output_fol': output_dir,
        'output_sub_fol': f"test_{param_id:03d}",
        'input_gt': os.path.join('003_groundtruth', 'tracking.jsonl'),
        'input_track': os.path.join('tracking_results', f'{param_id:03d}', 'tracking.jsonl'),
        'skip': 1,
        'tracker': f'{system}_{param_id:03d}',
        'seq_list': sorted(videos),
        'input_dir': sota_dir,
        'input_gt_dir': gt_execution_dir,
        'input_track_dir': sota_dir,
    }

    # Build the minimal TrackEval evaluator configuration for this task.
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

    # Build the TrackEval evaluator and dataset wrappers for this task.
    eval_dataset = Dataset(dataset_config)
    evaluator = trackeval.Evaluator(eval_config)

    # Suppress noisy TrackEval warnings so failures stay visible.
    with warnings.catch_warnings():
        # Silence TrackEval warning output for deterministic logs.
        warnings.filterwarnings("ignore")
        # Run TrackEval for the configured SOTA task.
        results = evaluator.evaluate([eval_dataset], [HOTA()])

    # Validate the TrackEval response structure and status.
    assert results and len(results) == 2, results
    assert results[1]['Dataset']['sort'] == 'Success', f"Evaluation failed: {results[1]}"

    # Extract the per-sequence tracker results from the nested TrackEval payload.
    tracker_results = results[0]['Dataset']['sort']
    expected_sequences = tuple(sorted(set(videos) | {'COMBINED_SEQ'}))
    actual_sequences = tuple(sorted(tracker_results.keys()))
    assert expected_sequences == actual_sequences, (
        f"Expected sequences {expected_sequences} do not match actual {actual_sequences}"
    )

    # Persist one JSON file per sequence so debugging retains per-video visibility.
    for seq, tracker_result in tracker_results.items():
        # Validate that TrackEval produced the vehicle-class payload.
        assert 'vehicle' in tracker_result, f"Vehicle results not found for {seq}"

        # Collect the HOTA and Count metric payloads for this sequence.
        seq_metrics: dict[str, object] = {}
        vehicle_results = tracker_result['vehicle']
        for metric in [HOTA(), Count()]:
            metric_name = metric.get_name()
            assert metric_name in vehicle_results, f"Metric {metric_name} not found in {vehicle_results}"
            seq_metrics[metric_name] = vehicle_results[metric_name]

        # Assemble the split-aware SOTA result row.
        result_data = {
            'video': None if seq == 'COMBINED_SEQ' else seq,
            'dataset': dataset,
            'videoset': 'test',
            'system': system,
            'param_id': param_id,
            'metrics': seq_metrics,
        }

        # Persist the combined row as DATASET.json and per-video rows under their filename.
        output_file = 'DATASET' if seq == 'COMBINED_SEQ' else seq
        with open(os.path.join(output_dir, f'{output_file}.json'), 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)

    # Return the worker slot once the task is complete.
    worker_id_queue.put(worker_id)


def build_sota_tasks(dataset: str, system: str) -> list[tuple[int, list[str], str]]:
    # Skip systems that have not been transformed for the current dataset.
    stat_path = cache.sota(system, dataset, 'stat.csv')
    if not os.path.exists(stat_path):
        return []

    # Load the transformed SOTA stat manifest to enumerate param IDs.
    stat_df = load_sota_stat_manifest(system, dataset)
    # Materialize the configured test videos from the dataset store.
    videos_df = build_split_video_manifest(datasets=[dataset], videosets=['test'])
    videos = sorted(videos_df['video'].drop_duplicates().tolist())

    # Fail fast when the configured test split has no videos.
    assert videos, f"No test videos found for dataset {dataset}"

    # Collect one validated accuracy task per SOTA param_id.
    task_rows: list[tuple[int, list[str], str]] = []
    for param_id in sorted(stat_df['param_id'].drop_duplicates().tolist()):
        # Validate the expected split-aware SOTA inputs before scheduling the task.
        validate_sota_task_inputs(dataset, system, int(param_id), videos)
        # Resolve the split-aware raw output directory for the current param_id.
        output_dir = os.path.join(cache.sota(system, dataset, 'accuracy'), 'raw', 'test', f'{int(param_id):03d}')
        task_rows.append((int(param_id), videos, output_dir))

    return task_rows


def main(args):
    # Log the configured datasets before SOTA evaluation starts.
    print(f"Starting OTIF and LEAP tracking accuracy evaluation for datasets: {DATASETS}")

    # Collect the split-aware SOTA accuracy tasks across datasets and systems.
    eval_tasks: list[Callable[[int, "mp.Queue"], None]] = []

    # Expand and schedule all configured dataset/system/task combinations.
    for dataset in DATASETS:
        for system in ['otif', 'leap']:
            task_rows = build_sota_tasks(dataset, system)
            if not task_rows:
                continue

            # Recreate the system-local accuracy directory before writing fresh results.
            evaluation_dir = cache.sota(system, dataset, 'accuracy')
            if os.path.exists(evaluation_dir):
                shutil.rmtree(evaluation_dir)
            os.makedirs(os.path.join(evaluation_dir, 'raw', 'test'), exist_ok=True)

            # Schedule one TrackEval task per SOTA param_id.
            for param_id, videos, output_dir in task_rows:
                eval_tasks.append(partial(
                    evaluate_sota_tracking_accuracy,
                    dataset,
                    videos,
                    param_id,
                    system,
                    output_dir,
                ))

    # Fail fast when no SOTA system produced any configured evaluation task.
    assert eval_tasks, "No OTIF or LEAP tracking results found. Please ensure p140_otif_transform.py has been run first."

    # Create the worker-id queue used to label concurrent evaluation tasks.
    worker_id_queue = mp.Queue()
    for worker_id in range(max(1, int(mp.cpu_count() * 0.8))):
        worker_id_queue.put(worker_id)

    # Run sequentially when the caller disables multiprocessing.
    if args.no_parallel:
        for eval_task in track(eval_tasks):
            worker_id = worker_id_queue.get()
            eval_task(worker_id, worker_id_queue)
        return

    # Track the spawned child processes for the parallel run.
    processes: list[mp.Process] = []

    # Start one process per validated SOTA task.
    for eval_task in eval_tasks:
        worker_id = worker_id_queue.get()
        process = mp.Process(target=eval_task, args=(worker_id, worker_id_queue))
        process.start()
        processes.append(process)

    # Wait for all child processes and terminate their resources.
    for process in track(processes):
        process.join()
        process.terminate()


if __name__ == '__main__':
    main(parse_args())
