#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import multiprocessing as mp
import sys
from typing import Callable

import numpy as np
from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import HOTA, CLEAR, Identity

from polyis.trackeval.dataset import Dataset
from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST


TILE_SIZES = [30, 60]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval and create visualizations')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR',  #,Identity',
                        help='Comma-separated list of metrics to evaluate')
    parser.add_argument('--no_parallel', action='store_true', default=False,
                        help='Whether to disable parallel processing')
    return parser.parse_args()


def find_tracking_results(cache_dir: str, dataset: str) -> tuple[set[str], set[tuple[str, int]]]:
    """
    Find all video files with tracking results for the specified dataset and tile size.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        tuple[set[str], set[tuple[str, int]]]: Set of video names and set of (classifier, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset, 'execution')
    assert os.path.exists(dataset_cache_dir), f"Dataset cache directory {dataset_cache_dir} does not exist"
    
    video_tile_combinations: list[tuple[str, str, int]] = []
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        assert os.path.isdir(video_dir), f"Video directory {video_dir} is not a directory"

        tracking_dir = os.path.join(video_dir, '060_uncompressed_tracks')
        assert os.path.exists(tracking_dir), f"Tracking directory {tracking_dir} does not exist"

        for classifier_tilesize in os.listdir(tracking_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            tracking_path = os.path.join(tracking_dir, f'{classifier}_{ts}', 'tracking.jsonl')
            groundtruth_path = os.path.join(video_dir, '000_groundtruth', 'tracking.jsonl')
            
            assert os.path.exists(tracking_path), f"Tracking path {tracking_path} does not exist"
            assert os.path.exists(groundtruth_path), f"Groundtruth path {groundtruth_path} does not exist"
            video_tile_combinations.append((video_filename, classifier, ts))
            print(f"Found tracking results: {video_filename} with tile size {ts}")
    
    classifier_tilesizes = set((cl, ts) for _, cl, ts in video_tile_combinations)
    videos = set(video for video, _, _ in video_tile_combinations)

    video_tile_combinations_set = set(video_tile_combinations)
    assert len(video_tile_combinations_set) == len(video_tile_combinations), \
        f"Duplicate video-tile combinations: {video_tile_combinations_set}"
    for video in videos:
        for cl, ts in classifier_tilesizes:
            assert (video, cl, ts) in video_tile_combinations_set, \
                f"Video-tile combination {video}-{cl}-{ts} not found"

    return videos, classifier_tilesizes


def evaluate_tracking_accuracy(dataset: str, video_name: str, classifier: str,
                               tile_size: int, metrics_list: list[str], 
                               output_dir: str) -> dict:
    """
    Evaluate tracking accuracy for a single video using TrackEval.
    
    Args:
        dataset (str): Dataset name
        video_name (str): Name of the video
        classifier (str): Classifier used
        tile_size (int): Tile size used
        metrics_list (List[str]): List of metrics to evaluate
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"Evaluating {video_name} with tile size {tile_size}")

    clts = f'{classifier}_{tile_size}'

    # Create dataset configuration
    dataset_config = {
        'output_fol': output_dir,
        'output_sub_fol': f'{video_name}_{clts}',
        'input_gt': os.path.join('000_groundtruth', 'tracking.jsonl'),
        'input_track': os.path.join('060_uncompressed_tracks', clts, 'tracking.jsonl'),
        'skip': 1,  # Process every frame
        'tracker': clts,
        'seq_list': [video_name],
        'input_dir': os.path.join(CACHE_DIR, dataset, 'execution')
    }
    
    # Create evaluator configuration
    eval_config = {
        'USE_PARALLEL': False,
        # 'NUM_PARALLEL_CORES': int(mp.cpu_count() * 0.8),
        'BREAK_ON_ERROR': True,
        'LOG_ON_ERROR': f'{dataset}_{video_name}_{clts}_error_log.txt',
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
        'OUTPUT_EMPTY_CLASSES': False,
    }
    
    # Create metrics
    metrics = []
    for metric_name in metrics_list:
        if metric_name == 'HOTA':
            metrics.append(HOTA({'THRESHOLD': 0.5}))
        elif metric_name == 'CLEAR':
            metrics.append(CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}))
        elif metric_name == 'Identity':
            metrics.append(Identity({'THRESHOLD': 0.5}))
    
    # Create dataset and evaluator
    eval_dataset = Dataset(dataset_config)
    evaluator = trackeval.Evaluator(eval_config)
    
    # Run evaluation
    results = evaluator.evaluate([eval_dataset], metrics)
    
    # Extract summary results
    summary_results = {}
    
    # The actual structure is: results[0]["Dataset"]["sort"]["COMBINED_SEQ"]["vehicle"]
    assert results and len(results) == 2, results

    # Get the first result which contains the B3D evaluation
    b3d_result = results[0].get('Dataset', {})
    sort_result = b3d_result.get('sort', {})
    
    # Look for the COMBINED_SEQ results which contain the vehicle class
    assert 'COMBINED_SEQ' in sort_result, sort_result
    assert 'vehicle' in sort_result['COMBINED_SEQ'], sort_result

    vehicle_results = sort_result['COMBINED_SEQ']['vehicle']
    
    # Extract metrics from the vehicle results
    for metric in metrics:
        metric_name = metric.get_name()
        if metric_name in vehicle_results:
            summary_results[metric_name] = vehicle_results[metric_name]
    
    result = {
        'video_name': video_name,
        'tile_size': tile_size,
        'classifier': classifier,
        'metrics': summary_results,
        'success': True,
        'output_dir': output_dir,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)


def main(args):
    """
    Main function that orchestrates the tracking accuracy evaluation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with tracking results for the specified datasets and tile size(s)
    2. Runs accuracy evaluation using TrackEval's B3D evaluation methods
    3. Creates summary reports of the accuracy results
    4. Optionally creates visualizations if requested and libraries are available
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tracking results from 060_exec_track.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/060_uncompressed_tracks/{classifier}_{tile_size}/tracking.jsonl
        - Groundtruth data should be in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl
        - Multiple metrics are evaluated: HOTA, CLEAR (MOTA), and Identity (IDF1)
    """
    print(f"Starting tracking accuracy evaluation for datasets: {args.datasets}")
    
    # Parse metrics (needed for both compute and no_recompute modes)
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    print(f"Evaluating metrics: {metrics_list}")
    
    # Find tracking results for all datasets
    eval_tasks: list[Callable[[], dict]] = []
    for dataset in args.datasets:
        print(f"Processing dataset: {dataset}")
        videos, classifier_tilesizes = find_tracking_results(CACHE_DIR, dataset)
        execution_dir = os.path.join(CACHE_DIR, dataset, 'execution')
        # Add dataset info to each combination
        for video in videos:
            for cl, ts in classifier_tilesizes:
                output_dir = os.path.join(execution_dir, video, '070_tracking_accuracy',
                                          f'{cl}_{ts}', 'accuracy')
                eval_tasks.append(partial(evaluate_tracking_accuracy, dataset, video,
                                          cl, ts, metrics_list, output_dir))
    
    assert len(eval_tasks) > 0, "No tracking results found. Please ensure 060_exec_track.py has been run first."
    print(f"Found {len(eval_tasks)} video-tile size combinations to evaluate")
    
    if args.no_parallel:
        for eval_task in eval_tasks:
            eval_task()
    else:
        processes: list[mp.Process] = []
        for eval_task in eval_tasks:
            process = mp.Process(target=eval_task)
            process.start()
            processes.append(process)

        for process in track(processes):
            process.join()
            process.terminate()


if __name__ == '__main__':
    main(parse_args())
