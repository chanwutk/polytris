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
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval')
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
    Find all videos and classifier/tile_size combinations with tracking results.
    
    Scans the execution directory to discover all available videos and their
    corresponding classifier/tile_size combinations that have both tracking
    results and groundtruth data available.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        tuple[set[str], set[tuple[str, int]]]: Set of video names and set of (classifier, tile_size) tuples
    """
    # Construct path to dataset execution directory
    dataset_cache_dir = os.path.join(cache_dir, dataset, 'execution')
    assert os.path.exists(dataset_cache_dir), f"Dataset cache directory {dataset_cache_dir} does not exist"
    
    # Collect all video-classifier-tile_size combinations
    video_tile_combinations: list[tuple[str, str, int]] = []
    
    # Iterate through all video directories in the dataset
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        assert os.path.isdir(video_dir), f"Video directory {video_dir} is not a directory"

        # Check for tracking results directory
        tracking_dir = os.path.join(video_dir, '060_uncompressed_tracks')
        assert os.path.exists(tracking_dir), f"Tracking directory {tracking_dir} does not exist"

        # Iterate through all classifier-tile_size combinations
        for classifier_tilesize in os.listdir(tracking_dir):
            # Parse classifier and tile size from directory name
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            
            # Construct paths to tracking and groundtruth files
            tracking_path = os.path.join(tracking_dir, f'{classifier}_{ts}', 'tracking.jsonl')
            groundtruth_path = os.path.join(video_dir, '000_groundtruth', 'tracking.jsonl')
            
            # Verify both tracking results and groundtruth exist
            assert os.path.exists(tracking_path), f"Tracking path {tracking_path} does not exist"
            assert os.path.exists(groundtruth_path), f"Groundtruth path {groundtruth_path} does not exist"
            
            # Add this combination to our list
            video_tile_combinations.append((video_filename, classifier, ts))
            print(f"Found tracking results: {video_filename} with tile size {ts}")
    
    # Extract unique classifier-tile_size combinations and video names
    classifier_tilesizes = set((cl, ts) for _, cl, ts in video_tile_combinations)
    videos = set(video for video, _, _ in video_tile_combinations)

    # Validate that all videos have results for all classifier-tile_size combinations
    # This ensures we have complete data for multi-video evaluation
    video_tile_combinations_set = set(video_tile_combinations)
    assert len(video_tile_combinations_set) == len(video_tile_combinations), \
        f"Duplicate video-tile combinations: {video_tile_combinations_set}"
    
    # Check completeness: every video should have results for every classifier-tile_size combination
    for video in videos:
        for cl, ts in classifier_tilesizes:
            assert (video, cl, ts) in video_tile_combinations_set, \
                f"Video-tile combination {video}-{cl}-{ts} not found"

    return videos, classifier_tilesizes


def evaluate_tracking_accuracy(dataset: str, videos: set[str], classifier: str,
                               tile_size: int, metrics_list: list[str], output_dir: str):
    """
    Evaluate tracking accuracy for multiple videos using TrackEval.
    
    Performs a single evaluation across all videos in the dataset for the given
    classifier and tile size combination. Generates both combined dataset results
    and individual video results in a flattened directory structure.
    
    Args:
        dataset (str): Dataset name
        videos (set[str]): Set of video names to evaluate (all videos in dataset)
        classifier (str): Classifier used
        tile_size (int): Tile size used
        metrics_list (list[str]): List of metrics to evaluate
        output_dir (str): Output directory for results
        
    Output Structure:
        - DATASET.json: Combined results across all videos
        - {video_name}.json: Individual video results
        - LOG.txt: Evaluation logs and errors
    """
    print(f"Evaluating {len(videos)} videos with classifier {classifier} and tile size {tile_size}")

    # Create classifier-tile_size identifier for naming
    clts = f'{classifier}_{tile_size}'

    # Create TrackEval dataset configuration
    # This configures how TrackEval will find and process the data files
    dataset_config = {
        'output_fol': output_dir,  # Where TrackEval will write its output
        'output_sub_fol': f'{dataset}_{clts}',  # Subdirectory name for this evaluation
        'input_gt': os.path.join('000_groundtruth', 'tracking.jsonl'),  # Relative path to groundtruth files
        'input_track': os.path.join('060_uncompressed_tracks', clts, 'tracking.jsonl'),  # Relative path to tracking files
        'skip': 1,  # Process every frame (no frame skipping)
        'tracker': clts,  # Tracker name identifier
        'seq_list': videos,  # List of sequences (videos) to evaluate
        'input_dir': os.path.join(CACHE_DIR, dataset, 'execution')  # Base directory for relative paths
    }
    
    # Create TrackEval evaluator configuration
    # This controls how the evaluation is performed and what output is generated
    eval_config = {
        'USE_PARALLEL': True,  # Enable parallel processing within TrackEval
        'NUM_PARALLEL_CORES': min(mp.cpu_count(), len(videos)),  # Limit cores to number of videos
        'BREAK_ON_ERROR': True,  # Stop evaluation if any error occurs
        'LOG_ON_ERROR': os.path.join(output_dir, 'LOG.txt'),  # Save error logs to this file
        'PRINT_RESULTS': False,  # Don't print results to console
        'PRINT_CONFIG': False,  # Don't print configuration to console
        'TIME_PROGRESS': False,  # Don't show time progress
        'OUTPUT_SUMMARY': False,  # Don't generate summary output files
        'OUTPUT_DETAILED': False,  # Don't generate detailed output files
        'PLOT_CURVES': False,  # Don't generate plot curves
        'OUTPUT_EMPTY_CLASSES': False,  # Don't output results for empty classes
    }
    
    # Create TrackEval metric objects based on requested metrics
    # Each metric is configured with appropriate thresholds and settings
    metrics = []
    for metric_name in metrics_list:
        if metric_name == 'HOTA':
            # Higher Order Tracking Accuracy metric with 0.5 IoU threshold
            metrics.append(HOTA({'THRESHOLD': 0.5}))
        elif metric_name == 'CLEAR':
            # CLEAR metrics (MOTA, MOTP, etc.) with 0.5 IoU threshold
            metrics.append(CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}))
        elif metric_name == 'Identity':
            # Identity metrics (IDF1, etc.) with 0.5 IoU threshold
            metrics.append(Identity({'THRESHOLD': 0.5}))
    
    # Create TrackEval dataset and evaluator objects
    # The dataset object handles data loading and preprocessing
    eval_dataset = Dataset(dataset_config)
    # The evaluator object handles the actual evaluation process
    evaluator = trackeval.Evaluator(eval_config)
    
    # Run the evaluation across all videos simultaneously
    # This returns results for both individual videos and combined dataset
    results = evaluator.evaluate([eval_dataset], metrics)
    
    # TrackEval returns results in structure: results[0]["Dataset"]["sort"][sequence]["vehicle"]
    # where sequence can be individual video names or "COMBINED_SEQ" for aggregated results
    # results[0] contains the actual evaluation results
    # results[1] contains the evaluation status
    assert results and len(results) == 2, results
    assert results[1]['Dataset']['sort'] == 'Success', f"Evaluation failed: {results[1]}"

    # Extract evaluation results from TrackEval output structure
    # Navigate through the nested result structure to get to the actual data
    dataset_result = results[0].get('Dataset', {})
    tracker_results = dataset_result.get('sort', {})
    
    # Ensure output directory exists for saving results
    os.makedirs(output_dir, exist_ok=True)

    # Validate that all expected sequences are present in results
    # This ensures we have results for all videos plus the combined result
    expected_sequences = tuple(sorted(videos | {'COMBINED_SEQ'}))
    actual_sequences = tuple(sorted(tracker_results.keys()))
    assert expected_sequences == actual_sequences, \
        f"Expected sequences {expected_sequences} do not match actual {actual_sequences}"
    
    # Process and save results for each sequence (individual videos + combined)
    # Iterate through all sequences in the results
    for seq, tracker_result in tracker_results.items():
        # Validate that vehicle results exist for this sequence
        assert 'vehicle' in tracker_result, f"Vehicle results not found for {seq}"
        # Validate that this is either a known video or the combined sequence
        assert seq in videos or seq == 'COMBINED_SEQ', f"Sequence {seq} not found in {videos}"
        
        # Initialize metrics dictionary for this sequence
        seq_metrics = {}
        vehicle_results = tracker_result['vehicle']
        
        # Extract metrics from TrackEval results for this sequence
        # Each metric object provides its name and the corresponding results
        for metric in metrics:
            metric_name = metric.get_name()
            assert metric_name in vehicle_results, \
                f"Metric {metric_name} not found in {vehicle_results}"

            # Store the metric results for this sequence
            seq_metrics[metric_name] = vehicle_results[metric_name]
        
        # Prepare result data structure with metadata
        # This creates a consistent structure for both individual and combined results
        result_data = {
            'video_name': None if seq == 'COMBINED_SEQ' else seq,  # None for combined results
            'dataset': dataset,
            'classifier': classifier,
            'tile_size': tile_size,
            'metrics': seq_metrics,
        }
        
        # Save results to appropriate file
        # DATASET.json for combined results, {video}.json for individual video results
        output_file = "DATASET" if seq == "COMBINED_SEQ" else seq
        with open(os.path.join(output_dir, f'{output_file}.json'), 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)


def main(args):
    """
    Main function that orchestrates the tracking accuracy evaluation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with tracking results for the specified datasets and classifier/tile_size combinations
    2. Groups videos by dataset and classifier/tile_size combination
    3. Runs accuracy evaluation using TrackEval for each combination (evaluating all videos simultaneously)
    4. Generates both combined dataset results and individual video results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tracking results from 060_exec_track.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/060_uncompressed_tracks/{classifier}_{tile_size}/tracking.jsonl
        - Groundtruth data should be in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl
        - Results are saved to:
          {CACHE_DIR}/{dataset}/evaluation/070_tracking_accuracy/{classifier}_{tile_size}/
          ├── DATASET.json (combined results)
          ├── {video_name}.json (individual video results)
          └── LOG.txt (evaluation logs)
        - Multiple metrics are evaluated: HOTA, CLEAR (MOTA), and Identity (IDF1)
    """
    print(f"Starting tracking accuracy evaluation for datasets: {args.datasets}")
    
    # Parse metrics from comma-separated string into list
    # Remove any whitespace and split by comma
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    print(f"Evaluating metrics: {metrics_list}")
    
    # Find tracking results for all datasets and create evaluation tasks
    eval_tasks: list[Callable[[], None]] = []
    
    # Process each dataset separately
    for dataset in args.datasets:
        print(f"Processing dataset: {dataset}")
        
        # Find all videos and classifier/tile_size combinations for this dataset
        videos, classifier_tilesizes = find_tracking_results(CACHE_DIR, dataset)
        
        # Create evaluation directory path for this dataset
        evaluation_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '070_tracking_accuracy')
        
        # Create one evaluation task per classifier/tile_size combination
        # Each task will evaluate all videos in the dataset for that combination
        for cl, ts in classifier_tilesizes:
            output_dir = os.path.join(evaluation_dir, f'{cl}_{ts}')
            # Create a partial function with all arguments bound except the function call
            eval_tasks.append(partial(evaluate_tracking_accuracy, dataset, videos,
                                      cl, ts, metrics_list, output_dir))
    
    # Validate that we found some evaluation tasks
    assert len(eval_tasks) > 0, "No tracking results found. Please ensure 060_exec_track.py has been run first."
    print(f"Found {len(eval_tasks)} classifier-tile size combinations to evaluate")
    
    # Execute evaluation tasks either sequentially or in parallel
    if args.no_parallel:
        # Sequential execution: run each task one after another
        for eval_task in eval_tasks:
            eval_task()
    else:
        # Parallel execution: start all processes simultaneously
        processes: list[mp.Process] = []
        
        # Start each evaluation task in a separate process
        for eval_task in eval_tasks:
            process = mp.Process(target=eval_task)
            process.start()
            processes.append(process)

        # Wait for all processes to complete and clean up
        for process in track(processes):
            process.join()  # Wait for process to finish
            process.terminate()  # Ensure process is terminated


if __name__ == '__main__':
    main(parse_args())
