#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import multiprocessing as mp
import shutil
import sys
from typing import Callable, override
import warnings

import numpy as np
from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.metrics import HOTA, Count

from polyis.trackeval.dataset import Dataset
from polyis.utilities import dataset_root_name, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']

DATASETS_IN_MAP = {
    'caldot1-y05': 'caldot1_y5',
    'caldot1-y11': 'caldot1_y11',
    'caldot2-y05': 'caldot2_y5',
    'caldot2-y11': 'caldot2_y11',
}


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


def find_sota_tracking_results(cache_dir: str, dataset: str, system: str) -> tuple[set[str], set[int]]:
    """
    Find all videos and param_id combinations with SOTA tracking results.

    Scans the SOTA directory (OTIF or LEAP) to discover all available videos and their
    corresponding param_id combinations that have both tracking results
    and groundtruth data available.

    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        system (str): System name ('otif' or 'leap')

    Returns:
        tuple[set[str], set[int]]: Set of video names and set of param_id values
    """
    # Construct path to SOTA dataset directory
    sota_dir = os.path.join(cache_dir, 'SOTA', system, dataset)
    if not os.path.exists(sota_dir):
        return set(), set()


    # Collect all video-param_id combinations
    video_param_combinations: list[tuple[str, int]] = []

    # Iterate through all video directories in the SOTA dataset directory
    for video_filename in os.listdir(sota_dir):
        video_dir = os.path.join(sota_dir, video_filename)
        if not os.path.isdir(video_dir):
            # assert video_filename == 'stat.csv', f"Video directory {video_dir} is not a directory"
            continue
            
        if not video_filename.endswith('.mp4'):
            assert video_filename == 'accuracy'
            continue

        # Check for tracking_results subdirectory
        tracking_results_dir = os.path.join(video_dir, 'tracking_results')
        assert os.path.exists(tracking_results_dir), f"Tracking results directory {tracking_results_dir} does not exist"

        # Check for param_id subdirectories in tracking_results
        for param_item in os.listdir(tracking_results_dir):
            # Parse param_id from directory name
            assert param_item.isdigit(), f"Param item {param_item} is not a digit"

            param_id = int(param_item)
            param_path = os.path.join(tracking_results_dir, param_item)
            assert os.path.isdir(param_path), f"Param path {param_path} is not a directory"

            # Construct paths to tracking and groundtruth files
            tracking_path = os.path.join(param_path, 'tracking.jsonl')
            # Resolve to root dataset name (e.g., caldot1-y05 -> caldot1) for GT path
            gt_dataset = dataset_root_name(dataset)
            groundtruth_path = os.path.join(cache_dir, gt_dataset, 'execution', video_filename, '003_groundtruth', 'tracking.jsonl')

            # Verify both tracking results and groundtruth exist
            assert os.path.exists(tracking_path), f"Tracking path {tracking_path} does not exist"
            assert os.path.exists(groundtruth_path), f"Groundtruth path {groundtruth_path} does not exist"

            # Add this combination to our list
            video_param_combinations.append((video_filename, param_id))
            print(f"Found {system.upper()} tracking results: {video_filename} with param_id {param_id}")

    # Extract unique param_ids and video names
    param_ids = set(param_id for _, param_id in video_param_combinations)
    videos = set(video for video, _ in video_param_combinations)

    # Validate that all videos have results for all param_ids (only for OTIF)
    if system == 'otif' and videos and param_ids:
        video_param_combinations_set = set(video_param_combinations)
        assert len(video_param_combinations_set) == len(video_param_combinations), \
            f"Duplicate video-param combinations: {video_param_combinations_set}"

        # Check completeness: every video should have results for every param_id
        for video in videos:
            for param_id in param_ids:
                assert (video, param_id) in video_param_combinations_set, \
                    f"Video-param combination {video}-{param_id} ({dataset}) not found"

    return videos, param_ids


def evaluate_sota_tracking_accuracy(dataset: str, videos: set[str], param_id: int,
                                     system: str, output_dir: str, worker_id: int, worker_id_queue: "mp.Queue"):
    """
    Evaluate tracking accuracy for SOTA results (OTIF or LEAP) for a specific param_id.

    Performs a single evaluation across all videos in the dataset for the given param_id.
    Generates both combined dataset results and individual video results in a flattened directory structure.

    Args:
        dataset (str): Dataset name
        videos (set[str]): Set of video names to evaluate (all videos in dataset)
        param_id (int): Parameter ID
        system (str): System name ('otif' or 'leap')
        output_dir (str): Output directory for results

    Output Structure:
        - DATASET.json: Combined results across all videos
        - {video}.json: Individual video results
        - LOG.txt: Evaluation logs and errors
    """
    print(f"Evaluating {len(videos)} videos with param_id {param_id} for {system.upper()}")

    # Use the SOTA directory as the base (tracking files and groundtruth files are already there)
    sota_dir = os.path.join(CACHE_DIR, 'SOTA', system, dataset)

    # Create tracker identifier
    tracker_name = system
    input_track = os.path.join('tracking_results', f'{param_id:03d}', 'tracking.jsonl')

    # Create TrackEval dataset configuration
    # This configures how TrackEval will find and process the data files
    dataset_config = {
        'output_fol': output_dir,  # Where TrackEval will write its output
        'output_sub_fol': f'{dataset}_{tracker_name}_{param_id:03d}',  # Subdirectory name for this evaluation
        'input_gt': os.path.join('003_groundtruth', 'tracking.jsonl'),  # Relative path to groundtruth files
        'input_track': input_track,  # Relative path to tracking files
        'skip': 1,  # Process every frame (no frame skipping)
        'tracker': tracker_name,  # Tracker name identifier
        'seq_list': videos,  # List of sequences (videos) to evaluate
        'input_dir': sota_dir  # Base directory for relative paths (SOTA directory with copied groundtruth)
    }

    # Create TrackEval evaluator configuration
    # This controls how the evaluation is performed and what output is generated
    eval_config = {
        'USE_PARALLEL': False,  # Enable parallel processing within TrackEval
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

    os.makedirs(output_dir, exist_ok=True)

    # Create TrackEval metric objects
    metrics = [HOTA()]

    # Create TrackEval dataset and evaluator objects
    # The dataset object handles data loading and preprocessing
    eval_dataset = Dataset(dataset_config)
    # The evaluator object handles the actual evaluation process
    evaluator = trackeval.Evaluator(eval_config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Run the evaluation across all videos simultaneously
        # This returns results for both individual videos and combined dataset
        results = evaluator.evaluate([eval_dataset], metrics)

    # TrackEval returns results in structure: results[0]["Dataset"]['sort'][sequence]["vehicle"]
    # where sequence can be individual video names or "COMBINED_SEQ" for aggregated results
    # results[0] contains the actual evaluation results
    # results[1] contains the evaluation status
    assert results and len(results) == 2, results
    assert results[1]['Dataset']['sort'] == 'Success', f"Evaluation failed: {results[1]}"

    # Extract evaluation results from TrackEval output structure
    # Navigate through the nested result structure to get to the actual data
    dataset_result = results[0]['Dataset']
    tracker_results = dataset_result['sort']

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
        for metric in metrics + [Count()]:
            metric_name = metric.get_name()
            assert metric_name in vehicle_results, \
                f"Metric {metric_name} not found in {vehicle_results}"

            # Store the metric results for this sequence
            seq_metrics[metric_name] = vehicle_results[metric_name]

        # Prepare result data structure with metadata
        # This creates a consistent structure for both individual and combined results
        result_data = {
            'video': None if seq == 'COMBINED_SEQ' else seq,  # None for combined results
            'dataset': dataset,
            'param_id': param_id,
            'metrics': seq_metrics,
        }

        # Save results to appropriate file
        # DATASET.json for combined results, {video}.json for individual video results
        output_file = "DATASET" if seq == "COMBINED_SEQ" else seq
        with open(os.path.join(output_dir, f'{output_file}.json'), 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)

    worker_id_queue.put(worker_id)


def main(args):
    """
    Main function that orchestrates the OTIF and LEAP tracking accuracy evaluation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with OTIF and LEAP tracking results for the specified datasets and param_id combinations
    2. Groups videos by dataset and param_id combination
    3. Runs accuracy evaluation using TrackEval for each combination (evaluating all videos simultaneously)
    4. Generates both combined dataset results and individual video results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects OTIF tracking results in:
          {CACHE_DIR}/SOTA/otif/{dataset}/{video_file}/tracking_results/{param_id:03d}/tracking.jsonl
        - The script expects LEAP tracking results in:
          {CACHE_DIR}/SOTA/leap/{dataset_in}/{video_file}/tracking_results/000/tracking.jsonl
        - Groundtruth data should be in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/003_groundtruth/tracking.jsonl
        - OTIF results are saved to:
          {CACHE_DIR}/SOTA/otif/{dataset}/accuracy/raw/{param_id:03d}/
        - LEAP results are saved to:
          {CACHE_DIR}/SOTA/leap/{dataset_in}/accuracy/raw/000/
        - Both include:
          ├── DATASET.json (combined results)
          ├── {video}.json (individual video results)
          └── LOG.txt (evaluation logs)
    """
    print(f"Starting OTIF and LEAP tracking accuracy evaluation for datasets: {DATASETS}")
    
    # Find tracking results for all datasets and create evaluation tasks
    eval_tasks: list[Callable[[int, "mp.Queue"], None]] = []
    
    # Process each dataset separately
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        
        # Process both OTIF and LEAP results
        for system in ['otif', 'leap']:
            videos, param_ids = find_sota_tracking_results(CACHE_DIR, dataset, system)
            print(videos, param_ids)
            
            if not videos or not param_ids:
                continue
            
            # Create evaluation directory path for this dataset
            evaluation_dir = os.path.join(CACHE_DIR, 'SOTA', system, dataset, 'accuracy')
            
            # Clear evaluation directory
            if os.path.exists(evaluation_dir):
                shutil.rmtree(evaluation_dir)
                print(f"Cleared existing {system.upper()} accuracy directory: {evaluation_dir}")
            os.makedirs(evaluation_dir, exist_ok=True)
            os.makedirs(os.path.join(evaluation_dir, 'raw'), exist_ok=True)
            
            # Copy groundtruth files to SOTA directory structure (before parallel execution to avoid race conditions)
            # TrackEval expects: {input_dir}/{video}/003_groundtruth/tracking.jsonl
            sota_dir = os.path.join(CACHE_DIR, 'SOTA', system, dataset)
            # Resolve to root dataset name (e.g., caldot1-y05 -> caldot1) for GT path
            gt_dataset = dataset_root_name(dataset)
            execution_dir = os.path.join(CACHE_DIR, gt_dataset, 'execution')
            for video_file in videos:
                # Construct paths for this video
                groundtruth_source = os.path.join(execution_dir, video_file, '003_groundtruth', 'tracking.jsonl')
                video_sota_dir = os.path.join(sota_dir, video_file)
                groundtruth_dest_dir = os.path.join(video_sota_dir, '003_groundtruth')
                groundtruth_dest = os.path.join(groundtruth_dest_dir, 'tracking.jsonl')
                
                # Check if groundtruth file exists
                assert os.path.exists(groundtruth_source), f"Groundtruth file not found: {groundtruth_source}"
                
                # Create groundtruth directory in SOTA directory for this video
                os.makedirs(groundtruth_dest_dir, exist_ok=True)
                
                # Copy groundtruth file to SOTA directory
                if os.path.exists(groundtruth_dest):
                    # Remove existing file if it exists
                    os.remove(groundtruth_dest)
                shutil.copy2(groundtruth_source, groundtruth_dest)
            
            # Create one evaluation task per param_id combination
            # Each task will evaluate all videos in the dataset for that param_id
            for param_id in param_ids:
                output_dir = os.path.join(evaluation_dir, 'raw', f'{param_id:03d}')
                # Create a partial function with all arguments bound except the function call
                eval_tasks.append(partial(evaluate_sota_tracking_accuracy, dataset, videos,
                                          param_id, system, output_dir))
    
    # Validate that we found some evaluation tasks
    assert len(eval_tasks) > 0, "No OTIF or LEAP tracking results found. Please ensure p140_otif_transform.py has been run first."
    print(f"Found {len(eval_tasks)} param_id combinations to evaluate")

    # Execute evaluation tasks either sequentially or in parallel
    # Parallel execution: start all processes simultaneously
    processes: list[mp.Process] = []
    worker_id_queue = mp.Queue()
    for i in range(int(mp.cpu_count() * 0.8)):
        worker_id_queue.put(i)

    # Start each evaluation task in a separate process
    for eval_task in eval_tasks:
        worker_id = worker_id_queue.get()
        process = mp.Process(target=eval_task, args=(worker_id, worker_id_queue))
        process.start()
        processes.append(process)

    # Wait for all processes to complete and clean up
    for process in track(processes):
        process.join()  # Wait for process to finish
        process.terminate()  # Ensure process is terminated


if __name__ == '__main__':
    main(parse_args())

