#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import multiprocessing as mp
import tempfile
import sys
from typing import Callable, Dict, List, Tuple, Any

import numpy as np

from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.datasets import B3D
from trackeval.metrics import HOTA, CLEAR, Identity

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


def find_tracking_results(cache_dir: str, dataset: str) -> List[Tuple[str, str, int]]:
    """
    Find all video files with tracking results for the specified dataset and tile size.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        List[Tuple[str, str, int]]: List of (video_name, classifier, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset, 'execution')
    if not os.path.exists(dataset_cache_dir):
        print(f"Dataset cache directory {dataset_cache_dir} does not exist, skipping...")
        return []
    
    video_tile_combinations: list[tuple[str, str, int]] = []
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        if not os.path.isdir(video_dir):
            continue
        tracking_dir = os.path.join(video_dir, '060_uncompressed_tracks')
        if not os.path.exists(tracking_dir):
            continue

        for classifier_tilesize in os.listdir(tracking_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            tracking_path = os.path.join(tracking_dir, f'{classifier}_{ts}', 'tracking.jsonl')
            groundtruth_path = os.path.join(video_dir, '000_groundtruth', 'tracking.jsonl')
            
            if os.path.exists(tracking_path) and os.path.exists(groundtruth_path):
                video_tile_combinations.append((video_filename, classifier, ts))
                print(f"Found tracking results: {video_filename} with tile size {ts}")
    
    return video_tile_combinations


def load_tracking_data(file_path: str) -> Dict[int, List[List[float]]]:
    """
    Load tracking data from JSONL file.
    
    Args:
        file_path (str): Path to the tracking JSONL file
        
    Returns:
        Dict[int, List[List[float]]]: Dictionary mapping frame indices to list of detections
    """
    frame_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                frame_idx = data['frame_idx']
                tracks = data['tracks']
                frame_data[frame_idx] = tracks
    
    # Pad with empty lists if first frame index is not 0
    if frame_data and min(frame_data.keys()) > 0:
        for i in range(min(frame_data.keys())):
            frame_data[i] = []
    
    return frame_data


def load_groundtruth_data(file_path: str) -> Dict[int, List[List[float]]]:
    """
    Load groundtruth data from JSONL file.
    
    Args:
        file_path (str): Path to the groundtruth JSONL file
        
    Returns:
        Dict[int, List[List[float]]]: Dictionary mapping frame indices to list of groundtruth detections
    """
    frame_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                frame_idx = data['frame_idx']
                detections = data['detections'] if 'detections' in data else data.get('tracks', [])
                frame_data[frame_idx] = detections
    
    # Pad with empty lists if first frame index is not 0
    if frame_data and min(frame_data.keys()) > 0:
        for i in range(min(frame_data.keys())):
            frame_data[i] = []
    
    return frame_data


def convert_to_trackeval_format(frame_data: Dict[int, List[List[float]]], is_gt: bool = False) -> str:
    """
    Convert frame data to TrackEval format and save to temporary file.
    
    Args:
        frame_data (Dict[int, List[List[float]]]): Frame data dictionary
        is_gt (bool): Whether this is groundtruth data
        
    Returns:
        str: Path to temporary file in TrackEval format
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    # Sort frames and write data
    for frame_idx in sorted(frame_data.keys()):
        detections = frame_data[frame_idx]
        temp_file.write(json.dumps([frame_idx, detections]) + '\n')
    
    temp_file.close()
    return temp_file.name


def evaluate_tracking_accuracy(video_name: str, classifier: str, tile_size: int, tracking_path: str, 
                              groundtruth_path: str, metrics_list: List[str], 
                              output_dir: str) -> Dict[str, Any]:
    """
    Evaluate tracking accuracy for a single video using TrackEval.
    
    Args:
        video_name (str): Name of the video
        classifier (str): Classifier used
        tile_size (int): Tile size used
        tracking_path (str): Path to tracking results
        groundtruth_path (str): Path to groundtruth data
        metrics_list (List[str]): List of metrics to evaluate
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"Evaluating {video_name} with tile size {tile_size}")
    
    # Load data
    tracking_data = load_tracking_data(tracking_path)
    groundtruth_data = load_groundtruth_data(groundtruth_path)
    
    # Convert to TrackEval format
    temp_tracking_file = convert_to_trackeval_format(tracking_data, is_gt=False)
    temp_groundtruth_file = convert_to_trackeval_format(groundtruth_data, is_gt=True)
    
    # Create dataset configuration
    dataset_config = {
        'output_fol': output_dir,
        'output_sub_fol': f'{video_name}_{classifier}_{tile_size}',
        'input_gt': temp_groundtruth_file,
        'input_track': temp_tracking_file,
        'skip': 1,  # Process every frame
        'tracker': f'{classifier}_{tile_size}'
    }
    
    # Create evaluator configuration
    eval_config = {
        'USE_PARALLEL': False,
        # 'NUM_PARALLEL_CORES': int(mp.cpu_count() * 0.8),
        'BREAK_ON_ERROR': True,
        'LOG_ON_ERROR': os.path.join(f'{video_name}_{classifier}_{tile_size}_error_log.txt'),  # if not None, save any errors into a log file.
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
    dataset = B3D(dataset_config)
    evaluator = trackeval.Evaluator(eval_config)
    
    # Run evaluation
    results = evaluator.evaluate([dataset], metrics)
    
    # Clean up temporary files
    os.unlink(temp_tracking_file)
    os.unlink(temp_groundtruth_file)
    
    # Extract summary results
    summary_results = {}
    
    # The actual structure is: results[0]["B3D"]["xsort"]["COMBINED_SEQ"]["car"]
    if results and len(results) > 0:
        # Get the first result which contains the B3D evaluation
        b3d_result = results[0].get('B3D', {})
        xsort_result = b3d_result.get('xsort', {})
        
        # Look for the COMBINED_SEQ results which contain the car class
        if 'COMBINED_SEQ' in xsort_result and 'car' in xsort_result['COMBINED_SEQ']:
            car_results = xsort_result['COMBINED_SEQ']['car']
            
            # Extract metrics from the car results
            for metric in metrics:
                metric_name = metric.get_name()
                if metric_name in car_results:
                    summary_results[metric_name] = car_results[metric_name]
    
    return {
        'video_name': video_name,
        'tile_size': tile_size,
        'classifier': classifier,
        'metrics': summary_results,
        'success': True,
        'output_dir': output_dir,
    }


def get_results(eval_task: Callable[[], dict], res_queue: "mp.Queue[dict]"):
    res_queue.put(eval_task())


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
    
    print(f"Metrics: {args.metrics}")
    print(f"Evaluating metrics: {metrics_list}")
    
    # Find tracking results for all datasets
    all_video_tile_combinations = []
    for dataset in args.datasets:
        print(f"Processing dataset: {dataset}")
        video_tile_combinations = find_tracking_results(CACHE_DIR, dataset)
        # Add dataset info to each combination
        for video_name, classifier, tile_size in video_tile_combinations:
            all_video_tile_combinations.append((dataset, video_name, classifier, tile_size))
    
    if not all_video_tile_combinations:
        print("No tracking results found. Please ensure 060_exec_track.py has been run first.")
        return
    
    print(f"Found {len(all_video_tile_combinations)} video-tile size combinations to evaluate")
    
    # Prepare arguments for parallel processing if requested
    # eval_args = []
    eval_tasks: list[Callable[[], dict]] = []
    for dataset, video_name, classifier, tile_size in sorted(all_video_tile_combinations):
        tracking_path = os.path.join(CACHE_DIR, dataset, 'execution', video_name, '060_uncompressed_tracks',
                                        f'{classifier}_{tile_size}', 'tracking.jsonl')
        groundtruth_path = os.path.join(CACHE_DIR, dataset, 'execution', video_name, 
                                        '000_groundtruth', 'tracking.jsonl')
        output_dir = os.path.join(CACHE_DIR, dataset, 'execution', video_name, '070_tracking_accuracy',
                                    f'{classifier}_{tile_size}', 'accuracy')

        # eval_args.append((video_name, classifier, tile_size, tracking_path, groundtruth_path, 
        #                  metrics_list, output_dir))
        eval_tasks.append(partial(evaluate_tracking_accuracy, video_name, classifier,
                                    tile_size, tracking_path, groundtruth_path, metrics_list, output_dir))
    
    
    if args.no_parallel:
        results = []
        for eval_task in eval_tasks:
            results.append(eval_task())
    else:
        # # Run evaluation in parallel
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     results = pool.starmap(evaluate_tracking_accuracy, eval_args)

        res_queue = mp.Queue()
        processes: list[mp.Process] = []
        for eval_task in eval_tasks:
            process = mp.Process(target=get_results, args=(eval_task, res_queue))
            process.start()
            processes.append(process)
        
        results = []
        for _ in track(range(len(eval_tasks)), total=len(eval_tasks)):
            results.append(res_queue.get())

        for process in track(processes):
            process.join()
            process.terminate()
    

    # Save individual results
    for result in results:
        print('save results to', result['output_dir'])
        os.makedirs(result['output_dir'], exist_ok=True)
        with open(os.path.join(result['output_dir'], 'detailed_results.json'), 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)


if __name__ == '__main__':
    main(parse_args())
