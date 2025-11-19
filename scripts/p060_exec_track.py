#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Callable

from polyis.utilities import create_tracker, format_time, ProgressBar, register_tracked_detections, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']
CLASSIFIERS = CONFIG['EXEC']['CLASSIFIERS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']
TILEPADDING_MODES = CONFIG['EXEC']['TILEPADDING_MODES']


def parse_args():
    parser = argparse.ArgumentParser(description='Execute object tracking on detection results')
    parser.add_argument('--max_age', type=int, default=10,
                        help='Maximum age for SORT tracker')
    parser.add_argument('--min_hits', type=int, default=3,
                        help='Minimum hits for SORT tracker')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IOU threshold for SORT tracker')
    parser.add_argument('--no_interpolate', action='store_true',
                        help='Whether to not perform trajectory interpolation')
    return parser.parse_args()


def load_detection_results(cache_dir: str, dataset: str, video_file: str, tilesize: int,
                           classifier: str, tilepadding: str | None = None, verbose: bool = False):
    """
    Load detection results from the uncompressed detections JSONL file.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        video_file (str): Video file name
        tilesize (int): Tile size used for detections
        classifier (str): Classifier name used for detections
        tilepadding (str): Whether padding was applied to classification results
        verbose (bool): Whether to print verbose output
    Returns:
        list[dict]: list of frame detection results
        
    Raises:
        FileNotFoundError: If no detection results file is found
    """
    tilepadding_str = ""
    if tilepadding is not None:
        tilepadding_str = f"_{tilepadding}"
    detection_path = os.path.join(cache_dir, dataset, 'execution', video_file,
                                  '050_uncompressed_detections',
                                  f'{classifier}_{tilesize}{tilepadding_str}',
                                  'detections.jsonl')
    
    if not os.path.exists(detection_path):
        raise FileNotFoundError(f"Detection results not found: {detection_path}")
    
    if verbose:
        print(f"Loading detection results from: {detection_path}")
    
    results: list[dict] = []
    with open(detection_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if verbose:
        print(f"Loaded {len(results)} frame detections")
    return results


def track(dataset: str, video: str, classifier: str, tilesize: int, tilepadding: str,
          no_interpolate: bool, gpu_id: int, command_queue: mp.Queue):
    """
    Process tracking for a single video/classifier/tilesize combination.
    This function is designed to be called in parallel.
    
    Args:
        dataset (str): Name of the dataset
        video (str): Name of the video file to process
        classifier (str): Classifier name used for detections
        tilesize (int): Tile size used for detections
        tilepadding (str): Whether padding was applied to classification results
        no_interpolate (bool): Whether to not perform trajectory interpolation
        gpu_id (int): GPU ID to use for processing
        command_queue (mp.Queue): Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    
    # Check if uncompressed detections exist
    detection_path = os.path.join(CACHE_DIR, dataset, 'execution', video, '050_uncompressed_detections',
                                  f'{classifier}_{tilesize}_{tilepadding}', 'detections.jsonl')
    assert os.path.exists(detection_path), f"Detections not found: {detection_path}"

    # Load detection results
    detection_results = load_detection_results(CACHE_DIR, dataset, video, tilesize, classifier, tilepadding)

    # Create output path for tracking results
    uncompressed_tracking_dir = os.path.join(CACHE_DIR, dataset, 'execution', video, '060_uncompressed_tracks')
    output_path = os.path.join(uncompressed_tracking_dir, f'{classifier}_{tilesize}_{tilepadding}', 'tracking.jsonl')
    
    # Create tracker
    tracker = create_tracker('sort')
    
    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    
    # print(f"Processing {len(detection_results)} frames for tracking...")
    
    # Send initial progress update
    command_queue.put((device, {
        'description': f"{video} {'sort'} {classifier} {tilesize}",
        'completed': 0,
        'total': len(detection_results)
    }))
    
    # Create runtime output file
    runtime_path = output_path.replace('tracking.jsonl', 'runtimes.jsonl')
    runtime_dir = os.path.dirname(runtime_path)
    os.makedirs(runtime_dir, exist_ok=True)
    
    with open(runtime_path, 'w') as runtime_file:
        # Process each frame
        mod = max(1, int(len(detection_results) * 0.05))
        for frame_result in detection_results:
            frame_idx = frame_result['frame_idx']
            bboxes = frame_result['bboxes']
            
            # Start timing for this frame
            step_times = {}
            
            # Profile: Convert detections to numpy array
            step_start = (time.time_ns() / 1e6)
            dets = np.array(bboxes)
            if dets.size > 0:
                dets = dets[:, :5]  # Take first 5 columns: x1, y1, x2, y2, score
            else:
                dets = np.empty((0, 5))
            step_times['convert_detections'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Update tracker
            step_start = (time.time_ns() / 1e6)
            tracked_dets = tracker.update(dets)
            step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Process tracking results
            step_start = (time.time_ns() / 1e6)
            register_tracked_detections(tracked_dets, frame_idx, frame_tracks,
                                        trajectories, no_interpolate)
            step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start
            
            # Save runtime data for this frame
            runtime_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_detections': len(bboxes),
                'num_tracks': tracked_dets.size if tracked_dets.size > 0 else 0
            }
            runtime_file.write(json.dumps(runtime_data) + '\n')
            
            # Send progress update
            if frame_idx % mod == 0:
                command_queue.put((device, {'completed': frame_idx + 1}))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        frame_ids = frame_tracks.keys()
        if len(frame_ids) == 0:
            return
        
        first_idx = min(frame_ids)
        last_idx = max(frame_ids)

        for frame_idx in range(first_idx, last_idx + 1):
            if frame_idx not in frame_tracks:
                frame_tracks[frame_idx] = []
                
            frame_data = {
                "frame_idx": frame_idx,
                "tracks": frame_tracks[frame_idx]
            }
            f.write(json.dumps(frame_data) + '\n')


def main(args: argparse.Namespace):
    """
    Main function that orchestrates the object tracking process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves tracking results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects uncompressed detection results from 050_exec_uncompress.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/050_uncompressed_detections/{classifier}_{tilesize}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/060_uncompressed_tracks/{classifier}_{tilesize}/tracking.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - The number of processes equals the number of available GPUs
    """
    mp.set_start_method('spawn', force=True)
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        videosets_dir = os.path.join(dataset_dir, 'test')
        
        # Find all videos with uncompressed detection results
        cache_dir = os.path.join(CACHE_DIR, dataset, 'execution')
        for video in os.listdir(videosets_dir):
            # uncompressed_tracking_dir = os.path.join(cache_dir, video, '060_uncompressed_tracks')
            # if os.path.exists(uncompressed_tracking_dir):
            #     shutil.rmtree(uncompressed_tracking_dir)

            for classifier in CLASSIFIERS:
                for tilesize in TILE_SIZES:
                    for tilepadding in TILEPADDING_MODES:
                        funcs.append(partial(track, dataset, video, classifier, tilesize, tilepadding, args.no_interpolate))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar
    ProgressBar(num_workers=int(mp.cpu_count() * 0.8), num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
