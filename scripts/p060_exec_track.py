#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time
import numpy as np
import tqdm
import multiprocessing as mp
from functools import partial
from typing import Callable
import torch

from polyis.utilities import create_tracker, format_time, interpolate_trajectory, CACHE_DIR, ProgressBar, register_tracked_detections


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tracker (str): Tracking algorithm to use (default: 'sort')
            - max_age (int): Maximum age for SORT tracker (default: 10)
            - min_hits (int): Minimum hits for SORT tracker (default: 3)
            - iou_threshold (float): IOU threshold for SORT tracker (default: 0.3)
            - no_interpolate (bool): Whether to not perform trajectory interpolation (default: False)
    """
    parser = argparse.ArgumentParser(description='Execute object tracking on uncompressed '
                                                 'detection results from 050_exec_uncompress.py')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tracker', required=False,
                        default='sort',
                        choices=['sort'],
                        help='Tracking algorithm to use')
    parser.add_argument('--max_age', type=int, default=10,
                        help='Maximum age for SORT tracker')
    parser.add_argument('--min_hits', type=int, default=3,
                        help='Minimum hits for SORT tracker')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IOU threshold for SORT tracker')
    parser.add_argument('--no_interpolate', action='store_true',
                        help='Whether to not perform trajectory interpolation')
    return parser.parse_args()


def load_detection_results(cache_dir: str, dataset: str, video_file: str, tile_size: int, classifier: str) -> list[dict]:
    """
    Load detection results from the uncompressed detections JSONL file.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        video_file (str): Video file name
        tile_size (int): Tile size used for detections
        classifier (str): Classifier name used for detections
        
    Returns:
        list[dict]: list of frame detection results
        
    Raises:
        FileNotFoundError: If no detection results file is found
    """
    detection_path = os.path.join(cache_dir, dataset, video_file,
                                  'uncompressed_detections',
                                  f'{classifier}_{tile_size}',
                                  'detections.jsonl')
    
    if not os.path.exists(detection_path):
        raise FileNotFoundError(f"Detection results not found: {detection_path}")
    
    print(f"Loading detection results from: {detection_path}")
    
    results = []
    with open(detection_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"Loaded {len(results)} frame detections")
    return results


def process_tracking_task(video_file: str, tile_size: int, classifier: str, 
                          args: argparse.Namespace, gpu_id: int, command_queue: mp.Queue):
    """
    Process tracking for a single video/classifier/tile_size combination.
    This function is designed to be called in parallel.
    
    Args:
        video_file (str): Name of the video file to process
        tile_size (int): Tile size used for detections
        classifier (str): Classifier name used for detections
        gpu_id (int): GPU ID to use for processing
        command_queue (mp.Queue): Queue for progress updates
        args: Command line arguments
    """
    device = f'cuda:{gpu_id}'
    video_name = os.path.basename(video_file)
    tracker_name = args.tracker
    max_age = args.max_age
    min_hits = args.min_hits
    iou_threshold = args.iou_threshold
    no_interpolate = args.no_interpolate
    
    # Check if uncompressed detections exist
    detection_path = os.path.join(CACHE_DIR, args.dataset, video_file,
                                  'uncompressed_detections', f'{classifier}_{tile_size}',
                                  'detections.jsonl')
    assert os.path.exists(detection_path)

    # Load detection results
    detection_results = load_detection_results(CACHE_DIR, args.dataset, video_file, tile_size, classifier)

    # Create output path for tracking results
    uncompressed_tracking_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'uncompressed_tracking')
    output_path = os.path.join(uncompressed_tracking_dir, f'{classifier}_{tile_size}', 'tracking.jsonl')
    
    # Create tracker
    tracker = create_tracker(tracker_name, max_age, min_hits, iou_threshold)
    
    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    
    print(f"Processing {len(detection_results)} frames for tracking...")
    
    # Send initial progress update
    command_queue.put((device, {
        'description': f"{video_name} {tracker_name} {classifier} {tile_size}",
        'completed': 0,
        'total': len(detection_results)
    }))
    
    # Create runtime output file
    runtime_path = output_path.replace('tracking.jsonl', 'runtimes.jsonl')
    runtime_dir = os.path.dirname(runtime_path)
    os.makedirs(runtime_dir, exist_ok=True)
    
    with open(runtime_path, 'w') as runtime_file:
        # Process each frame
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
            command_queue.put((device, {'completed': frame_idx + 1}))
    
    print(f"Tracking completed. Found {len(trajectories)} unique tracks.")
    
    # Save tracking results
    print(f"Saving tracking results to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        frame_ids = frame_tracks.keys()
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
    
    print(f"Tracking results saved successfully. Total frames: {len(frame_tracks)}")
    print(f"Runtime data saved to: {runtime_path}")


def main(args: argparse.Namespace):
    """
    Main function that orchestrates the object tracking process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Creates a list of all video/classifier/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves tracking results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects uncompressed detection results from 050_exec_uncompress.py in:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_detections/{classifier}_{tile_size}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/{classifier}_{tile_size}/tracking.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - The number of processes equals the number of available GPUs
    """
    mp.set_start_method('spawn', force=True)
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Using tracker: {args.tracker}")
    print(f"Tracker parameters: max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_threshold}")
    print(f"Interpolation: {'enabled' if not args.no_interpolate else 'disabled'}")
    
    # Find all videos with uncompressed detection results
    dataset_cache_dir = os.path.join(CACHE_DIR, args.dataset)
    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    # Create tasks list with all video/classifier/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if not os.path.isdir(item_path):
            continue

        uncompressed_detections_dir = os.path.join(item_path, 'uncompressed_detections')
        if not os.path.exists(uncompressed_detections_dir):
            continue

        uncompressed_tracking_dir = os.path.join(item_path, 'uncompressed_tracking')
        if os.path.exists(uncompressed_tracking_dir):
            shutil.rmtree(uncompressed_tracking_dir)

        for classifier_tilesize in sorted(os.listdir(uncompressed_detections_dir)):
            classifier, tile_size = classifier_tilesize.split('_')
            tile_size = int(tile_size)
            funcs.append(partial(process_tracking_task, item, tile_size, classifier, args))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.8)
    print(f"Using {num_processes} CPUs for parallel processing")
    
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
