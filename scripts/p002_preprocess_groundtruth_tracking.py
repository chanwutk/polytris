#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import torch
from functools import partial
from multiprocessing import Queue

from polyis.utilities import CACHE_DIR, create_tracker, format_time, load_detection_results, ProgressBar, register_tracked_detections


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (list): List of dataset names to process (default: ['caldot1', 'caldot2'])
            - tracker (str): Tracking algorithm to use (default: 'sort')
            - max_age (int): Maximum age for SORT tracker (default: 10)
            - min_hits (int): Minimum hits for SORT tracker (default: 3)
            - iou_threshold (float): IOU threshold for SORT tracker (default: 0.3)
    """
    parser = argparse.ArgumentParser(description='Execute object tracking on detection results')
    parser.add_argument('--datasets', required=False,
                        default=['caldot1', 'caldot2'],
                        nargs='+',
                        help='Dataset names (space-separated)')
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
    return parser.parse_args()


def track(video_file: str, args, cache_dir: str, dataset: str, gpu_id: int, command_queue: "Queue[tuple[str, dict]]"):
    """
    Execute object tracking on detection results and save tracking results to JSONL.

    Args:
        video_file (str): Name of the video file to process
        args: Command line arguments
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        gpu_id (int): GPU device ID to use for this process
        command_queue (Queue): Queue for progress updates
    """
    # Load detection results
    detection_results = load_detection_results(cache_dir, dataset, video_file)

    # Create output path for tracking results
    output_path = os.path.join(cache_dir, dataset, 'execution', video_file, 'groundtruth', 'tracking.jsonl')

    tracker_name = args.tracker
    max_age = args.max_age
    min_hits = args.min_hits
    iou_threshold = args.iou_threshold

    # print(f"Processing video: {video_file}")
    # Create tracker
    tracker = create_tracker(tracker_name, max_age, min_hits, iou_threshold)

    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    # print(f"Processing {len(detection_results)} frames for tracking...")
    # Send initial progress update
    command_queue.put((f'cuda:{gpu_id}', {
        'description': video_file,
        'completed': 0,
        'total': len(detection_results)
    }))

    runtime_path = output_path.replace('tracking.jsonl', 'tracking_runtimes.jsonl')
    with open(runtime_path, 'w') as runtime_file:
        # Process each frame
        mod = max(1, int(len(detection_results) * 0.05))
        for fidx, frame_result in enumerate(detection_results):
            frame_idx = frame_result['frame_idx']
            detections = frame_result['detections']

            # Start timing for this frame
            step_times = {}

            # Convert detections to numpy array format expected by SORT
            step_start = (time.time_ns() / 1e6)
            if detections:
                # SORT expects format: [[x1, y1, x2, y2, score], ...]
                dets = np.array(detections)
                if dets.size > 0:
                    # Ensure we have the right format
                    if dets.shape[1] >= 5:
                        dets = dets[:, :5]  # Take first 5 columns: x1, y1, x2, y2, score
                    else:
                        # If we don't have scores, add default score of 1.0
                        dets = np.column_stack([dets, np.ones(dets.shape[0])])
                else:
                    dets = np.empty((0, 5))
            else:
                dets = np.empty((0, 5))
            step_times['convert_detections'] = (time.time_ns() / 1e6) - step_start

            # Update tracker
            step_start = (time.time_ns() / 1e6)
            tracked_dets = tracker.update(dets)
            step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start

            # Process tracking results
            step_start = (time.time_ns() / 1e6)
            register_tracked_detections(tracked_dets, frame_idx, frame_tracks, trajectories, False)
            step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start
            runtime_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_detections': len(dets),
                'num_tracks': tracked_dets.size if tracked_dets.size > 0 else 0
            }
            runtime_file.write(json.dumps(runtime_data) + '\n')

            # Send progress update
            if fidx % mod == 0:
                command_queue.put((f'cuda:{gpu_id}', {'completed': fidx + 1}))

    # print(f"Tracking completed. Found {len(trajectories)} unique tracks.")

    # Save tracking results
    # print(f"Saving tracking results to: {output_path}")

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

    # print(f"Tracking results saved successfully. Total frames: {len(frame_tracks)}")


def main(args):
    """
    Main function that orchestrates the object tracking process.

    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Finds all videos with detection results
    4. Uses ProgressBar for parallel processing across available GPUs
    5. Executes tracking on each video and saves results

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (list): List of dataset names to process
            - tracker (str): Tracking algorithm to use
            - max_age (int): Maximum age for SORT tracker
            - min_hits (int): Minimum hits for SORT tracker
            - iou_threshold (float): IOU threshold for SORT tracker

    Note:
        - The script expects detection results from 001_preprocess_groundtruth_detection.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/groundtruth/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/groundtruth/tracking.jsonl
        - Linear interpolation is performed to fill missing detections in tracks
        - Processing is parallelized across available GPUs for improved performance
    """
    datasets = args.datasets

    print(f"Using tracker: {args.tracker}")
    print(f"Tracker parameters: max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_threshold}")

    # Create task functions
    funcs = []
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Find all videos with detection results
        dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
        if not os.path.exists(dataset_cache_dir):
            print(f"Dataset cache directory {dataset_cache_dir} does not exist, skipping...")
            continue

        # Look for directories that contain detection results in execution subdirectory
        execution_dir = os.path.join(dataset_cache_dir, 'execution')
        if not os.path.exists(execution_dir):
            print(f"Execution directory {execution_dir} does not exist, skipping dataset {dataset}...")
            continue

        video_dirs = []
        for item in os.listdir(execution_dir):
            item_path = os.path.join(execution_dir, item)
            if os.path.isdir(item_path):
                detection_path = os.path.join(item_path, 'groundtruth', 'detections.jsonl')
                if os.path.exists(detection_path):
                    video_dirs.append(item)

        if not video_dirs:
            print(f"No videos with detection results found in {execution_dir}")
            continue

        print(f"Found {len(video_dirs)} videos with detection results")

        funcs.extend(
            partial(track, video_file, args, CACHE_DIR, dataset)
            for video_file in video_dirs
        )

    assert len(funcs) > 0, "No videos found to process across all datasets"

    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")

    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)

    print("All datasets processed successfully!")


if __name__ == '__main__':
    main(parse_args())
