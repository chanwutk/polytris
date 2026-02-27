#!/usr/local/bin/python

import argparse
import os
import time
import numpy as np
from functools import partial
import queue
import multiprocessing as mp

from polyis.utilities import create_tracker, dedupe_datasets_by_root, get_video_resolution, load_detection_results, ProgressBar, register_tracked_detections, get_config, save_tracking_results


CONFIG = get_config()
EXEC_DATASETS = CONFIG['EXEC']['DATASETS']
VIDEO_SETS = CONFIG['EXEC']['VIDEO_SETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess groundtruth tracking data')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def prepare_frame_tracks_for_save(
    frame_tracks: dict[int, list[list[float]]],
    detection_frame_count: int,
    frame_rate: float
) -> dict[int, list[list[float]]]:
    # Keep original frame mapping for 15 FPS inputs.
    if frame_rate <= 15:
        return frame_tracks

    # Build output tracks sampled every two source frames with sequential indices.
    sampled_frame_tracks: dict[int, list[list[float]]] = {}
    output_frame_idx = 0
    for source_frame_idx in range(0, detection_frame_count, 2):
        sampled_frame_tracks[output_frame_idx] = frame_tracks.get(source_frame_idx, [])
        output_frame_idx += 1
    return sampled_frame_tracks


def track(dataset: str, video_file: str, gpu_id: int, command_queue: "queue.Queue[tuple[str, dict]]"):
    """
    Execute object tracking on detection results and save tracking results to JSONL.

    Args:
        dataset (str): Dataset name
        video_file (str): Name of the video file to process
        tracker_name (str): Name of the tracker to use
        gpu_id (int): GPU device ID to use for this process
        command_queue (Queue): Queue for progress updates
    """
    # Load detection results
    detection_results = load_detection_results(CACHE_DIR, dataset, video_file, filename='detection.jsonl', groundtruth=True)

    # Create output path for tracking results
    output_path = os.path.join(CACHE_DIR, dataset, 'execution', video_file, '003_groundtruth', 'tracking.jsonl')

    # print(f"Processing video: {video_file}")
    # Create tracker
    # tracker = create_tracker('sort')
    # tracker_cython = create_tracker('sort-cython')
    resolution = get_video_resolution(dataset, video_file)
    width, height = resolution
    # Use detection frame count to infer the effective video duration category.
    detection_frame_count = len(detection_results)
    # Use 15 FPS tracking parameters for shorter clips.
    if detection_frame_count < 1000:
        frame_rate = 15
    # Use 29.97 FPS only for JNC datasets; use 30 FPS for other long clips.
    else:
        frame_rate = 29.97 if dataset.startswith('jnc') else 30
    tracker = create_tracker('bytetrackcython', img_size=(height, width), frame_rate=frame_rate, track_buffer=40 if frame_rate == 15 else 20)

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

    # Process each frame
    mod = max(1, int(len(detection_results) * 0.05))
    for fidx, frame_result in enumerate(detection_results):
        frame_idx = frame_result['frame_idx']
        assert frame_idx == fidx, f"Frame index mismatch: {frame_idx} != {fidx}"
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

        # # Update tracker
        # dets_python = dets.copy()
        # step_start = (time.time_ns() / 1e6)
        # tracked_dets = tracker.update(dets_python)
        # step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start

        step_start = (time.time_ns() / 1e6)
        tracked_dets = tracker.update(dets)
        # step_times['tracker_update_cython'] = (time.time_ns() / 1e6) - step_start
        step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start

        # assert np.array_equal(tracked_dets, tracked_dets_cython), f"Tracking results mismatch: {tracked_dets} != {tracked_dets_cython}"

        # Process tracking results
        step_start = (time.time_ns() / 1e6)
        register_tracked_detections(tracked_dets, frame_idx, frame_tracks, trajectories)
        step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start

        # Send progress update
        if fidx % mod == 0:
            command_queue.put((f'cuda:{gpu_id}', {'completed': fidx + 1}))

    # print(f"Tracking completed. Found {len(trajectories)} unique tracks.")

    # Save tracking results
    # print(f"Saving tracking results to: {output_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare frame tracks for saving based on effective tracking frame rate.
    output_frame_tracks = prepare_frame_tracks_for_save(frame_tracks, detection_frame_count, frame_rate)
    save_tracking_results(output_frame_tracks, output_path)


def main():
    """
    Main function that orchestrates the object tracking process.

    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Finds all videos with detection results
    4. Uses ProgressBar for parallel processing across available GPUs
    5. Executes tracking on each video and saves results

    Note:
        - The script expects detection results from 001_preprocess_groundtruth_detection.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl
        - Linear interpolation is performed to fill missing detections in tracks
        - Processing is parallelized across available GPUs for improved performance
    """
    args = parse_args()
    
    # Determine which videosets to process based on arguments
    splits = []
    if args.test:
        splits.append('test')
    if args.train:
        splits.append('train')
    if args.valid:
        splits.append('valid')
    
    # If no videosets are specified, default to test
    if not splits:
        splits = ['test']
    
    # Resolve configured datasets to unique dataset roots.
    datasets_to_process = dedupe_datasets_by_root(EXEC_DATASETS)
    print(f'Resolved datasets for groundtruth tracking preprocessing: {datasets_to_process}')

    funcs = []
    for dataset in datasets_to_process:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"
        
        # Get all video files from the dataset directory
        video_files: list[str] = []
        for videoset in splits:
            videoset_dir = os.path.join(dataset_dir, videoset)
            assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"
            video_files.extend([videoset + '/' + f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        assert len(video_files) > 0, f"No video files found in {dataset_dir}"
        
        for video_file in video_files:
            video = video_file.split('/')[-1]
            funcs.append(partial(track, dataset, video))
    
    # Determine number of available GPUs
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), int(mp.cpu_count() * 0.8))
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=max_processes, num_tasks=len(funcs), refresh_per_second=10).run_all(funcs)


if __name__ == '__main__':
    main()
