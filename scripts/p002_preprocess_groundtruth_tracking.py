#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue

from scripts.utilities import CACHE_DIR, create_tracker, format_time, interpolate_trajectory, load_detection_results, progress_bars


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tracker (str): Tracking algorithm to use (default: 'sort')
            - max_age (int): Maximum age for SORT tracker (default: 1)
            - min_hits (int): Minimum hits for SORT tracker (default: 3)
            - iou_threshold (float): IOU threshold for SORT tracker (default: 0.3)
    """
    parser = argparse.ArgumentParser(description='Execute object tracking on detection results')
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
    return parser.parse_args()


def track_objects_in_video(video_index: int, video_file: str, detection_results: list[dict],
                           tracker_name: str, max_age: int, min_hits: int, iou_threshold: float,
                           output_path: str, progress_queue: Queue):
    """
    Execute object tracking on detection results and save tracking results to JSONL.
    
    Args:
        video_index (int): Index of the video (0-based)
        video_file (str): Name of the video file being processed
        detection_results (list[dict]): list of detection results from load_detection_results
        tracker_name (str): Name of the tracking algorithm
        max_age (int): Maximum age for SORT tracker
        min_hits (int): Minimum hits for SORT tracker
        iou_threshold (float): IOU threshold for SORT tracker
        output_path (str): Path where the output JSONL file will be saved
    """
    print(f"Processing video: {video_file}")
    
    # Create tracker
    tracker = create_tracker(tracker_name, max_age, min_hits, iou_threshold)
    
    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    
    print(f"Processing {len(detection_results)} frames for tracking...")
    
    # Send initial progress update
    progress_queue.put((f'cuda:{video_index}', {
        'description': video_file,
        'completed': 0,
        'total': len(detection_results)
    }))
    
    runtime_path = output_path.replace('tracking.jsonl', 'tracking_runtimes.jsonl')
    with open(runtime_path, 'w') as runtime_file:
        # Process each frame
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
            trackers = tracker.update(dets)
            step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start
            
            # Process tracking results
            step_start = (time.time_ns() / 1e6)
            if trackers.size > 0:
                for track in trackers:
                    # SORT returns: [x1, y1, x2, y2, track_id]
                    x1, y1, x2, y2, track_id = track
                    track_id = int(track_id)
                    
                    # Convert to detection format: [track_id, x1, y1, x2, y2]
                    detection = [track_id, x1, y1, x2, y2]
                    
                    # Add to frame tracks
                    if frame_idx not in frame_tracks:
                        frame_tracks[frame_idx] = []
                    # frame_tracks[frame_idx].append(detection)

                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    box_array = np.array([x1, y1, x2, y2], dtype=np.float32)

                    
                    extend = interpolate_trajectory(trajectories[track_id], (frame_idx, box_array))
                    
                    # Add interpolated points to frame tracks
                    for e in [*extend, (frame_idx, box_array)]:
                        e_frame_idx, e_box = e
                        if e_frame_idx not in frame_tracks:
                            frame_tracks[e_frame_idx] = []
                        
                        # Convert back to list format: [track_id, x1, y1, x2, y2]
                        e_detection = [track_id, *e_box.tolist()]
                        frame_tracks[e_frame_idx].append(e_detection)

                        # Add interpolated points to trajectories
                        trajectories[track_id].append((e_frame_idx, e_box))

            # Handle frames with no detections
            if frame_idx not in frame_tracks:
                frame_tracks[frame_idx] = []

            step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start
            runtime_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_detections': len(dets),
                'num_tracks': trackers.size if trackers.size > 0 else 0
            }
            runtime_file.write(json.dumps(runtime_data) + '\n')
            
            # Send progress update
            progress_queue.put((f'cuda:{video_index}', {'completed': fidx + 1}))
    
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


def process_video_tracking(video_index: int, video_file: str, args, cache_dir: str, dataset: str, progress_queue: Queue):
    """
    Process tracking for a single video file.
    
    Args:
        video_index (int): Index of the video (0-based)
        video_file (str): Name of the video file to process
        args: Command line arguments
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
    """
    # Load detection results
    detection_results = load_detection_results(cache_dir, dataset, video_file)
    
    # Create output path for tracking results
    output_path = os.path.join(cache_dir, dataset, video_file, 'groundtruth', 'tracking.jsonl')
    
    # Execute tracking
    track_objects_in_video(
        video_index, video_file, detection_results, args.tracker,
        args.max_age, args.min_hits, args.iou_threshold, output_path, progress_queue
    )
    
    print(f"Completed tracking for video: {video_file}")


def main(args):
    """
    Main function that orchestrates the object tracking process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Finds all videos with detection results
    3. Creates a process pool for parallel processing
    4. Executes tracking on each video and saves results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects detection results from 001_preprocess_groundtruth_detection.py in:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/detection.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/tracking.jsonl
        - Linear interpolation is performed to fill missing detections in tracks
        - Processing is parallelized for improved performance
    """
    print(f"Processing dataset: {args.dataset}")
    print(f"Using tracker: {args.tracker}")
    print(f"Tracker parameters: max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_threshold}")
    
    # Find all videos with detection results
    dataset_cache_dir = os.path.join(CACHE_DIR, args.dataset)
    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    # Look for directories that contain detection results
    video_dirs = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            detection_path = os.path.join(item_path, 'groundtruth', 'detection.jsonl')
            if os.path.exists(detection_path):
                video_dirs.append(item)
    
    if not video_dirs:
        print(f"No videos with detection results found in {dataset_cache_dir}")
        return
    
    print(f"Found {len(video_dirs)} videos with detection results")
    
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), len(video_dirs), 40)  # Cap at 40 processes
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create a pool of workers
    print(f"Creating process pool with {num_processes} workers...")
    
    # Prepare arguments for each video
    video_args = []
    for i, video_file in enumerate(video_dirs):
        video_args.append((i, video_file, args, CACHE_DIR, args.dataset))
        print(f"Prepared video {i}: {video_file}")
    
    # Create progress queue and start progress display
    progress_queue = Queue()
    
    # Start progress display in a separate process
    progress_process = mp.Process(target=progress_bars, args=(progress_queue, num_processes, len(video_dirs)))
    progress_process.start()
    
    # Create and start video processing processes
    processes: list[mp.Process] = []
    for i, (video_index, video_file, args, cache_dir, dataset) in enumerate(video_args):
        process = mp.Process(target=process_video_tracking, 
                           args=(video_index, video_file, args, cache_dir, dataset, progress_queue))
        process.start()
        processes.append(process)
    
    # Wait for all video processing to complete
    for process in processes:
        process.join()
        process.terminate()
    
    # Signal progress display to stop and wait for it
    progress_queue.put(None)
    progress_process.join()
    
    print("All videos tracked successfully!")


if __name__ == '__main__':
    main(parse_args())
