#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import tqdm
import multiprocessing as mp

from polyis.utilities import create_tracker, format_time, interpolate_trajectory, CACHE_DIR


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
    detection_path = os.path.join(cache_dir, dataset, video_file, 'uncompressed_detections', f'{classifier}_{tile_size}', 'detections.jsonl')
    
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


def track_objects_in_video(video_file: str, detection_results: list[dict], tracker_name: str, 
                           max_age: int, min_hits: int, iou_threshold: float, 
                           no_interpolate: bool, output_path: str, i: int):
    """
    Execute object tracking on detection results and save tracking results to JSONL.
    
    Args:
        video_file (str): Name of the video file being processed
        detection_results (list[dict]): list of detection results from load_detection_results
        tracker_name (str): Name of the tracking algorithm
        max_age (int): Maximum age for SORT tracker
        min_hits (int): Minimum hits for SORT tracker
        iou_threshold (float): IOU threshold for SORT tracker
        no_interpolate (bool): Whether to not perform trajectory interpolation
        output_path (str): Path where the output JSONL file will be saved
        i (int): Index of the video-tile combination
    """
    print(f"Processing video: {video_file}")
    
    # Create tracker
    tracker = create_tracker(tracker_name, max_age, min_hits, iou_threshold)
    
    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    
    print(f"Processing {len(detection_results)} frames for tracking...")
    
    # Create runtime output file
    runtime_path = output_path.replace('tracking.jsonl', 'runtimes.jsonl')
    runtime_dir = os.path.dirname(runtime_path)
    os.makedirs(runtime_dir, exist_ok=True)
    
    with open(runtime_path, 'w') as runtime_file:
        # Process each frame
        for frame_result in tqdm.tqdm(detection_results, desc=f"Tracking objects ({i})",
                                      position=i, leave=False):
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
            trackers = tracker.update(dets)
            step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Process tracking results
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
                    frame_tracks[frame_idx].append(detection)

                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    box_array = np.array([x1, y1, x2, y2], dtype=np.float32)
                    
                    # Add to trajectories for interpolation (if enabled)
                    if not no_interpolate:
                        extend = interpolate_trajectory(trajectories[track_id], (frame_idx, box_array))
                        
                        # Add interpolated points to frame tracks
                        for e in extend:
                            e_frame_idx, e_box = e
                            if e_frame_idx not in frame_tracks:
                                frame_tracks[e_frame_idx] = []
                            
                            # Convert back to list format: [track_id, x1, y1, x2, y2]
                            e_detection = [track_id, *e_box.tolist()]
                            frame_tracks[e_frame_idx].append(e_detection)

                            # Add interpolated points to trajectories
                            trajectories[track_id].append((e_frame_idx, e_box))

                    trajectories[track_id].append((frame_idx, box_array))

            # Handle frames with no detections
            if frame_idx not in frame_tracks:
                frame_tracks[frame_idx] = []
            
            step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start
            
            # Save runtime data for this frame
            runtime_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_detections': len(bboxes),
                'num_tracks': trackers.size if trackers.size > 0 else 0
            }
            runtime_file.write(json.dumps(runtime_data) + '\n')
    
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


def process_video_tracking(video_file: str, args, cache_dir: str, dataset: str, tile_size: int, classifier: str, i: int):
    """
    Process tracking for a single video file.
    
    Args:
        video_file (str): Name of the video file to process
        args: Command line arguments
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        tile_size (int): Tile size used for detections
        classifier (str): Classifier name used for detections
        i (int): Index of the video-tile combination
    """
    # Load detection results
    detection_results = load_detection_results(cache_dir, dataset, video_file, tile_size, classifier)
    
    # Create output path for tracking results
    output_path = os.path.join(cache_dir, dataset, video_file, 'uncompressed_tracking',
                               f'{classifier}_{tile_size}', 'tracking.jsonl')
    
    # Execute tracking
    track_objects_in_video(video_file, detection_results, args.tracker, args.max_age, args.min_hits,
                           args.iou_threshold, args.no_interpolate, output_path, i)


def main(args):
    """
    Main function that orchestrates the object tracking process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Finds all videos with uncompressed detection results for the specified tile size(s)
    3. Creates a process pool for parallel processing
    4. Executes tracking on each video and saves results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects uncompressed detection results from 050_exec_uncompress.py in:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_detections/{classifier}_{tile_size}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/{classifier}_{tile_size}/tracking.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - When tile_size is 'all', all available tile sizes are processed
    """
    print(f"Processing dataset: {args.dataset}")
    print(f"Using tracker: {args.tracker}")
    print(f"Tracker parameters: max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_threshold}")
    print(f"Interpolation: {'enabled' if not args.no_interpolate else 'disabled'}")
    
    # Find all videos with uncompressed detection results
    dataset_cache_dir = os.path.join(CACHE_DIR, args.dataset)
    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    # Look for directories that contain uncompressed detection results
    video_tile_combinations = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            uncompressed_detections_dir = os.path.join(item_path, 'uncompressed_detections')
            for classifier_tilesize in sorted(os.listdir(uncompressed_detections_dir)):
                classifier, tile_size = classifier_tilesize.split('_')
                tile_size = int(tile_size)
                video_tile_combinations.append((item, tile_size, classifier))
    
    if not video_tile_combinations:
        print(f"No videos with uncompressed detection results found in {dataset_cache_dir}")
        return
    
    print(f"Found {len(video_tile_combinations)} video-tile size combinations to process")
    
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), len(video_tile_combinations), 40)  # Cap at 40 processes
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create a pool of workers
    print(f"Creating process pool with {num_processes} workers...")
    
    # Prepare arguments for each video-tile combination
    video_args = []
    for i, (video_file, tile_size, classifier) in enumerate(video_tile_combinations):
        video_args.append((video_file, args, CACHE_DIR, args.dataset, tile_size, classifier, i))
        print(f"Prepared video: {video_file} with tile size: {tile_size} and classifier: {classifier} ({i+1}/{len(video_tile_combinations)})")
    
    # Use process pool to execute video tracking
    with mp.Pool(processes=num_processes) as pool:
        print(f"Starting video tracking with {num_processes} parallel workers...")
        results = pool.starmap(process_video_tracking, video_args)


if __name__ == '__main__':
    main(parse_args())
