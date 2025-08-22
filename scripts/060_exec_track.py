#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import tqdm
import multiprocessing as mp

from modules.b3d.b3d.external.sort import Sort

CACHE_DIR = '/polyis-cache'
TILE_SIZES = [64]


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (str): Tile size to use for tracking (choices: '64', '128', 'all')
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
    parser.add_argument('--tile_size', type=str, choices=['64', '128', 'all'], default='all',
                        help='Tile size to use for tracking (or "all" for all tile sizes)')
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


def load_detection_results(cache_dir: str, dataset: str, video_file: str, tile_size: int) -> list[dict]:
    """
    Load detection results from the uncompressed detections JSONL file.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        video_file (str): Video file name
        tile_size (int): Tile size used for detections
        
    Returns:
        list[dict]: list of frame detection results
        
    Raises:
        FileNotFoundError: If no detection results file is found
    """
    detection_path = os.path.join(cache_dir, dataset, video_file, 'uncompressed_detections', f'proxy_{tile_size}', 'detections.jsonl')
    
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


def create_tracker(tracker_name: str, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
    """
    Create a tracker instance based on the specified algorithm.
    
    Args:
        tracker_name (str): Name of the tracking algorithm
        max_age (int): Maximum age for SORT tracker
        min_hits (int): Minimum hits for SORT tracker
        iou_threshold (float): IOU threshold for SORT tracker
        
    Returns:
        Tracker instance
        
    Raises:
        ValueError: If the tracker name is not supported
    """
    if tracker_name == 'sort':
        print(f"Creating SORT tracker with max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
        return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")


def interpolate_trajectory(trajectory: list[tuple[int, np.ndarray]], nxt: tuple[int, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    """
    Perform linear interpolation between two trajectory points except the last point (nxt).
    
    Args:
        trajectory (list[tuple[int, np.ndarray]]): list of (frame_idx, detection) tuples
        nxt (tuple[int, np.ndarray]): Next detection point (frame_idx, detection)
        
    Returns:
        list[tuple[int, np.ndarray]]: list of interpolated points
    """
    extend: list[tuple[int, np.ndarray]] = []
    
    if len(trajectory) != 0:
        prv = trajectory[-1]
        assert prv[0] < nxt[0]
        prv_det = prv[1]
        nxt_det = nxt[1]
        dif_det = nxt_det - prv_det
        dif_det = dif_det.reshape(1, -1)

        scale = np.arange(0, nxt[0] - prv[0], dtype=np.float32).reshape(-1, 1) / (nxt[0] - prv[0])
        
        int_dets = (scale @ dif_det) + prv_det.reshape(1, -1)

        for idx, int_det in enumerate(int_dets[:-1]):
            extend.append((prv[0] + idx + 1, int_det))

    return extend


def track_objects_in_video(video_file: str, detection_results: list[dict], tracker_name: str, 
                           max_age: int, min_hits: int, iou_threshold: float, 
                           no_interpolate: bool, output_path: str):
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
        for frame_result in tqdm.tqdm(detection_results, desc="Tracking objects"):
            frame_idx = frame_result['frame_idx']
            bboxes = frame_result['bboxes']
            
            # Start timing for this frame
            frame_start_time = time.time()
            step_times = {}
            
            # Profile: Convert detections to numpy array
            step_start = time.time()
            dets = np.array(bboxes)
            if dets.size > 0:
                dets = dets[:, :5]  # Take first 5 columns: x1, y1, x2, y2, score
            else:
                dets = np.empty((0, 5))
            step_times['convert_detections'] = time.time() - step_start
            
            # Profile: Update tracker
            step_start = time.time()
            trackers = tracker.update(dets)
            step_times['tracker_update'] = time.time() - step_start
            
            # Profile: Process tracking results
            step_start = time.time()
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
            
            step_times['process_results'] = time.time() - step_start
            
            # Calculate total frame processing time
            step_times['total_frame_time'] = time.time() - frame_start_time
            
            # Save runtime data for this frame
            runtime_data = {
                'frame_idx': frame_idx,
                'step_times': step_times,
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


def process_video_tracking(video_file: str, args, cache_dir: str, dataset: str, tile_size: int):
    """
    Process tracking for a single video file.
    
    Args:
        video_file (str): Name of the video file to process
        args: Command line arguments
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        tile_size (int): Tile size used for detections
    """
    # Load detection results
    detection_results = load_detection_results(cache_dir, dataset, video_file, tile_size)
    
    # Create output path for tracking results
    output_path = os.path.join(cache_dir, dataset, video_file, 'uncompressed_tracking',
                               f'proxy_{tile_size}', 'tracking.jsonl')
    
    # Execute tracking
    track_objects_in_video(video_file, detection_results, args.tracker, args.max_age, args.min_hits,
                           args.iou_threshold, args.no_interpolate, output_path)


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
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_detections/proxy_{tile_size}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/proxy_{tile_size}/tracks.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - When tile_size is 'all', all available tile sizes are processed
    """
    print(f"Processing dataset: {args.dataset}")
    print(f"Using tracker: {args.tracker}")
    print(f"Tracker parameters: max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_threshold}")
    print(f"Interpolation: {'enabled' if not args.no_interpolate else 'disabled'}")
    
    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all available tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")
    
    # Find all videos with uncompressed detection results
    dataset_cache_dir = os.path.join(CACHE_DIR, args.dataset)
    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(f"Dataset cache directory {dataset_cache_dir} does not exist")
    
    # Look for directories that contain uncompressed detection results
    video_tile_combinations = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            for tile_size in tile_sizes_to_process:
                detection_path = os.path.join(item_path, 'uncompressed_detections', f'proxy_{tile_size}', 'detections.jsonl')
                if os.path.exists(detection_path):
                    video_tile_combinations.append((item, tile_size))
    
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
    for video_file, tile_size in video_tile_combinations:
        video_args.append((video_file, args, CACHE_DIR, args.dataset, tile_size))
        print(f"Prepared video: {video_file} with tile size: {tile_size}")
    
    # Use process pool to execute video tracking
    with mp.Pool(processes=num_processes) as pool:
        print(f"Starting video tracking with {num_processes} parallel workers...")
        
        # Map the work to the pool
        results = pool.starmap(process_video_tracking, video_args)
        
        print("All videos tracked successfully!")


if __name__ == '__main__':
    main(parse_args())
