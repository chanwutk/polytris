#!/usr/local/bin/python

import argparse
import os
import multiprocessing as mp
from multiprocessing import Queue
from functools import partial

from polyis.utilities import CACHE_DIR, DATASETS_DIR, PREFIX_TO_VIDEOSET, ProgressBar, create_tracking_visualization, load_detection_results, get_config


CONFIG = get_config()
EXEC_DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tracking results on original videos')
    parser.add_argument('--speed_up', type=int, default=4,
                        help='Speed up factor for visualization (process every Nth frame)')
    parser.add_argument('--track_ids', type=int, nargs='*', default=None,
                        help='List of track IDs to color (others will be grey)')
    parser.add_argument('--detection_only', action='store_true',
                        help='Only show detections without trajectories, all boxes in green without track IDs')
    return parser.parse_args()


def visualize_video(video_file: str, cache_dir: str, dataset: str, speed_up: int,
                    track_ids: list[int] | None, detection_only: bool, process_id: int, progress_queue: Queue):
    """
    Process visualization for a single video file.
    
    Args:
        video_file (str): Name of the video file to process
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        speed_up (int): Speed up factor for visualization (process every Nth frame)
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, only show detections without trajectories, all boxes in green without track IDs
        process_id (int): Process ID for logging
        progress_queue (Queue): Queue for progress updates
    """
    # Load tracking results
    tracking_results_raw = load_detection_results(cache_dir, dataset, video_file, tracking=True, filename='tracking.jsonl')
    
    # Convert to the format expected by create_tracking_visualization
    tracking_results = {}
    for frame_result in tracking_results_raw:
        frame_idx = frame_result['frame_idx']
        tracks = frame_result['tracks']
        tracking_results[frame_idx] = tracks
    
    # Get path to original video
    video_path = os.path.join(DATASETS_DIR, dataset, PREFIX_TO_VIDEOSET[video_file[:2]], video_file)
    assert os.path.exists(video_path), f"Original video not found for {video_path}"
    
    # Create output path for visualization
    output_path = os.path.join(cache_dir, dataset, 'execution', video_file,
                               '000_groundtruth', f'annotated_{video_file}')
    
    # Create visualization
    create_tracking_visualization(video_path, tracking_results, output_path, speed_up,
                                  process_id, progress_queue, track_ids, detection_only)


def main(args):
    """
    Main function that orchestrates the tracking visualization process.
    
    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Finds all videos with tracking results
    4. Creates a process pool for parallel processing
    5. Creates visualizations for each video and saves results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (list): List of dataset names to process
            - speed_up (int): Speed up factor for visualization
        
    Note:
        - The script expects tracking results from p002_preprocess_groundtruth_tracking.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl
        - Original videos are read from {DATASETS_DIR}/{dataset}/
        - Visualization videos are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/annotated_{video_file}
        - Each track ID gets a unique color from a predefined palette
        - Processing is parallelized for improved performance
    """
    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")
    
    # Create task functions
    funcs = []
    for dataset in EXEC_DATASETS:
        print(f"Processing dataset: {dataset}")
        
        # Find all videos with tracking results
        dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
        if not os.path.exists(dataset_cache_dir):
            print(f"Dataset cache directory {dataset_cache_dir} does not exist, skipping...")
            continue
        
        # Look for directories that contain tracking results in execution subdirectory
        execution_dir = os.path.join(dataset_cache_dir, 'execution')
        if not os.path.exists(execution_dir):
            print(f"Execution directory {execution_dir} does not exist, skipping dataset {dataset}...")
            continue
            
        video_dirs = []
        for item in os.listdir(execution_dir):
            item_path = os.path.join(execution_dir, item)
            if os.path.isdir(item_path):
                tracking_path = os.path.join(item_path, '000_groundtruth', 'tracking.jsonl')
                if os.path.exists(tracking_path):
                    video_dirs.append(item)
        
        if not video_dirs:
            print(f"No videos with tracking results found in {execution_dir}")
            continue
        
        print(f"Found {len(video_dirs)} videos with tracking results")
        
        funcs.extend(
            partial(visualize_video, video_file, CACHE_DIR, dataset, args.speed_up, args.track_ids, args.detection_only)
            for video_file in video_dirs
        )
    
    assert len(funcs) > 0, "No videos found to process across all datasets"
    
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), len(funcs), 40)  # Cap at 20 processes

    mp.set_start_method('spawn', force=True)
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
