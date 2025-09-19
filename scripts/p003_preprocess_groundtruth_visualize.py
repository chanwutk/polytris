#!/usr/local/bin/python

import argparse
import os
import multiprocessing as mp
from multiprocessing import Queue
from functools import partial

from polyis.utilities import CACHE_DIR, DATA_DIR, ProgressBar, create_tracking_visualization, load_tracking_results


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - speed_up (int): Speed up factor for visualization (default: 4)
    """
    parser = argparse.ArgumentParser(description='Visualize tracking results on original videos')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--speed_up', type=int, default=4,
                        help='Speed up factor for visualization (process every Nth frame)')
    return parser.parse_args()



def process_video_visualization(video_file: str, cache_dir: str,
                                dataset: str, speed_up: int,
                                process_id: int, progress_queue: Queue):
    """
    Process visualization for a single video file.
    
    Args:
        video_file (str): Name of the video file to process
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        speed_up (int): Speed up factor for visualization (process every Nth frame)
        process_id (int): Process ID for logging
    """
    # Load tracking results
    tracking_results = load_tracking_results(cache_dir, dataset, video_file)
    
    # Get path to original video
    video_path = os.path.join(DATA_DIR, dataset, video_file)
    assert os.path.exists(video_path), f"Original video not found for {video_file}"
    
    # Create output path for visualization
    output_path = os.path.join(cache_dir, dataset, video_file, 'groundtruth', f'annotated_groundtruth_{video_file}')
    
    # Create visualization
    create_tracking_visualization(video_path, tracking_results, output_path, speed_up, process_id, progress_queue)


def main(args):
    """
    Main function that orchestrates the tracking visualization process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Finds all videos with tracking results
    3. Creates a process pool for parallel processing
    4. Creates visualizations for each video and saves results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tracking results from 002_preprocess_groundtruth_tracking.py in:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/tracking.jsonl
        - Original videos are read from {DATA_DIR}/{dataset}/
        - Visualization videos are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/visualization.mp4
        - Each track ID gets a unique color from a predefined palette
        - Processing is parallelized for improved performance
    """
    print(f"Processing dataset: {args.dataset}")
    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")
    
    # Find all videos with tracking results
    dataset_cache_dir = os.path.join(CACHE_DIR, args.dataset)
    
    # Look for directories that contain tracking results
    funcs = []
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            tracking_path = os.path.join(item_path, 'groundtruth', 'tracking.jsonl')
            if os.path.exists(tracking_path):
                funcs.append(partial(process_video_visualization,
                             item, CACHE_DIR, args.dataset, args.speed_up))
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), len(funcs), 20)  # Cap at 20 processes

    mp.set_start_method('spawn', force=True)
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
