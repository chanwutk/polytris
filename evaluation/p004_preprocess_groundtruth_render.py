#!/usr/local/bin/python

import argparse
import os
import multiprocessing as mp
from multiprocessing import Queue
from functools import partial

from polyis.io import cache, store
from polyis.utilities import PREFIX_TO_VIDEOSET, ProgressBar, create_tracking_visualization, dedupe_datasets_by_root, load_detection_results, get_config


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


def visualize_video(video_file: str, dataset: str, speed_up: int,
                    track_ids: list[int] | None, detection_only: bool, groundtruth: bool,
                    process_id: int, progress_queue: Queue):
    """
    Process visualization for a single video file.

    Args:
        video_file (str): Name of the video file to process
        dataset (str): Dataset name
        speed_up (int): Speed up factor for visualization (process every Nth frame)
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, only show detections without trajectories, all boxes in green without track IDs
        process_id (int): Process ID for logging
        progress_queue (Queue): Queue for progress updates
    """
    # Load tracking results
    tracking_results_raw = load_detection_results(dataset, video_file, tracking=True,
                                                  filename='tracking.jsonl', groundtruth=groundtruth)
    
    # Convert to the format expected by create_tracking_visualization
    tracking_results = {}
    for frame_result in tracking_results_raw:
        frame_idx = frame_result['frame_idx']
        tracks = frame_result['tracks']
        tracking_results[frame_idx] = tracks
    
    # Get path to original video
    video_path = store.dataset(dataset, PREFIX_TO_VIDEOSET[video_file[:2]], video_file)
    assert video_path.exists(), f"Original video not found for {video_path}"

    # Create output path for visualization
    stage = 'groundtruth' if groundtruth else 'naive'
    output_path = cache.exec(dataset, stage, video_file, f'annotated_{video_file}')
    
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
          {CACHE_DIR}/{dataset}/execution/{video_file}/003_groundtruth/tracking.jsonl
        - Original videos are read from {DATASETS_DIR}/{dataset}/
        - Visualization videos are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/003_groundtruth/annotated_{video_file}
        - Each track ID gets a unique color from a predefined palette
        - Processing is parallelized for improved performance
    """
    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")
    
    # Resolve configured datasets to unique dataset roots.
    datasets_to_process = dedupe_datasets_by_root(EXEC_DATASETS)
    print(f'Resolved datasets for groundtruth render preprocessing: {datasets_to_process}')

    # Create task functions
    funcs = []
    for dataset in datasets_to_process:
        print(f"Processing dataset: {dataset}")
        
        # Find all videos with tracking results
        execution_dir = cache.execution(dataset)
        if not execution_dir.exists():
            print(f"Execution directory {execution_dir} does not exist, skipping dataset {dataset}...")
            continue

        video_dirs = []
        for item in os.listdir(execution_dir):
            if not item.startswith('te') or not (execution_dir / item).is_dir():
                continue
            tracking_path = cache.exec(dataset, 'groundtruth', item, 'tracking.jsonl')
            if tracking_path.exists():
                video_dirs.append(item)
        
        if not video_dirs:
            print(f"No videos with tracking results found in {execution_dir}")
            continue
        
        print(f"Found {len(video_dirs)} videos with tracking results")
        
        for groundtruth in [True]:
            funcs.extend(
                partial(visualize_video, video_file, dataset,
                        args.speed_up, args.track_ids, args.detection_only, groundtruth)
                for video_file in video_dirs
            )
    
    assert len(funcs) > 0, "No videos found to process across all datasets"
    
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), len(funcs), 40)  # Cap at 20 processes

    mp.set_start_method('spawn', force=True)
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
