#!/usr/local/bin/python

import argparse
import os
import multiprocessing as mp
from multiprocessing import Queue
from functools import partial

import cv2
import numpy as np

from polyis.utilities import (
    CACHE_DIR, DATASETS_DIR, PREFIX_TO_VIDEOSET, ProgressBar,
    load_detection_results, DATASETS_TO_TEST, to_h264,
    create_visualization_frame
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tracking results on frame difference videos')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--speed_up', type=int, default=4,
                        help='Speed up factor for visualization (process every Nth frame)')
    parser.add_argument('--track_ids', type=int, nargs='*', default=None,
                        help='List of track IDs to color (others will be grey)')
    parser.add_argument('--detection_only', action='store_true',
                        help='Only show detections without trajectories, all boxes in green without track IDs')
    return parser.parse_args()


def create_tracking_visualization_diff(video_path: str, tracking_results: dict[int, list[list[float]]],
                                       output_path: str, speed_up: int, process_id: int, progress_queue=None,
                                       track_ids: list[int] | None = None, detection_only: bool = False):
    """
    Create a visualization video showing tracking results overlaid on frame difference video.

    Args:
        video_path (str): Path to the input video file
        tracking_results (dict[int, list[list[float]]]): Tracking results from load_tracking_results
        output_path (str): Path where the output visualization video will be saved
        speed_up (int): Speed up factor for visualization (process every Nth frame)
        process_id (int): Process ID for logging
        progress_queue: Queue for progress updates
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, only show detections without trajectories, all boxes in green without track IDs
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        cap.release()
        return

    # Initialize trajectory history for all tracks with frame timestamps
    trajectory_history: dict[int, list[tuple[int, int, int]]] = {}  # track_id -> [(x, y, frame_idx), ...]

    # Initialize frame_idx for exception handling
    frame_idx = 0

    # Send initial progress update
    if progress_queue is not None:
        progress_queue.put((f'cuda:{process_id}', {
            'description': os.path.basename(video_path),
            'completed': 0,
            'total': frame_count
        }))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}")
        cap.release()
        writer.release()
        return

    # Process first frame (use black frame as diff since there's no previous frame)
    diff_frame = np.zeros_like(prev_frame, dtype=np.uint8)

    # Get tracking results for first frame
    tracks = tracking_results.get(0, [])

    # Create visualization frame with trajectory history
    vis_frame = create_visualization_frame(diff_frame, tracks, 0, trajectory_history,
                                           speed_up, track_ids, detection_only)

    # Write frame to video if not skipped
    if vis_frame is not None:
        writer.write(vis_frame)

    # Send progress update for first frame
    if progress_queue is not None:
        progress_queue.put((f'cuda:{process_id}', {'completed': 1}))

    # Process remaining frames
    for frame_idx in range(1, frame_count):
        # Read current frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Compute absolute difference between consecutive frames
        diff_frame = cv2.absdiff(prev_frame, curr_frame)

        # Get tracking results for this frame
        tracks = tracking_results.get(frame_idx, [])

        # Create visualization frame with trajectory history
        vis_frame = create_visualization_frame(diff_frame, tracks, frame_idx, trajectory_history,
                                               speed_up, track_ids, detection_only)

        # Write frame to video if not skipped
        if vis_frame is not None:
            writer.write(vis_frame)

        # Update previous frame for next iteration
        prev_frame = curr_frame

        # Send progress update
        if progress_queue is not None:
            progress_queue.put((f'cuda:{process_id}', {'completed': frame_idx + 1}))

    # Release resources
    cap.release()
    writer.release()

    # Convert to H.264 format for better compatibility and smaller file size
    to_h264(output_path)


def visualize_video(video_file: str, cache_dir: str, dataset: str, speed_up: int,
                    track_ids: list[int] | None, detection_only: bool, process_id: int, progress_queue: Queue):
    """
    Process visualization for a single video file with frame differences.

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
    tracking_results_raw = load_detection_results(cache_dir, dataset, video_file, tracking=True)

    # Convert to the format expected by create_tracking_visualization_diff
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
                               '000_groundtruth', f'diff_{video_file}')

    # Create visualization with frame differences
    create_tracking_visualization_diff(video_path, tracking_results, output_path, speed_up,
                                       process_id, progress_queue, track_ids, detection_only)


def main(args):
    """
    Main function that orchestrates the tracking visualization process on frame differences.

    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Finds all videos with tracking results
    4. Creates a process pool for parallel processing
    5. Creates visualizations for each video with frame differences and saves results

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (list): List of dataset names to process
            - speed_up (int): Speed up factor for visualization
            - track_ids (list): List of track IDs to color (others will be grey)
            - detection_only (bool): Only show detections without trajectories

    Note:
        - The script expects tracking results from p002_preprocess_groundtruth_tracking.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl
        - Original videos are read from {DATASETS_DIR}/{dataset}/
        - Visualization videos are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/diff_{video_file}
        - Each track ID gets a unique color from a predefined palette
        - Bounding boxes and trajectories are overlaid on frame differences
        - Processing is parallelized for improved performance
    """
    datasets = args.datasets

    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")

    # Create task functions
    funcs = []
    for dataset in datasets:
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
            if os.path.isdir(item_path) and item.startswith('te'):
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
    num_processes = min(mp.cpu_count(), len(funcs), 20)  # Cap at 20 processes

    mp.set_start_method('spawn', force=True)
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
