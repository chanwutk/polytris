#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable
import cv2
import numpy as np
import time
import multiprocessing as mp
from functools import partial

from polyis.utilities import CACHE_DIR, DATA_DIR, format_time, load_tracking_results, mark_detections, progress_bars, ProgressBar, DATASETS_TO_TEST, TILE_SIZES


def parse_args():
    parser = argparse.ArgumentParser(description='Execute trained proxy models to classify video tiles')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    return parser.parse_args()


def process_frame_tiles(frame: np.ndarray, detections: list[list[float]], tile_size: int) -> tuple[np.ndarray, list[dict]]:
    """
    Process a single video frame with groundtruth detections and return relevance scores.
    
    This function uses groundtruth bounding boxes to determine which tiles are relevant,
    rather than running inference with a trained model.
    
    Args:
        frame (np.ndarray): Input video frame as a numpy array with shape (H, W, 3)
        detections (list[list[float]]): List of bounding boxes for this frame
        tile_size (int): Size of tiles to use for processing (30, 60, or 120)
            
    Returns:
        tuple[np.ndarray, list[dict]]: A tuple containing:
            - 2D grid of relevance scores where each element is 1 for relevant tiles and 0 for irrelevant tiles
            - Runtime in seconds (always 0.0 since no model inference is performed)
            
    Note:
        - Frame dimensions are used to create the tile grid
        - Bounding boxes are converted to tile coordinates
        - Tiles overlapping with detections are marked as relevant (1.0)
        - Tiles without detections are marked as irrelevant (0.0)
    """
    start_time = (time.time_ns() / 1e6)
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create bitmap marking relevant tiles
    relevance_grid = mark_detections(detections, width, height, tile_size) * 255
    
    end_time = (time.time_ns() / 1e6)
    runtime = end_time - start_time
    
    return relevance_grid, format_time(inference=runtime, transform=0)
    

def process_video(video_path: str, video_file: str, tile_size: int,
                  dataset: str, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using groundtruth detection data, and saves the results in JSONL format.
    Each line in the output file represents one frame with its tile classifications.
    
    Args:
        video_path (str): Path to the input video file to process
        video_file (str): Video filename
        tile_size (int): Tile size used for processing (30, 60, or 120)
        dataset (str): Dataset name
        gpu_id (int): GPU ID to use for processing
        command_queue (mp.Queue): Queue for progress updates
    Note:
        - Video is processed frame by frame to minimize memory usage
        - Progress is displayed using a progress bar
        - Results are flushed to disk after each frame for safety
        - Video metadata (FPS, dimensions, frame count) is extracted and logged
        - Each frame entry includes frame index, timestamp, frame dimensions, tile classifications, and runtime
        - The function handles various video formats (.mp4, .avi, .mov, .mkv)
        
    Output Format:
        Each line in the JSONL file contains a JSON object with:
        - frame_idx (int): Zero-based frame index
        - timestamp (float): Frame timestamp in seconds
        - frame_size (list[int]): [height, width] of the frame
        - tile_size (int): Tile size used for processing (30, 60, or 120)
        - tile_classifications (list[list[float]]): Relevance scores grid for the specified tile size
        - runtime (float): Runtime in seconds (always 0.0 for groundtruth-based processing)
        - classification_size (list[int]): [height, width] of the classification grid
        - classification_hex (str): Hexadecimal representation of the classification grid
    """
    # Load the groundtruth tracking results for this video
    frame_detections = load_tracking_results(CACHE_DIR, dataset, video_file)
    
    # Create output directory structure
    output_dir = os.path.join(CACHE_DIR, dataset, 'execution', video_file, '020_relevancy')
    os.makedirs(output_dir, exist_ok=True)

    classifier_dir = os.path.join(output_dir, f'Perfect_{tile_size}')
    os.makedirs(classifier_dir, exist_ok=True)
    
    # Create score directory for this tile size
    score_dir = os.path.join(classifier_dir, 'score')
    os.makedirs(score_dir, exist_ok=True)
    output_path = os.path.join(score_dir, 'score.jsonl')
    
    # Process the video
    device = f'cuda:{gpu_id}'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    with open(output_path, 'w') as f:
        frame_idx = 0
        command_queue.put((device, {'description': f"{video_path.split('/')[-1]} {tile_size:>3}",
                                    'completed': 0, 'total': frame_count}))
        
        mod = int(max(1, frame_count * 0.02))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections for this frame (empty list if no detections)
            detections = frame_detections.get(frame_idx, [])
            
            # Process frame with groundtruth detections
            relevance_grid, runtime = process_frame_tiles(frame, detections, tile_size)
            
            # Create result entry for this frame
            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "frame_size": [height, width],
                "tile_size": tile_size,
                "runtime": runtime,
                "classification_size": relevance_grid.shape,
                "classification_hex": relevance_grid.flatten().tobytes().hex(),
            }
            
            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            if frame_idx % 100 == 0:
                f.flush()
            
            frame_idx += 1
            if frame_idx % mod == 0:
                command_queue.put((device, {'completed': frame_idx}))
    
    cap.release()
    # print(f"Completed processing {frame_idx} frames. Results saved to {output_path}")


def main(args):
    """
    Main function that orchestrates the video tile classification process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available CPUs
    4. Processes each video and saves classification results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            
    Note:
        - The script expects a specific directory structure:
          {DATA_DIR}/{dataset}/ - contains video files
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl - contains groundtruth tracking results
          where DATA_DIR is /polyis-data/video-datasets-low and CACHE_DIR is /polyis-cache
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Groundtruth tracking results are loaded for each video
        - When tile_size is 'all', all three tile sizes (30, 60, 120) are processed
        - Output files are saved in {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/Perfect_{tile_size}/score/score.jsonl
        - If no tracking results are found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)
    
    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")
    
    # Create functions list with all video/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue
        
        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in sorted(video_files):
            video_file_path = os.path.join(dataset_dir, video_file)
            for tile_size in tile_sizes_to_process:
                funcs.append(partial(process_video, video_file_path,
                             video_file, tile_size, dataset_name))
    
    print(f"Created {len(funcs)} tasks to process")
    
    # Set up multiprocessing with ProgressBar - use CPU count as we don't need GPUs for groundtruth processing
    ProgressBar(num_workers=mp.cpu_count(), num_tasks=len(funcs), refresh_per_second=2).run_all(funcs)
    
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
