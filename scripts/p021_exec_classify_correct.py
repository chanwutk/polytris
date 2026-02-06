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

import torch

from polyis.utilities import format_time, load_tracking_results, mark_detections, progress_bars, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
DATASETS = config['EXEC']['DATASETS']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']


def process_frame_tiles(width: int, height: int, detections: list[list[float]], tile_size: int) -> tuple[np.ndarray, list[dict]]:
    """
    Process a single video frame with groundtruth detections and return relevance scores.
    
    This function uses groundtruth bounding boxes to determine which tiles are relevant,
    rather than running inference with a trained model.
    
    Args:
        width (int): Width of the video frame
        height (int): Height of the video frame
        detections (list[list[float]]): List of bounding boxes for this frame
        tile_size (int): Size of tiles to use for processing (30, 60, or 120)
            
    Returns:
        tuple[np.ndarray, list[dict]]: A tuple containing:
            - 2D grid of relevance scores where each element is 1 for relevant tiles and 0 for irrelevant tiles
            - Runtime in seconds
            
    Note:
        - Frame dimensions are used to create the tile grid
        - Bounding boxes are converted to tile coordinates
        - Tiles overlapping with detections are marked as relevant (255)
        - Tiles without detections are marked as irrelevant (0)
    """
    start_time = (time.time_ns() / 1e6)
    
    # Create bitmap marking relevant tiles
    relevance_grid = mark_detections(detections, width, height, tile_size) * 255
    
    end_time = (time.time_ns() / 1e6)
    runtime = end_time - start_time
    
    return relevance_grid, format_time(inference=runtime, transform=0)
    

def process_video(dataset: str, videoset: str, video: str, tile_size: int, sample_rate: int, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using groundtruth detection data, and saves the results in JSONL format.
    Each line in the output file represents one frame with its tile classifications.
    
    Args:
        dataset (str): Dataset name
        videoset (str): Videoset name (test, train, or valid)
        video (str): Video filename
        tile_size (int): Tile size used for processing (30, 60, or 120)
        sample_rate (int): Sample rate for frame sampling (1 = all frames, 2 = every 2nd frame, etc.)
        gpu_id (int): GPU ID to use for processing
        command_queue (mp.Queue): Queue for progress updates
    Note:
        - Video is processed frame by frame to minimize memory usage
        - Progress is displayed using a progress bar
        - Results are flushed to disk after each frame for safety
        - Video metadata (FPS, dimensions, frame count) is extracted and logged
        - Each frame entry includes frame index, tile classification grid, and runtime
        - The function handles various video formats (.mp4, .avi, .mov, .mkv)
        
    Output Format:
        Each line in the JSONL file contains a JSON object with:
        - classification_size (list[int]): [height, width] of the classification grid
        - classification_hex (str): Hexadecimal representation of the classification grid
        - idx (int): Zero-based frame index
    """
    # Load the groundtruth tracking results for this video
    frame_detections = load_tracking_results(CACHE_DIR, dataset, video)
    
    # Create output directory structure
    output_dir = os.path.join(CACHE_DIR, dataset, 'execution', video, '020_relevancy')
    os.makedirs(output_dir, exist_ok=True)

    classifier_dir = os.path.join(output_dir, f'Perfect_{tile_size}_{sample_rate}')
    os.makedirs(classifier_dir, exist_ok=True)
    
    # Create score directory for this tile size
    score_dir = os.path.join(classifier_dir, 'score')
    os.makedirs(score_dir, exist_ok=True)
    output_path = os.path.join(score_dir, 'score.jsonl')
    runtime_path = os.path.join(score_dir, 'runtime.jsonl')
    
    # Process the video
    device = f'cuda:{gpu_id}'
    video_path = os.path.join(DATASETS_DIR, dataset, videoset, video)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error: Could not open video {video_path}"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Filter to sampled frames only (frame_idx % sample_rate == 0)
    sampled_indices = [idx for idx in range(frame_count) if idx % sample_rate == 0]
    # Always include the last frame
    last_idx = frame_count - 1
    if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
        sampled_indices.append(last_idx)

    # print(f"Video info: {width}x{height}, {frame_count} frames")
    with open(output_path, 'w') as f, open(runtime_path, 'w') as fr:
        description = f"{video_path.split('/')[-1]} {tile_size:>3} sr{sample_rate}"
        command_queue.put((device, {'description': description,
                                    'completed': 0, 'total': len(sampled_indices)}))

        mod = int(max(1, len(sampled_indices) * 0.02))
        for i, frame_idx in enumerate(sampled_indices):
            # Get detections for this frame (empty list if no detections)
            detections = frame_detections.get(frame_idx, [])

            # Process frame with groundtruth detections
            relevance_grid, runtime = process_frame_tiles(width, height, detections, tile_size)

            # Create result entry for this frame
            frame_entry = {
                "classification_size": relevance_grid.shape,
                "classification_hex": relevance_grid.flatten().tobytes().hex(),
                "idx": frame_idx,
            }

            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            fr.write(json.dumps(runtime) + '\n')
            if i % mod == 0:
                command_queue.put((device, {'completed': i}))


def parse_args():
    parser = argparse.ArgumentParser(description='Execute tile classification using groundtruth detections')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def main():
    """
    Main function that orchestrates the video tile classification process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available CPUs
    4. Processes each video and saves classification results
            
    Note:
        - The script expects a specific directory structure:
          {DATASETS_DIR}/{dataset}/ - contains video files
          {CACHE_DIR}/{dataset}/execution/{video_file}/000_groundtruth/tracking.jsonl - contains groundtruth tracking results
          where DATASETS_DIR is /polyis-data/video-datasets-low and CACHE_DIR is /polyis-cache
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Groundtruth tracking results are loaded for each video
        - When tile_size is 'all', all three tile sizes (30, 60, 120) are processed
        - Output files are saved in {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/Perfect_{tile_size}/score/score.jsonl
        - If no tracking results are found for a video, that video is skipped with a warning
    """
    args = parse_args()
    
    # Determine which videosets to process based on arguments
    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.train:
        selected_videosets.append('train')
    if args.valid:
        selected_videosets.append('valid')
    
    # If no videosets are specified, default to all three
    if not selected_videosets:
        selected_videosets = ['test']
    
    mp.set_start_method('spawn', force=True)
    
    # Create functions list with all video/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        for videoset in selected_videosets:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Videoset directory {videoset_dir} does not exist, skipping...")
                continue
            
            # Get all video files from the dataset directory
            videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video in sorted(videos):
                for tile_size in TILE_SIZES:
                    for sample_rate in SAMPLE_RATES:
                        funcs.append(partial(process_video, dataset, videoset, video, tile_size, sample_rate))
    
    print(f"Created {len(funcs)} tasks to process")
    num_processes = min(torch.cuda.device_count(), len(funcs))
    assert num_processes > 0, "No GPUs available"
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=10).run_all(funcs)
    
    print("All tasks completed!")


if __name__ == '__main__':
    main()
