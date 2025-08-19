#!/usr/local/bin/python

import argparse
import json
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import shutil


DATA_DIR = '/polyis-data/video-datasets-low'
CACHE_DIR = '/polyis-cache'
TILE_SIZES = [32, 64, 128]


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 32, 64, 128, 'all')
    """
    parser = argparse.ArgumentParser(description='Execute trained proxy models to classify video tiles')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['32', '64', '128', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    return parser.parse_args()


def load_groundtruth_tracking(video_path: str) -> dict[int, list[list[float]]]:
    """
    Load groundtruth tracking results from the JSONL file.
    
    This function reads the tracking results file and organizes them by frame index.
    
    Args:
        video_path (str): Path to the video directory containing groundtruth tracking results
        
    Returns:
        dict[int, list[list[float]]]: Dictionary mapping frame indices to lists of bounding boxes
            Each bounding box is formatted as [tracking_id, x1, y1, x2, y2]
            
    Raises:
        FileNotFoundError: If no tracking results file is found
    """
    tracking_path = os.path.join(video_path, 'groundtruth', 'tracking.jsonl')
    
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Tracking results not found: {tracking_path}")
    
    print(f"Loading groundtruth tracking results from {tracking_path}")
    
    frame_detections = {}
    with open(tracking_path, 'r') as f:
        for line in f:
            if line.strip():
                frame_data = json.loads(line)
                frame_idx = frame_data['frame_idx']
                tracks = frame_data['tracks']
                frame_detections[frame_idx] = tracks
    
    print(f"Loaded tracking results for {len(frame_detections)} frames")
    return frame_detections


def mark_detections(detections: list[list[float]], width: int, height: int, chunk_size: int) -> np.ndarray:
    """
    Mark tiles as relevant based on groundtruth detections.
    
    This function creates a bitmap where 1 indicates a tile with detection and 0 indicates no detection.
    Based on the mark_detections2 function from chunker.py.
    
    Args:
        detections (list[list[float]]): List of bounding boxes, each formatted as [tracking_id, x1, y1, x2, y2]
        width (int): Frame width
        height (int): Frame height
        chunk_size (int): Size of each tile
        
    Returns:
        np.ndarray: 2D array representing the grid of tiles, where 1 indicates relevant tiles
    """
    bitmap = np.zeros((height // chunk_size, width // chunk_size), dtype=np.int32)
    
    for bbox in detections:
        # Extract bounding box coordinates (ignore tracking_id)
        x1, y1, x2, y2 = bbox[1:5]  # Skip tracking_id at index 0
        
        # Convert to tile coordinates
        xfrom, xto = int(x1 // chunk_size), int(x2 // chunk_size)
        yfrom, yto = int(y1 // chunk_size), int(y2 // chunk_size)
        
        # Mark all tiles that overlap with the bounding box
        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def process_frame_tiles(frame: np.ndarray, detections: list[list[float]], tile_size: int) -> tuple[list[list[float]], float]:
    """
    Process a single video frame with groundtruth detections and return relevance scores.
    
    This function uses groundtruth bounding boxes to determine which tiles are relevant,
    rather than running inference with a trained model.
    
    Args:
        frame (np.ndarray): Input video frame as a numpy array with shape (H, W, 3)
        detections (list[list[float]]): List of bounding boxes for this frame
        tile_size (int): Size of tiles to use for processing (32, 64, or 128)
            
    Returns:
        tuple[list[list[float]], float]: A tuple containing:
            - 2D grid of relevance scores where each element is 1.0 for relevant tiles and 0.0 for irrelevant tiles
            - Runtime in seconds (always 0.0 since no model inference is performed)
            
    Note:
        - Frame dimensions are used to create the tile grid
        - Bounding boxes are converted to tile coordinates
        - Tiles overlapping with detections are marked as relevant (1.0)
        - Tiles without detections are marked as irrelevant (0.0)
    """
    start_time = time.time()
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create bitmap marking relevant tiles
    bitmap = mark_detections(detections, width, height, tile_size)
    
    # Convert bitmap to relevance scores (0.0 or 1.0)
    relevance_grid = bitmap.astype(np.float32)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return relevance_grid.tolist(), runtime


def process_video(video_path: str, frame_detections: dict[int, list[list[float]]], tile_size: int, output_path: str):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using groundtruth detection data, and saves the results in JSONL format.
    Each line in the output file represents one frame with its tile classifications.
    
    Args:
        video_path (str): Path to the input video file to process
        frame_detections (dict[int, list[list[float]]]): Dictionary mapping frame indices to detection lists
        tile_size (int): Tile size used for processing (32, 64, or 128)
        output_path (str): Path where the output JSONL file will be saved
            
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
        - tile_size (int): Tile size used for processing (32, 64, or 128)
        - tile_classifications (list[list[float]]): Relevance scores grid for the specified tile size
        - runtime (float): Runtime in seconds (always 0.0 for groundtruth-based processing)
    """
    print(f"Processing video: {video_path}")
    
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
        
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
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
                    "tile_classifications": relevance_grid,
                    "runtime": runtime
                }
                
                # Write to JSONL file
                f.write(json.dumps(frame_entry) + '\n')
                f.flush()
                
                frame_idx += 1
                pbar.update(1)
                
                # Optional: limit processing for testing
                # if frame_idx > 100:
                #     break
    
    cap.release()
    print(f"Completed processing {frame_idx} frames. Results saved to {output_path}")


def main(args):
    """
    Main function that orchestrates the video tile classification process.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Iterates through all videos in the dataset directory
    3. For each video, loads the appropriate trained proxy model(s) for the specified tile size(s)
    4. Processes each video and saves classification results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('32', '64', '128', or 'all')
            
    Note:
        - The script expects a specific directory structure:
          {DATA_DIR}/{dataset}/ - contains video files
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/tracking.jsonl - contains groundtruth tracking results
          where DATA_DIR is /polyis-data/video-datasets-low and CACHE_DIR is /polyis-cache
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Groundtruth tracking results are loaded for each video
        - When tile_size is 'all', all three tile sizes (32, 64, 128) are processed
        - Output files are saved in {CACHE_DIR}/{dataset}/{video_file}/relevancy/score/proxy_{tile_size}/score_correct.jsonl
        - If no tracking results are found for a video, that video is skipped with a warning
    """
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")
    
    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")
    
    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video file
    for video_file in video_files:
        video_file_path = os.path.join(dataset_dir, video_file)
        
        print(f"\nProcessing video file: {video_file}")
        
        # Process each tile size for this video
        for tile_size in tile_sizes_to_process:
            print(f"Processing tile size: {tile_size}")
            
            # Look for the groundtruth tracking results in the cache directory structure
            cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
            
            # Load the groundtruth tracking results for this video
            try:
                frame_detections = load_groundtruth_tracking(cache_video_dir)
                print(f"Successfully loaded groundtruth tracking for tile size {tile_size}")
            except FileNotFoundError as e:
                print(f"Warning: {e}, skipping tile size {tile_size} for video {video_file}")
                continue
            
            # Create output directory structure
            output_dir = os.path.join(cache_video_dir, 'relevancy')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create score directory for this tile size
            score_dir = os.path.join(output_dir, 'score', f'proxy_{tile_size}')
            if os.path.exists(score_dir):
                # Remove the entire directory
                shutil.rmtree(score_dir)
            os.makedirs(score_dir)
            output_path = os.path.join(score_dir, 'score_correct.jsonl')
            
            # Process the video
            process_video(video_file_path, frame_detections, tile_size, output_path)
            

if __name__ == '__main__':
    main(parse_args())
