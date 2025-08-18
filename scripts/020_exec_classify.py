#!/usr/local/bin/python

import argparse
import json
import os
import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
import shutil

from polyis.proxy import ClassifyRelevance
from polyis.images import splitHWC, padHWC


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


def load_model_for_video(video_path: str, tile_size: int) -> ClassifyRelevance:
    """
    Load trained proxy model for the specified tile size from a specific video directory.
    
    This function searches for a trained model in the expected directory structure:
    {video_path}/training/results/proxy_{tile_size}/model.pth
    
    Args:
        video_path (str): Path to the specific video directory
        tile_size (int): Tile size for which to load the model (32, 64, or 128)
        
    Returns:
        ClassifyRelevance: The loaded trained model for the specified tile size.
            The model is loaded to CUDA and set to evaluation mode.
            
    Raises:
        FileNotFoundError: If no trained model is found for the specified tile size
    """
    results_path = os.path.join(video_path, 'training', 'results', f'proxy_{tile_size}')
    model_path = os.path.join(results_path, 'model.pth')
    
    if os.path.exists(model_path):
        print(f"Loading model for tile size {tile_size} from {model_path}")
        model = torch.load(model_path, map_location='cuda', weights_only=False)
        model.eval()
        return model
    
    raise FileNotFoundError(f"No trained model found for tile size {tile_size} in {video_path}")


def process_frame_tiles(frame: np.ndarray, model: ClassifyRelevance, tile_size: int) -> tuple[list[list[float]], float]:
    """
    Process a single video frame with the specified tile size and return relevance scores and runtime.
    
    This function splits the input frame into tiles of the specified size, runs inference
    with the trained model, and returns relevance scores for each tile along with the runtime.
    
    Args:
        frame (np.ndarray): Input video frame as a numpy array with shape (H, W, 3)
            where H and W are the frame height and width, and 3 represents RGB channels
        model (ClassifyRelevance): Trained model for the specified tile size
        tile_size (int): Size of tiles to use for processing (32, 64, or 128)
            
    Returns:
        tuple[list[list[float]], float]: A tuple containing:
            - 2D grid of relevance scores where each element represents the relevance score
              (probability between 0 and 1) for the corresponding tile in the frame
            - Runtime in seconds for the ClassifyRelevance model inference
            
    Note:
        - Frame is padded if necessary to ensure divisibility by tile size
        - Input frame is normalized to [0, 1] range before inference
        - Model is expected to output logits, which are converted to probabilities using sigmoid
        - Runtime measurement includes only the model inference time, not preprocessing
    """
    # Convert frame to tensor and ensure it's in HWC format
    frame_tensor = torch.from_numpy(frame).float().to('cuda')
    
    # Pad frame to be divisible by tile_size
    padded_frame = padHWC(frame_tensor, tile_size, tile_size)  # type: ignore
    
    # Split frame into tiles
    tiles = splitHWC(padded_frame, tile_size, tile_size)  # type: ignore
    
    # Flatten tiles for batch processing
    num_tiles = tiles.shape[0] * tiles.shape[1]
    tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)
    
    # Normalize to [0, 1] range
    tiles_flat = tiles_flat / 255.0
    
    # Convert to NCHW format for the model
    tiles_nchw = tiles_flat.permute(0, 3, 1, 2)
    
    # Measure runtime for ClassifyRelevance model inference
    start_time = time.time()
    
    # Run inference
    with torch.no_grad():
        predictions = model(tiles_nchw)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(predictions).cpu().numpy().flatten()
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Reshape back to grid format
    grid_height, grid_width = tiles.shape[:2]
    relevance_grid = probabilities.reshape(grid_height, grid_width)
    
    return relevance_grid.tolist(), runtime


def process_video(video_path: str, model: ClassifyRelevance, tile_size: int, output_path: str):
    """
    Process a single video file and save tile classification results to a JSONL file.
    
    This function reads a video file frame by frame, processes each frame to classify
    tiles using the trained proxy model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.
    
    Args:
        video_path (str): Path to the input video file to process
        model (ClassifyRelevance): Trained model for the specified tile size
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
        - runtime (float): Runtime in seconds for the ClassifyRelevance model inference
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
                
                # Process frame with the model
                relevance_grid, runtime = process_frame_tiles(frame, model, tile_size)
                
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
          {DATA_CACHE}/{dataset}/{video_file_name}/training/results/proxy_{tile_size}/model.pth - contains trained models
          where DATA_DIR and DATA_CACHE are both /polyis-data/video-datasets-low
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - A separate model is loaded for each video directory
        - When tile_size is 'all', all three tile sizes (32, 64, 128) are processed
        - Output files are saved in {DATA_CACHE}/{dataset}/{video_file_name}/relevancy/score/proxy_{tile_size}/score.jsonl
        - If no trained model is found for a video, that video is skipped with a warning
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
            
            # Look for the trained model in the expected cache directory structure
            cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
            
            # Load the trained model for this specific video and tile size
            try:
                model = load_model_for_video(cache_video_dir, tile_size)
                print(f"Successfully loaded model for tile size {tile_size}")
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
            output_path = os.path.join(score_dir, 'score.jsonl')
            
            # Process the video
            process_video(video_file_path, model, tile_size, output_path)
            

if __name__ == '__main__':
    main(parse_args())
