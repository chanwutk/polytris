#!/usr/local/bin/python

import argparse
import os
import cv2
import numpy as np
import multiprocessing as mp

from polyis.utilities import CACHE_DIR, DATA_DIR, load_classification_results, ProgressBar


TILE_SIZES = [30, 60]  #, 120]


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - threshold (float): Threshold for classification visualization (default: 0.5)
            - processes (int): Number of processes to use for parallel processing
    """
    parser = argparse.ArgumentParser(description='Render annotated video with tile classification results')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    parser.add_argument('--frame_limit', type=int, default=None,
                        help='Maximum number of frames to process per video (default: process all frames)')
    return parser.parse_args()


def create_visualization_frame(frame: np.ndarray, classifications: np.ndarray,
                              tile_size: int, threshold: float) -> np.ndarray:
    """
    Create a visualization frame by adjusting tile brightness based on classification scores.

    Args:
        frame (np.ndarray): Original video frame (H, W, 3)
        classifications (np.ndarray): 2D grid of classification scores
        tile_size (int): Size of tiles used for classification
        threshold (float): Threshold value for visualization

    Returns:
        np.ndarray: Visualization frame with adjusted tile brightness
    """
    # Create a copy of the frame for visualization
    vis_frame = frame.copy().astype(np.float32)

    # Get grid dimensions
    grid_height = len(classifications)
    grid_width = len(classifications[0]) if grid_height > 0 else 0

    # Calculate frame dimensions after padding
    vis_height = grid_height * tile_size
    vis_width = grid_width * tile_size

    # Ensure frame is large enough (handle padding)
    if frame.shape[0] < vis_height or frame.shape[1] < vis_width:
        # Pad frame if necessary
        pad_height = max(0, vis_height - frame.shape[0])
        pad_width = max(0, vis_width - frame.shape[1])
        vis_frame = np.pad(vis_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    # Apply brightness adjustments to each tile
    for i in range(grid_height):
        for j in range(grid_width):
            # Get tile coordinates
            y_start = i * tile_size
            y_end = min(y_start + tile_size, vis_frame.shape[0])
            x_start = j * tile_size
            x_end = min(x_start + tile_size, vis_frame.shape[1])

            # Get classification score for this tile
            score = classifications[i][j]

            # Only apply red tint if score is below threshold
            if score <= threshold:
                # Calculate red tint factor based on score (lower score = more red)
                # Normalize score to 0-1 range, then invert so low scores get high red values
                red_intensity = 1.0 - score  # Higher red for lower probability
                
                # Get the tile region
                tile_region = vis_frame[y_start:y_end, x_start:x_end]
                
                # Apply red tint: reduce green and blue channels, enhance red channel
                if red_intensity > 0:
                    # Reduce green and blue channels based on red intensity
                    tile_region[:, :, :2] = tile_region[:, :, :2] * (1.0 - red_intensity * 0.7)  # Blue and Green
                    # Optionally enhance red channel slightly
                    tile_region[:, :, 2] = np.minimum(255, tile_region[:, :, 2] * (1.0 + red_intensity * 0.3))  # Red
                
                # # Update the frame
                # vis_frame[y_start:y_end, x_start:x_end] = tile_region
            # If score > threshold, leave the tile unchanged (same as original image)

    # Clip values to valid range and convert back to uint8
    vis_frame = np.clip(vis_frame, 0, 255).astype(np.uint8)

    return vis_frame


def render_scores(video_file: str, video_file_path: str, dataset_name: str, 
                  classifier_name: str, tile_size: int, threshold: float, 
                  frame_limit: int | None, worker_id: int, command_queue: mp.Queue):
    """
    Render scores for a single classifier-tile size combination.
    
    Args:
        video_file (str): Video file name
        video_file_path (str): Full path to video file
        dataset_name (str): Dataset name
        classifier_name (str): Classifier name
        tile_size (int): Tile size
        threshold (float): Threshold value
        frame_limit (int, optional): Maximum number of frames to process (default: process all frames)
        worker_id (int): Worker ID for progress tracking
        command_queue (mp.Queue): Queue for progress updates
    """
    # Load classification results
    results = load_classification_results(
        CACHE_DIR, dataset_name, video_file, tile_size, classifier_name)
    
    # Create output directory for visualizations
    vis_output_dir = os.path.join(
        CACHE_DIR, dataset_name, video_file, 'relevancy', f'{classifier_name}_{tile_size}')
    os.makedirs(vis_output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_file_path)
    assert cap.isOpened(), f"Could not open video {video_file_path}"

    # Get actual video frame count and properties
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results_frame_count = len(results)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert video_frame_count == results_frame_count, \
        f"Frame count mismatch: Video has {video_frame_count} frames, " \
        f"but results contain {results_frame_count} frames. " \
        f"This suggests the classification results don't match the video file."

    # Create video writer for brightness visualization
    brightness_video_path = os.path.join(vis_output_dir, 'visualization.mp4')
    # Use MP4V codec for compatibility
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    brightness_writer = cv2.VideoWriter(brightness_video_path, fourcc, fps, (width, height))
    assert brightness_writer.isOpened(), f"Could not create video writer for {brightness_video_path}"

    # Send initial progress update
    command_queue.put((f'cuda:{worker_id}', {
        'description': f"{video_file} {classifier_name} {tile_size}",
        'completed': 0,
        'total': video_frame_count
    }))

    # Process all frames
    for frame_idx in range(video_frame_count):
        # Get frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Get classification results for this frame
        frame_result = results[frame_idx]
        classifications = frame_result['classification_hex']
        classification_size = frame_result['classification_size']
        classifications = (np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
            .reshape(classification_size)
            .astype(np.float32) / 255.0)

        # Create brightness visualization frame
        brightness_frame = create_visualization_frame(frame, classifications, tile_size, threshold)

        # Write frame to video
        brightness_writer.write(brightness_frame)

        # Update progress
        command_queue.put((f'cuda:{worker_id}', {'completed': frame_idx + 1}))

        # Check frame limit
        if frame_limit is not None and frame_idx >= frame_limit - 1:
            break

    # Release resources
    cap.release()
    brightness_writer.release()


def _render_scores(video_file: str, video_file_path: str, dataset_name: str, 
                                 classifier_name: str, tile_size: int, threshold: float, 
                                 frame_limit: int | None, worker_id: int,
                                 progress_bar: ProgressBar):
    """
    Wrapper function for render_scores that handles worker ID management.
    """
    try:
        render_scores(video_file, video_file_path, dataset_name, 
                                   classifier_name, tile_size, threshold, 
                                   frame_limit, worker_id, progress_bar.command_queue)
    finally:
        progress_bar.worker_id_queue.put(worker_id)


def main(args):
    """
    Main function that orchestrates the video tile classification rendering process.

    This function serves as the entry point for the script. It: 
    1. Validates the dataset directory exists
    2. Iterates through all videos in the dataset directory
    3. For each video, loads the classification results for the specified tile size(s)
    4. Creates rendered videos showing tile scores and brightness adjustments

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)
            - frame_limit (int, optional): Maximum number of frames to process per video

         Note:
         - The script expects classification results from 020_exec_classify.py in:
           {CACHE_DIR}/{dataset}/{video_file}/relevancy/score/proxy_{tile_size}/
         - Looks for score.jsonl files
         - Videos are read from {DATA_DIR}/{dataset}/
         - Visualizations are saved to {CACHE_DIR}/{dataset}/{video_file}/relevancy/proxy_{tile_size}/
         - The script creates a video file (visualization.mp4) showing brightness-adjusted frames
    """
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")

    print(f"Using threshold: {args.threshold}")

    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return

    print(f"Found {len(video_files)} video files to process")

    # Collect all video-classifier-tile combinations for parallel processing
    all_tasks = []
    
    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)
        
        # Get classifier tile sizes for this video
        relevancy_dir = os.path.join(CACHE_DIR, args.dataset, video_file, 'relevancy')
        if not os.path.exists(relevancy_dir):
            print(f"Skipping {video_file}: No relevancy directory found")
            continue
            
        classifier_tilesizes: list[tuple[str, int]] = []
        for file in os.listdir(relevancy_dir):
            if '_' in file:
                classifier_name = file.split('_')[0]
                tile_size = int(file.split('_')[1])
                classifier_tilesizes.append((classifier_name, tile_size))
        
        classifier_tilesizes = sorted(classifier_tilesizes)
        
        if not classifier_tilesizes:
            print(f"Skipping {video_file}: No classifier tile sizes found")
            continue
            
        print(f"Found {len(classifier_tilesizes)} classifier tile sizes for {video_file}: {classifier_tilesizes}")
        
        # Add tasks for each classifier-tile size combination
        for classifier_name, tile_size in classifier_tilesizes:
            all_tasks.append((video_file, video_file_path, args.dataset,
                              classifier_name, tile_size, args.threshold,
                              args.frame_limit))

    if not all_tasks:
        print("No tasks to process")
        return

    # Determine number of processes to use
    num_processes = mp.cpu_count()
    print(f"Processing {len(all_tasks)} tasks in parallel using {num_processes} processes...")

    # Set up multiprocessing with ProgressBar
    with ProgressBar(num_workers=num_processes, num_tasks=len(all_tasks)) as pb:
        processes: list[mp.Process] = []
        for task in all_tasks:
            # Get a worker ID
            worker_id = pb.get_worker_id()
            
            # Update overall progress
            pb.update_overall_progress(1)
            
            # Start the worker process
            process = mp.Process(target=_render_scores, args=(*task, worker_id, pb))
            process.start()
            processes.append(process)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
            process.terminate()
    
    print(f"\nCompleted processing {len(all_tasks)} tasks")


if __name__ == '__main__':
    main(parse_args())
