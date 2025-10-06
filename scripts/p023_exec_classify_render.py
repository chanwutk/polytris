#!/usr/local/bin/python

import argparse
import os
import cv2
import numpy as np
import multiprocessing as mp
from functools import partial

from polyis.utilities import CACHE_DIR, DATA_DIR, load_classification_results, ProgressBar, to_h264, DATASETS_TO_TEST, TILE_SIZES


def parse_args():
    parser = argparse.ArgumentParser(description='Render annotated video with tile classification results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification visualization (0.0 to 1.0)')
    parser.add_argument('--frame_limit', type=int, default=None,
                        help='Maximum number of frames to process per video (default: process all frames)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Specific frame indices to visualize (saves as .jpg files)')
    parser.add_argument('--no_tint', action='store_true',
                        help='Color tiles black instead of red tint for scores below threshold')
    return parser.parse_args()


def create_visualization_frame(frame: np.ndarray, classifications: np.ndarray,
                              tile_size: int, threshold: float, no_tint: bool = False) -> np.ndarray:
    """
    Create a visualization frame by adjusting tile appearance based on classification scores.

    Args:
        frame (np.ndarray): Original video frame (H, W, 3)
        classifications (np.ndarray): 2D grid of classification scores
        tile_size (int): Size of tiles used for classification
        threshold (float): Threshold value for visualization
        no_tint (bool): If True, make low-score tiles black; if False, apply red tint

    Returns:
        np.ndarray: Visualization frame with adjusted tiles
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

    # Apply tile adjustments based on classification scores
    for i in range(grid_height):
        for j in range(grid_width):
            # Get tile coordinates
            y_start = i * tile_size
            y_end = min(y_start + tile_size, vis_frame.shape[0])
            x_start = j * tile_size
            x_end = min(x_start + tile_size, vis_frame.shape[1])

            # Get classification score for this tile
            score = classifications[i][j]

            # Apply different visualizations based on score and no_tint flag
            if score <= threshold:
                if no_tint:
                    # Make tile black if no_tint is True
                    vis_frame[y_start:y_end, x_start:x_end] = 0  # Black
                    continue

                # Apply red tint (original behavior)
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
                  frame_limit: int | None, frames: list[int] | None, no_tint: bool, worker_id: int, command_queue: mp.Queue):
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
        frames (list[int], optional): Specific frame indices to visualize (saves as .jpg files)
        no_tint (bool): If True, make low-score tiles black; if False, apply red tint
        worker_id (int): Worker ID for progress tracking
        command_queue (mp.Queue): Queue for progress updates
    """
    # Load classification results
    results = load_classification_results(
        CACHE_DIR, dataset_name, video_file, tile_size, classifier_name, execution_dir=True)
    
    # Create output directory for visualizations
    vis_output_dir = os.path.join(
        CACHE_DIR, dataset_name, 'execution', video_file,
        '020_relevancy', f'{classifier_name}_{tile_size}')
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

    # Create video writer for brightness visualization (only if not processing specific frames)
    brightness_writer = None

    # Determine which frames to process
    if frames is not None:
        frames_dir = os.path.join(vis_output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Process only specific frames
        frames_to_process = [f for f in frames if f < video_frame_count]
        if not frames_to_process:
            print(f"No valid frames found in {video_file} for indices {frames}")
            cap.release()
            return
        
        # Send initial progress update
        command_queue.put((f'cuda:{worker_id}', {
            'description': f"{video_file} {classifier_name} {tile_size} (frames: {len(frames_to_process)})",
            'completed': 0,
            'total': len(frames_to_process)
        }))
        
        # Process specific frames
        for frame_idx in frames_to_process:
            # Seek to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_idx} from {video_file}")
                continue

            # Get classification results for this frame
            frame_result = results[frame_idx]
            classifications = frame_result['classification_hex']
            classification_size = frame_result['classification_size']
            classifications = (np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
                .reshape(classification_size)
                .astype(np.float32) / 255.0)

            # Save original frame
            original_frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}_original.jpg')
            cv2.imwrite(original_frame_path, frame)
            
            # Create classification visualization frame
            classified_frame = create_visualization_frame(frame, classifications, tile_size, threshold, no_tint)
            
            # Save classified frame
            classified_frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}_classified.jpg')
            cv2.imwrite(classified_frame_path, classified_frame)

            # Update progress
            command_queue.put((f'cuda:{worker_id}', {'completed': frames_to_process.index(frame_idx) + 1}))
    else:
        brightness_video_path = os.path.join(vis_output_dir, 'visualization.mp4')
        # Use MP4V codec for compatibility
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        brightness_writer = cv2.VideoWriter(brightness_video_path, fourcc, fps, (width, height))
        assert brightness_writer.isOpened(), \
            f"Could not create video writer for {brightness_video_path}"

        # Process all frames (original behavior)
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
            brightness_frame = create_visualization_frame(frame, classifications, tile_size, threshold, no_tint)

            # Write frame to video
            brightness_writer.write(brightness_frame)

            # Update progress
            command_queue.put((f'cuda:{worker_id}', {'completed': frame_idx + 1}))

            # Check frame limit
            if frame_limit is not None and frame_idx >= frame_limit - 1:
                break

        brightness_writer.release()
        # Convert to H.264 using FFMPEG
        to_h264(brightness_video_path)

    # Release resources
    cap.release()


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
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - threshold (float): Threshold value for visualization (0.0 to 1.0)
            - frame_limit (int, optional): Maximum number of frames to process per video

         Note:
         - The script expects classification results from 020_exec_classify.py in:
           {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier}_{tile_size}/
         - Looks for score.jsonl files
         - Videos are read from {DATA_DIR}/{dataset}/
         - Visualizations are saved to {CACHE_DIR}/{dataset}/execution/{video_file}/020_relevancy/{classifier}_{tile_size}/
         - The script creates a video file (visualization.mp4) showing brightness-adjusted frames
    """
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Validate threshold
    assert 0.0 <= args.threshold <= 1.0, "Threshold must be between 0.0 and 1.0"
    print(f"Using threshold: {args.threshold}")

    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")

    # Collect all video-classifier-tile combinations for parallel processing
    funcs = []
    
    # Process each dataset
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue
            
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if len(video_files) == 0:
            print(f"No video files found in {dataset_dir}, skipping...")
            continue
            
        print(f"Found {len(video_files)} video files to process")
        
        for video_file in sorted(video_files):
            video_file_path = os.path.join(dataset_dir, video_file)
            
            # Get classifier tile sizes for this video
            relevancy_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file, '020_relevancy')
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
                funcs.append(partial(render_scores, video_file, video_file_path, dataset_name, 
                                    classifier_name, tile_size, args.threshold, args.frame_limit, args.frames, args.no_tint))

    assert len(funcs) > 0, "No tasks to process"
    
    # Set up multiprocessing with ProgressBar
    num_processes = int(mp.cpu_count() * 0.8)
    if len(funcs) < num_processes:
        num_processes = len(funcs)
    
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
