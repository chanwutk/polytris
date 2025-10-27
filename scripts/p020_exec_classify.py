#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable
import cv2
import torch
import numpy as np
import time
import shutil
import multiprocessing as mp
from functools import partial

from polyis.images import splitHWC, padHWC

from polyis.utilities import CACHE_DIR, CLASSIFIERS_CHOICES, CLASSIFIERS_TO_TEST, DATASETS_DIR, format_time, ProgressBar, DATASETS_TO_TEST, TILE_SIZES


def parse_args():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (List[str]): Dataset names to process (default: ['b3d'])
            - tile_size (int | str): Tile size to use for classification (choices: 30, 60, 120, 'all')
            - classifiers (List[str]): List of classifier models to use (default: multiple classifiers)
    """
    parser = argparse.ArgumentParser(description='Execute trained classifier models to classify video tiles')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all',
                        help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--classifiers', required=False, nargs='+',
                        default=CLASSIFIERS_TO_TEST,
                        choices=CLASSIFIERS_CHOICES,
                        help='Specific classifiers to analyze (if not specified, all classifiers will be analyzed)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear the output directory before processing')
    return parser.parse_args()


def load_model(dataset_name: str, tile_size: int, classifier_name: str, device: str) -> "torch.nn.Module":
    """
    Load trained classifier model for the specified tile size from the dataset indexing directory.

    This function searches for a trained model in the expected directory structure:
    {CACHE_DIR}/{dataset_name}/indexing/training/results/{classifier_name}_{tile_size}/model.pth

    Args:
        dataset_name (str): Name of the dataset
        tile_size (int): Tile size for which to load the model (30, 60, or 120)
        classifier_name (str): Name of the classifier model to use (default: 'SimpleCNN')
        device (str): Device to use for loading the model (e.g. 'cuda', 'cpu')

    Returns:
        The loaded trained model for the specified tile size.
            The model is loaded to CUDA and set to evaluation mode.

    Raises:
        FileNotFoundError: If no trained model is found for the specified tile size
        ValueError: If the classifier is not supported
    """
    results_path = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'training', 'results', f'{classifier_name}_{tile_size}')
    model_path = os.path.join(results_path, 'model.pth')

    if os.path.exists(model_path):
        print(f"Loading {classifier_name} model for tile size {tile_size} from {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model

    raise FileNotFoundError(f"No trained model found for {classifier_name} tile size {tile_size} in {results_path}")


def process_frame_tiles(frame: np.ndarray, model: torch.nn.Module, tile_size: int, device: str) -> tuple[np.ndarray, list[dict]]:
    """
    Process a single video frame with the specified tile size and return relevance scores and timing information.

    This function splits the input frame into tiles of the specified size, runs inference
    with the trained model, and returns relevance scores for each tile along with timing information.

    Args:
        frame (np.ndarray): Input video frame as a numpy array with shape (H, W, 3)
            where H and W are the frame height and width, and 3 represents RGB channels
        model (torch.nn.Module): Trained model for the specified tile size
        tile_size (int): Size of tiles to use for processing (30, 60, or 120)
        device (str): Device to use for processing

    Returns:
        tuple[np.ndarray, list[dict[str, float]]]: A tuple containing:
            - 2D grid of relevance scores where each element represents the relevance score
              (probability between 0 and 1) for the corresponding tile in the frame
            - List of dictionaries with 'op' (operation) and 'time' keys for preprocessing and model inference

    Note:
        - Frame is padded if necessary to ensure divisibility by tile size
        - Input frame is normalized to [0, 1] range before inference
        - Model outputs logits, which are converted to probabilities using sigmoid
        - Timing information includes preprocessing and model inference times
    """
    with torch.no_grad():
        start_time = (time.time_ns() / 1e6)
        # Convert frame to tensor and ensure it's in HWC format
        frame_tensor = torch.from_numpy(frame).to(device).float()

        # Pad frame to be divisible by tile_size
        padded_frame = padHWC(frame_tensor, tile_size, tile_size)  # type: ignore

        # Split frame into tiles
        tiles = splitHWC(padded_frame, tile_size, tile_size)

        # Flatten tiles for batch processing
        num_tiles = tiles.shape[0] * tiles.shape[1]
        tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, 3)

        # Normalize to [0, 1] range
        tiles_flat = tiles_flat / 255.0

        # Convert to NCHW format for the model
        tiles_nchw = tiles_flat.permute(0, 3, 1, 2)
        transform_runtime = (time.time_ns() / 1e6) - start_time

        # Run inference
        start_time = (time.time_ns() / 1e6)
        predictions = torch.sigmoid(model(tiles_nchw))
        probabilities = (predictions * 255).to(torch.uint8).cpu().numpy().flatten()

        # Reshape back to grid format
        grid_height, grid_width = tiles.shape[:2]
        relevance_grid = probabilities.reshape(grid_height, grid_width)
        end_time = (time.time_ns() / 1e6)
        inference_runtime = end_time - start_time

    return relevance_grid, format_time(transform=transform_runtime, inference=inference_runtime)


def process_video_task(video_path: str, cache_video_dir: str, dataset_name: str, classifier: str,
                      tile_size: int, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.

    This function reads a video file frame by frame, processes each frame to classify
    tiles using the trained classifier model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.

    Args:
        video_path: Path to the video file
        cache_video_dir: Path to the cache directory for this video
        classifier: Classifier name to use
        tile_size: Tile size to use
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates
        dataset_name: Name of the dataset (used to locate the trained model)

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
        - runtime (float): Runtime in seconds for the ClassifyRelevance model inference
    """
    device = f'cuda:{gpu_id}'

    # Load the trained model for this dataset, classifier, and tile size
    model = load_model(dataset_name, tile_size, classifier, device)
    model = model.to(device)
    # try:
    #     model.compile()
    #     # model = torch.compile(model)
    # except Exception as e:
    #     print(f"Failed to compile model: {e}")

    # Create output directory structure
    output_dir = os.path.join(cache_video_dir, '020_relevancy')

    # Create score directory for this classifier and tile size
    classifier_dir = os.path.join(output_dir, f'{classifier}_{tile_size}')
    if os.path.exists(classifier_dir):
        shutil.rmtree(classifier_dir)
    os.makedirs(classifier_dir)

    # Create score directory for this tile size
    score_dir = os.path.join(classifier_dir, 'score')
    if os.path.exists(score_dir):
        shutil.rmtree(score_dir)
    os.makedirs(score_dir)
    output_path = os.path.join(score_dir, 'score.jsonl')

    # print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    with open(output_path, 'w') as f:
        frame_idx = 0
        description = f"{video_path.split('/')[-1]} {tile_size:>3} {classifier}"
        command_queue.put((device, {'description': description,
                                    'completed': 0, 'total': frame_count}))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with the model
            relevance_grid, runtime = process_frame_tiles(frame, model, tile_size, device)

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
            command_queue.put((device, {'completed': frame_idx}))

    cap.release()
    # print(f"Completed processing {frame_idx} frames. Results saved to {output_path}")


def main(args):
    """
    Main function that orchestrates the video tile classification process using parallel processing.

    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves classification results

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (List[str]): Names of the datasets to process
            - tile_size (str): Tile size to use for classification ('30', '60', '120', or 'all')
            - classifiers (List[str]): List of classifier models to use (default: multiple classifiers)

    Note:
        - The script expects a specific directory structure:
          {DATASETS_DIR}/{dataset}/ - contains video files
          {DATA_CACHE}/{dataset}/{video_file_name}/training/results/{classifier_name}_{tile_size}/model.pth - contains trained models
          where DATASETS_DIR and DATA_CACHE are both /polyis-data/video-datasets-low
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - A separate model is loaded for each video directory, classifier, and tile size combination
        - When tile_size is 'all', all three tile sizes (30, 60, 120) are processed
        - Output files are saved in {DATA_CACHE}/{dataset}/{video_file_name}/020_relevancy/score/{classifier_name}_{tile_size}/score.jsonl
        - If no trained model is found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)

    # Determine which tile sizes to process
    if args.tile_size == 'all':
        tile_sizes_to_process = TILE_SIZES
        print(f"Processing all tile sizes: {tile_sizes_to_process}")
    else:
        tile_sizes_to_process = [int(args.tile_size)]
        print(f"Processing tile size: {tile_sizes_to_process[0]}")

    # Create tasks list with all video/classifier/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []

    for dataset_name in args.datasets:
        dataset_dir = os.path.join(DATASETS_DIR, dataset_name)

        for videoset in ['test']:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Dataset directory {videoset_dir} does not exist, skipping...")
                continue

            # Get all video files from the dataset directory
            video_files = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            for video_file in sorted(video_files):
                video_file_path = os.path.join(videoset_dir, video_file)
                cache_video_dir = os.path.join(CACHE_DIR, dataset_name, 'execution', video_file)

                output_dir = os.path.join(cache_video_dir, '020_relevancy')

                # Clear output directory if --clear flag is specified
                if args.clear and os.path.exists(output_dir):
                    print(f"Clearing output directory: {output_dir}")
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

                for classifier in args.classifiers:
                    for tile_size in tile_sizes_to_process:
                        func = partial(process_video_task, video_file_path, cache_video_dir,
                                    dataset_name, classifier, tile_size)
                        funcs.append(func)

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
