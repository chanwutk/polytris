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

from polyis.train.select_model_optimization import select_model_optimization
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']
DATASETS = config['EXEC']['DATASETS']


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
        # print(f"Loading {classifier_name} model for tile size {tile_size} from {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model

    raise FileNotFoundError(f"No trained model found for {classifier_name} tile size {tile_size} in {results_path}")


def process_frame_tiles(frame: np.ndarray, previous_frame: np.ndarray, model: torch.nn.Module, tile_size: int, device: str, 
                       normalize_mean: torch.Tensor, normalize_std: torch.Tensor) -> tuple[np.ndarray, list[dict]]:
    """
    Process a single video frame with the specified tile size and return relevance scores and timing information.

    This function splits the input frame into tiles of the specified size, runs inference
    with the trained model, and returns relevance scores for each tile along with timing information.

    Args:
        frame (np.ndarray): Input video frame as a numpy array with shape (H, W, 3)
            where H and W are the frame height and width, and 3 represents RGB channels
        previous_frame (np.ndarray): Previous video frame as a numpy array with shape (H, W, 3)
        model (torch.nn.Module): Trained model for the specified tile size
        tile_size (int): Size of tiles to use for processing (30, 60, or 120)
        device (str): Device to use for processing
        normalize_mean (torch.Tensor): Pre-created mean tensor for ImageNet normalization
        normalize_std (torch.Tensor): Pre-created std tensor for ImageNet normalization

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
        frame_tensor = torch.from_numpy(frame).to(device)
        previous_frame_tensor = torch.from_numpy(previous_frame).to(device)

        diff = torch.abs(frame_tensor.to(torch.int16) - previous_frame_tensor.to(torch.int16)).to(torch.uint8)
        frame_tensor = torch.cat([frame_tensor, diff], dim=2)
        channels = frame_tensor.shape[2]

        # Pad frame to be divisible by tile_size
        padded_frame = padHWC(frame_tensor, tile_size, tile_size)  # type: ignore

        # Split frame into tiles
        tiles = splitHWC(padded_frame, tile_size, tile_size)

        # Flatten tiles for batch processing
        num_tiles = tiles.shape[0] * tiles.shape[1]
        tiles_flat = tiles.reshape(num_tiles, tile_size, tile_size, channels)

        # Create position tensor representing (y, x) indices of tiles_flat
        grid_height, grid_width = tiles.shape[:2]
        y_indices = torch.arange(grid_height, device=device, dtype=torch.uint8).repeat_interleave(grid_width)
        x_indices = torch.arange(grid_width, device=device, dtype=torch.uint8).repeat(grid_height)
        positions = torch.stack([y_indices, x_indices], dim=1).float()

        # Identify all-black tiles (all pixels are zero)
        # Check if any pixel in each tile is non-zero
        non_black_mask = tiles_flat.reshape(num_tiles, -1).any(dim=1).to(torch.uint8)

        # Normalize to [0, 1] range (equivalent to ToTensor())
        tiles_flat = tiles_flat.float() / 255.0

        # Convert to NCHW format for the model
        tiles_nchw = tiles_flat.permute(0, 3, 1, 2)
        
        # Apply ImageNet normalization to match training transform
        # Using manual normalization for ~1.9x speedup vs torchvision.Normalize
        tiles_nchw = (tiles_nchw - normalize_mean) / normalize_std
        
        transform_runtime = (time.time_ns() / 1e6) - start_time

        # Run inference
        start_time = (time.time_ns() / 1e6)
        predictions = torch.sigmoid(model(tiles_nchw, positions))
        
        # Convert to uint8 and transfer to CPU
        probabilities = (predictions * 255).to(torch.uint8)
        predictions = predictions * non_black_mask.view(-1, 1)
        probabilities = probabilities.cpu().numpy().flatten()

        # Reshape back to grid format
        grid_height, grid_width = tiles.shape[:2]
        relevance_grid = probabilities.reshape(grid_height, grid_width)
        end_time = (time.time_ns() / 1e6)
        inference_runtime = end_time - start_time

    return relevance_grid, format_time(transform=transform_runtime, inference=inference_runtime)


def classify(dataset: str, video: str, classifier: str, tile_size: int, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.

    This function reads a video file frame by frame, processes each frame to classify
    tiles using the trained classifier model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.

    Args:
        dataset: Name of the dataset
        video: Name of the video
        classifier: Classifier name to use
        tile_size: Tile size to use
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates

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
    model = load_model(dataset, tile_size, classifier, device)
    model = model.to(device)

    video_path = os.path.join(DATASETS_DIR, dataset, 'test', video)
    cache_video_dir = os.path.join(CACHE_DIR, dataset, 'execution', video)

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
    runtime_path = os.path.join(score_dir, 'runtime.jsonl')

    # print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Select the best model optimization method based on the benchmark results and apply it to the model
    with open(os.path.join(CACHE_DIR, dataset, 'indexing', 'training', 'results',
                           f'{classifier}_{tile_size}', 'model_compilation.jsonl'), 'r') as f:
        benchmark_results = [json.loads(line) for line in f]
    model = select_model_optimization(model, benchmark_results, device, tile_size,
                                      (width * height) // (tile_size * tile_size))

    # Pre-create normalization tensors for ImageNet normalization (1.9x speedup vs torchvision.Normalize)
    normalize_mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=device).view(1, 6, 1, 1)
    normalize_std = torch.tensor([0.229, 0.224, 0.225] * 2, device=device).view(1, 6, 1, 1)

    # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    with open(output_path, 'w') as f, open(runtime_path, 'w') as fr:
        description = f"{video_path.split('/')[-1]} {tile_size:>3} {classifier}"
        command_queue.put((device, {'description': description,
                                    'completed': 0, 'total': frame_count}))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        assert len(frames) == frame_count, f"Expected {frame_count} frames, got {len(frames)}"

        # Warm up the model
        for frame_idx, frame in enumerate(frames[:16]):
            previous_frame = frames[frame_idx - 1] if frame_idx > 0 else frames[1]
            relevance_grid, runtime = process_frame_tiles(frame, previous_frame, model, tile_size, device, 
                                                         normalize_mean, normalize_std)  # type: ignore
        
        for frame_idx, frame in enumerate(frames):
            # Process frame with the model
            previous_frame = frames[frame_idx - 1] if frame_idx > 0 else frames[1]
            relevance_grid, runtime = process_frame_tiles(frame, previous_frame, model, tile_size, device, 
                                                         normalize_mean, normalize_std)  # type: ignore

            # Create result entry for this frame
            frame_entry = {
                "classification_size": relevance_grid.shape,
                "classification_hex": relevance_grid.flatten().tobytes().hex(),
                "idx": frame_idx,
            }

            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            fr.write(json.dumps(runtime) + '\n')
            command_queue.put((device, {'completed': frame_idx}))


def main():
    """
    Main function that orchestrates the video tile classification process using parallel processing.

    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tile_size combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves classification results

    Note:
        - The script expects a specific directory structure:
          {DATASETS_DIR}/{dataset}/ - contains video files
          {DATA_CACHE}/{dataset}/{video_file_name}/training/results/{classifier_name}_{tile_size}/model.pth - contains trained models
          where DATASETS_DIR and DATA_CACHE are both /polyis-data/video-datasets-low
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - A separate model is loaded for each video directory, classifier, and tile size combination
        - Output files are saved in {DATA_CACHE}/{dataset}/{video_file_name}/020_relevancy/score/{classifier_name}_{tile_size}/score.jsonl
        - If no trained model is found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)

    # Create tasks list with all video/classifier/tile_size combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)

        for videoset in ['test']:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Dataset directory {videoset_dir} does not exist, skipping...")
                continue

            # Get all video files from the dataset directory
            videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video in sorted(videos):
                for classifier in CLASSIFIERS:
                    for tile_size in TILE_SIZES:
                        func = partial(classify, dataset, video, classifier, tile_size)
                        funcs.append(func)

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main()
