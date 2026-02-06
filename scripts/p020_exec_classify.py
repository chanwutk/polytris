#!/usr/local/bin/python

import argparse
import json
import os
from typing import Callable, cast
import cv2
import torch
import numpy as np
import time
import shutil
import multiprocessing as mp
from functools import partial

from polyis.images import ImgNHWC, splitNHWC

from polyis.train.select_model_optimization import select_model_optimization
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']
DATASETS = config['EXEC']['DATASETS']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']
BATCH_SIZE = 16


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
    model_path = os.path.join(results_path, 'model_best.pth')

    if os.path.exists(model_path):
        # print(f"Loading {classifier_name} model for tile size {tile_size} from {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        model.half()
        return model

    raise FileNotFoundError(f"No trained model found for {classifier_name} tile size {tile_size} in {results_path}")


def classify_batch(
    grid_width: int,
    grid_height: int,
    positions: torch.Tensor,
    batch_frames: list[np.ndarray],
    batch_prev_frames: list[np.ndarray],
    model: torch.nn.Module,
    tile_size: int,
    device: str,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    always_relevant_mask: torch.Tensor,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Process up to BATCH_SIZE frames in one batch: prepare tiles for each frame,
    run inference once on all valid tiles, scatter results per frame.
    Returns (list of relevance_grids, runtime dict).
    """
    batch_size = len(batch_frames)
    num_tiles = grid_height * grid_width

    with torch.no_grad():
        send_start = time.time_ns() / 1e6
        # 1. Stack on CPU then send to GPU (one transfer is faster than N)
        frames_stacked = np.stack(batch_frames, axis=0)
        frames_tensor = torch.from_numpy(frames_stacked).to(device)
        # Stack previous frames from original video ordering and send to GPU
        prev_stacked = np.stack(batch_prev_frames, axis=0)
        prev_frames = torch.from_numpy(prev_stacked).to(device)
        send_runtime = time.time_ns() / 1e6 - send_start

        diff_start = time.time_ns() / 1e6
        # 2. Find diff
        diff = torch.abs(
            frames_tensor.to(torch.int16) - prev_frames.to(torch.int16)
        ).to(torch.uint8)
        # 4. Cat frames and diff
        frames_6ch = torch.cat([frames_tensor, diff], dim=-1)
        diff_runtime = time.time_ns() / 1e6 - diff_start

        reshape_start = time.time_ns() / 1e6
        # 5. Split
        tiles_nghwc = splitNHWC(cast(ImgNHWC, frames_6ch), tile_size, tile_size)
        # 6. Flat tiles (batch_size*num_tiles, tile_size, tile_size, 6)
        tiles_flat = tiles_nghwc.reshape(
            batch_size * num_tiles, tile_size, tile_size, 6
        )
        reshape_runtime = time.time_ns() / 1e6 - reshape_start

        mask_start = time.time_ns() / 1e6
        # 7. Same as non-batched: non_black, valid mask, normalize
        non_black_mask = (
            tiles_flat.reshape(batch_size * num_tiles, -1).any(dim=1).to(torch.bool)
        )
        always_relevant_expanded = (
            always_relevant_mask.to(torch.bool)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
        )
        valid_flat_idx = (non_black_mask & always_relevant_expanded).nonzero().squeeze(1)
        tiles_valid = tiles_flat[valid_flat_idx].float() / 255.0
        tiles_nchw_valid = tiles_valid.permute(0, 3, 1, 2)
        tiles_nchw_valid = (tiles_nchw_valid - normalize_mean) / normalize_std
        positions_expanded = (
            positions.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * num_tiles, 2)
        )
        all_positions = positions_expanded[valid_flat_idx]
        mask_runtime = (time.time_ns() / 1e6) - mask_start
        # transform_times = [transform_runtime / batch_size] * batch_size

        inference_start = time.time_ns() / 1e6
        all_tiles = tiles_nchw_valid.half()
        all_positions = all_positions.half()

        predictions = torch.sigmoid(model(all_tiles, all_positions))
        inference_runtime = time.time_ns() / 1e6 - inference_start

        collect_start = time.time_ns() / 1e6
        # Recover probabilities for all tiles, then split by frame
        predictions_uint8 = (predictions * 255).to(torch.uint8)
        probabilities_full = torch.zeros(
            batch_size * num_tiles, 1, device=device, dtype=torch.uint8
        )
        probabilities_full[valid_flat_idx] = predictions_uint8
        probabilities_per_frame = probabilities_full.reshape(
            batch_size, grid_height, grid_width
        )
        relevance_grids_np = probabilities_per_frame.cpu().numpy()
        relevance_grids = list(relevance_grids_np)

        collect_runtime = time.time_ns() / 1e6 - collect_start

        runtime = format_time(inference=inference_runtime, collect=collect_runtime, reshape=reshape_runtime, mask=mask_runtime, diff=diff_runtime, send=send_runtime)
    return relevance_grids, runtime


def classify(dataset: str, videoset: str, video: str, classifier: str, tile_size: int, sample_rate: int, gpu_id: int, command_queue: mp.Queue):
    """
    Process a single video file and save tile classification results to a JSONL file.

    This function reads a video file frame by frame, processes sampled frames to classify
    tiles using the trained classifier model for the specified tile size, and saves the
    results in JSONL format. Each line in the output file represents one frame with
    its tile classifications and runtime measurement.

    Args:
        dataset: Name of the dataset
        videoset: Videoset name (test, train, or valid)
        video: Name of the video
        classifier: Classifier name to use
        tile_size: Tile size to use
        sample_rate: Sample rate for frame sampling (1 = all frames, 2 = every 2nd frame, etc.)
        gpu_id: GPU ID to use for processing
        command_queue: Queue for progress updates

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

    video_path = os.path.join(DATASETS_DIR, dataset, videoset, video)
    cache_video_dir = os.path.join(CACHE_DIR, dataset, 'execution', video)

    # Create output directory structure
    output_dir = os.path.join(cache_video_dir, '020_relevancy')

    # Create score directory for this classifier, tile size, and sample rate
    classifier_dir = os.path.join(output_dir, f'{classifier}_{tile_size}_{sample_rate}')
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
    model, method_name = select_model_optimization(model, benchmark_results, device, tile_size,
                                      (width * height) // (tile_size * tile_size))

    # Pre-create normalization tensors for ImageNet normalization (1.9x speedup vs torchvision.Normalize); half for model.half()
    normalize_mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=device, dtype=torch.float16).view(1, 6, 1, 1)
    normalize_std = torch.tensor([0.229, 0.224, 0.225] * 2, device=device, dtype=torch.float16).view(1, 6, 1, 1)

    # Load always_relevant bitmap if available to filter out tiles that have never been relevant
    always_relevant_path = os.path.join(CACHE_DIR, dataset, 'indexing', 'always_relevant', f'{tile_size}_all.npy')
    assert os.path.exists(always_relevant_path), f"Always relevant bitmap not found for {dataset} {tile_size}"

    # Load the bitmap (2D array where 1 = relevant at some point, 0 = never relevant)
    always_relevant_bitmap = np.load(always_relevant_path)
    # Flatten to match the tile processing order and convert to tensor
    always_relevant_mask = torch.from_numpy(always_relevant_bitmap.flatten()).to(device).to(torch.uint8)

    assert width % tile_size == 0, f"Width {width} is not divisible by tile size {tile_size}"
    assert height % tile_size == 0, f"Height {height} is not divisible by tile size {tile_size}"
    grid_width = width // tile_size
    grid_height = height // tile_size

    y_indices = torch.arange(grid_height, device=device, dtype=torch.uint8).repeat_interleave(grid_width)
    x_indices = torch.arange(grid_width, device=device, dtype=torch.uint8).repeat(grid_height)
    positions = torch.stack([y_indices, x_indices], dim=1).float()

    # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    with open(output_path, 'w') as f, open(runtime_path, 'w') as fr:
        # Read all frames from video
        frames: list[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR to RGB
        cap.release()
        assert len(frames) == frame_count, f"Expected {frame_count} frames, got {len(frames)}"

        # Filter to sampled frames only (frame_idx % sample_rate == 0)
        sampled_indices = [idx for idx in range(len(frames)) if idx % sample_rate == 0]
        last_idx = len(frames) - 1
        if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
            sampled_indices.append(last_idx)
        sampled_frames = [frames[idx] for idx in sampled_indices]

        # Update progress tracking to use sampled frame count
        description = f"{video_path.split('/')[-1]} {tile_size:>3} {classifier} {method_name} sr{sample_rate}"
        command_queue.put((device, {'description': description,
                                    'completed': 0, 'total': len(sampled_frames)}))

        # Warm up the model with first 16 sampled frames (or fewer if not enough sampled frames)
        warmup_count = min(16, len(sampled_indices))
        for i in range(warmup_count):
            frame_idx = sampled_indices[i]
            frame = frames[frame_idx]
            # Get previous frame from original video ordering
            prev_frame = frames[frame_idx - 1] if frame_idx > 0 else frame
            _, _ = classify_batch(
                grid_width,
                grid_height,
                positions,
                [frame],
                [prev_frame],
                model,
                tile_size,
                device,
                normalize_mean,
                normalize_std,
                always_relevant_mask,
            )

        # Process sampled frames in batches of BATCH_SIZE; one inference per batch
        frame_idx_in_sampled = 0
        while frame_idx_in_sampled < len(sampled_frames):
            batch_end = min(frame_idx_in_sampled + BATCH_SIZE, len(sampled_frames))
            batch_frames = [sampled_frames[i] for i in range(frame_idx_in_sampled, batch_end)]
            batch_indices = [sampled_indices[i] for i in range(frame_idx_in_sampled, batch_end)]
            # Get previous frames from original video ordering for each frame in batch
            batch_prev_frames = [frames[idx - 1] if idx > 0 else frames[idx + 1] for idx in batch_indices]

            relevance_grids, runtime = classify_batch(
                grid_width,
                grid_height,
                positions,
                batch_frames,
                batch_prev_frames,
                model,
                tile_size,
                device,
                normalize_mean,
                normalize_std,
                always_relevant_mask,
            )
            fr.write(json.dumps(runtime) + '\n')
            for j, relevance_grid in enumerate(relevance_grids):
                absolute_idx = batch_indices[j]  # Use absolute frame index from original video
                frame_entry = {
                    "classification_size": relevance_grid.shape,
                    "classification_hex": relevance_grid.flatten().tobytes().hex(),
                    "idx": absolute_idx,  # CRITICAL: Store absolute frame index, not relative
                }
                f.write(json.dumps(frame_entry) + '\n')
                command_queue.put((device, {'completed': frame_idx_in_sampled + j + 1}))

            frame_idx_in_sampled = batch_end


def parse_args():
    parser = argparse.ArgumentParser(description='Execute tile classification using trained models')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


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

    # Create tasks list with all video/classifier/tile_size/sample_rate combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)

        for videoset in selected_videosets:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Dataset directory {videoset_dir} does not exist, skipping...")
                continue

            # Get all video files from the dataset directory
            videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video in sorted(videos):
                for classifier in CLASSIFIERS:
                    for tile_size in TILE_SIZES:
                        for sample_rate in SAMPLE_RATES:
                            func = partial(classify, dataset, videoset, video, classifier, tile_size, sample_rate)
                            funcs.append(func)

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main()
