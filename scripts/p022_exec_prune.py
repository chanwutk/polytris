#!/usr/local/bin/python

"""
Execution script for pruning polyominoes using temporal covering algorithms.

This script processes classification results from p020_exec_classify.py and
applies pruning algorithms to reduce the number of frames/polyominoes while
maintaining temporal coverage constraints.
"""

import argparse
import json
import os
import shutil
import numpy as np
import time
from typing import List, Tuple, Dict, Any
import multiprocessing as mp
from functools import partial

from polyis.utilities import get_config, ProgressBar, format_time
from polyis.sample import greedy_prune_polyominoes, ilp_prune_polyominoes
from polyis.pack.group_tiles import group_tiles, free_polyomino_array

# Load configuration
config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']
DATASETS = config['EXEC']['DATASETS']
TILEPADDING_VALUES = config['EXEC']['TILEPADDING_VALUES']


def load_relevance_scores(cache_video_dir: str, classifier: str, tile_size: int) -> Tuple[np.ndarray, List[int]]:
    """
    Load relevance scores from p020 classification results.

    Parameters:
        cache_video_dir: Cache directory for the video
        classifier: Classifier name
        tile_size: Tile size

    Returns:
        Tuple of (relevance_bitmaps, frame_indices)
        - relevance_bitmaps: 3D array [frames, height, width] of uint8 scores (0-255)
        - frame_indices: List of frame indices
    """
    # Load classification results
    score_dir = os.path.join(cache_video_dir, '020_relevancy', f'{classifier}_{tile_size}', 'score')
    score_path = os.path.join(score_dir, 'score.jsonl')

    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Classification results not found at {score_path}")

    # Read all frames
    frames_data = []
    with open(score_path, 'r') as f:
        for line in f:
            frames_data.append(json.loads(line))

    # Sort by frame index
    frames_data.sort(key=lambda x: x['idx'])

    # Extract dimensions from first frame
    first_frame = frames_data[0]
    grid_height, grid_width = first_frame['classification_size']

    # Create 3D array for all frames
    num_frames = len(frames_data)
    relevance_bitmaps = np.zeros((num_frames, grid_height, grid_width), dtype=np.uint8)
    frame_indices = []

    # Fill array with relevance scores
    for i, frame_data in enumerate(frames_data):
        frame_idx = frame_data['idx']
        frame_indices.append(frame_idx)

        # Decode hex string to numpy array
        hex_data = frame_data['classification_hex']
        flat_scores = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
        relevance_bitmaps[i] = flat_scores.reshape(grid_height, grid_width)

    return relevance_bitmaps, frame_indices


def prune_video(
    dataset: str,
    videoset: str,
    video: str,
    classifier: str,
    tile_size: int,
    tilepadding: str,
    algorithm: str,
    threshold: float,
    max_gap_uniform: int,
    gpu_id: int,
    command_queue: mp.Queue
):
    """
    Process a single video and apply pruning algorithm.

    Parameters:
        dataset: Dataset name
        videoset: Videoset name (test, train, valid)
        video: Video filename
        classifier: Classifier name
        tile_size: Tile size (30, 60, 120)
        tilepadding: Tile padding mode
        algorithm: Pruning algorithm ('greedy' or 'ilp')
        threshold: Threshold for converting relevance to binary
        max_gap_uniform: Uniform maximum sampling gap for all tiles
        gpu_id: GPU ID (unused but kept for consistency)
        command_queue: Queue for progress updates
    """
    # Set up paths
    cache_video_dir = os.path.join(CACHE_DIR, dataset, 'execution', video)
    output_dir = os.path.join(cache_video_dir, '022_pruned',
                              f'{classifier}_{tile_size}_{tilepadding}')

    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Load relevance scores from p020
    try:
        relevance_bitmaps, frame_indices = load_relevance_scores(cache_video_dir, classifier, tile_size)
    except FileNotFoundError as e:
        print(f"Skipping {video}: {e}")
        command_queue.put((f'cuda:{gpu_id}', {'completed': 1, 'total': 1}))
        return

    num_frames, grid_height, grid_width = relevance_bitmaps.shape

    # Create max_gaps array (uniform for now)
    max_gaps = np.full((grid_height, grid_width), max_gap_uniform, dtype=np.int32)

    # Convert relevance to binary and extract polyominoes for each frame
    polyomino_arrays = []
    start_time = time.time()

    for f in range(num_frames):
        # Convert to binary bitmap
        bitmap = (relevance_bitmaps[f] >= int(threshold * 255)).astype(np.uint8)

        # Extract polyominoes using group_tiles
        poly_array_ptr = group_tiles(bitmap, mode=0)  # No padding for initial grouping
        polyomino_arrays.append(poly_array_ptr)

    # Run pruning algorithm
    description = f"{video} {tile_size:>3} {classifier} {algorithm}"
    command_queue.put((f'cuda:{gpu_id}', {
        'description': description,
        'completed': 0,
        'total': 1
    }))

    if algorithm == 'greedy':
        # Use greedy algorithm
        if greedy_prune_polyominoes is None:
            print(f"Error: Cython greedy module not built. Run setup.py build_ext first.")
            command_queue.put((f'cuda:{gpu_id}', {'completed': 1}))
            return

        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays,
            relevance_bitmaps,
            max_gaps,
            threshold
        )
    elif algorithm == 'ilp':
        # Use ILP algorithm
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays,
            relevance_bitmaps,
            max_gaps,
            threshold,
            solver='CBC',  # Use CBC solver (comes with PuLP)
            time_limit=300  # 5 minute time limit
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    end_time = time.time()
    runtime = end_time - start_time

    # Calculate statistics
    reduction_percentage = 100.0 * (1 - len(selected_frames) / num_frames)
    total_tiles_original = num_frames * grid_height * grid_width
    total_tiles_selected = len(selected_frames) * grid_height * grid_width  # Simplified

    statistics = {
        'video': video,
        'dataset': dataset,
        'classifier': classifier,
        'tile_size': tile_size,
        'algorithm': algorithm,
        'threshold': threshold,
        'max_gap_uniform': max_gap_uniform,
        'num_frames_original': num_frames,
        'num_frames_selected': len(selected_frames),
        'reduction_percentage': reduction_percentage,
        'total_tiles_original': total_tiles_original,
        'total_tiles_selected': total_tiles_selected,
        'runtime_seconds': runtime
    }

    # Save results
    # Save selected frame indices
    selected_frames_path = os.path.join(output_dir, 'selected_frames.jsonl')
    with open(selected_frames_path, 'w') as f:
        for frame_idx in selected_frames:
            entry = {
                'frame_idx': frame_idx,
                'original_idx': frame_indices[frame_idx] if frame_idx < len(frame_indices) else frame_idx
            }
            f.write(json.dumps(entry) + '\n')

    # Save statistics
    statistics_path = os.path.join(output_dir, 'statistics.json')
    with open(statistics_path, 'w') as f:
        json.dump(statistics, f, indent=2)

    # Save runtime info
    runtime_path = os.path.join(output_dir, 'runtime.jsonl')
    with open(runtime_path, 'w') as f:
        runtime_entry = format_time(pruning=runtime)
        f.write(json.dumps(runtime_entry) + '\n')

    # Free polyomino arrays
    for poly_ptr in polyomino_arrays:
        free_polyomino_array(poly_ptr)

    command_queue.put((f'cuda:{gpu_id}', {'completed': 1}))

    print(f"Pruned {video}: {len(selected_frames)}/{num_frames} frames selected "
          f"({reduction_percentage:.1f}% reduction)")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Execute polyomino pruning using temporal covering algorithms'
    )
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    parser.add_argument('--algorithm', type=str, default='ilp',
                        choices=['greedy', 'ilp'],
                        help='Pruning algorithm to use (default: greedy)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for converting relevance to binary (default: 0.5)')
    parser.add_argument('--max-gap', type=int, default=30,
                        help='Maximum sampling gap in frames (default: 30)')
    return parser.parse_args()


def main():
    """
    Main function that orchestrates the pruning process.
    """
    args = parse_args()

    # Determine which videosets to process
    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.train:
        selected_videosets.append('train')
    if args.valid:
        selected_videosets.append('valid')

    # Default to test if none specified
    if not selected_videosets:
        selected_videosets = ['test']

    mp.set_start_method('spawn', force=True)

    # Create tasks list
    funcs = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)

        for videoset in selected_videosets:
            videoset_dir = os.path.join(dataset_dir, videoset)
            if not os.path.exists(videoset_dir):
                print(f"Dataset directory {videoset_dir} does not exist, skipping...")
                continue

            # Get all video files
            videos = [f for f in os.listdir(videoset_dir)
                     if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            for video in sorted(videos):
                for classifier in CLASSIFIERS:
                    for tile_size in TILE_SIZES:
                        for tilepadding in TILEPADDING_VALUES:
                            func = partial(
                                prune_video,
                                dataset, videoset, video, classifier, tile_size,
                                tilepadding, args.algorithm, args.threshold,
                                args.max_gap
                            )
                            funcs.append(func)

    # Run tasks with progress bar
    num_workers = min(mp.cpu_count(), 8)  # Use CPU workers since no GPU needed
    print(f"Processing {len(funcs)} tasks with {num_workers} workers...")
    ProgressBar(num_workers=num_workers, num_tasks=len(funcs)).run_all(funcs)

    print("Pruning complete!")


if __name__ == '__main__':
    main()