#!/usr/local/bin/python

import argparse
import itertools
import json
import os
import shutil
import time
from typing import Callable
import numpy as np
import multiprocessing as mp
from functools import partial

from polyis.pack.adapters import group_tiles_all
from polyis.utilities import format_time, ProgressBar, get_config, load_classification_results, build_param_str
from polyis.io import cache, store
from polyis.pareto import build_pareto_combo_filter
from polyis.sample.ilp.c.gurobi import solve_ilp


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
CLASSIFIERS: list[str] = config['EXEC']['CLASSIFIERS']
DATASETS: list[str] = config['EXEC']['DATASETS']
SAMPLE_RATES: list[int] = config['EXEC']['SAMPLE_RATES']
TRACKERS: list[str] = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS: list[float] = [t for t in config['EXEC']['TRACKING_ACCURACY_THRESHOLDS'] if t is not None]

TILEPADDING_MODE = 0  # 0: No padding, 1: Connected padding, 2: Disconnected padding

# Accuracy threshold indices into max_rate_table.npy (axis 2)
# Index: 0=60%, 1=70%, 2=80%, 3=90%, 4=95%, 5=100%
_ALL_ACCURACY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
# Mapping from float threshold to index in max_rate_table.npy
_ACCURACY_THRESHOLD_TO_IDX = {t: i for i, t in enumerate(_ALL_ACCURACY_THRESHOLDS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Group and prune polyominoes from classification results using ILP'
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--time-limit', type=float, default=None,
                        dest='time_limit',
                        help='ILP solver wall-clock time limit in seconds (default: 0.5)')
    return parser.parse_args()


def load_max_sampling_rate(dataset: str, tracker: str, tile_size: int, accuracy_idx: int) -> np.ndarray:
    """
    Load the max sampling rate table for each tile position at a given accuracy threshold.
    
    The table is stored as a 3D array [grid_height, grid_width, num_accuracy_thresholds]
    in: {CACHE_DIR}/{dataset}/indexing/track_rate/{tracker}_{tile_size}/max_rate_table.npy
    
    Args:
        dataset: Dataset name (e.g. 'caldot2-y05')
        tracker: Tracker name (e.g. 'bytetrackcython')
        tile_size: Tile size used for classification (e.g. 60)
        accuracy_idx: Index into accuracy thresholds [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
                      Default 0 (60% accuracy, most permissive)
    
    Returns:
        2D numpy array [grid_height, grid_width] of max sampling rates per tile
    """
    max_rate_path = cache.index(
        dataset, 'track_rates',
        f'{tracker}_{tile_size}', 'max_rate_table.npy'
    )
    
    if not os.path.exists(max_rate_path):
        raise FileNotFoundError(
            f"Max sampling rate table not found at {max_rate_path}. "
            f"Please run the tracking rate indexing step first."
        )
    
    max_rate_table = np.load(max_rate_path)  # shape: (grid_height, grid_width, num_thresholds)
    assert max_rate_table.ndim == 3, \
        f"Expected 3D array, got shape {max_rate_table.shape}"
    assert 0 <= accuracy_idx < max_rate_table.shape[2], \
        f"accuracy_idx {accuracy_idx} out of range [0, {max_rate_table.shape[2]})"
    
    return max_rate_table[:, :, accuracy_idx]


def process_video(
    dataset: str,
    videoset: str,
    video: str,
    classifier: str,
    tile_size: int,
    sample_rate: int,
    tracker: str,
    tracking_accuracy_threshold: float,
    time_limit: float | None,
):
    """
    Process a single video to group and prune polyominoes via ILP.

    Args:
        dataset: Dataset name
        videoset: Videoset name (test, train, or valid)
        video: Video filename
        classifier: Classifier name
        tile_size: Tile size used for classification
        sample_rate: Frame sampling stride used by upstream classification
        tracker: Tracker name
        tracking_accuracy_threshold: Accuracy threshold (float, e.g. 0.90)
        time_limit: ILP solver wall-clock time limit in seconds (None for default 0.5s)
    """
    # Assert that classification results exist before attempting to load.
    score_path = cache.exec(dataset, 'relevancy', video,
                            f'{classifier}_{tile_size}_{sample_rate}', 'score', 'score.jsonl')
    assert os.path.exists(score_path), f"Score path {score_path} does not exist"

    # Resolve float threshold to index in max_rate_table.npy
    accuracy_idx = _ACCURACY_THRESHOLD_TO_IDX[tracking_accuracy_threshold]

    # Load classification results
    print(f"[{video}] Loading classification results...", flush=True)
    raw_results = load_classification_results(
        dataset, video, tile_size, classifier, sample_rate, verbose=False
    )
    
    classifications = []
    frame_indices = []
    grid_height = None
    grid_width = None
    
    for frame_data in raw_results:
        frame_indices.append(frame_data['idx'])
        if grid_height is None or grid_width is None:
            grid_height, grid_width = frame_data['classification_size']
        
        # Create bitmap from classifications
        hex_data = frame_data['classification_hex']
        flat_data = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
        classification_grid = flat_data.reshape((grid_height, grid_width))
        # TODO: for now, classification threshold is 128
        binary_grid = (classification_grid >= 128).astype(np.uint8) * 255
        
        classifications.append(binary_grid)
    
    num_frames = len(classifications)
    print(f"[{video}] Loaded {num_frames} frames (grid: {grid_height}x{grid_width})", flush=True)
    
    # Load max sampling rate table (2D array per tile)
    max_sampling_distance = load_max_sampling_rate(
        dataset, tracker, tile_size, accuracy_idx
    )
    assert max_sampling_distance.shape == (grid_height, grid_width), \
        f"Max sampling rate shape {max_sampling_distance.shape} doesn't match grid ({grid_height}, {grid_width})"
    
    max_sampling_distance //= sample_rate
    max_sampling_distance = np.maximum(max_sampling_distance, 1)
    
    step_times = {}

    # Step 1: Prepare bitmaps
    print(f"[{video}] Step 1: Preparing bitmaps...", flush=True)
    step_start = (time.time_ns() / 1e6)
    bitmaps = np.stack(classifications, axis=0) // 255
    step_times['prepare_bitmaps'] = (time.time_ns() / 1e6) - step_start
    print(f"[{video}] Step 1 done: prepare_bitmaps={step_times['prepare_bitmaps']:.1f}ms", flush=True)

    # Step 2: Group tiles into polyominoes
    print(f"[{video}] Step 2: Grouping tiles into polyominoes...", flush=True)
    step_start = (time.time_ns() / 1e6)
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(
        bitmaps.astype(np.uint8),
        TILEPADDING_MODE
    )
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)  # convert memoryview -> numpy array
    step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start
    print(f"[{video}] Step 2 done: group_tiles={step_times['group_tiles']:.1f}ms", flush=True)
    
    # Step 3: Solve ILP to prune polyominoes
    total_polyominoes = sum(len(p) for p in polyomino_lengths)
    print(f"[{video}] Step 3: Solving ILP ({total_polyominoes} polyominoes across {num_frames} frames)...", flush=True)
    assert grid_height is not None
    assert grid_width is not None
    ilp_result = solve_ilp(
        tile_to_polyomino_id,
        polyomino_lengths,
        max_sampling_distance,
        grid_height,
        grid_width,
        time_limit_seconds=time_limit or 0.1,
    )
    selected = ilp_result.selected
    step_times['build_ilp'] = ilp_result.build_ms
    step_times['solve_ilp'] = ilp_result.solve_ms
    print(f"[{video}] Step 3 done: build_ilp={step_times['build_ilp']:.1f}ms solve_ilp={step_times['solve_ilp']:.1f}ms ({len(selected)} polyominoes selected)", flush=True)
    
    # Step 4: Convert selected polyominoes back to binary grids (vectorized)
    print(f"[{video}] Step 4: Extracting pruned grids...", flush=True)
    step_start = (time.time_ns() / 1e6)
    pruned_grids = []
    
    for b in range(num_frames):
        # Build set of selected polyomino IDs for this frame
        selected_ids = {pid for (frame, pid) in selected if frame == b}
        
        # Vectorized: check if each tile's polyomino is selected
        tile_ids = tile_to_polyomino_id[b]
        mask = np.isin(tile_ids, list(selected_ids)) & (tile_ids >= 0)
        grid = (mask.astype(np.uint8) * 255)
        
        pruned_grids.append(grid)
    step_times['extract_grids'] = (time.time_ns() / 1e6) - step_start
    print(f"[{video}] Step 4 done: extract_grids={step_times['extract_grids']:.1f}ms", flush=True)
    
    # Create output directory using build_param_str for consistent naming.
    # When a time limit is specified (i.e. the experiment mode), results go into a
    # separate cache stage so they never overwrite the standard pipeline output.
    param_str = build_param_str(classifier=classifier, tilesize=tile_size, sample_rate=sample_rate,
                                tracker=tracker, tracking_accuracy_threshold=tracking_accuracy_threshold)
    output_dir = cache.exec(
        dataset, 'pruned-polyominoes', video,
        param_str
    )
    if time_limit is not None:
        output_dir = cache.exec(
            dataset, 'pruned-polyominoes-tl', video,
            param_str, f'tl{time_limit}'
        )

    os.makedirs(output_dir, exist_ok=True)

    score_dir = os.path.join(output_dir, 'score')
    os.makedirs(score_dir, exist_ok=True)
    
    # Step 5: Save results
    print(f"[{video}] Step 5: Saving results to {score_dir}...", flush=True)
    # step_start = (time.time_ns() / 1e6)
    output_path = os.path.join(score_dir, 'score.jsonl')
    runtime_path = os.path.join(score_dir, 'runtime.jsonl')
    
    with open(output_path, 'w') as f:
        for frame_idx, grid in zip(frame_indices, pruned_grids):
            frame_entry = {
                "classification_size": grid.shape,
                "classification_hex": grid.flatten().tobytes().hex(),
                "idx": frame_idx,
            }
            f.write(json.dumps(frame_entry) + '\n')
    # step_times['save_results'] = (time.time_ns() / 1e6) - step_start
    # print(f"[{video}] Step 5 done: save_results={step_times['save_results']:.1f}ms", flush=True)

    # Save runtime data including the solver time limit used for this run.
    with open(runtime_path, 'w') as f:
        f.write(json.dumps({'runtime': format_time(**step_times), 'time_limit': time_limit}) + '\n')
    
    print(f"[{video}] All steps complete!", flush=True)


def process_all(
    dataset: str,
    videoset: str,
    videos: list[str],
    classifier: str,
    tile_size: int,
    sample_rate: int,
    tracker: str,
    tracking_accuracy_threshold: float,
    time_limit: float | None,
    gpu_id: int,
    command_queue: mp.Queue,
):
    device = f'cuda:{gpu_id}'
    # Build a human-readable description for the progress bar.
    description = f"{dataset} {classifier}_{tile_size} sr{sample_rate} {tracker} {tracking_accuracy_threshold}"
    # Report initial progress: 0 of N videos done.
    command_queue.put((device, {'completed': 0, 'total': len(videos), 'description': description}))
    # Iterate over all videos in the split for this parameter combination.
    for i, video in enumerate(videos):
        process_video(dataset, videoset, video, classifier, tile_size, sample_rate,
                      tracker, tracking_accuracy_threshold, time_limit)
        # Advance the progress bar by one unit after each video completes.
        command_queue.put((device, {'completed': i + 1, 'total': len(videos), 'description': description}))


def main():
    """
    Main function that orchestrates the polyomino grouping and pruning process.
    
    This function:
    1. Reads classification results from 020_relevancy
    2. Converts hex-encoded classifications to binary numpy arrays (threshold=128)
    3. Groups connected tiles into polyominoes using group_tiles_all
    4. Solves ILP to select minimum set of polyominoes satisfying temporal constraints
    5. Outputs pruned tiles in same format as p021 (hex-encoded binary grids)
    
    Input:
        - Classification results: {CACHE_DIR}/{dataset}/execution/{video}/020_relevancy/{classifier}_{tile_size}_{sample_rate}/score/score.jsonl
        - Max sampling rate: {CACHE_DIR}/{dataset}/indexing/track_rate/{tracker}_{tile_size}/max_rate_table.npy
    
    Output:
        - Pruned classification: {CACHE_DIR}/{dataset}/execution/{video}/022_pruned_polyominoes/{classifier}_{tile_size}_{sample_rate}_{tracking_accuracy_threshold}_{tracker}/score/score.jsonl
    """
    args = parse_args()

    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.valid:
        selected_videosets.append('valid')

    if not selected_videosets:
        # Default to valid-only to avoid running the full grid on test unnecessarily.
        selected_videosets = ['valid']

    # Solver time limit for this run (seconds).
    time_limit: float | None = args.time_limit

    mp.set_start_method('spawn', force=True)

    # Build allowed-combo set for the test pass (None means no filtering applies).
    allowed_combos = build_pareto_combo_filter(
        DATASETS, selected_videosets,
        ['classifier', 'tilesize', 'sample_rate', 'tracker', 'tracking_accuracy_threshold'],
    )

    funcs: list[Callable[[int, mp.Queue], None]] = []

    print(f"datasets: {DATASETS}")
    print(f"sample_rates: {SAMPLE_RATES}")
    print(f"trackers: {TRACKERS}")
    print(f"tracking_accuracy_thresholds: {TRACKING_ACCURACY_THRESHOLDS}")
    print(f"time_limit: {time_limit}s")
    for dataset, videoset in itertools.product(DATASETS, selected_videosets):
        videoset_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

        videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video in videos:
            shutil.rmtree(cache.exec(dataset, 'pruned-polyominoes', video), ignore_errors=True)
        for classifier, tile_size, sample_rate in itertools.product(CLASSIFIERS, TILE_SIZES, SAMPLE_RATES):
            # Iterate over all tracker × threshold combinations for each sample_rate.
            for tracker, threshold in itertools.product(TRACKERS, TRACKING_ACCURACY_THRESHOLDS):
                # Skip parameter combos not on the Pareto front during the test pass.
                combo = (classifier, tile_size, sample_rate, tracker, threshold)
                if allowed_combos is not None and combo not in allowed_combos[dataset]:
                    continue
                func = partial(process_all, dataset, videoset, sorted(videos), classifier,
                               tile_size, sample_rate, tracker, threshold, time_limit)
                funcs.append(func)
    
    print(f"Created {len(funcs)} tasks to process")
    
    num_workers = mp.cpu_count()
    num_workers = mp.cpu_count() // 2
    # num_workers = 2
    if len(funcs) > 0:
        ProgressBar(num_workers=num_workers, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)
    
    print("All tasks completed!")


if __name__ == '__main__':
    main()
