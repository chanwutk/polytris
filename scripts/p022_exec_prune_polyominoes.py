#!/usr/local/bin/python

import argparse
import json
import os
import time
from typing import Callable
import numpy as np
import multiprocessing as mp
from functools import partial
import pulp

from polyis.pack.adapters import group_tiles_all
from polyis.utilities import format_time, ProgressBar, get_config, load_classification_results, build_param_str


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
CLASSIFIERS = config['EXEC']['CLASSIFIERS']
DATASETS = config['EXEC']['DATASETS']
TRACKERS = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS = [t for t in config['EXEC']['TRACKING_ACCURACY_THRESHOLDS'] if t is not None]

TILEPADDING_MODE = 0  # 0: No padding, 1: Connected padding, 2: Disconnected padding

# Accuracy threshold indices into max_rate_table.npy (axis 2)
# Index: 0=60%, 1=70%, 2=80%, 3=90%, 4=95%, 5=100%
_ALL_ACCURACY_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
# Mapping from float threshold to index in max_rate_table.npy
_ACCURACY_THRESHOLD_TO_IDX = {t: i for i, t in enumerate(_ALL_ACCURACY_THRESHOLDS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Group and prune polyominoes from classification results using ILP'
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    return parser.parse_args()


def load_max_sampling_rate(dataset: str, tracker: str, tile_size: int, accuracy_idx: int) -> np.ndarray:
    """
    Load the max sampling rate table for each tile position at a given accuracy threshold.
    
    The table is stored as a 3D array [grid_height, grid_width, num_accuracy_thresholds]
    in: {CACHE_DIR}/{dataset}/indexing/tracking_rate/{tracker}_{tile_size}/max_rate_table.npy
    
    Args:
        dataset: Dataset name (e.g. 'caldot2-y05')
        tracker: Tracker name (e.g. 'bytetrackcython')
        tile_size: Tile size used for classification (e.g. 60)
        accuracy_idx: Index into accuracy thresholds [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
                      Default 0 (60% accuracy, most permissive)
    
    Returns:
        2D numpy array [grid_height, grid_width] of max sampling rates per tile
    """
    max_rate_path = os.path.join(
        CACHE_DIR, dataset, 'indexing', 'tracking_rate',
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


def solve_ilp(
    tile_to_polyomino_id: np.ndarray,
    polyomino_lengths: list[list[int]],
    max_sampling_distance: np.ndarray,
    grid_height: int,
    grid_width: int
) -> set[tuple[int, int]]:
    """
    Solve integer linear program to select minimum set of polyominoes.
    
    Args:
        tile_to_polyomino_id: 3D array [frame, row, col] -> polyomino_id (-1 if no polyomino)
        polyomino_lengths: polyomino_lengths[frame][polyomino_id] = number of cells
        max_sampling_distance: 2D array [grid_height, grid_width] of max sampling distance per tile
        grid_height: Height of the grid
        grid_width: Width of the grid
        
    Returns:
        Set of (frame_idx, polyomino_id) tuples representing selected polyominoes
    """
    B = len(polyomino_lengths)  # Number of frames
    
    # Create optimization problem
    prob = pulp.LpProblem("MinCells", pulp.LpMinimize)
    
    # Create variables: x[b, k] = 1 if polyomino k in frame b is selected
    x = {}
    for b in range(B):
        for k in range(len(polyomino_lengths[b])):
            x[(b, k)] = pulp.LpVariable(f"x_{b}_{k}", cat='Binary')
    
    # Objective: minimize total number of cells in selected polyominoes
    prob += pulp.lpSum([x[(b, k)] * polyomino_lengths[b][k] for (b, k) in x])
    
    # Temporal constraints: for each cell (n, m)
    for n in range(grid_height):
        for m in range(grid_width):
            # Find all frames where this cell is covered by a polyomino
            pos = [b for b in range(B) if tile_to_polyomino_id[b][n, m] >= 0]
            
            if len(pos) <= 1:
                continue  # No temporal constraint needed
            
            # For each consecutive pair of frames where this cell appears
            for i in range(len(pos) - 1):
                b_curr = pos[i]
                b_next_avail = pos[i + 1]
                t_limit = b_curr + max_sampling_distance[n, m]  # Per-tile sampling distance
                
                # Get the polyomino IDs at current and next frames
                k_curr = tile_to_polyomino_id[b_curr][n, m]
                k_next = tile_to_polyomino_id[b_next_avail][n, m]
                
                curr_var = x[(b_curr, k_curr)]
                
                if b_next_avail > t_limit:
                    # Gap too large: both polyominoes must be selected (mandatory bridge)
                    prob += curr_var == 1
                    prob += x[(b_next_avail, k_next)] == 1
                else:
                    # Window constraint: if current is selected, at least one in window must be selected
                    # Use pointer-based iteration to avoid O(n²) list slicing
                    j = i + 1
                    while j < len(pos) and pos[j] <= t_limit:
                        j += 1
                    window_frames = pos[i+1:j]  # Single slice, no filtering needed
                    if window_frames:
                        window_vars = [x[(b, tile_to_polyomino_id[b][n, m])] 
                                      for b in window_frames]
                        prob += pulp.lpSum(window_vars) >= curr_var

            # Always choose first and last time tile is covered
            b_first = pos[0]
            b_last = pos[-1]
            k_first = tile_to_polyomino_id[b_first][n, m]
            k_last = tile_to_polyomino_id[b_last][n, m]
            prob += x[(b_first, k_first)] == 1
            prob += x[(b_last, k_last)] == 1
    
    # Solve the ILP with a time limit to avoid infinite hangs
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    
    # Extract selected polyominoes
    selected = set()
    for (b, k), var in x.items():
        if pulp.value(var) == 1:
            selected.add((b, k))
    
    return selected

def process_video(
    dataset: str,
    videoset: str,
    video: str,
    classifier: str,
    tile_size: int,
    tracker: str,
    tracking_accuracy_threshold: float,
    gpu_id: int,
    command_queue: mp.Queue,
):
    """
    Process a single video to group and prune polyominoes via ILP.

    Args:
        dataset: Dataset name
        videoset: Videoset name (test, train, or valid)
        video: Video filename
        classifier: Classifier name
        tile_size: Tile size used for classification
        tracker: Tracker name
        tracking_accuracy_threshold: Accuracy threshold (float, e.g. 0.90)
        gpu_id: GPU ID (for progress tracking)
        command_queue: Queue for progress updates
    """
    # Resolve float threshold to index in max_rate_table.npy
    accuracy_idx = _ACCURACY_THRESHOLD_TO_IDX[tracking_accuracy_threshold]
    device = f'cuda:{gpu_id}'
    
    # Load classification results
    print(f"[{video}] Loading classification results...", flush=True)
    raw_results = load_classification_results(CACHE_DIR, dataset, video, tile_size, classifier, verbose=False)
    
    classifications = []
    grid_height = None
    grid_width = None
    
    for frame_data in raw_results:
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
    
    # Progress update
    description = f"{video} {tile_size:>3} {classifier}"
    command_queue.put((device, {
        'description': description,
        'completed': 0,
        'total': 3  # Group, Solve ILP, Save
    }))
    
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
    command_queue.put((device, {'completed': 1}))
    
    # Step 3: Solve ILP to prune polyominoes
    total_polyominoes = sum(len(p) for p in polyomino_lengths)
    print(f"[{video}] Step 3: Solving ILP ({total_polyominoes} polyominoes across {num_frames} frames)...", flush=True)
    step_start = (time.time_ns() / 1e6)
    assert grid_height is not None
    assert grid_width is not None
    selected = solve_ilp(
        tile_to_polyomino_id,
        polyomino_lengths,
        max_sampling_distance,
        grid_height,
        grid_width
    )
    step_times['solve_ilp'] = (time.time_ns() / 1e6) - step_start
    print(f"[{video}] Step 3 done: solve_ilp={step_times['solve_ilp']:.1f}ms ({len(selected)} polyominoes selected)", flush=True)
    command_queue.put((device, {'completed': 2}))
    
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
    
    # Create output directory using build_param_str for consistent naming
    param_str = build_param_str(classifier=classifier, tilesize=tile_size,
                                tracker=tracker, tracking_accuracy_threshold=tracking_accuracy_threshold)
    output_dir = os.path.join(
        CACHE_DIR, dataset, 'execution', video, '022_pruned_polyominoes',
        param_str
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
        for idx, grid in enumerate(pruned_grids):
            frame_entry = {
                "classification_size": grid.shape,
                "classification_hex": grid.flatten().tobytes().hex(),
                "idx": idx,
            }
            f.write(json.dumps(frame_entry) + '\n')
    # step_times['save_results'] = (time.time_ns() / 1e6) - step_start
    # print(f"[{video}] Step 5 done: save_results={step_times['save_results']:.1f}ms", flush=True)

    # Save runtime data
    with open(runtime_path, 'w') as f:
        f.write(json.dumps({'runtime': format_time(**step_times)}) + '\n')
    
    print(f"[{video}] All steps complete!", flush=True)
    command_queue.put((device, {'completed': 3}))


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
        - Classification results: {CACHE_DIR}/{dataset}/execution/{video}/020_relevancy/{classifier}_{tile_size}/score/score.jsonl
        - Max sampling rate: {CACHE_DIR}/{dataset}/indexing/tracking_rate/{tracker}_{tile_size}/max_rate_table.npy
    
    Output:
        - Pruned classification: {CACHE_DIR}/{dataset}/execution/{video}/022_pruned_polyominoes/{classifier}_{tile_size}/score/score.jsonl
    """
    args = parse_args()

    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.train:
        selected_videosets.append('train')
    if args.valid:
        selected_videosets.append('valid')

    if not selected_videosets:
        selected_videosets = ['test']

    mp.set_start_method('spawn', force=True)

    funcs: list[Callable[[int, mp.Queue], None]] = []

    print(f"datasets: {DATASETS}")
    print(f"trackers: {TRACKERS}")
    print(f"tracking_accuracy_thresholds: {TRACKING_ACCURACY_THRESHOLDS}")
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)

        for videoset in selected_videosets:
            videoset_dir = os.path.join(dataset_dir, videoset)
            print(f"videoset_dir: {videoset_dir}")
            if not os.path.exists(videoset_dir):
                print(f"Videoset directory {videoset_dir} does not exist, skipping...")
                continue

            videos = [f for f in os.listdir(videoset_dir)
                     if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            for video in sorted(videos):
                for classifier in CLASSIFIERS:
                    for tile_size in TILE_SIZES:
                        score_path = os.path.join(
                            CACHE_DIR, dataset, 'execution', video, '020_relevancy',
                            f'{classifier}_{tile_size}_1', 'score', 'score.jsonl'
                        )
                        if not os.path.exists(score_path):
                            continue
                        # Iterate over all tracker × threshold combinations
                        for tracker in TRACKERS:
                            for threshold in TRACKING_ACCURACY_THRESHOLDS:
                                func = partial(
                                    process_video,
                                    dataset, videoset, video, classifier, tile_size,
                                    tracker, threshold,
                                )
                                print(f"Added task for {video} {tile_size} {classifier} {tracker} {threshold}")
                                funcs.append(func)
    
    print(f"Created {len(funcs)} tasks to process")
    
    num_workers = mp.cpu_count()
    if len(funcs) > 0:
        ProgressBar(num_workers=num_workers, num_tasks=len(funcs)).run_all(funcs)
    
    print("All tasks completed!")


if __name__ == '__main__':
    main()