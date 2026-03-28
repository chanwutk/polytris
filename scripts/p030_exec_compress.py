#!/usr/local/bin/python

import argparse
from enum import IntEnum
import itertools
import json
import os
from typing import Callable, NamedTuple
import cv2
import numpy as np
import shutil
import time
import multiprocessing as mp
from functools import partial

import torch

from polyis import dtypes
from polyis.utilities import format_time, load_classification_results, load_pruned_classification_results, ProgressBar, get_config, TILEPADDING_MAPS, TilePadding, build_param_str, scale_to_percent
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.io import cache, store
from polyis.pareto import build_pareto_combo_filter


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
CLASSIFIERS: list[str] = config['EXEC']['CLASSIFIERS']
DATASETS: list[str] = config['EXEC']['DATASETS']
SAMPLE_RATES: list[int] = config['EXEC']['SAMPLE_RATES']
TILEPADDING_MODES: list[TilePadding] = config['EXEC']['TILEPADDING_MODES']
CANVAS_SCALES: list[float] = config['EXEC']['CANVAS_SCALE']
TRACKERS: list[str] = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS: list[float] = config['EXEC']['TRACKING_ACCURACY_THRESHOLDS']


class PackMode(IntEnum):
    """Packing mode options for bin packing algorithms."""
    Easiest_Fit = 0  # Pack into collage with most empty space
    First_Fit = 1    # Pack into first collage that fits
    Best_Fit = 2     # Pack into collage with least empty space that fits


class PolyominoPosition(NamedTuple):
    oy: int
    ox: int
    py: int
    px: int
    frame: int
    shape: np.ndarray


OUTPUT_DIR_MAP = {
    PackMode.Best_Fit: '033_compressed_frames',
    PackMode.Easiest_Fit: '034_compressed_frames',
    PackMode.First_Fit: '035_compressed_frames',
}

OffsetLookup = tuple[tuple[int, int], tuple[int, int], int]


def _compute_polyomino_tile_boundaries(
    oy: int,
    ox: int,
    py: int,
    px: int,
    i_coords: np.ndarray,
    j_coords: np.ndarray,
    src_grid_y_starts: np.ndarray,
    src_grid_y_ends: np.ndarray,
    src_grid_x_starts: np.ndarray,
    src_grid_x_ends: np.ndarray,
    dst_grid_y_starts: np.ndarray,
    dst_grid_y_ends: np.ndarray,
    dst_grid_x_starts: np.ndarray,
    dst_grid_x_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Compute source row boundaries via lookup to avoid repeated integer arithmetic.
    sy_starts = src_grid_y_starts[oy + i_coords]
    # Compute source column boundaries via lookup to avoid repeated integer arithmetic.
    sx_starts = src_grid_x_starts[ox + j_coords]
    # Compute source row end boundaries via lookup to avoid repeated integer arithmetic.
    sy_ends = src_grid_y_ends[oy + i_coords]
    # Compute source column end boundaries via lookup to avoid repeated integer arithmetic.
    sx_ends = src_grid_x_ends[ox + j_coords]
    # Compute destination row boundaries via lookup to avoid repeated integer arithmetic.
    dy_starts = dst_grid_y_starts[py + i_coords]
    # Compute destination column boundaries via lookup to avoid repeated integer arithmetic.
    dx_starts = dst_grid_x_starts[px + j_coords]
    # Compute destination row end boundaries via lookup to avoid repeated integer arithmetic.
    dy_ends = dst_grid_y_ends[py + i_coords]
    # Compute destination column end boundaries via lookup to avoid repeated integer arithmetic.
    dx_ends = dst_grid_x_ends[px + j_coords]
    return sy_starts, sx_starts, sy_ends, sx_ends, dy_starts, dx_starts, dy_ends, dx_ends


def _copy_same_shape_tiles(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
    same_shape: np.ndarray,
):
    # Iterate over tile indices where source and destination shapes already match.
    for idx in np.flatnonzero(same_shape):
        # Copy a same-size source tile directly into the destination tile.
        canvas[dy_starts[idx]:dy_ends[idx], dx_starts[idx]:dx_ends[idx]] = frame[sy_starts[idx]:sy_ends[idx], sx_starts[idx]:sx_ends[idx]]


def _copy_mismatched_tiles_with_edge_repeat(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
    src_hs: np.ndarray,
    src_ws: np.ndarray,
    dst_hs: np.ndarray,
    dst_ws: np.ndarray,
    same_shape: np.ndarray,
):
    # Iterate over tile indices where source and destination shapes are different.
    for idx in np.flatnonzero(~same_shape):
        # Read source row bounds for this tile.
        sy_start = sy_starts[idx]
        # Read source row end for this tile.
        sy_end = sy_ends[idx]
        # Read source column bounds for this tile.
        sx_start = sx_starts[idx]
        # Read source column end for this tile.
        sx_end = sx_ends[idx]
        # Read destination row bounds for this tile.
        dy_start = dy_starts[idx]
        # Read destination row end for this tile.
        dy_end = dy_ends[idx]
        # Read destination column bounds for this tile.
        dx_start = dx_starts[idx]
        # Read destination column end for this tile.
        dx_end = dx_ends[idx]

        # Read source tile height for this tile.
        src_h = src_hs[idx]
        # Read source tile width for this tile.
        src_w = src_ws[idx]
        # Read destination tile height for this tile.
        dst_h = dst_hs[idx]
        # Read destination tile width for this tile.
        dst_w = dst_ws[idx]

        # Skip degenerate tiles to avoid invalid indexing.
        if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
            continue

        # Compute copied height as the overlap between source and destination heights.
        copy_h = min(src_h, dst_h)
        # Compute copied width as the overlap between source and destination widths.
        copy_w = min(src_w, dst_w)

        # Create a destination tile view for in-place writes.
        dst_tile = canvas[dy_start:dy_end, dx_start:dx_end]
        # Create a source tile view for reading source pixels.
        src_tile = frame[sy_start:sy_end, sx_start:sx_end]

        # Copy the overlapping source area into the destination tile.
        dst_tile[:copy_h, :copy_w] = src_tile[:copy_h, :copy_w]

        # Extend the last copied row downward when destination height is larger.
        if dst_h > copy_h:
            dst_tile[copy_h:dst_h, :copy_w] = src_tile[copy_h - 1:copy_h, :copy_w]

        # Extend the last copied column rightward when destination width is larger.
        if dst_w > copy_w:
            dst_tile[:, copy_w:dst_w] = dst_tile[:, copy_w - 1:copy_w]


def _copy_polyomino_tiles_to_canvas(
    canvas: np.ndarray,
    frame: np.ndarray,
    sy_starts: np.ndarray,
    sx_starts: np.ndarray,
    sy_ends: np.ndarray,
    sx_ends: np.ndarray,
    dy_starts: np.ndarray,
    dx_starts: np.ndarray,
    dy_ends: np.ndarray,
    dx_ends: np.ndarray,
):
    # Precompute destination tile heights.
    dst_hs = (dy_ends - dy_starts)
    # Precompute destination tile widths.
    dst_ws = (dx_ends - dx_starts)
    # Precompute source tile heights.
    src_hs = (sy_ends - sy_starts)
    # Precompute source tile widths.
    src_ws = (sx_ends - sx_starts)
    # Identify tiles where source and destination shapes already match.
    same_shape = (src_hs == dst_hs) & (src_ws == dst_ws)

    # Copy same-shape tiles with direct slicing.
    _copy_same_shape_tiles(
        canvas=canvas,
        frame=frame,
        sy_starts=sy_starts,
        sx_starts=sx_starts,
        sy_ends=sy_ends,
        sx_ends=sx_ends,
        dy_starts=dy_starts,
        dx_starts=dx_starts,
        dy_ends=dy_ends,
        dx_ends=dx_ends,
        same_shape=same_shape,
    )

    # Copy mismatched tiles with in-place edge-repeat behavior.
    _copy_mismatched_tiles_with_edge_repeat(
        canvas=canvas,
        frame=frame,
        sy_starts=sy_starts,
        sx_starts=sx_starts,
        sy_ends=sy_ends,
        sx_ends=sx_ends,
        dy_starts=dy_starts,
        dx_starts=dx_starts,
        dy_ends=dy_ends,
        dx_ends=dx_ends,
        src_hs=src_hs,
        src_ws=src_ws,
        dst_hs=dst_hs,
        dst_ws=dst_ws,
        same_shape=same_shape,
    )


def _update_polyomino_metadata(
    index_map: dtypes.IndexMap,
    offset_lookup: list[OffsetLookup],
    gid: int,
    py: int,
    px: int,
    oy: int,
    ox: int,
    frame_idx: int,
    i_coords: np.ndarray,
    j_coords: np.ndarray,
):
    # Write the group id into index_map positions covered by this polyomino.
    index_map[py + i_coords, px + j_coords] = gid
    # Append the mapping tuple for decompression lookup.
    offset_lookup.append(((py, px), (oy, ox), frame_idx))


def parse_args():
    parser = argparse.ArgumentParser(description='Execute compression of video tiles into images based on classification results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification probability (0.0 to 1.0)')
    parser.add_argument('--mode', type=lambda x: PackMode[x], 
                        default=PackMode.Best_Fit,
                        help='Packing mode for the pack_all function. Options: Easiest_Fit, First_Fit, Best_Fit (default: Best_Fit)')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def save_packed_image(canvas: dtypes.NPImage, index_map: dtypes.IndexMap, offset_lookup: list[OffsetLookup],
                      collage_idx: int, start_idx: int, frame_idx: int, output_dir: str, step_times: dict):
    """
    Save the packed image, index_map, and offset_lookup.
    
    Args:
        canvas: The canvas to save
        index_map: The index map to save
        offset_lookup: The offset lookup to save
        start_idx: The start index of the packed image
        frame_idx: The end index of the packed image
        output_dir: The directory to save the files
        step_times: The step times to update
    """
    image_dir = os.path.join(output_dir, 'images')
    index_map_dir = os.path.join(output_dir, 'index_maps')
    offset_lookup_dir = os.path.join(output_dir, 'offset_lookups')

    # Profile: Save canvas
    step_start = (time.time_ns() / 1e6)
    img_path = os.path.join(image_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.jpg')
    cv2.imwrite(img_path, canvas)
    step_times['save_canvas'] = (time.time_ns() / 1e6) - step_start

    # Profile: Save index_map and offset_lookup
    step_start = (time.time_ns() / 1e6)
    index_map_path = os.path.join(index_map_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.npy')
    np.save(index_map_path, index_map)

    offset_lookup_path = os.path.join(offset_lookup_dir, f'{collage_idx:04d}_{start_idx:04d}_{frame_idx:04d}.jsonl')
    with open(offset_lookup_path, 'w') as f:
        for offset in offset_lookup:
            f.write(json.dumps(offset) + '\n')
    step_times['save_mapping_files'] = (time.time_ns() / 1e6) - step_start


# PolyominoPosition = tuple[int, int, int, int, int, int, np.ndarray]
Collage = list[PolyominoPosition]


def compress(dataset: str, videoset: str, video: str, classifier: str, tilesize: int,
             sample_rate: int, tilepadding: TilePadding, canvas_scale: float,
             tracker: str | None, tracking_accuracy_threshold: float | None,
             threshold: float, mode: PackMode):
    """
    Compress a single video by batch processing all sampled frames at once using pack_all.

    Args:
        dataset: Name of the dataset
        videoset: Videoset name (test, train, or valid)
        video: Name of the video
        classifier: Classifier name to use
        tilesize: Tile size to use
        tilepadding: Whether to apply padding to classification results
        sample_rate: Sample rate for frame sampling (1 = all frames)
        canvas_scale: Canvas grid scale factor relative to the source classification grid
        tracker: Tracker name for upstream pruning
        tracking_accuracy_threshold: Accuracy threshold for pruning (None = no pruning)
        threshold: Threshold for classification probability
        mode: Packing mode
    """
    video_path = store.dataset(dataset, videoset, video)

    # Load classification results from either p022 (pruned) or p020/p021 (unpruned).
    try:
        # Branch to pruned inputs when an accuracy threshold is configured.
        if tracking_accuracy_threshold is not None:
            # Read pruned score.jsonl for this (tracker, threshold) tuple.
            results = load_pruned_classification_results(
                dataset, video, tilesize, classifier, tracker,
                tracking_accuracy_threshold, sample_rate)
        else:
            # Read unpruned score.jsonl for this (classifier, tile size, sample rate) tuple.
            results = load_classification_results(
                dataset, video, tilesize, classifier, sample_rate)
    except FileNotFoundError as exc:
        # Re-raise so the caller sees the missing upstream cache as a hard failure.
        raise

    # Create output directory for compression results
    output_dir_name = OUTPUT_DIR_MAP[mode]
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate, tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker, tracking_accuracy_threshold=tracking_accuracy_threshold)
    # Only '033_compressed_frames' (Best_Fit) is mapped in cache.exec; other modes use manual paths
    if mode == PackMode.Best_Fit:
        output_dir = cache.exec(dataset, 'comp-frames', video, param_str)
    else:
        output_dir = cache.exec(dataset, 'comp-frames', video).parent / output_dir_name / param_str
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir)
    index_map_dir = os.path.join(output_dir, 'index_maps')
    os.makedirs(index_map_dir)
    offset_lookup_dir = os.path.join(output_dir, 'offset_lookups')
    os.makedirs(offset_lookup_dir)

    # Open video to get dimensions
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error: Could not open video {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Note: results contains only sampled frames (frame_idx % sample_rate == 0)
    # The total video frame count may be larger than len(results)

    # Calculate source grid dimensions from the original video using fixed tile size.
    src_grid_height = height // tilesize
    src_grid_width = width // tilesize
    # Calculate destination grid dimensions by scaling source grid tile counts.
    dst_grid_height = max(1, int(round(src_grid_height * canvas_scale)))
    dst_grid_width = max(1, int(round(src_grid_width * canvas_scale)))
    # Calculate canvas pixel dimensions from scaled grid dimensions and fixed tile size.
    canvas_height = dst_grid_height * tilesize
    canvas_width = dst_grid_width * tilesize

    # Step 1: Group tiles for all frames to get polyominoes
    timing_data = []

    # Create mapping from array index to absolute frame index
    # This is CRITICAL: results contains only sampled frames, but we need absolute frame indices
    array_idx_to_frame_idx = {i: result['idx'] for i, result in enumerate(results)}

    polyominoes_stacks = np.empty(len(results), dtype=np.uint64)
    for array_idx, frame_result in enumerate(results):
        step_times = {}

        # Get classification results
        step_start = (time.time_ns() / 1e6)
        classifications: str = frame_result['classification_hex']
        classification_size: tuple[int, int] = frame_result['classification_size']
        step_times['get_classifications'] = (time.time_ns() / 1e6) - step_start

        # Create bitmap from classifications
        step_start = (time.time_ns() / 1e6)
        bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
        bitmap_frame = bitmap_frame > (threshold * 255)
        bitmap_frame = bitmap_frame.astype(np.uint8)
        assert dtypes.is_bitmap(bitmap_frame), bitmap_frame.shape
        step_times['create_bitmap'] = (time.time_ns() / 1e6) - step_start

        # Group connected tiles into polyominoes
        step_start = (time.time_ns() / 1e6)
        polyominoes = group_tiles(bitmap_frame, TILEPADDING_MAPS[tilepadding])
        polyominoes_stacks[array_idx] = polyominoes
        step_times['group_tiles'] = (time.time_ns() / 1e6) - step_start

        # Use absolute frame index in timing data
        absolute_frame_idx = array_idx_to_frame_idx[array_idx]
        timing_data.append({'step': 'group_tiles', 'frame_idx': absolute_frame_idx, 'runtime': format_time(**step_times)})

        # # Update progress
        # if frame_idx % max(1, len(results) // 100) == 0:
        #     command_queue.put((device, {'description': description + ' grouping', 'completed': frame_idx}))

    # Step 2: Pack all polyominoes in batches
    num_batches = 1
    batch_size = len(polyominoes_stacks) // num_batches
    # Handle case where len(polyominoes_stacks) < num_batches
    if batch_size == 0:
        batch_size = 1
        num_batches = len(polyominoes_stacks)

    # Initialize empty list to store all collages from all batches
    collages = []
    total_pack_time = 0.0

    # Process each batch
    for batch_idx in range(num_batches):
        # Calculate batch boundaries
        start_idx = batch_idx * batch_size
        # For the last batch, include any remaining frames
        if batch_idx == num_batches - 1:
            end_idx = len(polyominoes_stacks)
        else:
            end_idx = start_idx + batch_size

        # Extract batch of polyominoes
        batch_polyominoes = polyominoes_stacks[start_idx:end_idx]

        # Pack this batch
        batch_start = (time.time_ns() / 1e6)
        batch_collages_ = pack(batch_polyominoes, dst_grid_height, dst_grid_width, int(mode))
        batch_pack_time = (time.time_ns() / 1e6) - batch_start
        total_pack_time += batch_pack_time

        # Adjust frame indices in batch_collages to use absolute frame indices
        # pack_all returns frame indices relative to the batch (0-indexed within batch)
        # We need to map these through our array_idx_to_frame_idx mapping to get absolute frame indices
        batch_collages: list[list[PolyominoPosition]] = []
        for collage in batch_collages_:
            batch_collages.append([
                PolyominoPosition(oy=poly_pos.oy, ox=poly_pos.ox,
                                  py=poly_pos.py, px=poly_pos.px,
                                  # Map from batch-relative index to absolute frame index
                                  frame=array_idx_to_frame_idx[poly_pos.frame + start_idx],
                                  shape=poly_pos.shape)
                for poly_pos in collage
            ])

        # Merge batch collages into the overall collages list
        collages.extend(batch_collages)

        # Record timing for this batch
        timing_data.append({
            'step': f'pack_batch_{batch_idx}',
            'frames': f'{start_idx}-{end_idx-1}',
            'runtime': format_time(pack_batch=batch_pack_time)
        })

    # # Record total packing time
    # timing_data.append({'step': 'pack_all_total', 'runtime': format_time(pack_all_total=total_pack_time)})

    # Step 3: Read ONLY sampled frames from video (only frames in results)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create set of sampled frame indices for fast lookup
    sampled_indices_set = {result['idx'] for result in results}
    # Create mapping from absolute frame index to frame data
    # This allows us to access frames by their absolute index in the original video
    frame_idx_to_frame: dict[int, np.ndarray] = {}

    for idx in range(num_frames_total):
        ret, frame = cap.read()
        if not ret:
            break

        # Only store frames that are in the sampled set
        if idx in sampled_indices_set:
            assert dtypes.is_np_image(frame), frame.shape
            frame_idx_to_frame[idx] = frame

    assert len(frame_idx_to_frame) == len(results), f"Expected {len(results)} sampled frames, got {len(frame_idx_to_frame)}"
    assert cap.read()[0] is False, "Expected no more frames"
    cap.release()

    # Step 4: Render and save each collage

    # Build source grid row indices to precompute source tile pixel boundaries.
    src_grid_rows = np.arange(src_grid_height, dtype=np.int32)
    # Build source grid column indices to precompute source tile pixel boundaries.
    src_grid_cols = np.arange(src_grid_width, dtype=np.int32)
    # Precompute source row starts as exact fixed-size tile boundaries.
    src_grid_y_starts = src_grid_rows * tilesize
    # Precompute source row ends as exact fixed-size tile boundaries.
    src_grid_y_ends = src_grid_y_starts + tilesize
    # Precompute source column starts as exact fixed-size tile boundaries.
    src_grid_x_starts = src_grid_cols * tilesize
    # Precompute source column ends as exact fixed-size tile boundaries.
    src_grid_x_ends = src_grid_x_starts + tilesize

    # Build destination grid row indices to precompute destination tile pixel boundaries.
    dst_grid_rows = np.arange(dst_grid_height, dtype=np.int32)
    # Build destination grid column indices to precompute destination tile pixel boundaries.
    dst_grid_cols = np.arange(dst_grid_width, dtype=np.int32)
    # Precompute destination row starts as exact fixed-size tile boundaries.
    dst_grid_y_starts = dst_grid_rows * tilesize
    # Precompute destination row ends as exact fixed-size tile boundaries.
    dst_grid_y_ends = dst_grid_y_starts + tilesize
    # Precompute destination column starts as exact fixed-size tile boundaries.
    dst_grid_x_starts = dst_grid_cols * tilesize
    # Precompute destination column ends as exact fixed-size tile boundaries.
    dst_grid_x_ends = dst_grid_x_starts + tilesize

    for collage_idx, collage in enumerate(collages):
        assert len(collage) > 0, f"Expected at least one polyomino in collage {collage_idx}"
        step_times = {}

        # Initialize canvas and metadata structures
        step_start = (time.time_ns() / 1e6)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        assert dtypes.is_np_image(canvas), canvas.shape
        index_map = np.zeros((dst_grid_height, dst_grid_width), dtype=np.uint16)
        assert dtypes.is_index_map(index_map), index_map.shape
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        step_times['initialize_canvas'] = (time.time_ns() / 1e6) - step_start

        # Get frame range for this collage
        frame_indices = [pos.frame for pos in collage]
        start_frame = min(frame_indices)
        end_frame = max(frame_indices)

        # Process each polyomino in this collage
        step_start = (time.time_ns() / 1e6)
        render_fetch_time = 0.0
        render_tile_bound_time = 0.0
        render_tile_size_time = 0.0
        render_crop_time = 0.0
        render_pad_time = 0.0
        render_copy_time = 0.0
        render_metadata_time = 0.0
        for gid, poly_pos in enumerate(collage, start=1):
            step_start = (time.time_ns() / 1e6)
            oy, ox, py, px, frame_idx, shape = poly_pos

            # Get source frame using absolute frame index
            frame = frame_idx_to_frame[frame_idx]

            # Tile rendering: scale grid positions to raw video resolution
            i_coords = shape[:, 0]
            j_coords = shape[:, 1]
            render_fetch_time += (time.time_ns() / 1e6) - step_start

            step_start = (time.time_ns() / 1e6)
            # Compute all source and destination tile boundaries for this polyomino.
            sy_starts, sx_starts, sy_ends, sx_ends, dy_starts, dx_starts, dy_ends, dx_ends = _compute_polyomino_tile_boundaries(
                oy=oy,
                ox=ox,
                py=py,
                px=px,
                i_coords=i_coords,
                j_coords=j_coords,
                src_grid_y_starts=src_grid_y_starts,
                src_grid_y_ends=src_grid_y_ends,
                src_grid_x_starts=src_grid_x_starts,
                src_grid_x_ends=src_grid_x_ends,
                dst_grid_y_starts=dst_grid_y_starts,
                dst_grid_y_ends=dst_grid_y_ends,
                dst_grid_x_starts=dst_grid_x_starts,
                dst_grid_x_ends=dst_grid_x_ends,
            )
            render_tile_bound_time += (time.time_ns() / 1e6) - step_start

            step_start = (time.time_ns() / 1e6)
            # Copy all tiles for this polyomino and collect detailed copy timings.
            _copy_polyomino_tiles_to_canvas(
                canvas=canvas,
                frame=frame,
                sy_starts=sy_starts,
                sx_starts=sx_starts,
                sy_ends=sy_ends,
                sx_ends=sx_ends,
                dy_starts=dy_starts,
                dx_starts=dx_starts,
                dy_ends=dy_ends,
                dx_ends=dx_ends,
            )
            # Accumulate time spent in tile-size preparation.
            render_copy_time += (time.time_ns() / 1e6) - step_start

            step_start = (time.time_ns() / 1e6)
            # Update all metadata structures for this polyomino placement.
            _update_polyomino_metadata(
                index_map=index_map,
                offset_lookup=offset_lookup,
                gid=gid,
                py=py,
                px=px,
                oy=oy,
                ox=ox,
                frame_idx=frame_idx,
                i_coords=i_coords,
                j_coords=j_coords,
            )
            render_metadata_time += (time.time_ns() / 1e6) - step_start
        # step_times['render_tiles'] = (time.time_ns() / 1e6) - step_start
        step_times['render_fetch'] = render_fetch_time
        step_times['render_tile_bound'] = render_tile_bound_time
        step_times['render_tile_size'] = render_tile_size_time
        step_times['render_crop'] = render_crop_time
        step_times['render_pad'] = render_pad_time
        step_times['render_copy'] = render_copy_time
        step_times['render_metadata'] = render_metadata_time

        # Save the collage
        step_start = (time.time_ns() / 1e6)
        save_packed_image(canvas, index_map, offset_lookup, collage_idx, start_frame, end_frame, output_dir, step_times)
        step_times['save_collage'] = (time.time_ns() / 1e6) - step_start

        timing_data.append({'step': 'process_collage', 'runtime': format_time(**step_times)})

    # # Free polyomino stacks
    # print('free polyominoes')
    # step_start = (time.time_ns() / 1e6)
    # command_queue.put((device, {'description': description + ' freeing polyominoes', 'completed': 0, 'total': len(polyominoes_stacks)}))
    # for idx, polyominoes in enumerate(polyominoes_stacks):
    #     free_polyimino_stack(polyominoes)
    #     command_queue.put((device, {'description': description + ' freeing polyominoes', 'completed': idx}))
    # end_time = (time.time_ns() / 1e6)
    # timing_data.append({'step': 'free_polyominoes', 'runtime': format_time(free_polyominoes=end_time - step_start)})
    # print('free polyominoes done')

    # Save runtime data
    runtime_file = os.path.join(output_dir, 'runtime.jsonl')
    with open(runtime_file, 'w') as f:
        for data in timing_data:
            f.write(json.dumps(data) + '\n')


def compress_all(dataset: str, videoset: str, videos: list[str], classifier: str, tilesize: int,
                 sample_rate: int, tilepadding: TilePadding, canvas_scale: float,
                 tracker: str | None, tracking_accuracy_threshold: float | None,
                 threshold: float, mode: PackMode, gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'
    # Build a human-readable description for the progress bar.
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker,
                                tracking_accuracy_threshold=tracking_accuracy_threshold)
    description = f"{dataset} {param_str}"
    # Report initial progress: 0 of N videos done.
    command_queue.put((device, {'completed': 0, 'total': len(videos), 'description': description}))
    # Iterate over all videos in the split for this parameter combination.
    for i, video in enumerate(videos):
        compress(dataset, videoset, video, classifier, tilesize, sample_rate,
                 tilepadding, canvas_scale, tracker, tracking_accuracy_threshold,
                 threshold, mode)
        # Advance the progress bar by one unit after each video completes.
        command_queue.put((device, {'completed': i + 1, 'total': len(videos), 'description': description}))


def main(args):
    """
    Main function that orchestrates the video tile compression process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves compression results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - threshold (float): Threshold for classification probability (0.0 to 1.0)
            - mode (PackMode): Packing mode for the pack_all function. Options: Easiest_Fit, First_Fit, Best_Fit (default: Best_Fit)
            
    Note:
        - The script expects classification results from 020_exec_classify.py in:
          {CACHE_DIR}/{dataset}/execution/{video}/020_relevancy/{classifier}_{tilesize}/score/
        - Looks for score.jsonl files
        - Videos are read from {DATASETS_DIR}/{dataset}/
        - Compressed images are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/images/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/index_maps/
        - Mappings are saved to {CACHE_DIR}/{dataset}/execution/{video}/03{mode+3}_compressed_frames/{classifier}_{tilesize}/offset_lookups/
        - If no classification results are found for a video, that video is skipped with a warning
    """
    mp.set_start_method('spawn', force=True)
    prediction_threshold = args.threshold
    mode = args.mode
    
    # Determine which videosets to process based on arguments
    videosets = []
    if args.test:
        videosets.append('test')
    if args.train:
        videosets.append('train')
    if args.valid:
        videosets.append('valid')
    
    # If no videosets are specified, default to valid only
    if not videosets:
        videosets = ['valid']

    # Build allowed-combo set for the test pass (None means no filtering applies).
    allowed_combos = build_pareto_combo_filter(
        DATASETS, videosets,
        ['classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale',
         'tracker', 'tracking_accuracy_threshold'],
        collapse_tracker_when_no_threshold=True,
    )

    # Create tasks list with all video/classifier/tilesize/sample_rate combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset, videoset in itertools.product(DATASETS, videosets):
        videoset_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

        # Get all video files from the dataset directory
        videos = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video in videos:
            shutil.rmtree(cache.exec(dataset, 'comp-frames', video), ignore_errors=True)
        for classifier, tilesize, tilepadding, sample_rate, canvas_scale, threshold in itertools.product(
            CLASSIFIERS, TILE_SIZES, TILEPADDING_MODES, SAMPLE_RATES, CANVAS_SCALES, TRACKING_ACCURACY_THRESHOLDS):
            for tracker in [None] if threshold is None else TRACKERS:
                # Skip parameter combos not on the Pareto front during the test pass.
                combo = (classifier, tilesize, sample_rate, tilepadding, canvas_scale, tracker, threshold)
                if allowed_combos is not None and combo not in allowed_combos[dataset]:
                    continue
                funcs.append(partial(compress_all, dataset, videoset, sorted(videos), classifier, tilesize, sample_rate,
                                     tilepadding, canvas_scale, tracker, threshold, prediction_threshold, mode))

    print(f"Created {len(funcs)} tasks to process")
    
    device_count = torch.cuda.device_count()
    device_count = 40
    ProgressBar(num_workers=device_count, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
