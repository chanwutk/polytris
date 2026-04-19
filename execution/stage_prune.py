"""
Stage 3: Polyomino pruning via ILP (p022).  Optional.

When tracking_accuracy_threshold is set, this stage groups tiles into
polyominoes and solves an ILP to select the minimum set satisfying
temporal sampling constraints.  When the threshold is None, the pipeline
bypasses this stage entirely (the queue is wired directly from Classify
to Compress).
"""

from __future__ import annotations

import os

import numpy as np
import torch.multiprocessing as mp

from polyis.io import cache
from polyis.pack.adapters import group_tiles_all
from polyis.sample.ilp.c.gurobi import solve_ilp

from execution.pipeline import (
    PipelineConfig,
    VideoClassifications,
)

# Accuracy threshold -> index mapping (matching p022).
_ALL_ACCURACY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
_ACCURACY_THRESHOLD_TO_IDX = {t: i for i, t in enumerate(_ALL_ACCURACY_THRESHOLDS)}

TILEPADDING_MODE = 0  # 0: No padding


def prune_process(
    *,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Prune stage process entry-point.

    Reads VideoClassifications from *in_queue*, runs polyomino grouping and
    ILP-based pruning, then sends modified VideoClassifications to *out_queue*.

    ``None`` on *in_queue* propagates and the process exits.
    """
    tile_size = config.tile_size
    sample_rate = config.sample_rate
    tracker = config.tracker
    threshold = config.tracking_accuracy_threshold
    assert threshold is not None, "Prune stage should not be spawned without a threshold"
    accuracy_idx = _ACCURACY_THRESHOLD_TO_IDX[threshold]

    while True:
        msg = in_queue.get()
        if msg is None:
            out_queue.put(None)
            return

        assert isinstance(msg, VideoClassifications)
        pruned = _prune_video(msg, tile_size, sample_rate, tracker, accuracy_idx,
                              config.dataset, config.relevance_threshold)
        out_queue.put(pruned)


def _prune_video(
    msg: VideoClassifications,
    tile_size: int,
    sample_rate: int,
    tracker: str | None,
    accuracy_idx: int,
    dataset: str,
    relevance_threshold: float,
) -> VideoClassifications:
    """Apply polyomino grouping + ILP pruning to one video's classifications."""
    classifications = msg.classifications

    # Decode hex-encoded classification grids into binary numpy arrays.
    bitmaps_list: list[np.ndarray] = []
    frame_indices: list[int] = []
    grid_height: int | None = None
    grid_width: int | None = None

    # Binarize at T_r (same convention as p022 / stage_compress).
    cutoff = int(relevance_threshold * 255)
    for frame_data in classifications:
        frame_indices.append(frame_data['idx'])
        if grid_height is None or grid_width is None:
            grid_height, grid_width = frame_data['classification_size']
        hex_data = frame_data['classification_hex']
        flat = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
        grid = flat.reshape((grid_height, grid_width))
        binary = (grid >= cutoff).astype(np.uint8) * 255
        bitmaps_list.append(binary)

    num_frames = len(bitmaps_list)
    assert grid_height is not None and grid_width is not None
    bitmaps = np.stack(bitmaps_list, axis=0) // 255

    # Group tiles into polyominoes.
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(
        bitmaps.astype(np.uint8), TILEPADDING_MODE,
    )
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)

    # Load per-tile max sampling distance.
    assert tracker is not None
    max_rate_path = cache.index(
        dataset, 'track_rates', f'{tracker}_{tile_size}', 'max_rate_table.npy',
    )
    assert os.path.exists(max_rate_path), f"Max rate table not found: {max_rate_path}"
    max_rate_table = np.load(max_rate_path)
    max_sampling_distance = max_rate_table[:, :, accuracy_idx]
    assert max_sampling_distance.shape == (grid_height, grid_width)
    max_sampling_distance = max_sampling_distance // sample_rate
    max_sampling_distance = np.maximum(max_sampling_distance, 1)

    # Solve ILP to select minimum polyomino set.
    ilp_result = solve_ilp(
        tile_to_polyomino_id,
        polyomino_lengths,
        max_sampling_distance,
        grid_height,
        grid_width,
        time_limit_seconds=0.1,
    )
    selected = ilp_result.selected

    # Convert selected polyominoes back to binary grids.
    pruned_classifications: list[dict] = []
    for b in range(num_frames):
        selected_ids = {pid for (frame, pid) in selected if frame == b}
        tile_ids = tile_to_polyomino_id[b]
        mask = np.isin(tile_ids, list(selected_ids)) & (tile_ids >= 0)
        grid = (mask.astype(np.uint8) * 255)
        pruned_classifications.append({
            "classification_size": grid.shape,
            "classification_hex": grid.flatten().tobytes().hex(),
            "idx": frame_indices[b],
        })

    return VideoClassifications(
        video=msg.video,
        classifications=pruned_classifications,
        frame_buffer=msg.frame_buffer,
        width=msg.width,
        height=msg.height,
        frame_count=msg.frame_count,
        sampled_indices=msg.sampled_indices,
    )
