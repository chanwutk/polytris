"""Prune worker process pool: polyomino grouping + ILP via Gurobi.

Each worker is its own ``mp.Process``.  Workers pull ``VideoClassifications``
from the shared input queue (one whole video per task), run
``group_tiles_all`` + ``solve_ilp``, and push the pruned
``VideoClassifications`` to the output queue.

Sentinel ``None`` on the input queue propagates ``None`` to the output queue.
The pool spawner ensures one ``None`` is forwarded to the output queue per
worker so downstream sees a single eventual ``None`` (we explicitly merge in
the spawner; see :func:`spawn_prune_pool`).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from typing import Optional

import numpy as np

from polyis.io import cache
from polyis.pack.adapters import group_tiles_all
from polyis.sample.ilp.c.gurobi import solve_ilp

from execution.config import PipelineConfig
from execution.messages import (
    PipelineError,
    StageTiming,
    VideoClassifications,
)


# Mapping from float accuracy threshold to the integer index used in the
# precomputed max_rate_table (matches scripts/p022).
_ALL_ACCURACY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
_ACCURACY_THRESHOLD_TO_IDX = {t: i for i, t in enumerate(_ALL_ACCURACY_THRESHOLDS)}

# Polyomino tile padding mode used in pruning (matches scripts/p022).
TILEPADDING_MODE = 0

# Solver time limit per video (seconds); matches scripts/p022 default.
_DEFAULT_TIME_LIMIT_S = 0.1


def prune_worker(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    config: PipelineConfig,
    error_q: mp.Queue,
    timings_q: Optional[mp.Queue],
):
    """Entry point for one prune worker process."""
    current_video: str | None = None
    try:
        assert config.tracking_accuracy_threshold is not None, \
            "Prune workers should not be spawned without a threshold"
        accuracy_idx = _ACCURACY_THRESHOLD_TO_IDX[config.tracking_accuracy_threshold]

        # Preload the max-rate table (whole-video constant); table is keyed
        # by tracker and tile size.
        max_rate_table = _load_max_rate_table(config)

        while True:
            msg = in_queue.get()
            if msg is None:
                # Propagate sentinel and exit.
                out_queue.put(None)
                return

            assert isinstance(msg, VideoClassifications)
            current_video = msg.video

            start_ns = time.time_ns()
            pruned = _prune_one_video(
                msg=msg,
                config=config,
                accuracy_idx=accuracy_idx,
                max_rate_table=max_rate_table,
            )
            out_queue.put(pruned)
            if timings_q is not None:
                timings_q.put(StageTiming(
                    stage='prune',
                    video=current_video,
                    duration_ms=(time.time_ns() - start_ns) / 1e6,
                ))
            current_video = None

    except BaseException:
        error_q.put(PipelineError(
            stage='prune',
            video=current_video,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _load_max_rate_table(config: PipelineConfig) -> np.ndarray:
    """Load the [H, W, num_thresholds] max-rate table for this (tracker, tile)."""
    max_rate_path = cache.index(
        config.dataset, 'track_rates',
        f'{config.tracker}_{config.tile_size}', 'max_rate_table.npy',
    )
    assert os.path.exists(max_rate_path), \
        f"Max rate table not found at {max_rate_path}"
    table = np.load(max_rate_path)
    assert table.ndim == 3, f"Expected 3D max_rate_table, got shape {table.shape}"
    return table


def _prune_one_video(
    msg: VideoClassifications,
    config: PipelineConfig,
    accuracy_idx: int,
    max_rate_table: np.ndarray,
) -> VideoClassifications:
    """Run group_tiles_all + ILP for one video; return new VideoClassifications."""
    # Decode hex-encoded classification grids into binary numpy arrays.
    bitmaps_list: list[np.ndarray] = []
    frame_indices: list[int] = []
    grid_height: int | None = None
    grid_width: int | None = None

    # Strict ``>`` comparison must match stage_compress's binarization so the
    # tiles pruned here are the same tiles compress sees as relevant.
    cutoff = config.relevance_threshold * 255
    for entry in msg.classifications:
        frame_indices.append(entry['idx'])
        if grid_height is None or grid_width is None:
            grid_height, grid_width = entry['classification_size']
        hex_data = entry['classification_hex']
        flat = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
        grid = flat.reshape((grid_height, grid_width))
        binary = (grid > cutoff).astype(np.uint8) * 255
        bitmaps_list.append(binary)

    num_frames = len(bitmaps_list)
    assert grid_height is not None and grid_width is not None
    bitmaps = np.stack(bitmaps_list, axis=0) // 255

    # Group tiles across all frames into polyominoes.
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(
        bitmaps.astype(np.uint8), TILEPADDING_MODE,
    )
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)

    # Slice the per-tile max sampling distance for this accuracy threshold.
    max_sampling_distance = max_rate_table[:, :, accuracy_idx]
    assert max_sampling_distance.shape == (grid_height, grid_width), \
        f"max_rate shape {max_sampling_distance.shape} != ({grid_height},{grid_width})"
    max_sampling_distance = max_sampling_distance // config.sample_rate
    max_sampling_distance = np.maximum(max_sampling_distance, 1)

    # Solve the ILP to select a minimum set of polyominoes covering temporal constraints.
    ilp_result = solve_ilp(
        tile_to_polyomino_id,
        polyomino_lengths,
        max_sampling_distance,
        grid_height,
        grid_width,
        time_limit_seconds=_DEFAULT_TIME_LIMIT_S,
    )
    selected = ilp_result.selected

    # Convert selected polyominoes back to binary grids for downstream compress.
    pruned: list[dict] = []
    for b in range(num_frames):
        selected_ids = {pid for (frame, pid) in selected if frame == b}
        tile_ids = tile_to_polyomino_id[b]
        mask = np.isin(tile_ids, list(selected_ids)) & (tile_ids >= 0)
        grid = (mask.astype(np.uint8) * 255)
        pruned.append({
            'classification_size': grid.shape,
            'classification_hex': grid.flatten().tobytes().hex(),
            'idx': frame_indices[b],
        })

    return VideoClassifications(
        video=msg.video,
        classifications=pruned,
        frame_shm=msg.frame_shm,
        width=msg.width,
        height=msg.height,
        frame_count=msg.frame_count,
        sampled_indices=msg.sampled_indices,
        buffer_frame_indices=msg.buffer_frame_indices,
    )


