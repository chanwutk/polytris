"""Compress worker process pool: group_tiles + pack + render canvases on CPU.

Each worker is its own ``mp.Process``.  Workers pull
``VideoClassifications`` from the shared input queue (one whole video per
task), attach to the frame buffer in shared memory, group tiles into
polyominoes, pack collages, render canvases on CPU, and emit one
``CollageReady`` per collage.  After emitting the last collage for a video,
a ``VideoCompressDone`` is sent on a dedicated channel back to the main
process so it can unlink the frame buffer and release the semaphore.

Sentinel ``None`` on the input queue propagates ``None`` to the output queue.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from typing import Optional

import numpy as np

from polyis import dtypes
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.pack.render import (
    precompute_grid_boundaries,
    render_collage_cpu,
)
from polyis.utilities import TILEPADDING_MAPS

from execution import shm as shm_mod
from execution.config import PipelineConfig
from execution.messages import (
    CollageReady,
    PipelineError,
    ShmRef,
    StageTiming,
    VideoClassifications,
    VideoCompressDone,
)


# Best-fit packing mode (matches the default in scripts/p030).
_PACK_MODE_BEST_FIT = 2


def compress_worker(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    config: PipelineConfig,
    compress_done_q: mp.Queue,
    error_q: mp.Queue,
    timings_q: Optional[mp.Queue],
):
    """Entry point for one compress worker process."""
    current_video: str | None = None
    try:
        while True:
            msg = in_queue.get()
            if msg is None:
                out_queue.put(None)
                return
            assert isinstance(msg, VideoClassifications)
            current_video = msg.video

            start_ns = time.time_ns()
            _compress_one_video(
                msg=msg,
                config=config,
                out_queue=out_queue,
                compress_done_q=compress_done_q,
            )
            if timings_q is not None:
                timings_q.put(StageTiming(
                    stage='compress',
                    video=current_video,
                    duration_ms=(time.time_ns() - start_ns) / 1e6,
                ))
            current_video = None

    except BaseException:
        error_q.put(PipelineError(
            stage='compress',
            video=current_video,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _compress_one_video(
    msg: VideoClassifications,
    config: PipelineConfig,
    out_queue: mp.Queue,
    compress_done_q: mp.Queue,
) -> None:
    """Process one video: group tiles, pack, render collages, emit messages."""
    # Source/destination grid dimensions and canvas pixel size.
    src_grid_height = msg.height // config.tile_size
    src_grid_width = msg.width // config.tile_size
    dst_grid_height = max(1, int(round(src_grid_height * config.canvas_scale)))
    dst_grid_width = max(1, int(round(src_grid_width * config.canvas_scale)))
    canvas_height = dst_grid_height * config.tile_size
    canvas_width = dst_grid_width * config.tile_size

    # Pre-decode classification grids into polyomino stacks for pack().
    array_idx_to_frame_idx = {i: r['idx'] for i, r in enumerate(msg.classifications)}
    polyominoes_stacks = np.empty(len(msg.classifications), dtype=np.uint64)
    cutoff = config.relevance_threshold * 255

    for array_idx, entry in enumerate(msg.classifications):
        hex_data: str = entry['classification_hex']
        cls_size: tuple[int, int] = entry['classification_size']
        bitmap = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8).reshape(cls_size)
        bitmap = (bitmap > cutoff).astype(np.uint8)
        assert dtypes.is_bitmap(bitmap), bitmap.shape
        polyominoes_stacks[array_idx] = group_tiles(bitmap, TILEPADDING_MAPS[config.tilepadding])

    # Pack polyominoes into collages.  Frame indices in the result are
    # array-relative; remap to absolute frame indices for downstream.
    raw_collages = pack(polyominoes_stacks, dst_grid_height, dst_grid_width, _PACK_MODE_BEST_FIT)
    collages: list[list[tuple]] = []
    for raw in raw_collages:
        collages.append([
            (pos.oy, pos.ox, pos.py, pos.px,
             array_idx_to_frame_idx[pos.frame], pos.shape)
            for pos in raw
        ])
    total_collages = len(collages)

    # Attach to the frame buffer in shared memory.  Close on exit from this
    # function; the main process owns unlinking.
    frame_shm_handle, frame_view = shm_mod.attach(
        msg.frame_shm.name, msg.frame_shm.shape,
    )
    try:
        # Buffer position lookup: absolute frame index -> buffer slot.
        idx_to_buf_pos = {
            abs_idx: pos for pos, abs_idx in enumerate(msg.buffer_frame_indices)
        }

        # If the entire video had no relevant tiles after pruning, emit a
        # single sentinel CollageReady so downstream can still close out
        # the video lifecycle without deadlocking.
        if total_collages == 0:
            empty_shm, _ = shm_mod.alloc_handoff(shape=(0, 0, 3))
            try:
                out_queue.put(CollageReady(
                    video=msg.video,
                    collage_idx=0,
                    total_collages=0,
                    is_last=True,
                    canvas_shm=ShmRef(name=empty_shm.name, shape=(0, 0, 3)),
                    index_map=np.empty((0, 0), dtype=np.uint16),
                    offset_lookup=[],
                    num_frames=msg.frame_count,
                    tile_size=config.tile_size,
                ))
            finally:
                empty_shm.close()
            compress_done_q.put(VideoCompressDone(
                video=msg.video,
                frame_shm_name=msg.frame_shm.name,
            ))
            return

        # Precompute tile-pixel boundary arrays once per video.
        src_y_starts, src_y_ends, src_x_starts, src_x_ends = \
            precompute_grid_boundaries(src_grid_height, src_grid_width, config.tile_size)
        dst_y_starts, dst_y_ends, dst_x_starts, dst_x_ends = \
            precompute_grid_boundaries(dst_grid_height, dst_grid_width, config.tile_size)

        # Frame fetch closure: returns the buffer slot for an absolute frame index.
        def fetch_frame(abs_frame_idx: int) -> np.ndarray:
            return frame_view[idx_to_buf_pos[abs_frame_idx]]

        # Render each collage; emit a CollageReady per collage.
        canvas_shape = (canvas_height, canvas_width, 3)
        for collage_idx, collage in enumerate(collages):
            # Allocate canvas in a separate shared memory block.  Ownership
            # is transferred to detect, which unlinks after consumption.
            canvas_shm, canvas_view = shm_mod.alloc_handoff(shape=canvas_shape)
            try:
                # Zero canvas (alloc_handoff doesn't guarantee zero-fill on every OS).
                canvas_view[:] = 0

                index_map = np.zeros((dst_grid_height, dst_grid_width), dtype=np.uint16)
                offset_lookup: list = []

                render_collage_cpu(
                    canvas=canvas_view,
                    index_map=index_map,
                    offset_lookup=offset_lookup,
                    collage=collage,
                    fetch_frame=fetch_frame,
                    src_y_starts=src_y_starts, src_y_ends=src_y_ends,
                    src_x_starts=src_x_starts, src_x_ends=src_x_ends,
                    dst_y_starts=dst_y_starts, dst_y_ends=dst_y_ends,
                    dst_x_starts=dst_x_starts, dst_x_ends=dst_x_ends,
                )

                is_last = (collage_idx == total_collages - 1)
                out_queue.put(CollageReady(
                    video=msg.video,
                    collage_idx=collage_idx,
                    total_collages=total_collages,
                    is_last=is_last,
                    canvas_shm=ShmRef(name=canvas_shm.name, shape=canvas_shape),
                    index_map=index_map,
                    offset_lookup=offset_lookup,
                    num_frames=msg.frame_count,
                    tile_size=config.tile_size,
                ))
            finally:
                # Drop the local numpy reference (so close doesn't fail), then
                # release the worker's mapping.  Block stays alive on tmpfs.
                canvas_view = None
                canvas_shm.close()
    finally:
        frame_shm_handle.close()

    # Signal main process that this video's frame buffer can be unlinked.
    compress_done_q.put(VideoCompressDone(
        video=msg.video,
        frame_shm_name=msg.frame_shm.name,
    ))
