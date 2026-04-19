"""
Stage 4: Tile compression / packing (p030).

Groups classified tiles into polyominoes (Cython, CPU), packs them into
collages (Cython, CPU), then renders the collage canvases using GPU tensor
operations instead of numpy.

The rendered canvas stays on GPU as a torch.Tensor and is sent directly
to the Detect stage, eliminating disk I/O for compressed images.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp

from polyis import dtypes
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.utilities import TILEPADDING_MAPS

from execution.pipeline import (
    CollageReady,
    PipelineConfig,
    VideoClassifications,
)

# Best-fit packing mode (matching the default in p030).
_PACK_MODE_BEST_FIT = 2


def compress_process(
    *,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Compress stage process entry-point.

    Reads VideoClassifications from *in_queue*, groups tiles, packs
    polyominoes into collages, renders canvases on GPU, and sends
    CollageReady messages to *out_queue*.

    After all collages for a video are sent, the frame_buffer reference
    is dropped so GPU memory can be reclaimed.

    ``None`` on *in_queue* propagates and the process exits.
    """
    device = f'cuda:{gpu_id}'
    tile_size = config.tile_size
    tilepadding = config.tilepadding
    canvas_scale = config.canvas_scale
    threshold = config.relevance_threshold

    while True:
        msg = in_queue.get()
        if msg is None:
            out_queue.put(None)
            return

        assert isinstance(msg, VideoClassifications)
        _compress_video(msg, device, tile_size, tilepadding, canvas_scale,
                        threshold, out_queue)


def _compress_video(
    msg: VideoClassifications,
    device: str,
    tile_size: int,
    tilepadding: str,
    canvas_scale: float,
    threshold: float,
    out_queue: mp.Queue,
):
    """Group, pack, and render all collages for one video."""
    classifications = msg.classifications
    frame_buffer = msg.frame_buffer
    width = msg.width
    height = msg.height
    sampled_indices = msg.sampled_indices

    # --- Source and destination grid dimensions ---
    src_grid_height = height // tile_size
    src_grid_width = width // tile_size
    dst_grid_height = max(1, int(round(src_grid_height * canvas_scale)))
    dst_grid_width = max(1, int(round(src_grid_width * canvas_scale)))
    canvas_height = dst_grid_height * tile_size
    canvas_width = dst_grid_width * tile_size

    # --- Step 1: Group tiles into polyominoes (CPU / Cython) ---
    # Build mapping from array index to absolute frame index.
    array_idx_to_frame_idx = {i: r['idx'] for i, r in enumerate(classifications)}

    polyominoes_stacks = np.empty(len(classifications), dtype=np.uint64)
    for array_idx, frame_result in enumerate(classifications):
        hex_data: str = frame_result['classification_hex']
        cls_size: tuple[int, int] = frame_result['classification_size']
        bitmap = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8).reshape(cls_size)
        bitmap = (bitmap > int(threshold * 255)).astype(np.uint8)
        assert dtypes.is_bitmap(bitmap), bitmap.shape
        polyominoes_stacks[array_idx] = group_tiles(bitmap, TILEPADDING_MAPS[tilepadding])

    # --- Step 2: Pack polyominoes into collages (CPU / Cython) ---
    collages_raw = pack(polyominoes_stacks, dst_grid_height, dst_grid_width, _PACK_MODE_BEST_FIT)

    # Remap frame indices from batch-relative to absolute.
    collages = []
    for collage in collages_raw:
        collages.append([
            (pos.oy, pos.ox, pos.py, pos.px,
             array_idx_to_frame_idx[pos.frame],
             pos.shape)
            for pos in collage
        ])

    total_collages = len(collages)

    # --- Handle the zero-collage case (all tiles pruned/classified away) ---
    # Send a sentinel so detect_uncompress doesn't deadlock waiting for collages.
    if total_collages == 0:
        out_queue.put(CollageReady(
            video=msg.video,
            collage_idx=0,
            total_collages=0,
            canvas=torch.empty(0, 0, 3, device=device, dtype=torch.uint8),
            index_map=np.empty((0, 0), dtype=np.uint16),
            offset_lookup=[],
            num_frames=msg.frame_count,
            tile_size=tile_size,
        ))
        del frame_buffer
        return

    # --- Build the frame lookup from the shared GPU buffer ---
    # sampled_indices maps buffer position -> absolute frame index.
    # We also need frames for non-sampled indices referenced by collages;
    # however, polyominoes only reference sampled frames (pack operates on
    # sampled frame indices).  Build a reverse map.
    frame_idx_to_buf_pos: dict[int, int] = {}
    # The buffer from the decoder contains *all needed frames* (sampled + prev).
    # sampled_indices only lists the sampled ones.  However, collages only
    # reference sampled frame indices (since polyominoes come from classification
    # results which are only for sampled frames).
    #
    # The decoder stores frames contiguously including prev-frames.  We need to
    # find the buffer position for each absolute frame index.  The decoder uses
    # sorted(needed_indices) where needed_indices = sampled ∪ prev(sampled).
    # We don't have that mapping here, so we reconstruct it.
    _needed: set[int] = set()
    for idx in sampled_indices:
        _needed.add(idx)
        prev = idx - 1 if idx > 0 else idx + 1
        _needed.add(prev)
    _sorted_needed = sorted(_needed)
    for buf_pos, abs_idx in enumerate(_sorted_needed):
        frame_idx_to_buf_pos[abs_idx] = buf_pos

    # --- Precompute grid boundary arrays (same logic as p030) ---
    src_rows = np.arange(src_grid_height, dtype=np.int32)
    src_cols = np.arange(src_grid_width, dtype=np.int32)
    src_y_starts = torch.from_numpy(src_rows * tile_size).to(device)
    src_y_ends = src_y_starts + tile_size
    src_x_starts = torch.from_numpy(src_cols * tile_size).to(device)
    src_x_ends = src_x_starts + tile_size

    dst_rows = np.arange(dst_grid_height, dtype=np.int32)
    dst_cols = np.arange(dst_grid_width, dtype=np.int32)
    dst_y_starts = torch.from_numpy(dst_rows * tile_size).to(device)
    dst_y_ends = dst_y_starts + tile_size
    dst_x_starts = torch.from_numpy(dst_cols * tile_size).to(device)
    dst_x_ends = dst_x_starts + tile_size

    # --- Step 3: Render each collage on GPU ---
    for collage_idx, collage in enumerate(collages):
        assert len(collage) > 0

        # Allocate GPU canvas.
        canvas = torch.zeros(
            canvas_height, canvas_width, 3, device=device, dtype=torch.uint8,
        )
        # Index map and offset lookup stay on CPU (small metadata).
        index_map = np.zeros((dst_grid_height, dst_grid_width), dtype=np.uint16)
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = []

        for gid, (oy, ox, py, px, frame_idx, shape) in enumerate(collage, start=1):
            # Get source frame from GPU buffer.
            buf_pos = frame_idx_to_buf_pos[frame_idx]
            frame_gpu = frame_buffer[buf_pos]   # [H, W, 3] on device

            i_coords = shape[:, 0].astype(np.int64)
            j_coords = shape[:, 1].astype(np.int64)

            # Copy each tile of this polyomino from source to canvas on GPU.
            for k in range(len(i_coords)):
                sy_s = int(src_y_starts[oy + i_coords[k]])
                sy_e = int(src_y_ends[oy + i_coords[k]])
                sx_s = int(src_x_starts[ox + j_coords[k]])
                sx_e = int(src_x_ends[ox + j_coords[k]])
                dy_s = int(dst_y_starts[py + i_coords[k]])
                dy_e = int(dst_y_ends[py + i_coords[k]])
                dx_s = int(dst_x_starts[px + j_coords[k]])
                dx_e = int(dst_x_ends[px + j_coords[k]])

                src_h = sy_e - sy_s
                src_w = sx_e - sx_s
                dst_h = dy_e - dy_s
                dst_w = dx_e - dx_s

                if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
                    continue

                copy_h = min(src_h, dst_h)
                copy_w = min(src_w, dst_w)

                # Direct GPU-to-GPU tile copy.
                canvas[dy_s:dy_s + copy_h, dx_s:dx_s + copy_w] = \
                    frame_gpu[sy_s:sy_s + copy_h, sx_s:sx_s + copy_w]

                # Edge-repeat for size mismatches.
                if dst_h > copy_h:
                    canvas[dy_s + copy_h:dy_e, dx_s:dx_s + copy_w] = \
                        frame_gpu[sy_s + copy_h - 1:sy_s + copy_h, sx_s:sx_s + copy_w]
                if dst_w > copy_w:
                    canvas[dy_s:dy_e, dx_s + copy_w:dx_e] = \
                        canvas[dy_s:dy_e, dx_s + copy_w - 1:dx_s + copy_w]

            # Update metadata.
            index_map[py + i_coords, px + j_coords] = gid
            offset_lookup.append(((int(py), int(px)), (int(oy), int(ox)), int(frame_idx)))

        # Send collage downstream.
        out_queue.put(CollageReady(
            video=msg.video,
            collage_idx=collage_idx,
            total_collages=total_collages,
            canvas=canvas,
            index_map=index_map,
            offset_lookup=offset_lookup,
            num_frames=msg.frame_count,
            tile_size=tile_size,
        ))

    # Drop frame buffer reference so GPU memory can be reclaimed.
    del frame_buffer
