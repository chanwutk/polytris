"""Tile relevance classification shared by scripts/p020_exec_classify.py and execution2."""

import time
from typing import cast

import numpy as np
import torch

from polyis.images import ImgNHWC, splitNHWC
from polyis.utilities import format_time


def load_classifier(model_path) -> torch.nn.Module:
    """Load a trained classifier checkpoint in eval mode and fp16, on CPU."""
    # Load to CPU first to avoid CUDA init errors in spawned subprocesses;
    # caller moves the model to the target GPU device afterward.
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    model.half()
    return model


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
) -> tuple[torch.Tensor, list[dict]]:
    """
    Process up to BATCH_SIZE frames in one batch: prepare tiles for each frame,
    run inference once on all valid tiles, scatter results per frame.
    Returns (list of relevance_grids, runtime dict).
    """
    batch_size = len(batch_frames)
    num_tiles = grid_height * grid_width

    # 1. Send individual frames to GPU then stack on GPU
    # Stack on GPU
    send_start = time.time_ns() / 1e6
    frames_gpu = [torch.from_numpy(f).to(device, non_blocking=True) for f in batch_frames]
    frames_tensor = torch.stack(frames_gpu, dim=0)

    prev_gpu = [torch.from_numpy(f).to(device, non_blocking=True) for f in batch_prev_frames]
    prev_frames = torch.stack(prev_gpu, dim=0)
    send_runtime = time.time_ns() / 1e6 - send_start

    # Resize images to (tile_size * grid_height, tile_size * grid_width) to ensure exact tile alignment
    resize_start = time.time_ns() / 1e6
    target_h = tile_size * grid_height
    target_w = tile_size * grid_width
    current_h, current_w = frames_tensor.shape[1:3]

    # Only resize if dimensions don't match
    if (current_h, current_w) != (target_h, target_w):
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor.permute(0, 3, 1, 2).half(),
            size=(target_h, target_w), mode='bilinear',
            align_corners=False
        ).to(torch.uint8).permute(0, 2, 3, 1)
        prev_frames = torch.nn.functional.interpolate(
            prev_frames.permute(0, 3, 1, 2).half(),
            size=(target_h, target_w), mode='bilinear',
            align_corners=False
        ).to(torch.uint8).permute(0, 2, 3, 1)
    resize_runtime = time.time_ns() / 1e6 - resize_start

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
    collect_runtime = time.time_ns() / 1e6 - collect_start

    runtime = format_time(inference=inference_runtime, collect=collect_runtime,
                          reshape=reshape_runtime, mask=mask_runtime,
                          diff=diff_runtime, send=send_runtime, resize=resize_runtime)
    return probabilities_per_frame, runtime
