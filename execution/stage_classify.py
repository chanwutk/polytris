"""
Stage 2: Tile classification (p020).

Loads the classifier model once, then processes frame batches streamed
from the Decoder.  Frames are read directly from the shared GPU buffer
(no CPU round-trip).  When all frames for a video are classified, the
accumulated results are sent downstream as a VideoClassifications message.
"""

from __future__ import annotations

import os
from typing import cast

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from polyis.images import ImgNHWC, splitNHWC
from polyis.io import cache, store
from polyis.train.select_model_optimization import select_model_optimization

from execution.pipeline import (
    FrameBatch,
    PipelineConfig,
    VideoClassifications,
    VideoEnd,
    VideoStart,
)

# Reuse model loading from the original classify script.
from scripts.p020_exec_classify import load_model


def classify_process(
    *,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Classification stage process entry-point.

    Reads VideoStart / FrameBatch / VideoEnd messages from *in_queue*.
    Accumulates per-frame classification results and, when a VideoEnd
    arrives, sends a single VideoClassifications to *out_queue*.

    ``None`` on *in_queue* propagates to *out_queue* and the process exits.
    """
    device = f'cuda:{gpu_id}'
    tile_size = config.tile_size

    # ---- Load and optimise the classifier model once ----
    model = load_model(config.dataset, tile_size, config.classifier, device)
    model = model.to(device)

    # Load compilation benchmarks and select the best optimisation.
    bench_path = cache.index(
        config.dataset, 'training', 'results',
        f'{config.classifier}_{tile_size}', 'model_compilation.jsonl',
    )
    import json
    with open(bench_path, 'r') as f:
        benchmark_results = [json.loads(line) for line in f]

    # Use the first video's resolution to pick an optimisation method.
    first_video_path = store.dataset(
        config.dataset, config.videoset,
        sorted(os.listdir(store.dataset(config.dataset, config.videoset)))[0],
    )
    first_cap = cv2.VideoCapture(first_video_path)
    assert first_cap.isOpened()
    vid_w = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    model, _method = select_model_optimization(
        model, benchmark_results, device, tile_size,
        (vid_w // tile_size) * (vid_h // tile_size),
    )

    # Pre-create normalisation tensors (fp16).
    normalize_mean = torch.tensor(
        [0.485, 0.456, 0.406] * 2, device=device, dtype=torch.float16,
    ).view(1, 6, 1, 1)
    normalize_std = torch.tensor(
        [0.229, 0.224, 0.225] * 2, device=device, dtype=torch.float16,
    ).view(1, 6, 1, 1)

    # Load never-relevant bitmap.
    always_relevant_path = cache.index(
        config.dataset, 'never-relevant', f'{tile_size}_all.npy',
    )
    assert os.path.exists(always_relevant_path), \
        f"Always relevant bitmap not found for {config.dataset} {tile_size}"
    always_relevant_bitmap = np.load(always_relevant_path)
    always_relevant_mask = (
        torch.from_numpy(always_relevant_bitmap.flatten()).to(device).to(torch.uint8)
    )

    # ---- Per-video state ----
    current_video: str | None = None
    current_buffer: torch.Tensor | None = None
    current_width = 0
    current_height = 0
    current_frame_count = 0
    current_sampled_indices: list[int] = []
    accumulated_classifications: list[dict] = []
    # Grid and position tensors (recomputed per-video).
    grid_width = 0
    grid_height = 0
    positions: torch.Tensor | None = None
    with torch.no_grad():
        while True:
            msg = in_queue.get()

            # --- shutdown sentinel ---
            if msg is None:
                out_queue.put(None)
                return

            # --- VideoStart: new video arriving ---
            if isinstance(msg, VideoStart):
                current_video = msg.video
                current_buffer = msg.frame_buffer
                current_width = msg.width
                current_height = msg.height
                current_frame_count = msg.frame_count
                current_sampled_indices = list(msg.sampled_indices)
                accumulated_classifications = []

                # Compute grid and position tensors for this resolution.
                grid_width = current_width // tile_size
                grid_height = current_height // tile_size
                y_idx = torch.arange(grid_height, device=device, dtype=torch.uint8)
                x_idx = torch.arange(grid_width, device=device, dtype=torch.uint8)
                y_rep = y_idx.repeat_interleave(grid_width)
                x_rep = x_idx.repeat(grid_height)
                positions = torch.stack([y_rep, x_rep], dim=1).float()
                continue

            # --- VideoEnd: flush accumulated results downstream ---
            if isinstance(msg, VideoEnd):
                assert current_video is not None
                assert current_buffer is not None
                out_queue.put(VideoClassifications(
                    video=current_video,
                    classifications=accumulated_classifications,
                    frame_buffer=current_buffer,
                    width=current_width,
                    height=current_height,
                    frame_count=current_frame_count,
                    sampled_indices=current_sampled_indices,
                ))
                current_video = None
                current_buffer = None
                continue

            # --- FrameBatch: classify a batch of frames ---
            assert isinstance(msg, FrameBatch)

            assert current_buffer is not None
            assert positions is not None

            # Run classification on this batch.
            probs, _runtime = _classify_batch_gpu(
                grid_width, grid_height, positions,
                current_buffer, msg.batch_positions, msg.prev_positions,
                model, tile_size, device,
                normalize_mean, normalize_std, always_relevant_mask,
            )

            # Track how many frames we've accumulated so far, and use that
            # to index back into sampled_indices for absolute frame indices.
            batch_start_in_sampled = len(accumulated_classifications)
            for j, relevance_grid in enumerate(probs.cpu().numpy()):
                sampled_pos = batch_start_in_sampled + j
                absolute_idx = current_sampled_indices[sampled_pos]
                frame_entry = {
                    "classification_size": tuple(relevance_grid.shape),
                    "classification_hex": relevance_grid.flatten().tobytes().hex(),
                    "idx": absolute_idx,
                }
                accumulated_classifications.append(frame_entry)


def _classify_batch_gpu(
    grid_width: int,
    grid_height: int,
    positions: torch.Tensor,
    frame_buffer: torch.Tensor,
    batch_positions: list[int],
    prev_positions: list[int],
    model: torch.nn.Module,
    tile_size: int,
    device: str,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    always_relevant_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    GPU-native variant of classify_batch from p020.

    Instead of receiving ``list[np.ndarray]`` and transferring to GPU, this
    reads frames directly from the shared *frame_buffer* already on the GPU.
    The rest of the logic (resize, diff, tile split, mask, inference) is
    identical to ``scripts.p020_exec_classify.classify_batch``.
    """
    batch_size = len(batch_positions)
    num_tiles = grid_height * grid_width

    # 1. Gather frames from the shared GPU buffer (already on device).
    frames_tensor = frame_buffer[batch_positions]    # [B, H, W, 3]
    prev_frames = frame_buffer[prev_positions]       # [B, H, W, 3]

    # 2. Resize to exact tile-aligned dimensions if needed.
    target_h = tile_size * grid_height
    target_w = tile_size * grid_width
    current_h, current_w = frames_tensor.shape[1:3]

    if (current_h, current_w) != (target_h, target_w):
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor.permute(0, 3, 1, 2).half(),
            size=(target_h, target_w), mode='bilinear', align_corners=False,
        ).to(torch.uint8).permute(0, 2, 3, 1)
        prev_frames = torch.nn.functional.interpolate(
            prev_frames.permute(0, 3, 1, 2).half(),
            size=(target_h, target_w), mode='bilinear', align_corners=False,
        ).to(torch.uint8).permute(0, 2, 3, 1)

    # 3. Compute frame diff and concatenate to 6-channel input.
    diff = torch.abs(
        frames_tensor.to(torch.int16) - prev_frames.to(torch.int16),
    ).to(torch.uint8)
    frames_6ch = torch.cat([frames_tensor, diff], dim=-1)

    # 4. Split into tiles.
    tiles_nghwc = splitNHWC(cast(ImgNHWC, frames_6ch), tile_size, tile_size)
    tiles_flat = tiles_nghwc.reshape(batch_size * num_tiles, tile_size, tile_size, 6)

    # 5. Mask: skip black and never-relevant tiles.
    non_black_mask = (
        tiles_flat.reshape(batch_size * num_tiles, -1).any(dim=1).to(torch.bool)
    )
    always_relevant_expanded = (
        always_relevant_mask.to(torch.bool)
        .unsqueeze(0).expand(batch_size, -1).reshape(-1)
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

    # 6. Inference.
    all_tiles = tiles_nchw_valid.half()
    all_positions = all_positions.half()
    predictions = torch.sigmoid(model(all_tiles, all_positions))

    # 7. Scatter predictions back to full grid.
    predictions_uint8 = (predictions * 255).to(torch.uint8)
    probabilities_full = torch.zeros(
        batch_size * num_tiles, 1, device=device, dtype=torch.uint8,
    )
    probabilities_full[valid_flat_idx] = predictions_uint8
    probabilities_per_frame = probabilities_full.reshape(
        batch_size, grid_height, grid_width,
    )

    runtime: dict = {}   # runtime tracking omitted in pipeline mode
    return probabilities_per_frame, runtime


