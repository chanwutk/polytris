"""Classify thread: GPU tile classifier on frame batches from the decoder.

Runs in the main process as a single thread bound to ``config.classify_gpu``.
Reuses ``load_model`` and ``classify_batch`` from ``scripts/p020`` so the
inference body is identical to the sequential path.

Per-video state is accumulated as ``FrameBatch`` messages arrive; on
``VideoEnd`` the accumulated classifications are sent downstream as a single
``VideoClassifications`` message.
"""

from __future__ import annotations

import json
import os
import queue
import time
import traceback
from typing import cast

import cv2
import numpy as np
import torch

from polyis.images import ImgNHWC
from polyis.io import cache, store
from polyis.train.select_model_optimization import select_model_optimization

# Reuse the canonical model loader + inference body from the script.
from scripts.p020_exec_classify import classify_batch as _classify_batch
from scripts.p020_exec_classify import load_model

from execution import shm as shm_mod
from execution.config import PipelineConfig
from execution.messages import (
    FrameBatch,
    PipelineError,
    ShmRef,
    StageTiming,
    VideoClassifications,
    VideoEnd,
    VideoStart,
)


def classify_stage(
    *,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
    error_q,
    timings_q,
):
    """Main loop of the classify thread."""
    current_video: str | None = None
    try:
        # Pin this thread to the classify GPU; the device is thread-local.
        device = f'cuda:{config.classify_gpu}'
        torch.cuda.set_device(config.classify_gpu)

        # Load the trained classifier and apply the optimization that the
        # training step benchmarked as fastest for this tile size + grid size.
        model = _load_and_optimize_model(config, device)

        # ImageNet 6-channel normalization (RGB + diff stacked) on the device.
        normalize_mean = torch.tensor(
            [0.485, 0.456, 0.406] * 2, device=device, dtype=torch.float16,
        ).view(1, 6, 1, 1)
        normalize_std = torch.tensor(
            [0.229, 0.224, 0.225] * 2, device=device, dtype=torch.float16,
        ).view(1, 6, 1, 1)

        # Tiles that are never relevant for any frame in this dataset are
        # filtered out of inference (an upstream offline computation).
        always_relevant_path = cache.index(
            config.dataset, 'never-relevant', f'{config.tile_size}_all.npy',
        )
        assert os.path.exists(always_relevant_path), \
            f"Always relevant bitmap not found at {always_relevant_path}"
        always_relevant_mask = (
            torch.from_numpy(np.load(always_relevant_path).flatten())
            .to(device).to(torch.uint8)
        )

        # Per-video accumulator state, reset on each VideoStart.
        current_frame_view: np.ndarray | None = None
        current_frame_shm = None
        current_frame_shm_ref: ShmRef | None = None
        current_width = 0
        current_height = 0
        current_frame_count = 0
        current_sampled_indices: list[int] = []
        current_buffer_indices: list[int] = []
        accumulated: list[dict] = []
        positions: torch.Tensor | None = None
        grid_width = 0
        grid_height = 0
        video_start_time_ns = 0

        with torch.no_grad():
            while True:
                msg = in_queue.get()
                if msg is None:
                    out_queue.put(None)
                    return

                if isinstance(msg, VideoStart):
                    current_video = msg.video
                    video_start_time_ns = time.time_ns()
                    # Attach to the shared frame buffer.  Mapping is closed on
                    # VideoEnd so we don't hold mmap handles indefinitely.
                    current_frame_shm_ref = msg.frame_shm
                    current_frame_shm, current_frame_view = shm_mod.attach(
                        msg.frame_shm.name, msg.frame_shm.shape,
                    )
                    current_width = msg.width
                    current_height = msg.height
                    current_frame_count = msg.frame_count
                    current_sampled_indices = list(msg.sampled_indices)
                    current_buffer_indices = list(msg.buffer_frame_indices)
                    accumulated = []

                    # Build position tensors for the tile grid of this video.
                    grid_width = current_width // config.tile_size
                    grid_height = current_height // config.tile_size
                    y_idx = torch.arange(grid_height, device=device, dtype=torch.uint8)
                    x_idx = torch.arange(grid_width, device=device, dtype=torch.uint8)
                    y_rep = y_idx.repeat_interleave(grid_width)
                    x_rep = x_idx.repeat(grid_height)
                    positions = torch.stack([y_rep, x_rep], dim=1).float()
                    continue

                if isinstance(msg, VideoEnd):
                    assert current_video is not None
                    assert current_frame_shm_ref is not None
                    out_queue.put(VideoClassifications(
                        video=current_video,
                        classifications=accumulated,
                        frame_shm=current_frame_shm_ref,
                        width=current_width,
                        height=current_height,
                        frame_count=current_frame_count,
                        sampled_indices=current_sampled_indices,
                        buffer_frame_indices=current_buffer_indices,
                    ))
                    # Release the mmap handle in this thread; main process
                    # still holds the original shm + will unlink later.
                    current_frame_shm.close()
                    current_frame_view = None
                    current_frame_shm = None
                    current_frame_shm_ref = None

                    if timings_q is not None:
                        dur_ms = (time.time_ns() - video_start_time_ns) / 1e6
                        timings_q.put(StageTiming(
                            stage='classify',
                            video=current_video,
                            duration_ms=dur_ms,
                        ))
                    current_video = None
                    continue

                assert isinstance(msg, FrameBatch)
                assert current_frame_view is not None
                assert positions is not None

                # Build list-of-array views into the shared memory buffer (no
                # copy) for the batched classifier wrapper.
                batch_frames = [current_frame_view[p] for p in msg.batch_positions]
                batch_prev_frames = [current_frame_view[p] for p in msg.prev_positions]

                probs, _runtime = _classify_batch(
                    grid_width=grid_width,
                    grid_height=grid_height,
                    positions=positions,
                    batch_frames=batch_frames,
                    batch_prev_frames=batch_prev_frames,
                    model=model,
                    tile_size=config.tile_size,
                    device=device,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                    always_relevant_mask=always_relevant_mask,
                )

                # Sampled positions correspond to absolute frame indices via
                # the running cursor into sampled_indices.
                start_sampled = len(accumulated)
                probs_np = probs.cpu().numpy()
                for j, grid in enumerate(probs_np):
                    sampled_pos = start_sampled + j
                    accumulated.append({
                        'classification_size': tuple(grid.shape),
                        'classification_hex': grid.flatten().tobytes().hex(),
                        'idx': current_sampled_indices[sampled_pos],
                    })

    except BaseException:
        error_q.put(PipelineError(
            stage='classify',
            video=current_video,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _load_and_optimize_model(config: PipelineConfig, device: str) -> torch.nn.Module:
    """Load the classifier model and apply the benchmarked optimization."""
    # Base model from disk.
    model = load_model(config.dataset, config.tile_size, config.classifier, device)
    model = model.to(device)

    # Load the precomputed compilation benchmarks for this classifier+tile.
    bench_path = cache.index(
        config.dataset, 'training', 'results',
        f'{config.classifier}_{config.tile_size}', 'model_compilation.jsonl',
    )
    with open(bench_path, 'r') as f:
        benchmark_results = [json.loads(line) for line in f]

    # Probe the first video to determine the grid size used to choose an
    # optimization.  This matches the existing scripts/p020 logic.
    videoset_dir = store.dataset(config.dataset, config.videoset)
    first_video = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )[0]
    cap = cv2.VideoCapture(store.dataset(config.dataset, config.videoset, first_video))
    assert cap.isOpened()
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model, _method = select_model_optimization(
        model, benchmark_results, device, config.tile_size,
        (vid_w // config.tile_size) * (vid_h // config.tile_size),
    )
    return model


