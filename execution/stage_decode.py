"""
Stage 1: Video frame decoder.

Reads video files with OpenCV, converts BGR -> RGB, transfers frames to a
shared GPU tensor buffer.  Sends FrameBatch messages to the Classify stage
in chunks of BATCH_SIZE so classification can start before the entire
video is decoded.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from polyis.io import store

from execution.pipeline import (
    FrameBatch,
    PipelineConfig,
    VideoEnd,
    VideoStart,
)

BATCH_SIZE = 16


def decoder_process(
    *,
    video_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Decode videos and stream frame batches to the Classify stage.

    For each video received on *video_queue*:
    1. Open the video with OpenCV and read frames.
    2. Apply sample_rate filtering (keep every Nth frame + last frame).
    3. Convert BGR -> RGB and transfer to a GPU tensor buffer.
    4. Send a VideoStart with the buffer reference, then FrameBatch
       messages for each batch of BATCH_SIZE frames, and finally VideoEnd.

    A ``None`` on *video_queue* signals shutdown: propagate ``None`` to
    *out_queue* and return.
    """
    device = f'cuda:{gpu_id}'
    sample_rate = config.sample_rate

    if config.preload:
        # Preload mode: decode ALL videos into GPU buffers first, then replay.
        _run_preload(video_queue, out_queue, device, config)
        return

    # Normal (streaming) mode.
    # Items on video_queue are either a video name (str) or a
    # (video_name, max_frames) tuple for warmup decoding with limited frames.
    while True:
        item = video_queue.get()
        if item is None:
            # No more videos; propagate shutdown downstream.
            out_queue.put(None)
            return

        if isinstance(item, tuple):
            video, max_frames = item
            _decode_one_video(video, out_queue, device, config,
                              max_frames=max_frames)
        else:
            _decode_one_video(item, out_queue, device, config)


def _decode_one_video(
    video: str,
    out_queue: mp.Queue,
    device: str,
    config: PipelineConfig,
    max_frames: int | None = None,
):
    """Decode a single video and stream frame batches to *out_queue*.

    When *max_frames* is set, only the first *max_frames* frames are
    considered (used for pipeline warmup).
    """
    video_path = store.dataset(config.dataset, config.videoset, video)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Limit frame count for warmup decoding.
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build the list of sampled frame indices (matching p020 logic).
    sampled_indices = [idx for idx in range(frame_count) if idx % config.sample_rate == 0]
    last_idx = frame_count - 1
    if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
        sampled_indices.append(last_idx)

    # Build a set of all frame indices we need to decode:
    # each sampled frame + its previous frame (for the diff channel in p020).
    needed_indices: set[int] = set()
    prev_map: dict[int, int] = {}   # sampled_idx -> prev_idx
    for idx in sampled_indices:
        needed_indices.add(idx)
        prev = idx - 1 if idx > 0 else idx + 1
        needed_indices.add(prev)
        prev_map[idx] = prev

    # Decode only the needed frames from the video.
    # Position-to-buffer mapping: needed frames are stored contiguously.
    sorted_needed = sorted(needed_indices)
    idx_to_buf_pos: dict[int, int] = {idx: pos for pos, idx in enumerate(sorted_needed)}
    num_needed = len(sorted_needed)

    # Allocate the GPU frame buffer for this video.
    frame_buffer = torch.zeros(
        num_needed, height, width, 3, device=device, dtype=torch.uint8,
    )

    # Read frames sequentially, only keeping needed ones.
    needed_set = set(sorted_needed)
    buf_pos = 0
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_set:
            # Convert BGR -> RGB and transfer to GPU.
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            frame_buffer[idx_to_buf_pos[frame_idx]] = torch.from_numpy(rgb).to(
                device, non_blocking=True,
            )
    cap.release()

    # Send VideoStart with buffer reference and metadata.
    out_queue.put(VideoStart(
        video=video,
        frame_buffer=frame_buffer,
        width=width,
        height=height,
        frame_count=frame_count,
        sampled_indices=sampled_indices,
    ))

    # Send FrameBatch messages in chunks.
    for batch_start in range(0, len(sampled_indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(sampled_indices))
        batch_sampled = sampled_indices[batch_start:batch_end]

        batch_positions = [idx_to_buf_pos[idx] for idx in batch_sampled]
        prev_positions = [idx_to_buf_pos[prev_map[idx]] for idx in batch_sampled]

        out_queue.put(FrameBatch(
            video=video,
            batch_positions=batch_positions,
            prev_positions=prev_positions,
        ))

    # Signal end of this video.
    out_queue.put(VideoEnd(video=video))


def _run_preload(
    video_queue: mp.Queue,
    out_queue: mp.Queue,
    device: str,
    config: PipelineConfig,
):
    """
    Preload mode: decode ALL videos into GPU memory first, then replay
    the messages so the timer only measures pipeline computation.
    """
    # Collect all video names.
    video_names: list[str] = []
    while True:
        video = video_queue.get()
        if video is None:
            break
        video_names.append(video)

    # Decode all videos and store their messages.
    all_messages: list[list] = []
    for video in video_names:
        messages: list = []
        # Decode into GPU buffer (reuse the same helper).
        video_path = store.dataset(config.dataset, config.videoset, video)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Could not open video {video_path}"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sampled_indices = [idx for idx in range(frame_count) if idx % config.sample_rate == 0]
        last_idx = frame_count - 1
        if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
            sampled_indices.append(last_idx)

        needed_indices: set[int] = set()
        prev_map: dict[int, int] = {}
        for idx in sampled_indices:
            needed_indices.add(idx)
            prev = idx - 1 if idx > 0 else idx + 1
            needed_indices.add(prev)
            prev_map[idx] = prev

        sorted_needed = sorted(needed_indices)
        idx_to_buf_pos = {idx: pos for pos, idx in enumerate(sorted_needed)}

        frame_buffer = torch.zeros(
            len(sorted_needed), height, width, 3, device=device, dtype=torch.uint8,
        )

        needed_set = set(sorted_needed)
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in needed_set:
                rgb = np.ascontiguousarray(frame[:, :, ::-1])
                frame_buffer[idx_to_buf_pos[frame_idx]] = torch.from_numpy(rgb).to(
                    device, non_blocking=True,
                )
        cap.release()

        # Build messages for this video.
        messages.append(VideoStart(
            video=video,
            frame_buffer=frame_buffer,
            width=width,
            height=height,
            frame_count=frame_count,
            sampled_indices=sampled_indices,
        ))
        for batch_start in range(0, len(sampled_indices), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(sampled_indices))
            batch_sampled = sampled_indices[batch_start:batch_end]
            batch_positions = [idx_to_buf_pos[idx] for idx in batch_sampled]
            prev_positions = [idx_to_buf_pos[prev_map[idx]] for idx in batch_sampled]
            messages.append(FrameBatch(
                video=video,
                batch_positions=batch_positions,
                prev_positions=prev_positions,
            ))
        messages.append(VideoEnd(video=video))
        all_messages.append(messages)

    # Synchronize GPU to ensure all transfers are complete.
    torch.cuda.synchronize()

    # Now replay all messages (this is when the timer starts in Classify).
    for messages in all_messages:
        for msg in messages:
            out_queue.put(msg)

    # Shutdown sentinel.
    out_queue.put(None)
