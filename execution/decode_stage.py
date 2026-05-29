"""Decoder thread: reads videos with OpenCV, writes RGB frames to shared memory.

Runs in the main process as a single thread.  Reads video filenames from
``video_queue``; for each video, allocates a CPU shared memory block sized
to the union of (sampled frames + their prev frames), fills it, then emits
``VideoStart`` / ``FrameBatch`` (BATCH_SIZE chunks) / ``VideoEnd`` messages
on ``out_queue``.

Sentinel ``None`` on ``video_queue`` propagates ``None`` to ``out_queue``.
A tuple ``(video, max_frames)`` triggers warmup decoding limited to
``max_frames`` source frames.
"""

from __future__ import annotations

import queue
import traceback

import cv2
import numpy as np

from polyis.io import store

from execution import shm as shm_mod
from execution.config import PipelineConfig
from execution.messages import (
    FrameBatch,
    PipelineError,
    ShmRef,
    VideoEnd,
    VideoStart,
)

BATCH_SIZE = 16


def decode_stage(
    *,
    video_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
    error_q,
):
    """Main loop of the decoder thread."""
    current_item = None
    try:
        while True:
            item = video_queue.get()
            current_item = item
            if item is None:
                # Propagate shutdown to the next stage and exit.
                out_queue.put(None)
                return

            # Either "video" or ("video", max_frames) for warmup decoding.
            if isinstance(item, tuple):
                video, max_frames = item
            else:
                video, max_frames = item, None

            _decode_one_video(video, max_frames, out_queue, config)
    except BaseException:
        # Any exception is forwarded to the error queue; main raises on receipt.
        video_name = None
        if isinstance(current_item, str):
            video_name = current_item
        elif isinstance(current_item, tuple):
            video_name = current_item[0]
        error_q.put(PipelineError(
            stage='decode',
            video=video_name,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _decode_one_video(
    video: str,
    max_frames: int | None,
    out_queue: queue.Queue,
    config: PipelineConfig,
) -> None:
    """Decode one video into a fresh shared memory block and emit messages."""
    video_path = store.dataset(config.dataset, config.videoset, video)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    # Total frame count; warmup paths cap this at max_frames.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sampled frame indices (matching scripts/p020 logic: every N-th + last).
    sampled_indices = [idx for idx in range(frame_count) if idx % config.sample_rate == 0]
    last_idx = frame_count - 1
    if last_idx >= 0 and (not sampled_indices or sampled_indices[-1] != last_idx):
        sampled_indices.append(last_idx)

    # We also need each sampled frame's "prev" frame for the diff channel in
    # classify.  Union sampled + prev gives the set of frames to keep.
    needed_indices: set[int] = set()
    prev_map: dict[int, int] = {}
    for idx in sampled_indices:
        needed_indices.add(idx)
        prev = idx - 1 if idx > 0 else idx + 1
        needed_indices.add(prev)
        prev_map[idx] = prev

    # Buffer stores frames in absolute-index sorted order; build the lookup.
    buffer_frame_indices = sorted(needed_indices)
    idx_to_buf_pos = {idx: pos for pos, idx in enumerate(buffer_frame_indices)}
    num_needed = len(buffer_frame_indices)

    # Allocate the CPU shared memory frame buffer for this video.
    shape = (num_needed, height, width, 3)
    shm, frames = shm_mod.alloc(shape=shape)

    # Read every frame sequentially; copy needed ones into the buffer.
    needed_set = set(buffer_frame_indices)
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_set:
            # BGR -> RGB and copy into the shared buffer slot for this index.
            np.copyto(frames[idx_to_buf_pos[frame_idx]], frame[:, :, ::-1])
    cap.release()

    # VideoStart carries the shared memory reference and per-video metadata.
    out_queue.put(VideoStart(
        video=video,
        frame_shm=ShmRef(name=shm.name, shape=shape),
        width=width,
        height=height,
        frame_count=frame_count,
        sampled_indices=sampled_indices,
        buffer_frame_indices=buffer_frame_indices,
    ))

    # Stream sampled frames in BATCH_SIZE chunks for classify.
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

    # End-of-video sentinel for downstream accumulation.
    out_queue.put(VideoEnd(video=video))
