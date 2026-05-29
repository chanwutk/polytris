"""NamedTuple message types for queues between pipeline stages.

Messages carry small Python objects only; large data (frames, canvases) lives
in shared memory referenced by name via :class:`ShmRef`.  Sentinel ``None``
on any queue signals shutdown to that stage.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ShmRef(NamedTuple):
    """Reference to a shared memory block (uint8) by name + shape."""
    name: str
    shape: tuple[int, ...]


class VideoStart(NamedTuple):
    """Decoder -> Classify, beginning of a new video."""
    video: str
    frame_shm: ShmRef
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]
    buffer_frame_indices: list[int]


class FrameBatch(NamedTuple):
    """Decoder -> Classify, one batch of frames ready for inference."""
    video: str
    batch_positions: list[int]
    prev_positions: list[int]


class VideoEnd(NamedTuple):
    """Decoder -> Classify, end of stream for a video."""
    video: str


class VideoClassifications(NamedTuple):
    """Classify -> Prune (or directly Compress when prune is bypassed).

    Carries all per-frame classification grids plus the frame buffer reference
    so Compress workers can attach to the same frames.
    """
    video: str
    classifications: list[dict]
    frame_shm: ShmRef
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]
    buffer_frame_indices: list[int]


class CollageReady(NamedTuple):
    """Compress -> Detect, one packed collage ready for detection."""
    video: str
    collage_idx: int
    total_collages: int
    is_last: bool
    canvas_shm: ShmRef
    index_map: np.ndarray
    offset_lookup: list
    num_frames: int
    tile_size: int


class VideoCompressDone(NamedTuple):
    """Compress -> Main, signal that this worker emitted the last collage.

    Main process unlinks the frame shared memory and releases the
    max-videos-in-flight semaphore on receipt.
    """
    video: str
    frame_shm_name: str


class VideoDetections(NamedTuple):
    """Detect -> Track, accumulated per-frame detections for one video."""
    video: str
    frame_detections: dict[int, list[list[float]]]
    num_frames: int


class TrackingResult(NamedTuple):
    """Track -> Main, final tracker output for one video."""
    video: str
    frame_tracks: dict[int, list[list[float]]]


class PipelineError(NamedTuple):
    """Any worker -> error_q on unhandled exception."""
    stage: str
    video: str | None
    traceback: str


class StageTiming(NamedTuple):
    """Per-task duration sample emitted by any stage worker for aggregation."""
    stage: str
    video: str
    duration_ms: float
