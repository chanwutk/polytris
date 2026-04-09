"""
Stage 6: Object tracking (p060).

Receives accumulated per-video detections from Detect+Uncompress and
runs sequential tracking (SORT / ByteTrack / OC-SORT).  Results are
held in memory and sent back to the pipeline orchestrator.
"""

from __future__ import annotations

import numpy as np
import torch.multiprocessing as mp

from polyis.utilities import (
    create_tracker,
    get_video_resolution,
    register_tracked_detections,
)

from execution.pipeline import (
    PipelineConfig,
    TrackingResult,
    VideoDetections,
)


def track_process(
    *,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Track stage process entry-point.

    Reads VideoDetections from *in_queue*, runs the tracker sequentially
    over all frames, and sends TrackingResult to *out_queue*.

    ``None`` on *in_queue* propagates and the process exits.
    """
    # The tracker is always specified in the config (main.py iterates over
    # all trackers even when pruning is disabled).
    tracker_name = config.tracker
    assert tracker_name is not None, "tracker must be set in PipelineConfig"

    no_interpolate = config.no_interpolate

    while True:
        msg = in_queue.get()
        if msg is None:
            out_queue.put(None)
            return

        assert isinstance(msg, VideoDetections)
        frame_tracks = _track_video(
            msg, tracker_name, config.dataset, no_interpolate,
        )
        out_queue.put(TrackingResult(
            video=msg.video,
            frame_tracks=frame_tracks,
        ))


def _track_video(
    msg: VideoDetections,
    tracker_name: str,
    dataset: str,
    no_interpolate: bool,
) -> dict[int, list[list[float]]]:
    """Run the tracker over one video's detections."""
    resolution = get_video_resolution(dataset, msg.video)
    width, height = resolution
    tracker = create_tracker(tracker_name, img_size=(height, width))

    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    # Process frames in order.
    sorted_frame_indices = sorted(msg.frame_detections.keys())
    for frame_idx in sorted_frame_indices:
        bboxes = msg.frame_detections[frame_idx]
        dets = np.array(bboxes)
        if dets.size > 0:
            dets = dets[:, :5]  # x1, y1, x2, y2, score
        else:
            dets = np.empty((0, 5))

        tracked_dets = tracker.update(dets)
        register_tracked_detections(
            tracked_dets, frame_idx, frame_tracks, trajectories,
            interpolate=not no_interpolate,
        )

    return frame_tracks
