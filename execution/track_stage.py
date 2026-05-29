"""Track thread: sequential tracker run per video.

Runs in the main process as a single thread.  Pulls ``VideoDetections``
messages, runs the configured tracker over all frames in absolute-index
order, and emits ``TrackingResult`` for each completed video.
"""

from __future__ import annotations

import queue
import time
import traceback

import numpy as np

from polyis.utilities import (
    create_tracker,
    get_video_resolution,
    register_tracked_detections,
)

from execution.config import PipelineConfig
from execution.messages import (
    PipelineError,
    StageTiming,
    TrackingResult,
    VideoDetections,
)


def track_stage(
    *,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
    error_q,
    timings_q,
):
    """Main loop of the track thread."""
    current_video: str | None = None
    try:
        while True:
            msg = in_queue.get()
            if msg is None:
                out_queue.put(None)
                return

            assert isinstance(msg, VideoDetections)
            current_video = msg.video
            start_ns = time.time_ns()
            frame_tracks = _track_one_video(msg, config)
            out_queue.put(TrackingResult(
                video=msg.video,
                frame_tracks=frame_tracks,
            ))
            if timings_q is not None:
                dur_ms = (time.time_ns() - start_ns) / 1e6
                timings_q.put(StageTiming(
                    stage='track',
                    video=msg.video,
                    duration_ms=dur_ms,
                ))
            current_video = None

    except BaseException:
        error_q.put(PipelineError(
            stage='track',
            video=current_video,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _track_one_video(
    msg: VideoDetections,
    config: PipelineConfig,
) -> dict[int, list[list[float]]]:
    """Run the tracker over one video's detections."""
    width, height = get_video_resolution(config.dataset, msg.video)
    tracker = create_tracker(config.tracker, img_size=(height, width))

    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}

    # Tracker is sequential within a video; iterate by absolute frame index.
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
            interpolate=not config.no_interpolate,
        )

    return frame_tracks
