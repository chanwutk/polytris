"""Per-stage test for ``execution.track_stage``.

Feeds a synthetic VideoDetections message with hand-built bboxes and asserts
that a TrackingResult comes back on the output queue with one frame_tracks
entry per frame.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading

import pytest

from execution.config import PipelineConfig
from execution.messages import TrackingResult, VideoDetections
from execution.track_stage import track_stage


def _build_config() -> PipelineConfig:
    """A minimal config; tracker name must match what create_tracker accepts."""
    return PipelineConfig(
        dataset='caldot2-y05',  # only used to resolve resolution
        videoset='valid',
        classifier='ShuffleNet05',
        tile_size=60,
        sample_rate=4,
        tilepadding='none',
        canvas_scale=1.0,
        tracker='sortcython',
        tracking_accuracy_threshold=None,
        relevance_threshold=0.5,
        classify_gpu=0,
        detect_gpu=0,
        prune_workers=1,
        compress_workers=1,
        max_videos_in_flight=1,
        no_interpolate=True,
        warmup=False,
    )


def test_track_stage_emits_tracking_result(tmp_path):
    """A VideoDetections in -> a TrackingResult out, with matching video name."""
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    error_q: mp.Queue = mp.Queue()
    timings_q: mp.Queue = mp.Queue()
    config = _build_config()

    # Synthetic detections: 3 frames, one bbox per frame moving rightward.
    # Format: [x1, y1, x2, y2, score].
    msg = VideoDetections(
        video='va00.mp4',
        frame_detections={
            0: [[100.0, 100.0, 150.0, 150.0, 0.95]],
            1: [[110.0, 100.0, 160.0, 150.0, 0.95]],
            2: [[120.0, 100.0, 170.0, 150.0, 0.95]],
        },
        num_frames=3,
    )

    t = threading.Thread(
        target=track_stage,
        kwargs=dict(
            in_queue=in_q, out_queue=out_q,
            config=config, error_q=error_q, timings_q=timings_q,
        ),
        daemon=True,
    )
    t.start()

    in_q.put(msg)
    in_q.put(None)

    result = out_q.get(timeout=20)
    assert isinstance(result, TrackingResult)
    assert result.video == 'va00.mp4'
    # Tracker should have produced at least one track entry across the frames.
    assert sum(len(v) for v in result.frame_tracks.values()) >= 1

    # Shutdown sentinel comes through.
    assert out_q.get(timeout=5) is None
    t.join(timeout=5)
    # No errors.
    assert error_q.empty()
