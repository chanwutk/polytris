"""Detect thread: batched object detection + coordinate uncompression.

Runs in the main process as a single thread bound to ``config.detect_gpu``.
Pulls ``CollageReady`` messages from any compress worker, batches up to
``BATCH_SIZE`` canvases, copies them to GPU, runs the detector, then unpacks
detections back to original-frame coordinates (p050 logic).  Per-video
state accumulates until all expected collages for a video have been
processed; at that point a ``VideoDetections`` message is emitted.

Canvas shared memory is unlinked by this stage after copying to GPU.
"""

from __future__ import annotations

import queue
import time
import traceback
from typing import Optional

import numpy as np
import torch

import polyis.dtypes
import polyis.models.detector
from scripts.p050_exec_uncompress import unpack_detections

from execution import shm as shm_mod
from execution.config import PipelineConfig
from execution.messages import (
    CollageReady,
    PipelineError,
    StageTiming,
    VideoDetections,
)

BATCH_SIZE = 4


def detect_stage(
    *,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
    error_q,
    timings_q,
):
    """Main loop of the detect thread."""
    current_video: str | None = None
    try:
        device = f'cuda:{config.detect_gpu}'
        torch.cuda.set_device(config.detect_gpu)

        # Initialize the detector once.  num_images=100 is a representative
        # warmup hint; the actual batch is at most BATCH_SIZE per call.
        detector = polyis.models.detector.get_detector(
            config.dataset, config.detect_gpu, BATCH_SIZE, num_images=100,
        )

        # Per-video accumulator state.
        video_detections: dict[str, dict[int, list[list[float]]]] = {}
        video_received: dict[str, int] = {}
        video_expected: dict[str, int] = {}
        video_num_frames: dict[str, int] = {}
        video_start_ns: dict[str, int] = {}

        # Pending batch of (msg, bgr_canvas) tuples awaiting detect.
        pending_msgs: list[CollageReady] = []
        pending_bgrs: list[np.ndarray] = []

        with torch.no_grad(), torch.inference_mode():
            while True:
                msg = in_queue.get()
                if msg is None:
                    # Flush any pending canvases before exiting.
                    if pending_msgs:
                        _flush_batch(
                            pending_msgs, pending_bgrs, detector,
                            video_detections, video_received,
                            video_expected, video_num_frames,
                            video_start_ns, out_queue, timings_q,
                        )
                    out_queue.put(None)
                    return

                assert isinstance(msg, CollageReady)
                current_video = msg.video

                # Initialise per-video state on first arrival.
                if msg.video not in video_detections:
                    video_detections[msg.video] = {i: [] for i in range(msg.num_frames)}
                    video_received[msg.video] = 0
                    video_expected[msg.video] = msg.total_collages
                    video_num_frames[msg.video] = msg.num_frames
                    video_start_ns[msg.video] = time.time_ns()

                # Zero-collage sentinel: skip directly to VideoDetections.
                if msg.total_collages == 0:
                    out_queue.put(VideoDetections(
                        video=msg.video,
                        frame_detections=video_detections.pop(msg.video),
                        num_frames=video_num_frames.pop(msg.video),
                    ))
                    video_received.pop(msg.video, None)
                    video_expected.pop(msg.video, None)
                    if timings_q is not None:
                        dur_ms = (time.time_ns() - video_start_ns.pop(msg.video)) / 1e6
                        timings_q.put(StageTiming(
                            stage='detect',
                            video=msg.video,
                            duration_ms=dur_ms,
                        ))
                    # The sentinel canvas (shape (0,0,3)) still has a shm to unlink.
                    shm_mod.unlink(msg.canvas_shm.name)
                    continue

                # Attach to canvas, copy out as BGR (detector expects BGR), unlink.
                with shm_mod.attached_view(msg.canvas_shm.name, msg.canvas_shm.shape) as canvas_rgb:
                    canvas_bgr = np.ascontiguousarray(canvas_rgb[:, :, ::-1])
                assert polyis.dtypes.is_np_image(canvas_bgr)
                shm_mod.unlink(msg.canvas_shm.name)

                pending_msgs.append(msg)
                pending_bgrs.append(canvas_bgr)

                # Decide whether to flush: batch full, or this collage completes
                # *some* video (so we don't keep that video waiting for the rest).
                last_video_in_batch = msg.video
                this_video_pending = sum(1 for m in pending_msgs if m.video == last_video_in_batch)
                will_complete_video = (
                    video_received[last_video_in_batch] + this_video_pending
                    >= video_expected[last_video_in_batch]
                )
                if len(pending_msgs) >= BATCH_SIZE or will_complete_video:
                    _flush_batch(
                        pending_msgs, pending_bgrs, detector,
                        video_detections, video_received,
                        video_expected, video_num_frames,
                        video_start_ns, out_queue, timings_q,
                    )
                    pending_msgs = []
                    pending_bgrs = []

    except BaseException:
        error_q.put(PipelineError(
            stage='detect',
            video=current_video,
            traceback=traceback.format_exc(),
        ))
        out_queue.put(None)
        return


def _flush_batch(
    msgs: list[CollageReady],
    bgrs: list[np.ndarray],
    detector,
    video_detections: dict[str, dict[int, list[list[float]]]],
    video_received: dict[str, int],
    video_expected: dict[str, int],
    video_num_frames: dict[str, int],
    video_start_ns: dict[str, int],
    out_queue: queue.Queue,
    timings_q,
):
    """Run detector on the pending batch, unpack results, emit completed videos."""
    # Single batched detector call across all pending canvases.
    batch_output = polyis.models.detector.detect_batch(bgrs, detector)

    # Per-collage: unpack detections via p050 logic and accumulate into per-video state.
    for i, msg in enumerate(msgs):
        detections = batch_output[i].tolist()
        frame_dets, _not_in_tile, _center_not = unpack_detections(
            detections, msg.index_map, msg.offset_lookup, msg.tile_size,
        )
        for frame_idx, bboxes in frame_dets.items():
            video_detections[msg.video][frame_idx].extend(bboxes)
        video_received[msg.video] += 1

        # Emit VideoDetections when all expected collages for this video arrived.
        if video_received[msg.video] >= video_expected[msg.video]:
            out_queue.put(VideoDetections(
                video=msg.video,
                frame_detections=video_detections.pop(msg.video),
                num_frames=video_num_frames.pop(msg.video),
            ))
            video_received.pop(msg.video, None)
            video_expected.pop(msg.video, None)
            if timings_q is not None:
                dur_ms = (time.time_ns() - video_start_ns.pop(msg.video)) / 1e6
                timings_q.put(StageTiming(
                    stage='detect',
                    video=msg.video,
                    duration_ms=dur_ms,
                ))
