"""
Stage 5: Object detection + coordinate uncompression (p040 + p050 merged).

Receives GPU canvas tensors from Compress, runs the detector (batched),
remaps detections to original frame coordinates, and accumulates per-video
results before sending to Track.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp

import polyis.dtypes
import polyis.models.detector
from scripts.p050_exec_uncompress import unpack_detections

from execution.pipeline import (
    CollageReady,
    PipelineConfig,
    VideoDetections,
)


def detect_uncompress_process(
    *,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    gpu_id: int,
    config: PipelineConfig,
):
    """
    Detect+Uncompress stage entry-point.

    Reads CollageReady messages from *in_queue*.  For each collage:
    1. Transfer canvas to CPU as numpy BGR.
    2. Run detector (batched up to *batch_size* collages per call).
    3. Remap detections via unpack_detections (p050).

    When all collages for a video are processed, sends VideoDetections
    to *out_queue*.

    ``None`` on *in_queue* propagates and the process exits.
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    # ---- Initialise detector ----
    batch_size = 4
    detector = polyis.models.detector.get_detector(
        config.dataset, gpu_id, batch_size, num_images=100,
    )

    # ---- Per-video state ----
    # {video: {frame_idx: [[x1,y1,x2,y2,...], ...]}}
    video_detections: dict[str, dict[int, list[list[float]]]] = {}
    video_num_frames: dict[str, int] = {}
    video_expected_collages: dict[str, int] = {}
    video_received_collages: dict[str, int] = {}

    # ---- Pending batch of collages for batched detection ----
    pending_msgs: list[CollageReady] = []
    pending_bgrs: list[np.ndarray] = []

    with torch.no_grad(), torch.inference_mode():
        while True:
            msg = in_queue.get()

            # --- shutdown sentinel ---
            if msg is None:
                # Flush any remaining pending collages before exiting.
                if pending_msgs:
                    _flush_batch(pending_msgs, pending_bgrs, detector,
                                 video_detections, video_received_collages,
                                 video_expected_collages, video_num_frames,
                                 out_queue)
                out_queue.put(None)
                return

            assert isinstance(msg, CollageReady)
            video = msg.video

            # Initialise tracking for a new video.
            if video not in video_detections:
                video_detections[video] = {i: [] for i in range(msg.num_frames)}
                video_num_frames[video] = msg.num_frames
                video_expected_collages[video] = msg.total_collages
                video_received_collages[video] = 0

            # --- Handle zero-collage videos (sentinel from compress) ---
            if msg.total_collages == 0:
                out_queue.put(VideoDetections(
                    video=video,
                    frame_detections=video_detections.pop(video),
                    num_frames=video_num_frames.pop(video),
                ))
                del video_expected_collages[video]
                del video_received_collages[video]
                continue

            # Transfer canvas from GPU to CPU numpy BGR for the detector.
            canvas_np = msg.canvas.cpu().numpy()
            canvas_bgr = np.ascontiguousarray(canvas_np[:, :, ::-1])
            assert polyis.dtypes.is_np_image(canvas_bgr)

            # Add to pending batch.
            pending_msgs.append(msg)
            pending_bgrs.append(canvas_bgr)

            # Flush when batch is full or when we've received the last collage
            # for the current video.
            is_last_for_video = (
                video_received_collages[video] + len(
                    [m for m in pending_msgs if m.video == video]
                ) >= video_expected_collages[video]
            )
            if len(pending_msgs) >= batch_size or is_last_for_video:
                _flush_batch(pending_msgs, pending_bgrs, detector,
                             video_detections, video_received_collages,
                             video_expected_collages, video_num_frames,
                             out_queue)
                pending_msgs = []
                pending_bgrs = []


def _flush_batch(
    msgs: list[CollageReady],
    bgrs: list[np.ndarray],
    detector,
    video_detections: dict[str, dict[int, list[list[float]]]],
    video_received_collages: dict[str, int],
    video_expected_collages: dict[str, int],
    video_num_frames: dict[str, int],
    out_queue: mp.Queue,
):
    """Run batched detection and unpack results for all pending collages."""
    # Run detection on the full batch.
    batch_output = polyis.models.detector.detect_batch(bgrs, detector)

    # Process each collage's detections.
    for i, msg in enumerate(msgs):
        video = msg.video
        detections = batch_output[i].tolist()

        # Unpack detections (p050 logic): remap compressed coords to original.
        frame_dets, _not_in_tile, _center_not = unpack_detections(
            detections, msg.index_map, msg.offset_lookup, msg.tile_size,
        )

        # Merge into accumulated per-frame detections.
        for frame_idx, bboxes in frame_dets.items():
            video_detections[video][frame_idx].extend(bboxes)

        video_received_collages[video] += 1

        # Emit VideoDetections when all collages for a video are processed.
        if video_received_collages[video] >= video_expected_collages[video]:
            out_queue.put(VideoDetections(
                video=video,
                frame_detections=video_detections.pop(video),
                num_frames=video_num_frames.pop(video),
            ))
            del video_expected_collages[video]
            del video_received_collages[video]
