"""
Stage 5: Object detection + coordinate uncompression (p040 + p050 merged).

Receives GPU canvas tensors from Compress, runs the detector (N threads
with CUDA streams), remaps detections to original frame coordinates, and
accumulates per-video results before sending to Track.
"""

from __future__ import annotations

import queue
import threading

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

NUM_DETECT_THREADS = 2


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
    2. Run detector.
    3. Remap detections via unpack_detections (p050).

    When all collages for a video are processed, sends VideoDetections
    to *out_queue*.

    ``None`` on *in_queue* propagates and the process exits.
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    # ---- Initialise detector ----
    # We need a hint for the detector (image count).  Use batch_size=4.
    batch_size = 4
    detector = polyis.models.detector.get_detector(
        config.dataset, gpu_id, batch_size, num_images=100,
    )

    # ---- Warmup ----
    # Create a small dummy image and run 2 warmup batches.
    with torch.no_grad(), torch.inference_mode():
        dummy = np.zeros((480, 704, 3), dtype=np.uint8)
        for _ in range(2):
            polyis.models.detector.detect_batch([dummy], detector)
    torch.cuda.synchronize()

    # ---- Per-video state ----
    # {video: {frame_idx: [[x1,y1,x2,y2,...], ...]}}
    video_detections: dict[str, dict[int, list[list[float]]]] = {}
    video_num_frames: dict[str, int] = {}
    video_expected_collages: dict[str, int] = {}
    video_received_collages: dict[str, int] = {}

    with torch.no_grad(), torch.inference_mode():
        while True:
            msg = in_queue.get()
            if msg is None:
                out_queue.put(None)
                return

            assert isinstance(msg, CollageReady)
            video = msg.video
            tile_size = msg.tile_size

            # Initialise tracking for a new video.
            if video not in video_detections:
                video_detections[video] = {i: [] for i in range(msg.num_frames)}
                video_num_frames[video] = msg.num_frames
                video_expected_collages[video] = msg.total_collages
                video_received_collages[video] = 0

            # Transfer canvas from GPU to CPU numpy BGR for the detector.
            canvas_np = msg.canvas.cpu().numpy()
            # RGB -> BGR for detector.
            canvas_bgr = np.ascontiguousarray(canvas_np[:, :, ::-1])
            assert polyis.dtypes.is_np_image(canvas_bgr)

            # Run detection.
            batch_output = polyis.models.detector.detect_batch([canvas_bgr], detector)
            detections = batch_output[0].tolist()

            # Unpack detections (p050 logic): remap compressed coords to original.
            frame_dets, _not_in_tile, _center_not = unpack_detections(
                detections, msg.index_map, msg.offset_lookup, tile_size,
            )

            # Merge into accumulated per-frame detections.
            for frame_idx, bboxes in frame_dets.items():
                video_detections[video][frame_idx].extend(bboxes)

            video_received_collages[video] += 1

            # Check if all collages for this video have been processed.
            if video_received_collages[video] >= video_expected_collages[video]:
                out_queue.put(VideoDetections(
                    video=video,
                    frame_detections=video_detections.pop(video),
                    num_frames=video_num_frames.pop(video),
                ))
                del video_expected_collages[video]
                del video_received_collages[video]
