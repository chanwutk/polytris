#!/usr/bin/env python3
"""
Debug IOU and fuse_score computation at frame 665 for track 32.
"""

import json
import os
import numpy as np

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython, STrack as STrackPython
from polyis.tracker.bytetrack import matching as matching_python
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython, STrackPy
from polyis.tracker.bytetrack.cython import matching as matching_cython
from polyis.tracker.bytetrack.cython.bytetrack import reset_tracker_count, joint_stracks as joint_stracks_cython
from polyis.tracker.bytetrack.byte_tracker import joint_stracks as joint_stracks_python
from polyis.utilities import get_config


def test_iou_computation():
    """Debug IOU computation for track 32."""

    # Load configuration
    config = get_config()
    cache_dir = config['DATA']['CACHE_DIR']

    # Path to detection results file
    detection_path = os.path.join(
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '002_naive', 'detection.jsonl'
    )

    # Create args
    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    args = Args()

    # Initialize Python tracker and run to frame 664
    from polyis.tracker.bytetrack.basetrack import BaseTrack
    BaseTrack._count = 0
    tracker_python = BYTETrackerPython(args)

    img_info = (1080, 1920)
    img_size = (1080, 1920)

    # Run Python tracker to frame 664
    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 665:
                break
            if line.strip():
                frame_result = json.loads(line)
                detections = frame_result.get('detections', [])
                if len(detections) > 0:
                    dets = np.array(detections, dtype=np.float64)
                    if dets.shape[1] < 5:
                        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                        dets = np.concatenate([dets, scores], axis=1)
                    dets = dets[:, :5]
                else:
                    dets = np.empty((0, 5), dtype=np.float64)
                tracker_python.update(dets, img_info, img_size)

    # Run Cython tracker to frame 664
    reset_tracker_count()
    tracker_cython = BYTETrackerCython(args)

    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 665:
                break
            if line.strip():
                frame_result = json.loads(line)
                detections = frame_result.get('detections', [])
                if len(detections) > 0:
                    dets = np.array(detections, dtype=np.float64)
                    if dets.shape[1] < 5:
                        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                        dets = np.concatenate([dets, scores], axis=1)
                    dets = dets[:, :5]
                else:
                    dets = np.empty((0, 5), dtype=np.float64)
                tracker_cython.update(dets, img_info, img_size)

    # Get frame 665 detections
    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 665:
                frame_result = json.loads(line)
                detections = frame_result.get('detections', [])
                if len(detections) > 0:
                    dets = np.array(detections, dtype=np.float64)
                    if dets.shape[1] < 5:
                        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                        dets = np.concatenate([dets, scores], axis=1)
                    dets = dets[:, :5]
                else:
                    dets = np.empty((0, 5), dtype=np.float64)
                break

    print("=== Frame 665 IOU Computation Debug ===")

    # Process like in update()
    scores = dets[:, 4]
    bboxes = dets[:, :4]

    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes = bboxes / scale

    remain_inds = scores > args.track_thresh
    dets_high = bboxes[remain_inds]
    scores_high = scores[remain_inds]

    # === Python ===
    python_unconfirmed = []
    python_tracked = []
    for track in tracker_python.tracked_stracks:
        if not track.is_activated:
            python_unconfirmed.append(track)
        else:
            python_tracked.append(track)

    python_pool = joint_stracks_python(python_tracked, tracker_python.lost_stracks)
    python_detections = [STrackPython(STrackPython.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_high, scores_high)]

    # Predict
    STrackPython.multi_predict(python_pool)

    # Compute IOU distance BEFORE fuse_score
    python_iou_dists = matching_python.iou_distance(python_pool, python_detections)

    # Apply fuse_score
    python_fused_dists = matching_python.fuse_score(python_iou_dists.copy(), python_detections)

    # === Cython ===
    cython_unconfirmed = []
    cython_tracked = []
    for track in tracker_cython.tracked_stracks:
        if not track.is_activated:
            cython_unconfirmed.append(track)
        else:
            cython_tracked.append(track)

    cython_pool = joint_stracks_cython(cython_tracked, tracker_cython.lost_stracks)
    cython_detections = [STrackPy(tracker_cython._tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_high, scores_high)]

    # Predict
    for track in cython_pool:
        track.predict()

    # Compute IOU distance BEFORE fuse_score
    cython_iou_dists = matching_cython.iou_distance(cython_pool, cython_detections)

    # Apply fuse_score
    cython_fused_dists = matching_cython.fuse_score(cython_iou_dists.copy(), cython_detections)

    # Compare for track 32, detection 15
    track_idx = 17
    det_idx = 15

    print(f"\nTrack 32 (index {track_idx}) to Detection 15 (index {det_idx}):")
    print(f"\nPython:")
    print(f"  IOU distance: {python_iou_dists[track_idx, det_idx]:.10f}")
    print(f"  Fused cost: {python_fused_dists[track_idx, det_idx]:.10f}")
    print(f"  Detection score: {python_detections[det_idx].score:.10f}")

    print(f"\nCython:")
    print(f"  IOU distance: {cython_iou_dists[track_idx, det_idx]:.10f}")
    print(f"  Fused cost: {cython_fused_dists[track_idx, det_idx]:.10f}")
    print(f"  Detection score: {cython_detections[det_idx].score:.10f}")

    print(f"\nDifferences:")
    print(f"  IOU distance diff: {abs(python_iou_dists[track_idx, det_idx] - cython_iou_dists[track_idx, det_idx]):.10f}")
    print(f"  Fused cost diff: {abs(python_fused_dists[track_idx, det_idx] - cython_fused_dists[track_idx, det_idx]):.10f}")

    # Get track 32 tlbr after prediction
    python_track_32 = python_pool[track_idx]
    cython_track_32 = cython_pool[track_idx]

    print(f"\nTrack 32 TLBR after prediction:")
    print(f"  Python: {python_track_32.tlbr}")
    print(f"  Cython: {cython_track_32.tlbr}")
    print(f"  Diff: {np.abs(python_track_32.tlbr - cython_track_32.tlbr)}")

    # Get detection 15 tlbr
    python_det_15 = python_detections[det_idx]
    cython_det_15 = cython_detections[det_idx]

    print(f"\nDetection 15 TLBR:")
    print(f"  Python: {python_det_15.tlbr}")
    print(f"  Cython: {cython_det_15.tlbr}")
    print(f"  Diff: {np.abs(python_det_15.tlbr - cython_det_15.tlbr)}")

    # Manually compute IOU
    def compute_iou(box1, box2):
        """Compute IOU between two boxes [x1,y1,x2,y2]."""
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])

        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-9)

    python_manual_iou = compute_iou(python_track_32.tlbr, python_det_15.tlbr)
    cython_manual_iou = compute_iou(cython_track_32.tlbr, cython_det_15.tlbr)

    print(f"\nManual IOU computation:")
    print(f"  Python: {python_manual_iou:.10f}")
    print(f"  Cython: {cython_manual_iou:.10f}")
    print(f"  Manual IOU distance (Python): {1 - python_manual_iou:.10f}")
    print(f"  Manual IOU distance (Cython): {1 - cython_manual_iou:.10f}")


if __name__ == '__main__':
    test_iou_computation()
