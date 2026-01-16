#!/usr/bin/env python3
"""
Debug matching at frame 665 to understand why track 32 isn't re-activated in Cython.
"""

import json
import os
import numpy as np

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
from polyis.tracker.bytetrack import matching as matching_python
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython
from polyis.tracker.bytetrack.cython import matching as matching_cython
from polyis.tracker.bytetrack.cython.bytetrack import reset_tracker_count
from polyis.utilities import get_config


def test_matching_frame_665():
    """Debug matching at frame 665."""

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

    print("=== Frame 665 Matching Debug ===")
    print(f"Number of detections: {len(dets)}")

    # Check track 32 in both trackers
    python_track_32 = None
    for t in tracker_python.lost_stracks:
        if t.track_id == 32:
            python_track_32 = t
            break

    cython_track_32 = None
    for t in tracker_cython.lost_stracks:
        if t.track_id == 32:
            cython_track_32 = t
            break

    if python_track_32:
        print(f"\nPython Track 32 (before update):")
        print(f"  tlbr: {python_track_32.tlbr}")
        print(f"  tlwh: {python_track_32.tlwh}")
        print(f"  score: {python_track_32.score}")
        print(f"  state: {python_track_32.state}")

    if cython_track_32:
        print(f"\nCython Track 32 (before update):")
        print(f"  tlbr: {cython_track_32.tlbr}")
        print(f"  tlwh: {cython_track_32.tlwh}")
        print(f"  score: {cython_track_32.score}")
        print(f"  state: {cython_track_32.state}")

    # Check if there's a difference in coordinates
    if python_track_32 and cython_track_32:
        tlbr_diff = np.abs(python_track_32.tlbr - cython_track_32.tlbr)
        print(f"\n  tlbr difference: {tlbr_diff}")
        print(f"  Max difference: {tlbr_diff.max():.6f}")

    # Compare IOU distances
    if python_track_32 and cython_track_32:
        print("\n=== IOU Distance Comparison ===")

        # Get one detection to test
        det_tlbr = dets[0, :4]
        print(f"\nTesting with detection 0: {det_tlbr}")

        # Compute IOU manually
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

        python_iou = compute_iou(python_track_32.tlbr, det_tlbr)
        cython_iou = compute_iou(cython_track_32.tlbr, det_tlbr)

        print(f"Python IOU with det 0: {python_iou:.6f}")
        print(f"Cython IOU with det 0: {cython_iou:.6f}")
        print(f"IOU difference: {abs(python_iou - cython_iou):.6f}")


if __name__ == '__main__':
    test_matching_frame_665()
