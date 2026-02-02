#!/usr/bin/env python3
"""
Debug test to compare ByteTrack implementations frame by frame.
"""

import json
import os
import numpy as np

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython
from polyis.tracker.bytetrack.cython.bytetrack import reset_tracker_count
from polyis.utilities import get_config


def test_first_frames():
    """Test the first few frames to see where divergence starts."""

    # Load configuration
    config = get_config()
    cache_dir = config['DATA']['CACHE_DIR']

    # Path to detection results file
    detection_path = os.path.join(
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '002_naive', 'detection.jsonl'
    )

    # Load first 700 frames
    detection_results = []
    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 700:
                break
            if line.strip():
                detection_results.append(json.loads(line))

    # Create args
    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    args = Args()

    # Initialize trackers
    from polyis.tracker.bytetrack.basetrack import BaseTrack
    BaseTrack._count = 0
    tracker_python = BYTETrackerPython(args)

    reset_tracker_count()
    tracker_cython = BYTETrackerCython(args)

    img_info = (1080, 1920)
    img_size = (1080, 1920)

    # Process frame by frame
    for frame_result in detection_results:
        frame_idx = frame_result['frame_idx']
        detections = frame_result.get('detections', [])

        if len(detections) > 0:
            dets = np.array(detections, dtype=np.float64)
            if dets.shape[1] < 5:
                scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                dets = np.concatenate([dets, scores], axis=1)
            dets = dets[:, :5]
        else:
            dets = np.empty((0, 5), dtype=np.float64)

        # Run both trackers
        python_result = tracker_python.update(dets, img_info, img_size)
        cython_result = tracker_cython.update(dets, img_info, img_size)

        # Compare results
        if isinstance(python_result, list):
            python_ids = sorted([int(t.track_id) for t in python_result])
        else:
            python_ids = sorted(python_result[:, 4].astype(int).tolist()) if len(python_result) > 0 else []

        cython_ids = sorted(cython_result[:, 4].astype(int).tolist()) if len(cython_result) > 0 else []

        if python_ids != cython_ids or frame_idx >= 660:
            if python_ids != cython_ids:
                status = "MISMATCH"
            else:
                status = "OK"
            print(f"\n=== Frame {frame_idx} - {status} ===")
            print(f"Detections: {len(dets)}")
            print(f"Python tracks: {len(python_result)}")
            print(f"Cython tracks: {len(cython_result)}")
            print(f"Python IDs: {python_ids}")
            print(f"Cython IDs: {cython_ids}")
            if python_ids != cython_ids:
                print(f"⚠ IDs DIFFER!")
                print(f"  Missing in Cython: {set(python_ids) - set(cython_ids)}")
                print(f"  Extra in Cython: {set(cython_ids) - set(python_ids)}")

                # Show tracker state
                print(f"\nPython tracker state:")
                print(f"  tracked_stracks: {len(tracker_python.tracked_stracks)}")
                print(f"  lost_stracks: {len(tracker_python.lost_stracks)}")
                print(f"  removed_stracks: {len(tracker_python.removed_stracks)}")

                print(f"\nCython tracker state:")
                print(f"  tracked_stracks: {len(tracker_cython.tracked_stracks)}")
                print(f"  lost_stracks: {len(tracker_cython.lost_stracks)}")
                print(f"  removed_stracks: {len(tracker_cython.removed_stracks)}")

                break  # Stop at first mismatch
        elif frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: ✓ {len(python_ids)} tracks match")


if __name__ == '__main__':
    test_first_frames()
