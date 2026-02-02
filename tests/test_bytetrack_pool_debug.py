#!/usr/bin/env python3
"""
Debug strack_pool ordering at frame 665.
"""

import json
import os
import numpy as np

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython, STrack as STrackPython
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython
from polyis.tracker.bytetrack.cython.bytetrack import reset_tracker_count, joint_stracks as joint_stracks_cython
from polyis.tracker.bytetrack.byte_tracker import joint_stracks as joint_stracks_python
from polyis.utilities import get_config


def test_pool_ordering():
    """Debug strack_pool ordering at frame 665."""

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

    print("=== Frame 665 strack_pool Debug ===")

    # Separate tracked and unconfirmed in Python
    python_unconfirmed = []
    python_tracked = []
    for track in tracker_python.tracked_stracks:
        if not track.is_activated:
            python_unconfirmed.append(track)
        else:
            python_tracked.append(track)

    # Create strack_pool for Python
    python_pool = joint_stracks_python(python_tracked, tracker_python.lost_stracks)

    print(f"\nPython:")
    print(f"  tracked_stracks: {len(tracker_python.tracked_stracks)} tracks")
    print(f"  lost_stracks: {len(tracker_python.lost_stracks)} tracks")
    print(f"  confirmed tracked: {len(python_tracked)} tracks")
    print(f"  strack_pool size: {len(python_pool)} tracks")
    print(f"  strack_pool IDs: {[t.track_id for t in python_pool]}")

    # Find track 32 in pool
    track_32_idx_python = None
    for i, t in enumerate(python_pool):
        if t.track_id == 32:
            track_32_idx_python = i
            break
    print(f"  Track 32 index in pool: {track_32_idx_python}")

    # Separate tracked and unconfirmed in Cython
    cython_unconfirmed = []
    cython_tracked = []
    for track in tracker_cython.tracked_stracks:
        if not track.is_activated:
            cython_unconfirmed.append(track)
        else:
            cython_tracked.append(track)

    # Create strack_pool for Cython
    cython_pool = joint_stracks_cython(cython_tracked, tracker_cython.lost_stracks)

    print(f"\nCython:")
    print(f"  tracked_stracks: {len(tracker_cython.tracked_stracks)} tracks")
    print(f"  lost_stracks: {len(tracker_cython.lost_stracks)} tracks")
    print(f"  confirmed tracked: {len(cython_tracked)} tracks")
    print(f"  strack_pool size: {len(cython_pool)} tracks")
    print(f"  strack_pool IDs: {[t.track_id for t in cython_pool]}")

    # Find track 32 in pool
    track_32_idx_cython = None
    for i, t in enumerate(cython_pool):
        if t.track_id == 32:
            track_32_idx_cython = i
            break
    print(f"  Track 32 index in pool: {track_32_idx_cython}")

    # Compare
    python_ids = [t.track_id for t in python_pool]
    cython_ids = [t.track_id for t in cython_pool]

    if python_ids == cython_ids:
        print(f"\n✓ strack_pool IDs match!")
    else:
        print(f"\n⚠ strack_pool IDs differ!")
        print(f"  Python has: {set(python_ids) - set(cython_ids)}")
        print(f"  Cython has: {set(cython_ids) - set(python_ids)}")


if __name__ == '__main__':
    test_pool_ordering()
