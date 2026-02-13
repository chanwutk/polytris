#!/usr/bin/env python3
"""
Debug first association at frame 665.
"""

import json
import os
import numpy as np

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython, STrack as STrackPython
from polyis.tracker.bytetrack import matching as matching_python
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython, STrackPy
from polyis.tracker.bytetrack.cython import matching as matching_cython
from polyis.tracker.bytetrack.cython.bytetrack import joint_stracks as joint_stracks_cython
from polyis.tracker.bytetrack.byte_tracker import joint_stracks as joint_stracks_python
from polyis.utilities import get_config


def test_association():
    """Debug first association at frame 665."""

    # Load configuration
    config = get_config()
    cache_dir = config['DATA']['CACHE_DIR']
    cache_dir = '/polyis-cache/ORIGINAL'

    # Path to detection results file
    detection_path = os.path.join(
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '002_naive', 'detection.jsonl'
    )
    detection_path = './tests/data/detection.jsonl'

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
    # reset_tracker_count()
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

    print("=== Frame 665 First Association Debug ===")

    # Process like in update()
    scores = dets[:, 4]
    bboxes = dets[:, :4]

    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes = bboxes / scale

    remain_inds = scores > args.track_thresh
    dets_high = bboxes[remain_inds]
    scores_high = scores[remain_inds]

    print(f"High score detections: {len(dets_high)}")

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

    # Compute distances
    python_dists = matching_python.iou_distance(python_pool, python_detections)
    if not args.mot20:
        python_dists = matching_python.fuse_score(python_dists, python_detections)

    # Match
    python_matches, python_u_track, python_u_detection = matching_python.linear_assignment(python_dists, thresh=args.match_thresh)

    print(f"\nPython:")
    print(f"  Matches: {len(python_matches)}")
    print(f"  Unmatched tracks: {len(python_u_track)}")
    print(f"  Unmatched detections: {len(python_u_detection)}")

    # Check if track 32 was matched
    track_32_idx = 17
    track_32_matched_python = False
    for itracked, idet in python_matches:
        if itracked == track_32_idx:
            track_32_matched_python = True
            print(f"  ✓ Track 32 (index {track_32_idx}) matched with detection {idet}")
            print(f"    Detection score: {scores_high[idet]:.3f}")
            print(f"    Cost: {python_dists[track_32_idx, idet]:.6f}")
            break

    if not track_32_matched_python:
        print(f"  ✗ Track 32 (index {track_32_idx}) NOT matched")
        if track_32_idx in python_u_track:
            print(f"    Track 32 is in unmatched tracks")
        # Show costs for track 32
        print(f"    Costs for track 32 to all detections:")
        for idet in range(len(python_detections)):
            print(f"      det {idet}: {python_dists[track_32_idx, idet]:.6f}")

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

    # Compute distances
    cython_dists = matching_cython.iou_distance(cython_pool, cython_detections)
    if not args.mot20:
        cython_dists = matching_cython.fuse_score(cython_dists, cython_detections)

    # Match
    cython_matches, cython_u_track, cython_u_detection = matching_cython.linear_assignment(cython_dists, thresh=args.match_thresh)

    print(f"\nCython:")
    print(f"  Matches: {len(cython_matches)}")
    print(f"  Unmatched tracks: {len(cython_u_track)}")
    print(f"  Unmatched detections: {len(cython_u_detection)}")

    # Check if track 32 was matched
    track_32_matched_cython = False
    for itracked, idet in cython_matches:
        if itracked == track_32_idx:
            track_32_matched_cython = True
            print(f"  ✓ Track 32 (index {track_32_idx}) matched with detection {idet}")
            print(f"    Detection score: {scores_high[idet]:.3f}")
            print(f"    Cost: {cython_dists[track_32_idx, idet]:.6f}")
            break

    if not track_32_matched_cython:
        print(f"  ✗ Track 32 (index {track_32_idx}) NOT matched")
        if track_32_idx in cython_u_track:
            print(f"    Track 32 is in unmatched tracks")
        # Show costs for track 32
        print(f"    Costs for track 32 to all detections:")
        for idet in range(len(cython_detections)):
            print(f"      det {idet}: {cython_dists[track_32_idx, idet]:.6f}")

    # Compare cost matrices
    print(f"\n=== Cost Matrix Comparison ===")
    cost_diff = np.abs(python_dists - cython_dists)
    print(f"Max cost difference: {cost_diff.max():.10f}")
    print(f"Mean cost difference: {cost_diff.mean():.10f}")

    if cost_diff.max() > 1e-6:
        print(f"⚠ Significant cost differences detected!")
        # Find where differences are largest
        max_idx = np.unravel_index(cost_diff.argmax(), cost_diff.shape)
        print(f"  Largest difference at track {max_idx[0]}, det {max_idx[1]}")
        print(f"    Python cost: {python_dists[max_idx]:.10f}")
        print(f"    Cython cost: {cython_dists[max_idx]:.10f}")


if __name__ == '__main__':
    test_association()
