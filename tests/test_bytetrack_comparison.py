#!/usr/bin/env python3
"""
Test suite for comparing tracking results from two ByteTrack implementations:
- polyis/tracker/bytetrack/byte_tracker.py (Python implementation)
- polyis/tracker/bytetrack/cython/bytetrack.pyx (Cython implementation)

This test loads detection results from a JSONL file and runs both trackers
on the same detections to compare their outputs.
"""

import json
import os
import time
from typing import Any
import pytest
import numpy as np
import numpy.typing as npt

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as BYTETrackerCython  # type: ignore
# from polyis.tracker.bytetrack.cython.bytetrack import reset_tracker_count
from polyis.utilities import CACHE_DIR, get_config


def load_detection_results(detection_path: str) -> list[dict]:
    """
    Load detection results from a JSONL file.

    Args:
        detection_path: Path to the detection.jsonl file

    Returns:
        list[dict]: List of frame detection results, each containing
                   'frame_idx' and 'detections' (list of [x1, y1, x2, y2, score])
    """
    if not os.path.exists(detection_path):
        pytest.skip(f"Detection file not found: {detection_path}")

    results = []
    with open(detection_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return results


def run_tracker(
    tracker,
    detection_results: list[dict],
    img_info: tuple[int, int] = (1080, 1920),
    img_size: tuple[int, int] = (1080, 1920)
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, Any] | None]:
    """
    Run a tracker on detection results and collect tracking outputs.

    Args:
        tracker: Tracker instance (BYTETrackerPython or BYTETrackerCython)
        detection_results: List of frame detection results
        img_info: Image info tuple (height, width)
        img_size: Image size tuple (height, width)

    Returns:
        tuple: (tracking_results, performance_metrics)
               - tracking_results: Dictionary mapping frame_idx to tracking results
                                  Each result is an array of [x1, y1, x2, y2, track_id]
               - performance_metrics: Dictionary with timing information
    """
    tracking_results: dict[int, npt.NDArray[np.floating]] = {}
    performance_metrics = {
        'total_time': 0.0,
        'frame_times': [],
        'num_frames': 0,
        'num_detections': [],
    }

    # Measure total execution time
    start_total = time.perf_counter()

    for frame_result in detection_results:
        frame_idx = frame_result['frame_idx']
        # Handle both 'detections' and 'bboxes' keys (different files use different keys)
        detections = frame_result.get('detections', frame_result.get('bboxes', []))

        # Convert detections to numpy array format [x1, y1, x2, y2, score]
        if len(detections) > 0:
            dets = np.array(detections, dtype=np.float64)
            if dets.size > 0:
                # Ensure we have at least 5 columns (x1, y1, x2, y2, score)
                if dets.shape[1] < 5:
                    # Pad with score column if missing
                    scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                    dets = np.concatenate([dets, scores], axis=1)
                dets = dets[:, :5]  # Take first 5 columns
            else:
                dets = np.empty((0, 5), dtype=np.float64)
        else:
            dets = np.empty((0, 5), dtype=np.float64)

        # Measure frame processing time
        start_frame = time.perf_counter()
        performance_metrics['num_detections'].append(len(dets))

        # Update tracker and get tracked detections
        tracked_objs = tracker.update(dets, img_info, img_size)

        # Convert tracked objects to numpy array
        # Check if we got objects or already a numpy array
        if isinstance(tracked_objs, np.ndarray):
            # Already a numpy array (Cython implementation)
            tracked_dets = tracked_objs
        elif len(tracked_objs) > 0:
            # List of track objects (Python implementation)
            tracked_dets = np.array([[t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3], t.track_id]
                                     for t in tracked_objs], dtype=np.float64)
        else:
            tracked_dets = np.empty((0, 5), dtype=np.float64)

        # Record frame timing
        frame_time = time.perf_counter() - start_frame
        performance_metrics['frame_times'].append(frame_time)
        performance_metrics['num_frames'] += 1

        # Store results for this frame
        tracking_results[frame_idx] = tracked_dets

    # Record total time
    performance_metrics['total_time'] = time.perf_counter() - start_total
    # Calculate statistics
    if performance_metrics['frame_times']:
        frame_times_array = np.array(performance_metrics['frame_times'])
        performance_metrics['avg_frame_time'] = float(np.mean(frame_times_array))
        performance_metrics['min_frame_time'] = float(np.min(frame_times_array))
        performance_metrics['max_frame_time'] = float(np.max(frame_times_array))
        performance_metrics['std_frame_time'] = float(np.std(frame_times_array))
        performance_metrics['median_frame_time'] = float(np.median(frame_times_array))
        performance_metrics['p95_frame_time'] = float(np.percentile(frame_times_array, 95))
        performance_metrics['p99_frame_time'] = float(np.percentile(frame_times_array, 99))
    else:
        performance_metrics['avg_frame_time'] = 0.0
        performance_metrics['min_frame_time'] = 0.0
        performance_metrics['max_frame_time'] = 0.0
        performance_metrics['std_frame_time'] = 0.0
        performance_metrics['median_frame_time'] = 0.0
        performance_metrics['p95_frame_time'] = 0.0
        performance_metrics['p99_frame_time'] = 0.0

    return tracking_results, performance_metrics


def compare_tracking_results(
    results_python: dict[int, npt.NDArray[np.floating]],
    results_cython: dict[int, npt.NDArray[np.floating]],
    tolerance: float = 1e-6
) -> dict[str, Any]:
    """
    Compare tracking results from two trackers.

    Args:
        results_python: Tracking results from Python implementation
        results_cython: Tracking results from Cython implementation
        tolerance: Numerical tolerance for comparing bounding boxes

    Returns:
        dict: Comparison statistics and differences
    """
    comparison = {
        'frames_compared': 0,
        'frames_match': 0,
        'frames_differ': 0,
        'total_tracks_python': 0,
        'total_tracks_cython': 0,
        'frame_differences': [],
    }

    # Get all frame indices from all results
    all_frames = set(results_python.keys()) | set(results_cython.keys())

    for frame_idx in sorted(all_frames):
        comparison['frames_compared'] += 1

        python_result = results_python.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        cython_result = results_cython.get(frame_idx, np.empty((0, 5), dtype=np.float64))

        comparison['total_tracks_python'] += len(python_result)
        comparison['total_tracks_cython'] += len(cython_result)

        # Compare results
        match = _compare_two_results(python_result, cython_result, tolerance, 'python', 'cython')

        if match:
            comparison['frames_match'] += 1
        else:
            comparison['frames_differ'] += 1
            # Record differences
            comparison['frame_differences'].append({
                'frame_idx': frame_idx,
                'type': 'python_vs_cython_mismatch',
                'python_count': len(python_result),
                'cython_count': len(cython_result),
            })

    return comparison


def _compare_two_results(
    result1: npt.NDArray[np.floating],
    result2: npt.NDArray[np.floating],
    tolerance: float,
    name1: str,
    name2: str
) -> bool:
    """Compare two tracking results and return True if they match."""
    # Compare number of tracks
    if len(result1) != len(result2):
        return False

    # If both are empty, they match
    if len(result1) == 0:
        return True

    # Sort by track ID for comparison
    # Track ID is in the last column (index 4)
    result1_sorted = result1[result1[:, 4].argsort()]
    result2_sorted = result2[result2[:, 4].argsort()]

    # Compare track IDs
    ids1 = result1_sorted[:, 4]
    ids2 = result2_sorted[:, 4]

    if not np.array_equal(ids1, ids2):
        return False

    # Compare bounding boxes (first 4 columns)
    bboxes1 = result1_sorted[:, :4]
    bboxes2 = result2_sorted[:, :4]

    if not np.allclose(bboxes1, bboxes2, atol=tolerance):
        return False

    return True


def test_bytetrack_comparison():
    """
    Test comparing tracking results from Python and Cython ByteTrack implementations.

    This test:
    1. Loads detection results from the specified JSONL file
    2. Initializes both trackers with the same parameters
    3. Runs both trackers on the same detections
    4. Compares the results frame by frame
    """

    # Load configuration to get tracker parameters
    config = get_config()
    cache_dir = config['DATA']['CACHE_DIR']

    # Path to detection results file
    detection_path = os.path.join(
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '002_naive', 'detection.jsonl'
    )

    # Load detection results
    detection_results = load_detection_results(detection_path)

    # Skip test if file doesn't exist
    if not detection_results:
        pytest.skip(f"No detection results found in {detection_path}")

    # Create a simple args object with default values
    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    args = Args()

    # Initialize both trackers with the same parameters
    tracker_python = BYTETrackerPython(args)
    tracker_cython = BYTETrackerCython(args)

    # Reset tracker counters to ensure consistent IDs
    from polyis.tracker.bytetrack.basetrack import BaseTrack
    BaseTrack._count = 0
    # reset_tracker_count()

    # Image info and size (default values, can be adjusted based on actual data)
    img_info = (1080, 1920)
    img_size = (1080, 1920)

    print("\n=== Running Python ByteTrack ===")
    results_python, perf_python = run_tracker(
        tracker_python, detection_results, img_info, img_size
    )

    # Reset counters again for fair comparison
    BaseTrack._count = 0
    # reset_tracker_count()

    # Reinitialize trackers
    tracker_python = BYTETrackerPython(args)
    tracker_cython = BYTETrackerCython(args)

    print("\n=== Running Cython ByteTrack ===")
    results_cython, perf_cython = run_tracker(
        tracker_cython, detection_results, img_info, img_size
    )

    # Compare results
    comparison = compare_tracking_results(results_python, results_cython)

    # Print comparison summary
    print(f"\n=== Tracking Comparison Summary ===")
    print(f"Frames compared: {comparison['frames_compared']}")
    print(f"Frames match: {comparison['frames_match']}")
    print(f"Frames differ: {comparison['frames_differ']}")
    print(f"Total tracks (Python): {comparison['total_tracks_python']}")
    print(f"Total tracks (Cython): {comparison['total_tracks_cython']}")

    if comparison['frame_differences']:
        print(f"\nFirst 20 frame differences:")
        for diff in comparison['frame_differences'][:20]:
            print(f"  Frame {diff['frame_idx']}: {diff['type']}")
            print(f"    Python: {diff.get('python_count', 'N/A')} tracks")
            print(f"    Cython: {diff.get('cython_count', 'N/A')} tracks")

    # Print performance comparison
    if perf_python and perf_cython:
        print(f"\n=== Performance Comparison ===")
        print(f"\nPython ByteTrack Performance:")
        print(f"  Total time: {perf_python['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_python['num_frames']}")
        print(f"  Average time per frame: {perf_python['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_python['median_frame_time']*1000:.4f} ms")
        if perf_python['num_frames'] > 0:
            print(f"  Throughput: {perf_python['num_frames']/perf_python['total_time']:.2f} frames/second")

        print(f"\nCython ByteTrack Performance:")
        print(f"  Total time: {perf_cython['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_cython['num_frames']}")
        print(f"  Average time per frame: {perf_cython['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_cython['median_frame_time']*1000:.4f} ms")
        if perf_cython['num_frames'] > 0:
            print(f"  Throughput: {perf_cython['num_frames']/perf_cython['total_time']:.2f} frames/second")

        # Calculate speedup/slowdown
        if perf_python['total_time'] > 0 and perf_cython['total_time'] > 0:
            speedup_cython = perf_python['total_time'] / perf_cython['total_time']

            print(f"\n=== Speedup Ratios ===")
            print(f"Cython vs Python: {speedup_cython:.4f}x")

            if speedup_cython > 1.0:
                print(f"  → Cython ByteTrack is {speedup_cython:.2f}x faster than Python")

    # Assert that results are identical (or at least very similar)
    assert comparison['frames_compared'] > 0, "No frames were compared"

    # Check if results match exactly
    if comparison['frames_differ'] == 0:
        print("\n✓ All frames match exactly across both implementations!")
    else:
        print(f"\n⚠ {comparison['frames_differ']} frames differ between implementations")
        # For debugging, let's check a few specific frames
        if comparison['frame_differences']:
            print("\n=== Debugging First Difference ===")
            first_diff = comparison['frame_differences'][0]
            frame_idx = first_diff['frame_idx']
            print(f"Frame {frame_idx}:")
            print(f"  Python result shape: {results_python.get(frame_idx, np.array([])).shape}")
            print(f"  Cython result shape: {results_cython.get(frame_idx, np.array([])).shape}")
            if frame_idx in results_python:
                print(f"  Python result:\n{results_python[frame_idx][:, :-1]}")
            if frame_idx in results_cython:
                print(f"  Cython result:\n{results_cython[frame_idx][:, :-1]}")

            if frame_idx in results_python:
                print(f"  Python idx:\n{results_python[frame_idx][:, -1]}")
            if frame_idx in results_cython:
                print(f"  Cython idx:\n{results_cython[frame_idx][:, -1]}")

    if comparison['frames_differ'] > 0:
        pytest.fail(f"Tracking results differ between implementations: {comparison['frames_differ']} frames differ")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
