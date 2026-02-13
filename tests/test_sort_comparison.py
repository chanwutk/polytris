#!/usr/bin/env python3
"""
Test suite for comparing tracking results from three SORT implementations:
- polyis/b3d/sort.py
- polyis/tracker/sort.py
- polyis/tracker/cython/_sort.pyx (Cython implementation)

This test loads detection results from a JSONL file and runs all trackers
on the same detections to compare their outputs.
"""

import json
import os
import time
from typing import Any
import pytest
import numpy as np
import numpy.typing as npt

from polyis.b3d.sort import Sort as SortB3D
# from polyis.tracker.sort import Sort as SortTracker

from polyis.tracker.sort.cython.sort import Sort as SortCython  # type: ignore
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
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, Any] | None]:
    """
    Run a tracker on detection results and collect tracking outputs.
    
    Args:
        tracker: Tracker instance (SortB3D or SortTracker)
        detection_results: List of frame detection results
        measure_performance: If True, measure and return performance metrics
        
    Returns:
        tuple: (tracking_results, performance_metrics)
               - tracking_results: Dictionary mapping frame_idx to tracking results
                                  Each result is an array of [x1, y1, x2, y2, track_id]
               - performance_metrics: Dictionary with timing information (if measure_performance=True)
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
                dets = dets[:, :5]  # Take first 5 columns
            else:
                dets = np.empty((0, 5), dtype=np.float64)
        else:
            dets = np.empty((0, 5), dtype=np.float64)
        
        # Measure frame processing time
        start_frame = time.perf_counter()
        performance_metrics['num_detections'].append(len(dets))
        
        # Update tracker and get tracked detections
        tracked_dets = tracker.update(dets)
        
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
    results_b3d: dict[int, npt.NDArray[np.floating]],
    results_tracker: dict[int, npt.NDArray[np.floating]],
    results_cython: dict[int, npt.NDArray[np.floating]] | None = None,
    tolerance: float = 1e-6
) -> dict[str, Any]:
    """
    Compare tracking results from two or three trackers.
    
    Args:
        results_b3d: Tracking results from b3d/sort.py
        results_tracker: Tracking results from tracker/sort.py
        results_cython: Tracking results from cython/_sort.pyx (optional)
        tolerance: Numerical tolerance for comparing bounding boxes
        
    Returns:
        dict: Comparison statistics and differences
    """
    comparison = {
        'frames_compared': 0,
        'frames_match_b3d_tracker': 0,
        'frames_match_all': 0,
        'frames_differ': 0,
        'total_tracks_b3d': 0,
        'total_tracks_tracker': 0,
        'total_tracks_cython': 0,
        'frame_differences': [],
    }
    
    # Get all frame indices from all results
    all_frames = set(results_b3d.keys()) | set(results_tracker.keys())
    if results_cython is not None:
        all_frames |= set(results_cython.keys())
    
    for frame_idx in sorted(all_frames):
        comparison['frames_compared'] += 1
        
        b3d_result = results_b3d.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        tracker_result = results_tracker.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        cython_result = results_cython.get(frame_idx, np.empty((0, 5), dtype=np.float64)) if results_cython is not None else None
        
        comparison['total_tracks_b3d'] += len(b3d_result)
        comparison['total_tracks_tracker'] += len(tracker_result)
        if cython_result is not None:
            comparison['total_tracks_cython'] += len(cython_result)
        
        # Compare b3d vs tracker
        b3d_tracker_match = _compare_two_results(b3d_result, tracker_result, tolerance, 'b3d', 'tracker')
        
        # Compare with cython if available
        b3d_cython_match = True
        if cython_result is not None:
            b3d_cython_match = _compare_two_results(b3d_result, cython_result, tolerance, 'b3d', 'cython')
        
        if b3d_tracker_match and b3d_cython_match:
            comparison['frames_match_all'] += 1
            comparison['frames_match_b3d_tracker'] += 1
        else:
            comparison['frames_differ'] += 1
            # Record differences
            if not b3d_tracker_match:
                comparison['frame_differences'].append({
                    'frame_idx': frame_idx,
                    'type': 'b3d_vs_tracker_mismatch',
                    'b3d_count': len(b3d_result),
                    'tracker_count': len(tracker_result),
                })
            if not b3d_cython_match:
                comparison['frame_differences'].append({
                    'frame_idx': frame_idx,
                    'type': 'b3d_vs_cython_mismatch',
                    'b3d_count': len(b3d_result),
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


def test_sort_comparison():
    """
    Test comparing tracking results from b3d/sort.py, tracker/sort.py, and cython/_sort.pyx.
    
    This test:
    1. Loads detection results from the specified JSONL file
    2. Initializes all trackers with the same parameters
    3. Runs all trackers on the same detections
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
    
    # Load tracker configuration
    tracker_config_path = os.path.join('configs', 'trackers.yaml')
    if os.path.exists(tracker_config_path):
        import yaml
        with open(tracker_config_path, 'r') as f:
            tracker_config = yaml.safe_load(f)['sort']
        max_age = tracker_config['max_age']
        min_hits = tracker_config['min_hits']
        iou_threshold = tracker_config['iou_threshold']
    else:
        # Default values if config file doesn't exist
        max_age = 20
        min_hits = 1
        iou_threshold = 0.1
    
    # Initialize all trackers with the same parameters
    tracker_b3d = SortB3D(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    # tracker_tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracker_cython = SortCython(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)  # type: ignore[call-arg]
    
    # Reset tracker counters to ensure consistent IDs
    from polyis.b3d.sort import KalmanBoxTracker as KalmanBoxTrackerB3D
    # from polyis.tracker.sort import KalmanBoxTracker as KalmanBoxTrackerTracker
    # from polyis.tracker.sort.cython.sort import reset_tracker_count
    KalmanBoxTrackerB3D.count = 0
    # KalmanBoxTrackerTracker.count = 0
    # reset_tracker_count()
    
    print("\n=== Running Cython SORT ===")
    results_cython, perf_cython = run_tracker(tracker_cython, detection_results)
    
    # Run all trackers on the same detections with performance measurement
    print("\n=== Running B3D SORT ===")
    results_b3d, perf_b3d = run_tracker(tracker_b3d, detection_results)
    
    # Reset counters again for fair comparison
    KalmanBoxTrackerB3D.count = 0
    # KalmanBoxTrackerTracker.count = 0
    # reset_tracker_count()
    
    print("\n=== Running Tracker SORT ===")
    # results_tracker, perf_tracker = run_tracker(tracker_tracker, detection_results)
    perf_tracker = None
    
    # Compare results
    comparison = compare_tracking_results(results_b3d, results_cython, None)

    # Print comparison summary
    print(f"\n=== Tracking Comparison Summary ===")
    print(f"Frames compared: {comparison['frames_compared']}")
    print(f"Frames match (b3d vs tracker): {comparison['frames_match_b3d_tracker']}")
    print(f"Frames match (all three): {comparison['frames_match_all']}")
    print(f"Frames differ: {comparison['frames_differ']}")
    print(f"Total tracks (b3d): {comparison['total_tracks_b3d']}")
    print(f"Total tracks (tracker): {comparison['total_tracks_tracker']}")
    print(f"Total tracks (cython): {comparison['total_tracks_cython']}")
    
    if comparison['frame_differences']:
        print(f"\nFirst 20 frame differences:")
        for diff in comparison['frame_differences'][:20]:
            print(f"  Frame {diff['frame_idx']}: {diff['type']}")
            if 'b3d_count' in diff:
                print(f"    b3d: {diff.get('b3d_count', 'N/A')} tracks")
            if 'tracker_count' in diff:
                print(f"    tracker: {diff.get('tracker_count', 'N/A')} tracks")
            if 'cython_count' in diff:
                print(f"    cython: {diff.get('cython_count', 'N/A')} tracks")

    if comparison['frames_differ'] > 0:
        pytest.fail(f"Tracking results differ between implementations: {comparison['frames_differ']} frames differ")
    
    # Print performance comparison
    # if perf_b3d and perf_tracker and perf_cython:
    if perf_b3d and perf_cython:
        print(f"\n=== Performance Comparison ===")
        print(f"\nB3D SORT Performance:")
        print(f"  Total time: {perf_b3d['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_b3d['num_frames']}")
        print(f"  Average time per frame: {perf_b3d['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_b3d['median_frame_time']*1000:.4f} ms")
        if perf_b3d['num_frames'] > 0:
            print(f"  Throughput: {perf_b3d['num_frames']/perf_b3d['total_time']:.2f} frames/second")
        
        # print(f"\nTracker SORT Performance:")
        # print(f"  Total time: {perf_tracker['total_time']:.4f} seconds")
        # print(f"  Number of frames: {perf_tracker['num_frames']}")
        # print(f"  Average time per frame: {perf_tracker['avg_frame_time']*1000:.4f} ms")
        # print(f"  Median time per frame: {perf_tracker['median_frame_time']*1000:.4f} ms")
        # if perf_tracker['num_frames'] > 0:
        #     print(f"  Throughput: {perf_tracker['num_frames']/perf_tracker['total_time']:.2f} frames/second")
        
        print(f"\nCython SORT Performance:")
        print(f"  Total time: {perf_cython['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_cython['num_frames']}")
        print(f"  Average time per frame: {perf_cython['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_cython['median_frame_time']*1000:.4f} ms")
        if perf_cython['num_frames'] > 0:
            print(f"  Throughput: {perf_cython['num_frames']/perf_cython['total_time']:.2f} frames/second")
        
        # Calculate speedup/slowdown
        # if perf_b3d['total_time'] > 0 and perf_tracker['total_time'] > 0 and perf_cython['total_time'] > 0:
        if perf_b3d['total_time'] > 0 and perf_cython['total_time'] > 0:
            # speedup_tracker = perf_b3d['total_time'] / perf_tracker['total_time']
            speedup_cython = perf_b3d['total_time'] / perf_cython['total_time']
            
            print(f"\n=== Speedup Ratios ===")
            # print(f"Tracker vs B3D: {speedup_tracker:.4f}x")
            print(f"Cython vs B3D: {speedup_cython:.4f}x")
            
            if speedup_cython > 1.0:
                print(f"  → Cython SORT is {speedup_cython:.2f}x faster than B3D")
            # if speedup_tracker > 1.0:
            #     print(f"  → Tracker SORT is {speedup_tracker:.2f}x faster than B3D")
    
    # Assert that results are identical (or at least very similar)
    assert comparison['frames_compared'] > 0, "No frames were compared"
    
    # Check if results match exactly
    if comparison['frames_differ'] == 0:
        print("\n✓ All frames match exactly across all implementations!")
    else:
        print(f"\n⚠ {comparison['frames_differ']} frames differ between implementations")
        # For debugging, let's check a few specific frames
        if comparison['frame_differences']:
            print("\n=== Debugging First Difference ===")
            first_diff = comparison['frame_differences'][0]
            frame_idx = first_diff['frame_idx']
            print(f"Frame {frame_idx}:")
            print(f"  B3D result shape: {results_b3d.get(frame_idx, np.array([])).shape}")
            print(f"  Tracker result shape: {results_tracker.get(frame_idx, np.array([])).shape}")
            print(f"  Cython result shape: {results_cython.get(frame_idx, np.array([])).shape}")
            if frame_idx in results_b3d:
                print(f"  B3D result:\n{results_b3d[frame_idx]}")
            if frame_idx in results_tracker:
                print(f"  Tracker result:\n{results_tracker[frame_idx]}")
            if frame_idx in results_cython:
                print(f"  Cython result:\n{results_cython[frame_idx]}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

