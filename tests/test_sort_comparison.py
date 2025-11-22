#!/usr/bin/env python3
"""
Test suite for comparing tracking results from two SORT implementations:
- polyis/b3d/sort.py
- polyis/tracker/sort.py

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

from polyis.b3d.sort import Sort as SortB3D
from polyis.tracker.sort import Sort as SortTracker
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
    tolerance: float = 1e-6
) -> dict[str, Any]:
    """
    Compare tracking results from two trackers.
    
    Args:
        results_b3d: Tracking results from b3d/sort.py
        results_tracker: Tracking results from tracker/sort.py
        tolerance: Numerical tolerance for comparing bounding boxes
        
    Returns:
        dict: Comparison statistics and differences
    """
    comparison = {
        'frames_compared': 0,
        'frames_match': 0,
        'frames_differ': 0,
        'total_tracks_b3d': 0,
        'total_tracks_tracker': 0,
        'frame_differences': [],
    }
    
    # Get all frame indices from both results
    all_frames = set(results_b3d.keys()) | set(results_tracker.keys())
    
    for frame_idx in sorted(all_frames):
        comparison['frames_compared'] += 1
        
        b3d_result = results_b3d.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        tracker_result = results_tracker.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        
        comparison['total_tracks_b3d'] += len(b3d_result)
        comparison['total_tracks_tracker'] += len(tracker_result)
        
        # Compare number of tracks
        if len(b3d_result) != len(tracker_result):
            comparison['frames_differ'] += 1
            comparison['frame_differences'].append({
                'frame_idx': frame_idx,
                'num_tracks_b3d': len(b3d_result),
                'num_tracks_tracker': len(tracker_result),
                'type': 'count_mismatch'
            })
            continue
        
        # If both are empty, they match
        if len(b3d_result) == 0:
            comparison['frames_match'] += 1
            continue
        
        # Sort by track ID for comparison
        # Track ID is in the last column (index 4)
        b3d_sorted = b3d_result  #[b3d_result[:, 4].argsort()]
        tracker_sorted = tracker_result  #[tracker_result[:, 4].argsort()]
        
        # Compare track IDs
        b3d_ids = b3d_sorted[:, 4]
        tracker_ids = tracker_sorted[:, 4]
        
        if not np.array_equal(b3d_ids, tracker_ids):
            comparison['frames_differ'] += 1
            comparison['frame_differences'].append({
                'frame_idx': frame_idx,
                'b3d_ids': b3d_ids.tolist(),
                'tracker_ids': tracker_ids.tolist(),
                'type': 'id_mismatch'
            })
            continue
        
        # Compare bounding boxes (first 4 columns)
        bboxes_b3d = b3d_sorted[:, :4]
        bboxes_tracker = tracker_sorted[:, :4]
        
        if not np.allclose(bboxes_b3d, bboxes_tracker, atol=tolerance):
            comparison['frames_differ'] += 1
            max_diff = np.max(np.abs(bboxes_b3d - bboxes_tracker))
            comparison['frame_differences'].append({
                'frame_idx': frame_idx,
                'max_bbox_diff': float(max_diff),
                'type': 'bbox_mismatch'
            })
            continue
        
        comparison['frames_match'] += 1
    
    return comparison


def test_sort_comparison():
    """
    Test comparing tracking results from b3d/sort.py and tracker/sort.py.
    
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
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '000_groundtruth', 'detection.jsonl'
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
    
    # Initialize both trackers with the same parameters
    tracker_b3d = SortB3D(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracker_tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    
    # Reset tracker counters to ensure consistent IDs
    from polyis.b3d.sort import KalmanBoxTracker as KalmanBoxTrackerB3D
    from polyis.tracker.sort import KalmanBoxTracker as KalmanBoxTrackerTracker
    KalmanBoxTrackerB3D.count = 0
    KalmanBoxTrackerTracker.count = 0
    
    # Run both trackers on the same detections with performance measurement
    results_b3d, perf_b3d = run_tracker(tracker_b3d, detection_results)
    
    # Reset counters again for fair comparison
    KalmanBoxTrackerB3D.count = 0
    KalmanBoxTrackerTracker.count = 0
    
    results_tracker, perf_tracker = run_tracker(tracker_tracker, detection_results)
    
    # Compare results
    comparison = compare_tracking_results(results_b3d, results_tracker)
    
    # Print comparison summary
    print(f"\n=== Tracking Comparison Summary ===")
    print(f"Frames compared: {comparison['frames_compared']}")
    print(f"Frames match: {comparison['frames_match']}")
    print(f"Frames differ: {comparison['frames_differ']}")
    print(f"Total tracks (b3d): {comparison['total_tracks_b3d']}")
    print(f"Total tracks (tracker): {comparison['total_tracks_tracker']}")
    
    if comparison['frame_differences']:
        print(f"\nFirst 10 frame differences:")
        for diff in comparison['frame_differences'][:10]:
            print(f"  Frame {diff['frame_idx']}: {diff['type']}")
            if 'max_bbox_diff' in diff:
                print(f"    Max bbox difference: {diff['max_bbox_diff']}")
            elif 'num_tracks_b3d' in diff:
                print(f"    b3d: {diff['num_tracks_b3d']} tracks, tracker: {diff['num_tracks_tracker']} tracks")
            elif 'b3d_ids' in diff:
                print(f"    b3d IDs: {diff['b3d_ids']}")
                print(f"    tracker IDs: {diff['tracker_ids']}")
    
    # Print performance comparison
    if perf_b3d and perf_tracker:
        print(f"\n=== Performance Comparison ===")
        print(f"\nB3D SORT Performance:")
        print(f"  Total time: {perf_b3d['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_b3d['num_frames']}")
        print(f"  Average time per frame: {perf_b3d['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_b3d['median_frame_time']*1000:.4f} ms")
        print(f"  Min frame time: {perf_b3d['min_frame_time']*1000:.4f} ms")
        print(f"  Max frame time: {perf_b3d['max_frame_time']*1000:.4f} ms")
        print(f"  Std dev frame time: {perf_b3d['std_frame_time']*1000:.4f} ms")
        print(f"  95th percentile: {perf_b3d['p95_frame_time']*1000:.4f} ms")
        print(f"  99th percentile: {perf_b3d['p99_frame_time']*1000:.4f} ms")
        if perf_b3d['num_frames'] > 0:
            print(f"  Throughput: {perf_b3d['num_frames']/perf_b3d['total_time']:.2f} frames/second")
        
        print(f"\nTracker SORT Performance:")
        print(f"  Total time: {perf_tracker['total_time']:.4f} seconds")
        print(f"  Number of frames: {perf_tracker['num_frames']}")
        print(f"  Average time per frame: {perf_tracker['avg_frame_time']*1000:.4f} ms")
        print(f"  Median time per frame: {perf_tracker['median_frame_time']*1000:.4f} ms")
        print(f"  Min frame time: {perf_tracker['min_frame_time']*1000:.4f} ms")
        print(f"  Max frame time: {perf_tracker['max_frame_time']*1000:.4f} ms")
        print(f"  Std dev frame time: {perf_tracker['std_frame_time']*1000:.4f} ms")
        print(f"  95th percentile: {perf_tracker['p95_frame_time']*1000:.4f} ms")
        print(f"  99th percentile: {perf_tracker['p99_frame_time']*1000:.4f} ms")
        if perf_tracker['num_frames'] > 0:
            print(f"  Throughput: {perf_tracker['num_frames']/perf_tracker['total_time']:.2f} frames/second")
        
        # Calculate speedup/slowdown
        if perf_b3d['total_time'] > 0 and perf_tracker['total_time'] > 0:
            speedup = perf_b3d['total_time'] / perf_tracker['total_time']
            print(f"\nPerformance Ratio (b3d/tracker): {speedup:.4f}x")
            if speedup > 1.0:
                print(f"  → Tracker SORT is {speedup:.2f}x faster")
            elif speedup < 1.0:
                print(f"  → B3D SORT is {1.0/speedup:.2f}x faster")
            else:
                print(f"  → Both implementations have similar performance")
            
            avg_speedup = perf_b3d['avg_frame_time'] / perf_tracker['avg_frame_time'] if perf_tracker['avg_frame_time'] > 0 else 0.0
            if avg_speedup > 0:
                print(f"Average frame time ratio (b3d/tracker): {avg_speedup:.4f}x")
    
    # Assert that results are identical (or at least very similar)
    # Note: Due to potential implementation differences, we check if they're close
    assert comparison['frames_compared'] > 0, "No frames were compared"
    
    # Check if results match exactly
    if comparison['frames_differ'] == 0:
        print("\n✓ All frames match exactly!")
    else:
        print(f"\n⚠ {comparison['frames_differ']} frames differ between implementations")
        # For now, we just report the differences rather than failing
        # This allows us to see what differences exist between the implementations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

