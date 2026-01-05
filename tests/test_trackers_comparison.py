#!/usr/bin/env python3
"""
Test suite for comparing tracking results and performance from multiple tracker implementations:
- polyis/tracker/sort/sort.py (SORT Python)
- polyis/tracker/sort/cython/sort.pyx (SORT Cython)
- polyis/tracker/ocsort/ocsort.py (OC-SORT Python)
- polyis/tracker/ocsort/cython/ocsort.pyx (OC-SORT Cython)

This test:
1. Compares Python vs Cython correctness for both SORT and OC-SORT
2. Compares performance (speed) across all four implementations
"""

import json
import os
import time
from typing import Any
import pytest
import numpy as np
import numpy.typing as npt

from polyis.b3d.sort import Sort as SortB3D
from polyis.tracker.cython.sort import Sort as SortCython  # type: ignore
from polyis.tracker.cython.sort import reset_tracker_count as reset_sort_count
from polyis.tracker.ocsort.ocsort import OCSort as OCSortPython
from polyis.tracker.ocsort.cython.ocsort import OCSort as OCSortCython  # type: ignore
from polyis.tracker.ocsort.cython.ocsort import reset_tracker_count as reset_ocsort_count
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


def run_sort_tracker(
    tracker, 
    detection_results: list[dict]
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, Any] | None]:
    """
    Run a SORT tracker on detection results and collect tracking outputs.
    
    Args:
        tracker: Tracker instance (SortB3D or SortCython)
        detection_results: List of frame detection results
        
    Returns:
        tuple: (tracking_results, performance_metrics)
    """
    tracking_results: dict[int, npt.NDArray[np.floating]] = {}
    performance_metrics = {
        'total_time': 0.0,
        'frame_times': [],
        'num_frames': 0,
        'num_detections': [],
    }
    
    start_total = time.perf_counter()
    
    for frame_result in detection_results:
        frame_idx = frame_result['frame_idx']
        detections = frame_result.get('detections', frame_result.get('bboxes', []))
        
        if len(detections) > 0:
            dets = np.array(detections, dtype=np.float64)
            if dets.size > 0:
                dets = dets[:, :5]
            else:
                dets = np.empty((0, 5), dtype=np.float64)
        else:
            dets = np.empty((0, 5), dtype=np.float64)
        
        start_frame = time.perf_counter()
        performance_metrics['num_detections'].append(len(dets))
        
        tracked_dets = tracker.update(dets)
        
        frame_time = time.perf_counter() - start_frame
        performance_metrics['frame_times'].append(frame_time)
        performance_metrics['num_frames'] += 1
        
        tracking_results[frame_idx] = tracked_dets
    
    performance_metrics['total_time'] = time.perf_counter() - start_total
    
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


def run_ocsort_tracker(
    tracker, 
    detection_results: list[dict],
    img_info: tuple[int, int] = (1080, 1920),
    img_size: tuple[int, int] = (1080, 1920)
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, Any] | None]:
    """
    Run an OC-SORT tracker on detection results and collect tracking outputs.
    
    Args:
        tracker: Tracker instance (OCSortPython or OCSortCython)
        detection_results: List of frame detection results
        img_info: Image info tuple (height, width)
        img_size: Image size tuple (height, width)
        
    Returns:
        tuple: (tracking_results, performance_metrics)
    """
    tracking_results: dict[int, npt.NDArray[np.floating]] = {}
    performance_metrics = {
        'total_time': 0.0,
        'frame_times': [],
        'num_frames': 0,
        'num_detections': [],
    }
    
    start_total = time.perf_counter()
    
    for frame_result in detection_results:
        frame_idx = frame_result['frame_idx']
        detections = frame_result.get('detections', frame_result.get('bboxes', []))
        
        if len(detections) > 0:
            dets = np.array(detections, dtype=np.float64)
            if dets.size > 0:
                if dets.shape[1] < 5:
                    scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                    dets = np.concatenate([dets, scores], axis=1)
                dets = dets[:, :5]
            else:
                dets = np.empty((0, 5), dtype=np.float64)
        else:
            dets = np.empty((0, 5), dtype=np.float64)
        
        start_frame = time.perf_counter()
        performance_metrics['num_detections'].append(len(dets))
        
        tracked_dets = tracker.update(dets, img_info, img_size)
        
        frame_time = time.perf_counter() - start_frame
        performance_metrics['frame_times'].append(frame_time)
        performance_metrics['num_frames'] += 1
        
        tracking_results[frame_idx] = tracked_dets
    
    performance_metrics['total_time'] = time.perf_counter() - start_total
    
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
    Compare tracking results from Python and Cython implementations.
    
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
    
    all_frames = set(results_python.keys()) | set(results_cython.keys())
    
    for frame_idx in sorted(all_frames):
        comparison['frames_compared'] += 1
        
        python_result = results_python.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        cython_result = results_cython.get(frame_idx, np.empty((0, 5), dtype=np.float64))
        
        comparison['total_tracks_python'] += len(python_result)
        comparison['total_tracks_cython'] += len(cython_result)
        
        match = _compare_two_results(python_result, cython_result, tolerance)
        
        if match:
            comparison['frames_match'] += 1
        else:
            comparison['frames_differ'] += 1
            comparison['frame_differences'].append({
                'frame_idx': frame_idx,
                'python_count': len(python_result),
                'cython_count': len(cython_result),
            })
    
    return comparison


def _compare_two_results(
    result1: npt.NDArray[np.floating],
    result2: npt.NDArray[np.floating],
    tolerance: float
) -> bool:
    """Compare two tracking results and return True if they match."""
    if len(result1) != len(result2):
        return False
    
    if len(result1) == 0:
        return True
    
    # Sort by track ID for comparison
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


def test_trackers_comparison():
    """
    Test comparing tracking results and performance from all tracker implementations.
    
    This test:
    1. Loads detection results from the specified JSONL file
    2. Initializes all trackers with the same parameters
    3. Runs all trackers on the same detections
    4. Compares Python vs Cython correctness for both SORT and OC-SORT
    5. Compares performance across all implementations
    """
    
    # Load configuration
    config = get_config()
    cache_dir = config['DATA']['CACHE_DIR']
    
    # Path to detection results file
    detection_path = os.path.join(
        cache_dir, 'jnc0', 'execution', 'te04.mp4', '002_naive', 'detection.jsonl'
    )
    
    # Load detection results
    detection_results = load_detection_results(detection_path)
    
    if not detection_results:
        pytest.skip(f"No detection results found in {detection_path}")
    
    # Load tracker configuration
    tracker_config_path = os.path.join('configs', 'trackers.yaml')
    if os.path.exists(tracker_config_path):
        import yaml
        with open(tracker_config_path, 'r') as f:
            tracker_config = yaml.safe_load(f)
        
        # SORT config
        sort_config = tracker_config.get('sort', {})
        sort_max_age = sort_config.get('max_age', 20)
        sort_min_hits = sort_config.get('min_hits', 1)
        sort_iou_threshold = sort_config.get('iou_threshold', 0.1)
        
        # OC-SORT config
        ocsort_config = tracker_config.get('ocsort', {})
        ocsort_max_age = ocsort_config.get('max_age', 30)
        ocsort_min_hits = ocsort_config.get('min_hits', 3)
        ocsort_iou_threshold = ocsort_config.get('iou_threshold', 0.3)
        ocsort_det_thresh = ocsort_config.get('det_thresh', 0.3)
        ocsort_delta_t = ocsort_config.get('delta_t', 3)
        ocsort_asso_func = ocsort_config.get('asso_func', 'iou')
        ocsort_inertia = ocsort_config.get('inertia', 0.2)
        ocsort_use_byte = ocsort_config.get('use_byte', False)
    else:
        # Default values
        sort_max_age = 20
        sort_min_hits = 1
        sort_iou_threshold = 0.1
        ocsort_max_age = 30
        ocsort_min_hits = 3
        ocsort_iou_threshold = 0.3
        ocsort_det_thresh = 0.3
        ocsort_delta_t = 3
        ocsort_asso_func = 'iou'
        ocsort_inertia = 0.2
        ocsort_use_byte = False
    
    # Image info for OC-SORT
    img_info = (1080, 1920)
    img_size = (1080, 1920)
    
    # Initialize SORT trackers
    from polyis.b3d.sort import KalmanBoxTracker as KalmanBoxTrackerB3D
    KalmanBoxTrackerB3D.count = 0
    reset_sort_count()
    
    sort_python = SortB3D(
        max_age=sort_max_age,
        min_hits=sort_min_hits,
        iou_threshold=sort_iou_threshold
    )
    sort_cython = SortCython(
        max_age=sort_max_age,
        min_hits=sort_min_hits,
        iou_threshold=sort_iou_threshold
    )
    
    # Initialize OC-SORT trackers
    from polyis.tracker.ocsort.ocsort import KalmanBoxTracker as KalmanBoxTrackerPython
    KalmanBoxTrackerPython.count = 0
    reset_ocsort_count()
    
    ocsort_python = OCSortPython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    ocsort_cython = OCSortCython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    
    # Run all trackers
    print("\n=== Running SORT Python ===")
    results_sort_python, perf_sort_python = run_sort_tracker(sort_python, detection_results)
    
    KalmanBoxTrackerB3D.count = 0
    reset_sort_count()
    sort_python = SortB3D(
        max_age=sort_max_age,
        min_hits=sort_min_hits,
        iou_threshold=sort_iou_threshold
    )
    sort_cython = SortCython(
        max_age=sort_max_age,
        min_hits=sort_min_hits,
        iou_threshold=sort_iou_threshold
    )
    
    print("\n=== Running SORT Cython ===")
    results_sort_cython, perf_sort_cython = run_sort_tracker(sort_cython, detection_results)
    
    KalmanBoxTrackerPython.count = 0
    reset_ocsort_count()
    ocsort_python = OCSortPython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    ocsort_cython = OCSortCython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    
    print("\n=== Running OC-SORT Python ===")
    results_ocsort_python, perf_ocsort_python = run_ocsort_tracker(
        ocsort_python, detection_results, img_info, img_size
    )
    
    KalmanBoxTrackerPython.count = 0
    reset_ocsort_count()
    ocsort_python = OCSortPython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    ocsort_cython = OCSortCython(
        det_thresh=ocsort_det_thresh,
        max_age=ocsort_max_age,
        min_hits=ocsort_min_hits,
        iou_threshold=ocsort_iou_threshold,
        delta_t=ocsort_delta_t,
        asso_func=ocsort_asso_func,
        inertia=ocsort_inertia,
        use_byte=ocsort_use_byte
    )
    
    print("\n=== Running OC-SORT Cython ===")
    results_ocsort_cython, perf_ocsort_cython = run_ocsort_tracker(
        ocsort_cython, detection_results, img_info, img_size
    )
    
    # Compare SORT Python vs Cython
    print("\n=== SORT Python vs Cython Comparison ===")
    sort_comparison = compare_tracking_results(results_sort_python, results_sort_cython)
    print(f"Frames compared: {sort_comparison['frames_compared']}")
    print(f"Frames match: {sort_comparison['frames_match']}")
    print(f"Frames differ: {sort_comparison['frames_differ']}")
    print(f"Total tracks (Python): {sort_comparison['total_tracks_python']}")
    print(f"Total tracks (Cython): {sort_comparison['total_tracks_cython']}")
    
    if sort_comparison['frames_differ'] == 0:
        print("✓ All SORT frames match exactly!")
    else:
        print(f"⚠ {sort_comparison['frames_differ']} SORT frames differ")
        if sort_comparison['frame_differences']:
            print(f"First 5 differences:")
            for diff in sort_comparison['frame_differences'][:5]:
                print(f"  Frame {diff['frame_idx']}: Python={diff['python_count']}, Cython={diff['cython_count']}")
    
    # Compare OC-SORT Python vs Cython
    print("\n=== OC-SORT Python vs Cython Comparison ===")
    ocsort_comparison = compare_tracking_results(results_ocsort_python, results_ocsort_cython)
    print(f"Frames compared: {ocsort_comparison['frames_compared']}")
    print(f"Frames match: {ocsort_comparison['frames_match']}")
    print(f"Frames differ: {ocsort_comparison['frames_differ']}")
    print(f"Total tracks (Python): {ocsort_comparison['total_tracks_python']}")
    print(f"Total tracks (Cython): {ocsort_comparison['total_tracks_cython']}")
    
    if ocsort_comparison['frames_differ'] == 0:
        print("✓ All OC-SORT frames match exactly!")
    else:
        print(f"⚠ {ocsort_comparison['frames_differ']} OC-SORT frames differ")
        if ocsort_comparison['frame_differences']:
            print(f"First 5 differences:")
            for diff in ocsort_comparison['frame_differences'][:5]:
                print(f"  Frame {diff['frame_idx']}: Python={diff['python_count']}, Cython={diff['cython_count']}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    all_perfs = {
        'SORT Python': perf_sort_python,
        'SORT Cython': perf_sort_cython,
        'OC-SORT Python': perf_ocsort_python,
        'OC-SORT Cython': perf_ocsort_cython,
    }
    
    for name, perf in all_perfs.items():
        if perf:
            print(f"\n{name}:")
            print(f"  Total time: {perf['total_time']:.4f} seconds")
            print(f"  Number of frames: {perf['num_frames']}")
            print(f"  Average time per frame: {perf['avg_frame_time']*1000:.4f} ms")
            print(f"  Median time per frame: {perf['median_frame_time']*1000:.4f} ms")
            if perf['num_frames'] > 0:
                print(f"  Throughput: {perf['num_frames']/perf['total_time']:.2f} frames/second")
    
    # Calculate speedup ratios
    print("\n=== Speedup Ratios ===")
    if perf_sort_python and perf_sort_cython and perf_sort_python['total_time'] > 0 and perf_sort_cython['total_time'] > 0:
        sort_speedup = perf_sort_python['total_time'] / perf_sort_cython['total_time']
        print(f"SORT Cython vs Python: {sort_speedup:.4f}x")
        if sort_speedup > 1.0:
            print(f"  → SORT Cython is {sort_speedup:.2f}x faster than Python")
    
    if perf_ocsort_python and perf_ocsort_cython and perf_ocsort_python['total_time'] > 0 and perf_ocsort_cython['total_time'] > 0:
        ocsort_speedup = perf_ocsort_python['total_time'] / perf_ocsort_cython['total_time']
        print(f"OC-SORT Cython vs Python: {ocsort_speedup:.4f}x")
        if ocsort_speedup > 1.0:
            print(f"  → OC-SORT Cython is {ocsort_speedup:.2f}x faster than Python")
    
    # Cross-comparison (relative performance)
    if (perf_sort_python and perf_ocsort_python and 
        perf_sort_python['total_time'] > 0 and perf_ocsort_python['total_time'] > 0):
        python_ratio = perf_ocsort_python['total_time'] / perf_sort_python['total_time']
        print(f"\nOC-SORT Python vs SORT Python: {python_ratio:.4f}x")
    
    if (perf_sort_cython and perf_ocsort_cython and 
        perf_sort_cython['total_time'] > 0 and perf_ocsort_cython['total_time'] > 0):
        cython_ratio = perf_ocsort_cython['total_time'] / perf_sort_cython['total_time']
        print(f"OC-SORT Cython vs SORT Cython: {cython_ratio:.4f}x")
    
    # Assertions
    assert sort_comparison['frames_compared'] > 0, "No SORT frames were compared"
    assert ocsort_comparison['frames_compared'] > 0, "No OC-SORT frames were compared"
    
    # Check correctness (Python vs Cython should match)
    if sort_comparison['frames_differ'] > 0:
        pytest.fail(f"SORT Python and Cython results differ: {sort_comparison['frames_differ']} frames differ")
    
    if ocsort_comparison['frames_differ'] > 0:
        pytest.fail(f"OC-SORT Python and Cython results differ: {ocsort_comparison['frames_differ']} frames differ")
    
    print("\n✓ All correctness checks passed!")
    print("✓ Python and Cython implementations produce identical results for both SORT and OC-SORT!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

