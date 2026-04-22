#!/usr/local/bin/python

import argparse
import itertools
import json
import os
import shutil
import time
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Callable

import torch

from polyis.utilities import create_tracker, format_time, ProgressBar, register_tracked_detections, get_config, save_tracking_results, get_video_resolution, build_param_str, TilePadding
from polyis.io import cache, store
from polyis.pareto import build_pareto_combo_filter


CONFIG = get_config()
DATASETS: list[str] = CONFIG['EXEC']['DATASETS']
CLASSIFIERS: list[str] = CONFIG['EXEC']['CLASSIFIERS']
TILE_SIZES: list[int] = CONFIG['EXEC']['TILE_SIZES']
TILEPADDING_MODES: list[TilePadding] = CONFIG['EXEC']['TILEPADDING_MODES']
SAMPLE_RATES: list[int] = CONFIG['EXEC']['SAMPLE_RATES']
TRACKERS: list[str] = CONFIG['EXEC']['TRACKERS']
CANVAS_SCALES: list[float] = CONFIG['EXEC']['CANVAS_SCALE']
TRACKING_ACCURACY_THRESHOLDS: list[float] = CONFIG['EXEC']['TRACKING_ACCURACY_THRESHOLDS']
RELEVANCE_THRESHOLDS: list[float] = CONFIG['EXEC']['RELEVANCE_THRESHOLDS']

def parse_args():
    parser = argparse.ArgumentParser(description='Execute object tracking on detection results')
    parser.add_argument('--max_age', type=int, default=10,
                        help='Maximum age for SORT tracker')
    parser.add_argument('--min_hits', type=int, default=3,
                        help='Minimum hits for SORT tracker')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IOU threshold for SORT tracker')
    parser.add_argument('--no_interpolate', action='store_true',
                        help='Whether to not perform trajectory interpolation')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    return parser.parse_args()


def load_detection_results(dataset: str, video_file: str, tilesize: int,
                           classifier: str, sample_rate: int = 1, tilepadding: str | None = None,
                           canvas_scale: float = 1.0, tracker: str | None = None,
                           tracking_accuracy_threshold: float | None = None,
                           *, relevance_threshold: float, verbose: bool = False):
    """
    Load detection results from the uncompressed detections JSONL file.

    Args:
        dataset (str): Dataset name
        video_file (str): Video file name
        tilesize (int): Tile size used for detections
        classifier (str): Classifier name used for detections
        tilepadding (str): Whether padding was applied to classification results
        sample_rate (int): Sample rate for frame sampling (default: 1)
        canvas_scale (float): Canvas scale used for compression outputs
        tracker (str | None): Tracker name for upstream pruning
        tracking_accuracy_threshold (float | None): Accuracy threshold for pruning
        relevance_threshold (float): Relevance binarization threshold T_r for cache paths
        verbose (bool): Whether to print verbose output
    Returns:
        list[dict]: list of frame detection results

    Raises:
        FileNotFoundError: If no detection results file is found
    """
    # Build the shared key used by 050 and 060 stage folders.
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate, tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker, tracking_accuracy_threshold=tracking_accuracy_threshold, relevance_threshold=relevance_threshold)
    detection_path = cache.exec(dataset, 'ucomp-dets', video_file,
                                param_str, 'detections.jsonl')
    
    if not os.path.exists(detection_path):
        raise FileNotFoundError(f"Detection results not found: {detection_path}")
    
    if verbose:
        print(f"Loading detection results from: {detection_path}")
    
    results: list[dict] = []
    with open(detection_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if verbose:
        print(f"Loaded {len(results)} frame detections")
    return results


def track(dataset: str, video: str, classifier: str, tilesize: int, sample_rate: int,
          tilepadding: str, canvas_scale: float, tracker_name: str,
          tracking_accuracy_threshold: float | None, relevance_threshold: float,
          no_interpolate: bool):
    """
    Process tracking for a single video/classifier/tilesize/tracker combination.
    This function is designed to be called in parallel.

    Args:
        dataset (str): Name of the dataset
        video (str): Name of the video file to process
        classifier (str): Classifier name used for detections
        tilesize (int): Tile size used for detections
        tilepadding (str): Whether padding was applied to classification results
        sample_rate (int): Sample rate for tracking
        canvas_scale (float): Canvas scale used for compression outputs
        tracker_name (str): Name of the tracker to use
        tracking_accuracy_threshold (float | None): Accuracy threshold for pruning (None = no pruning)
        relevance_threshold (float): Relevance classifier binarization threshold T_r (0–1)
        no_interpolate (bool): Whether to not perform trajectory interpolation
    """
    # Input from p050: tracker only in param_str when pruning is active
    input_tracker = tracker_name if tracking_accuracy_threshold is not None else None
    input_param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                      tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=input_tracker,
                                      tracking_accuracy_threshold=tracking_accuracy_threshold,
                                      relevance_threshold=relevance_threshold)
    # Output always includes tracker (p060 produces different results per tracker)
    output_param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                       tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker_name,
                                       tracking_accuracy_threshold=tracking_accuracy_threshold,
                                       relevance_threshold=relevance_threshold)

    # Check if uncompressed detections exist
    detection_path = cache.exec(dataset, 'ucomp-dets', video,
                                input_param_str, 'detections.jsonl')
    assert os.path.exists(detection_path), f"Detections not found: {detection_path}"

    # Load detection results using input params (tracker=None when no pruning)
    detection_results = load_detection_results(dataset, video, tilesize, classifier,
                                               sample_rate, tilepadding, canvas_scale,
                                               input_tracker, tracking_accuracy_threshold,
                                               relevance_threshold=relevance_threshold)

    # Create output path for tracking results
    output_path = cache.exec(dataset, 'ucomp-tracks', video, output_param_str, 'tracking.jsonl')
    
    # Create tracker
    resolution = get_video_resolution(dataset, video)
    width, height = resolution
    tracker = create_tracker(tracker_name, img_size=(height, width))
    
    # Initialize tracking data structures
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    
    # print(f"Processing {len(detection_results)} frames for tracking...")

    # Create runtime output file
    runtime_path = output_path.with_name('runtimes.jsonl')
    runtime_dir = runtime_path.parent
    os.makedirs(runtime_dir, exist_ok=True)
    
    with open(runtime_path, 'w') as runtime_file:
        # Note: Sampling is now applied at classification stage (p020)
        # Non-sampled frames will have empty bboxes arrays, which naturally
        # result in empty detections that the tracker skips
        # Process each frame
        for frame_result in detection_results:
            frame_idx = frame_result['frame_idx']
            bboxes = frame_result['bboxes']

            # Start timing for this frame
            step_times = {}
            
            # Profile: Convert detections to numpy array
            step_start = (time.time_ns() / 1e6)
            dets = np.array(bboxes)
            if dets.size > 0:
                dets = dets[:, :5]  # Take first 5 columns: x1, y1, x2, y2, score
            else:
                dets = np.empty((0, 5))
            step_times['convert_detections'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Update tracker
            step_start = (time.time_ns() / 1e6)
            tracked_dets = tracker.update(dets)
            step_times['tracker_update'] = (time.time_ns() / 1e6) - step_start
            
            # Profile: Process tracking results
            step_start = (time.time_ns() / 1e6)
            register_tracked_detections(tracked_dets, frame_idx, frame_tracks,
                                        trajectories, interpolate=not no_interpolate)
            step_times['interpolate_trajectory'] = (time.time_ns() / 1e6) - step_start
            
            # Save runtime data for this frame
            runtime_data = {
                'frame_idx': frame_idx,
                'runtime': format_time(**step_times),
                'num_detections': len(bboxes),
                'num_tracks': tracked_dets.size if tracked_dets.size > 0 else 0
            }
            runtime_file.write(json.dumps(runtime_data) + '\n')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    save_tracking_results(frame_tracks, output_path)


def track_all(dataset: str, videos: list[str], classifier: str, tilesize: int, sample_rate: int,
              tilepadding: str, canvas_scale: float, tracker_name: str,
              tracking_accuracy_threshold: float | None, relevance_threshold: float,
              no_interpolate: bool, gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'
    # Build a human-readable description for the progress bar.
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker_name,
                                tracking_accuracy_threshold=tracking_accuracy_threshold,
                                relevance_threshold=relevance_threshold)
    description = f"{dataset} {param_str}"
    # Report initial progress: 0 of N videos done.
    command_queue.put((device, {'completed': 0, 'total': len(videos), 'description': description}))
    # Iterate over all videos in the split for this parameter combination.
    for i, video in enumerate(videos):
        track(dataset, video, classifier, tilesize, sample_rate,
              tilepadding, canvas_scale, tracker_name, tracking_accuracy_threshold,
              relevance_threshold,
              no_interpolate)
        # Advance the progress bar by one unit after each video completes.
        command_queue.put((device, {'completed': i + 1, 'total': len(videos), 'description': description}))


def main(args: argparse.Namespace):
    """
    Main function that orchestrates the object tracking process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves tracking results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects uncompressed detection results from 050_exec_uncompress.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/050_uncompressed_detections/{classifier}_{tilesize}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/060_uncompressed_tracks/{classifier}_{tilesize}/tracking.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - The number of processes equals the number of available GPUs
    """
    mp.set_start_method('spawn', force=True)

    # Determine which videosets to process based on arguments.
    selected_videosets = []
    if args.test:
        selected_videosets.append('test')
    if args.valid:
        selected_videosets.append('valid')
    # Default to valid only when no flags are provided.
    if not selected_videosets:
        selected_videosets = ['valid']

    # Build allowed-combo set for the test pass (None means no filtering applies).
    allowed_combos = build_pareto_combo_filter(
        DATASETS, selected_videosets,
        ['classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale',
         'tracker', 'tracking_accuracy_threshold', 'relevance_threshold'],
    )

    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset, videoset in itertools.product(DATASETS, selected_videosets):
        print(f"Processing dataset: {dataset}")
        videosets_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videosets_dir), f"Videoset directory {videosets_dir} does not exist"

        # Find all videos with uncompressed detection results
        videos = [f for f in os.listdir(videosets_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video in videos:
            shutil.rmtree(cache.exec(dataset, 'ucomp-tracks', video), ignore_errors=True)

        for classifier, tilesize, tilepadding, sample_rate, canvas_scale, acc_threshold, relevance_threshold, tracker in itertools.product(
            CLASSIFIERS, TILE_SIZES, TILEPADDING_MODES, SAMPLE_RATES, CANVAS_SCALES, TRACKING_ACCURACY_THRESHOLDS, RELEVANCE_THRESHOLDS, TRACKERS):
            # Skip parameter combos not on the Pareto front during the test pass.
            combo = (classifier, tilesize, sample_rate, tilepadding, canvas_scale, tracker, acc_threshold, relevance_threshold)
            if allowed_combos is not None and combo not in allowed_combos[dataset]:
                continue
            funcs.append(partial(track_all, dataset, sorted(videos), classifier, tilesize, sample_rate,
                                 tilepadding, canvas_scale, tracker, acc_threshold, relevance_threshold, args.no_interpolate))

    print(f"Created {len(funcs)} tasks to process")

    num_gpus = torch.cuda.device_count()
    num_gpus = 40
    
    # Set up multiprocessing with ProgressBar
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main(parse_args())
