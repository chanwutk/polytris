import json
import os
import shutil
import subprocess
import time
import typing
import multiprocessing as mp
import functools
import queue
import random
import contextlib
import sys
import inspect
import logging
import yaml
from xml.etree import ElementTree
import pathlib

import cv2
from matplotlib.path import Path
import numpy as np
from rich import progress
import torch

from polyis.io import VideoCapture, VideoWriter

if typing.TYPE_CHECKING:
    import altair as alt
    import pandas as pd

SOURCE_DIR = '/polyis-data/sources'
DATASETS_DIR = '/polyis-data/datasets'
DATA_RAW_DIR = '/polyis-data/video-datasets-raw'
DATA_DIR = '/polyis-data/video-datasets'
CACHE_DIR = '/polyis-cache'
TILE_SIZES = [60]

GS_DATASETS_DIR = 'gs://polytris/polyis-data/datasets'
GS_CACHE = 'gs://polytris/polyis-cache'

GC_DATASETS_DIR = '/data/chanwutk/data/polyis-data/datasets'
GC_CACHE = '/data/chanwutk/data/polyis-cache'

# Define 10 distinct colors for track visualization (BGR format for OpenCV)
TRACK_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (255, 0, 128),  # Pink
]


video_frame_counts: dict[tuple[str, str], int] = {}
video_resolutions: dict[tuple[str, str], tuple[int, int]] = {}
source_video_frame_counts: dict[tuple[str, str], int] = {}


def dataset_root_name(dataset: str) -> str:
    # Normalize Amsterdam detector variants to the shared dataset key.
    if dataset.startswith('ams'):
        return 'ams'
    # Drop detector suffixes so all variants share one preprocessed dataset.
    return dataset.split('-')[0]


def resolve_otif_dataset_name(dataset: str) -> str:
    # Resolve the normalized dataset key first.
    dataset_root = dataset_root_name(dataset)
    # Map the Amsterdam key to the OTIF directory name.
    if dataset_root == 'ams':
        return 'amsterdam'
    # Use the normalized root directly for all other datasets.
    return dataset_root


def dedupe_datasets_by_root(datasets: list[str]) -> list[str]:
    # Track dataset roots that were already selected.
    seen_roots: set[str] = set()
    # Iterate configured datasets and keep only the first variant per root.
    for dataset in datasets:
        # Compute the shared root for this configured dataset.
        dataset_root = dataset_root_name(dataset)
        # Mark this root as already selected.
        seen_roots.add(dataset_root)
    # Return unique dataset roots.
    return list(seen_roots)


def get_segment_frame_range(total_frames: int, segment_idx: int, num_segments: int) -> tuple[int, int]:
    # Validate that the segment index is within the allowed range.
    assert 0 <= segment_idx < num_segments, (
        f'Invalid segment index {segment_idx} for {num_segments} segments'
    )
    # Compute fractional segment size in source-frame units.
    segment_size_frames = total_frames / num_segments
    # Compute inclusive start frame for this segment.
    start_frame = int(segment_idx * segment_size_frames)
    # Compute exclusive end frame for this segment and clamp to total frames.
    end_frame = min(int((segment_idx + 1) * segment_size_frames), total_frames)
    # Return [start, end) frame interval.
    return start_frame, end_frame


def build_b3d_mask_and_crop(
    file_name: str,
    annotations_path: str,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int, np.ndarray]:
    # Load annotation XML for B3D masks.
    root = ElementTree.parse(annotations_path).getroot()
    # Resolve the image annotation entry for this video.
    img = root.find(f'.//image[@name="{file_name.replace(".mp4", ".jpg")}"]')
    assert img is not None
    # Resolve the crop box used by the original preprocessing pipeline.
    box = img.find('.//box[@label="crop"]')
    assert box is not None
    # Parse crop bounds.
    left = round(float(box.attrib['xtl']))
    top = round(float(box.attrib['ytl']))
    right = round(float(box.attrib['xbr']))
    bottom = round(float(box.attrib['ybr']))

    # Build per-domain bitmap masks.
    domains = img.findall('.//polygon[@label="domain"]')
    bitmaps = []
    for domain in domains:
        assert isinstance(domain, ElementTree.Element)
        points = domain.attrib['points']
        points = points.replace(';', ',')
        points_array = np.array([float(pt) for pt in points.split(',')]).reshape((-1, 2))
        domain_poly = Path(points_array)
        x, y = np.meshgrid(np.arange(frame_width), np.arange(frame_height))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((x, y)).T
        bitmap = domain_poly.contains_points(pixel_points)
        bitmap = bitmap.reshape((1, frame_height, frame_width, 1))
        bitmaps.append(bitmap)

    # Merge all domain masks and crop to the crop area.
    bitmap = bitmaps[0]
    for current_bitmap in bitmaps[1:]:
        bitmap |= current_bitmap
    bitmap = bitmap.astype(np.uint8) * 255
    bitmap = bitmap[:, top:bottom, left:right, :]
    # Return crop geometry and mask bitmap.
    return top, bottom, left, right, bitmap


def get_video_frame_count(dataset: str, video: str) -> int:
    """
    Get the total number of frames in a video using OpenCV.

    Args:
        dataset (str): Dataset name
        video_name (str): Video name (with extension, e.g., 'te01.mp4')

    Returns:
        int: Total number of frames in the video
    """
    if (dataset, video) in video_frame_counts:
        return video_frame_counts[(dataset, video)]

    # Resolve videoset directory from video filename prefix.
    if video.startswith('va'):
        videoset = 'valid'
    elif video.startswith('tr'):
        videoset = 'train'
    else:
        videoset = 'test'
    video_path = os.path.join(DATASETS_DIR, dataset, videoset, video)
    assert os.path.exists(video_path), f"Video file not found for {dataset}/{video}"

    # Open video and get frame count
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_frame_counts[(dataset, video)] = frame_count
    return frame_count


def get_video_resolution(dataset: str, video: str) -> tuple[int, int]:
    """
    Get the resolution (width, height) of a video using OpenCV.

    Args:
        dataset (str): Dataset name
        video (str): Video name (with extension, e.g., 'te01.mp4')

    Returns:
        tuple[int, int]: Video resolution as (width, height)
    """
    if (dataset, video) in video_resolutions:
        return video_resolutions[(dataset, video)]

    split = {'te': 'test', 'va': 'valid', 'tr': 'train'}
    video_path = os.path.join(DATASETS_DIR, dataset, split[video[:2]], video)
    assert os.path.exists(video_path), f"Video file not found for {dataset}/{video}"

    # Open video and get resolution
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    resolution = (width, height)
    video_resolutions[(dataset, video)] = resolution
    return resolution


def get_num_frames(video_file_path: str | pathlib.Path) -> int:
    """
    Get the number of frames from a video file.

    Args:
        video_file_path (str): Path to the video file

    Returns:
        int: Number of frames in the video

    Raises:
        AssertionError: If the video file cannot be opened
    """
    cap = cv2.VideoCapture(video_file_path)
    assert cap.isOpened(), f"Could not open video {video_file_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def format_time(**kwargs: float | int) -> list[dict[str, float | int | str]]:
    """
    Format timing information into a list of dictionaries.

    Args:
        **kwargs: Keyword arguments where keys are operation names and values are timing values

    Returns:
        list: List of dictionaries with 'op' (operation) and 'time' keys for each input argument

    Example:
        >>> format_time(read=1.5, detect=2.3)
        [{'op': 'read', 'time': 1.5}, {'op': 'detect', 'time': 2.3}]
    """
    return [{'op': op, 'time': time} for op, time in kwargs.items()]


def load_detection_results(dataset: str, video_file: str, tracking: bool = False,
                           verbose: bool = False, filename: str | None = None, groundtruth: bool = False) -> list[dict]:
    """
    Load detection results from the JSONL file generated by 001_preprocess_groundtruth_detection.py.

    Args:
        dataset (str): Dataset name
        video_file (str): Video file name
        tracking (bool): Whether to load tracking results instead of detection results
        verbose (bool): Whether to print verbose output
        filename (str | None): Name of the file to load (default: 'tracking.jsonl' if tracking else 'detection.jsonl')
    Returns:
        list[dict]: list of frame detection results

    Raises:
        FileNotFoundError: If no detection results file is found
    """
    from polyis.io import cache
    if filename is None:
        filename = 'tracking.jsonl' if tracking else 'detection.jsonl'
    # Build path using centralized cache path builder
    stage = 'groundtruth' if groundtruth else 'naive'
    detection_path = cache.exec(dataset, stage, video_file, filename)

    if not os.path.exists(detection_path):
        raise FileNotFoundError(f"Detection results not found: {detection_path}")

    if verbose:
        print(f"Loading detection results from: {detection_path}")

    results = []
    with open(detection_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if verbose:
        print(f"Loaded {len(results)} frame detections")
    return results


def load_tracking_results(dataset: str, video_file: str, verbose: bool = False) -> dict[int, list[list[float]]]:
    """
    Load tracking results from the JSONL file generated by 002_preprocess_groundtruth_tracking.py.

    Args:
        dataset (str): Dataset name
        video_file (str): Video file name
        verbose (bool): Whether to print verbose output
    Returns:
        dict[int, list[list[float]]]: dictionary mapping frame indices to lists of tracks

    Raises:
        FileNotFoundError: If no tracking results file is found
    """
    from polyis.io import cache
    # Build path using centralized cache path builder
    tracking_path = cache.exec(dataset, 'naive', video_file, 'tracking.jsonl')

    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Tracking results not found: {tracking_path}")

    if verbose:
        print(f"Loading tracking results from: {tracking_path}")

    frame_tracks = {}
    with open(tracking_path, 'r') as f:
        for line in f:
            if line.strip():
                frame_data = json.loads(line)
                frame_idx = frame_data['frame_idx']
                tracks = frame_data['tracks']
                frame_tracks[frame_idx] = tracks

    if verbose:
        print(f"Loaded tracking results for {len(frame_tracks)} frames")
    return frame_tracks


def interpolate_trajectory(trajectory: list[tuple[int, np.ndarray]], nxt: tuple[int, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    """
    Perform linear interpolation between two trajectory points except the last point (nxt).

    Args:
        trajectory (list[tuple[int, np.ndarray]]): list of (frame_idx, detection) tuples
        nxt (tuple[int, np.ndarray]): Next detection point (frame_idx, detection)

    Returns:
        list[tuple[int, np.ndarray]]: list of interpolated points
    """
    extend: list[tuple[int, np.ndarray]] = []

    if len(trajectory) != 0:
        prv = trajectory[-1]
        assert prv[0] < nxt[0]
        prv_det = prv[1]
        nxt_det = nxt[1]
        dif_det = nxt_det - prv_det
        dif_det = dif_det.reshape(1, -1)

        scale = np.arange(1, nxt[0] - prv[0], dtype=np.float32).reshape(-1, 1) / (nxt[0] - prv[0])

        int_dets = (scale @ dif_det) + prv_det.reshape(1, -1)

        for idx, int_det in enumerate(int_dets):
            extend.append((prv[0] + idx + 1, int_det))

    return extend


def register_tracked_detections(
    tracked_dets: np.ndarray | list[tuple[float, float, float, float, int]],
    frame_idx: int,
    frame_tracks: dict[int, list[list[float]]],
    trajectories: dict[int, list[tuple[int, np.ndarray]]],
    interpolate: bool = True
):
    """
    Register tracked detections to frame tracks and trajectories.

    Args:
        tracked_dets (np.ndarray): Tracked detections
        frame_idx (int): Frame index
        frame_tracks (dict[int, list[list[float]]]): Frame tracks
        trajectories (dict[int, list[tuple[int, np.ndarray]]]): Trajectories
        interpolate (bool): Whether to perform trajectory interpolation (default: True)
    """

    if len(tracked_dets) == 0:
        return

    if frame_idx not in frame_tracks:
        frame_tracks[frame_idx] = []

    for track in tracked_dets:
        # SORT returns: [x1, y1, x2, y2, track_id]
        x1, y1, x2, y2, track_id = track
        track_id = int(track_id)

        # # Convert to detection format: [track_id, x1, y1, x2, y2]
        # detection = [track_id, x1, y1, x2, y2]

        # # Add to frame tracks
        # if frame_idx not in frame_tracks:
        #     frame_tracks[frame_idx] = []
        # frame_tracks[frame_idx].append(detection)

        if track_id not in trajectories:
            trajectories[track_id] = []
        box_array = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Add to trajectories for interpolation (if enabled)
        if interpolate:
            extend = interpolate_trajectory(trajectories[track_id], (frame_idx, box_array))
        else:
            extend = []

        # Add interpolated points to frame tracks
        for e in extend + [(frame_idx, box_array)]:
            e_frame_idx, e_box = e
            if e_frame_idx not in frame_tracks:
                frame_tracks[e_frame_idx] = []

            # Convert back to list format: [track_id, x1, y1, x2, y2]
            e_detection = [track_id, *e_box.tolist()]
            frame_tracks[e_frame_idx].append(e_detection)

            # Add interpolated points to trajectories
            trajectories[track_id].append((e_frame_idx, e_box))


def save_tracking_results(frame_tracks: dict[int, list[list[float]]], output_path: str | pathlib.Path):
    """
    Save tracking results to a JSONL file.

    Args:
        frame_tracks (dict[int, list[list[float]]]): Frame tracks
        output_path (str): Path to save the tracking results
    """
    with open(output_path, 'w') as f:
        frame_ids = frame_tracks.keys()
        if len(frame_ids) == 0:
            return

        first_idx = min(frame_ids)
        last_idx = max(frame_ids)

        for frame_idx in range(first_idx, last_idx + 1):
            if frame_idx not in frame_tracks:
                frame_tracks[frame_idx] = []

            frame_data = {
                "frame_idx": frame_idx,
                "tracks": frame_tracks[frame_idx]
            }
            f.write(json.dumps(frame_data) + '\n')


def get_track_color(track_id: int, track_ids: list[int] | None = None) -> tuple[int, int, int]:
    """
    Get a color for a track ID by cycling through the predefined colors.
    If track_ids is specified, only those track IDs get colors, others get grey.

    Args:
        track_id (int): Track ID
        track_ids (list[int] | None): List of track IDs to color (others will be grey)

    Returns:
        tuple[int, int, int]: BGR color tuple
    """
    # If track_ids is specified and this track_id is not in the list, return grey
    if track_ids is not None and track_id not in track_ids:
        return (128, 128, 128)  # Grey color in BGR format

    # Otherwise, use the normal color cycling
    color_index = track_id % len(TRACK_COLORS)
    return TRACK_COLORS[color_index]


def overlapi(interval1: tuple[int, int], interval2: tuple[int, int]):
    """
    Check if two 1D intervals overlap.

    Args:
        interval1 (tuple[int, int]): First interval as (start, end)
        interval2 (tuple[int, int]): Second interval as (start, end)

    Returns:
        bool: True if the intervals overlap, False otherwise
    """
    return (
        (interval1[0] <= interval2[0] <= interval1[1]) or
        (interval1[0] <= interval2[1] <= interval1[1]) or
        (interval2[0] <= interval1[0] <= interval2[1]) or
        (interval2[0] <= interval1[1] <= interval2[1])
    )


def overlap(b1, b2):
    """
    Check if two 2D bounding boxes overlap.

    Args:
        b1: First bounding box as (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right
        b2: Second bounding box as (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right

    Returns:
        bool: True if the bounding boxes overlap in both x and y dimensions, False otherwise
    """
    return overlapi((b1[0], b1[2]), (b2[0], b2[2])) and overlapi((b1[1], b1[3]), (b2[1], b2[3]))


def get_precision(tp: int, fp: int) -> float:
    """
    Calculate precision.

    Args:
        tp (int): True positives
        fp (int): False positives
    """
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)

def get_recall(tp: int, fn: int) -> float:
    """
    Calculate recall.

    Args:
        tp (int): True positives
        fn (int): False negatives
    """
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def get_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:

    """
    Calculate accuracy.

    Args:
        tp (int): True positives
        tn (int): True negatives
        fp (int): False positives
        fn (int): False negatives
    """
    if (tp + tn + fp + fn) == 0:
        return 0.0
    return (tp + tn) / (tp + tn + fp + fn)

def get_f1_score(tp: int, fp: int, fn: int) -> float:

    """
    Calculate F1 score.

    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives
    """
    if (get_precision(tp, fp) + get_recall(tp, fn)) == 0:
        return 0.0
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    return 2. * (precision * recall) / (precision + recall)


def load_classification_results(dataset: str, video_file: str,
                                tilesize: int, classifier: str, sample_rate: int = 1, verbose: bool = False) -> list:
    """
    Load classification results from the JSONL file generated by 020_exec_classify.py or 021_exec_classify_correct.py.

    Args:
        dataset (str): Dataset name
        video_file (str): Video file name
        tilesize (int): Tile size used for classification
        classifier (str): Classifier name to use
        sample_rate (int): Sample rate for frame sampling (default: 1, process all frames)
        verbose (bool): Whether to print verbose output

    Returns:
        list: List of frame classification results, each containing frame data and classifications

    Raises:
        FileNotFoundError: If no classification results file is found
    """
    from polyis.io import cache
    # Build path using centralized cache path builder
    results_file = cache.exec(dataset, 'relevancy', video_file, f'{classifier}_{tilesize}_{sample_rate}', 'score', 'score.jsonl')

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Classification results file not found: {results_file}")

    if verbose:
        print(f"Loading classification results from: {results_file}")

    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if verbose:
        print(f"Loaded {len(results)} frame classifications")
    return results


def load_pruned_classification_results(dataset: str, video_file: str,
                                        tilesize: int, classifier: str, tracker: str,
                                        tracking_accuracy_threshold: float,
                                        sample_rate: int = 1, verbose: bool = False) -> list:
    """
    Load pruned classification results from the JSONL file generated by p022_exec_prune_polyominoes.py.

    Args:
        dataset (str): Dataset name
        video_file (str): Video file name
        tilesize (int): Tile size used for classification
        classifier (str): Classifier name to use
        tracker (str): Tracker name used for pruning
        tracking_accuracy_threshold (float): Accuracy threshold used for pruning
        sample_rate (int): Sample rate for frame sampling (default: 1, process all frames)
        verbose (bool): Whether to print verbose output

    Returns:
        list: List of frame classification results after pruning

    Raises:
        FileNotFoundError: If no pruned classification results file is found
    """
    from polyis.io import cache
    # Build param string for the pruned output directory
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                tracker=tracker, tracking_accuracy_threshold=tracking_accuracy_threshold)
    # Build path using centralized cache path builder
    results_file = cache.exec(dataset, 'pruned-polyominoes', video_file, param_str, 'score', 'score.jsonl')

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Pruned classification results file not found: {results_file}")

    if verbose:
        print(f"Loading pruned classification results from: {results_file}")

    # Read all results and filter by sample rate
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                results.append(entry)

    if verbose:
        print(f"Loaded {len(results)} pruned frame classifications (sample_rate={sample_rate})")
    return results


def scale_to_percent(canvas_scale: float) -> int:
    # Convert a floating-point canvas scale to an integer percentage for stable folder names.
    return int(round(float(canvas_scale) * 100))


def format_tracking_accuracy_threshold(threshold: float) -> str:
    # Convert a float accuracy threshold to a 3-digit zero-padded percentage string.
    return f'{int(round(threshold * 100)):03d}'


def dataset_name_for_videoset(dataset: str, videoset: str) -> str:
    # Use a dedicated dataset name for validation split in evaluation outputs.
    if videoset == 'valid':
        return f'{dataset}-val'
    # Keep the original dataset name for all non-validation splits.
    return dataset


def split_dataset_name(dataset_name: str) -> tuple[str, str]:
    # Map dataset names with "-val" suffix back to base dataset and videoset.
    if dataset_name.endswith('-val'):
        return dataset_name[:-4], 'valid'
    # Default all non-suffixed dataset names to the test split for evaluation.
    return dataset_name, 'test'


def build_param_str(*, classifier: str, tilesize: int,
                    sample_rate: int | None = None,
                    tilepadding: str | None = None,
                    canvas_scale: float | None = None,
                    tracker: str | None = None,
                    tracking_accuracy_threshold: float | None = None) -> str:
    # Build a parameter string for pipeline stage directory naming.
    parts = [classifier, str(tilesize)]
    if sample_rate is not None:
        parts.append(str(sample_rate))
    if tracking_accuracy_threshold is not None:
        parts.append(format_tracking_accuracy_threshold(tracking_accuracy_threshold))
    if tilepadding is not None:
        parts.append(str(tilepadding))
    if canvas_scale is not None:
        parts.append(f's{scale_to_percent(canvas_scale)}')
    if tracker is not None:
        parts.append(tracker)
    return '_'.join(parts)


def parse_execution_param_str(param_str: str) -> dict[str, typing.Any]:
    # Split the encoded parameter string into underscore-delimited tokens.
    parts = param_str.split('_')
    # Require at least classifier and tile size tokens.
    assert len(parts) >= 2, f"Invalid parameter string: {param_str}"

    # Read classifier token.
    classifier = parts[0]
    # Parse tile size token.
    tilesize = int(parts[1])
    # Initialize cursor after required classifier/tile size tokens.
    cursor = 2

    # Initialize optional fields with defaults for missing dimensions.
    sample_rate: int | None = None
    tracking_accuracy_threshold: float | None = None
    tilepadding: str | None = None
    canvas_scale: float | None = None
    tracker: str | None = None

    # Parse optional sample rate token when present.
    if cursor < len(parts) and parts[cursor].isdigit():
        sample_rate = int(parts[cursor])
        cursor += 1

    # Parse optional threshold token (three-digit percentage) when present.
    if cursor < len(parts) and len(parts[cursor]) == 3 and parts[cursor].isdigit():
        tracking_accuracy_threshold = int(parts[cursor]) / 100.0
        cursor += 1

    # Collect remaining tokens for tilepadding/canvas_scale/tracker parsing.
    remaining = parts[cursor:]
    # Locate optional canvas-scale token (e.g., "s100") in remaining tokens.
    scale_idx = next((i for i, token in enumerate(remaining)
                      if token.startswith('s') and token[1:].isdigit()), None)

    # Handle formats that include a canvas-scale token (stages 030-060 outputs).
    if scale_idx is not None:
        # Parse optional tilepadding token that precedes canvas-scale.
        if scale_idx > 0:
            tilepadding = remaining[0]
        # Parse canvas-scale token to floating-point ratio.
        canvas_scale = int(remaining[scale_idx][1:]) / 100.0
        # Parse optional tracker token(s) that follow canvas-scale.
        if scale_idx + 1 < len(remaining):
            tracker = '_'.join(remaining[scale_idx + 1:])
    # Handle formats without canvas-scale token (e.g., stage 022 outputs).
    elif remaining:
        # Parse the remaining token(s) as tracker identifier.
        tracker = '_'.join(remaining)

    # Return the normalized parameter dictionary.
    return {
        'classifier': classifier,
        'tilesize': tilesize,
        'sample_rate': sample_rate,
        'tracking_accuracy_threshold': tracking_accuracy_threshold,
        'tilepadding': tilepadding,
        'canvas_scale': canvas_scale,
        'tracker': tracker,
        'param_str': param_str,
    }


def create_tracker(tracker_name: str, img_size: tuple[int, int], **override_params):
    """
    Create a tracker instance based on the specified algorithm.

    Args:
        tracker_name (str): Name of the tracking algorithm
        img_size (tuple[int, int]): Image size as (height, width)

    Returns:
        Tracker instance

    Raises:
        ValueError: If the tracker name is not supported
    """
    with open('configs/trackers.yaml', 'r') as f:
        configs = yaml.safe_load(f)
        if tracker_name == 'sort':
            # print(f"Creating SORT tracker with max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
            from polyis.b3d.sort import Sort
            config = configs['sort']
            return Sort(max_age=config['max_age'], min_hits=config['min_hits'], iou_threshold=config['iou_threshold'])
        if tracker_name == 'sortcython':
            from polyis.tracker.sort.cython.sort import Sort as SortCython
            config = configs['sort']
            return SortCython(max_age=config['max_age'], min_hits=config['min_hits'], iou_threshold=config['iou_threshold'])
        if tracker_name == 'ocsort':
            from polyis.tracker.ocsort.ocsort_wrapper import OCSort
            config = configs['ocsort']
            return OCSort(
                img_size=img_size,
                det_thresh=config['det_thresh'],
                max_age=config['max_age'],
                min_hits=config['min_hits'],
                iou_threshold=config['iou_threshold'],
                delta_t=config['delta_t'],
                asso_func=config['asso_func'],
                inertia=config['inertia'],
                use_byte=config['use_byte']
            )
        if tracker_name == 'ocsortcython':
            from polyis.tracker.ocsort.cython.ocsort_wrapper import OCSort as OCSortCython
            config = configs['ocsort']
            return OCSortCython(
                img_size=img_size,
                det_thresh=config['det_thresh'],
                max_age=config['max_age'],
                min_hits=config['min_hits'],
                iou_threshold=config['iou_threshold'],
                delta_t=config['delta_t'],
                asso_func=config['asso_func'],
                inertia=config['inertia'],
                use_byte=config['use_byte']
            )
        if tracker_name == 'bytetrack':
            from polyis.tracker.bytetrack import ByteTrack
            config = configs['bytetrack']
            return ByteTrack(
                img_size=img_size,
                track_thresh=config['track_thresh'],
                match_thresh=config['match_thresh'],
                track_buffer=override_params.get('track_buffer', config['track_buffer']),
                frame_rate=override_params.get('frame_rate', config['frame_rate']),
                # mot20=config['mot20']
            )
        if tracker_name == 'bytetrackcython':
            from polyis.tracker.bytetrack.cython.bytetrack_wrapper import ByteTrack as ByteTrackCython
            config = configs['bytetrack']
            return ByteTrackCython(
                img_size=img_size,
                track_thresh=config['track_thresh'],
                match_thresh=config['match_thresh'],
                track_buffer=override_params.get('track_buffer', config['track_buffer']),
                frame_rate=override_params.get('frame_rate', config['frame_rate']),
                # mot20=config['mot20']
            )
        else:
            raise ValueError(f"Unknown tracker: {tracker_name}")


def create_visualization_frame(frame: np.ndarray, tracks: list[list[float]], frame_idx: int,
                               trajectory_history: dict[int, list[tuple[int, int, int]]],
                               speed_up: int, track_ids: list[int] | None, detection_only: bool = False) -> np.ndarray | None:
    """
    Create a visualization frame by drawing bounding boxes and trajectories for all tracks.

    Args:
        frame (np.ndarray): Original video frame (H, W, 3)
        tracks (list[list[float]]): list of tracks for this frame
        frame_idx (int): Frame index for logging
        trajectory_history (dict[int, list[tuple[int, int, int]]]): History of track centers with frame timestamps
        speed_up (int): Speed up factor (process every Nth frame)
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, only show detections without trajectories, all boxes in green without track IDs

    Returns:
        np.ndarray | None: Frame with bounding boxes and trajectories drawn, or None if frame should be skipped
    """
    # First loop: Update trajectory history for all tracks
    for track in tracks:
        if len(track) >= 5:  # Ensure we have track_id, x1, y1, x2, y2
            track_id, x1, y1, x2, y2 = track[:5]
            track_id = int(track_id)

            # Calculate center of bounding box
            center_x = int((x1 + x2) // 2)
            center_y = int((y1 + y2) // 2)

            # Update trajectory history with frame timestamp
            if track_id not in trajectory_history:
                trajectory_history[track_id] = []
            trajectory_history[track_id].append((center_x, center_y, frame_idx))

    if frame_idx % speed_up != 0:
        return None

    # Create a copy of the frame for visualization
    vis_frame = frame.copy()

    if detection_only:
        # In detection-only mode, draw all bounding boxes in green without track IDs
        draw_track_bounding_boxes(vis_frame, tracks, track_ids, detection_only=True)
    else:
        # Draw bounding boxes and labels for tracks not in track_ids (grey)
        draw_track_bounding_boxes(vis_frame, tracks, [])

        # Draw all trajectories with gradual fading
        draw_trajectories(vis_frame, trajectory_history, frame_idx, track_ids)

        # Draw bounding boxes and labels for tracks in track_ids (colored)
        tracks_in_track_ids = [track for track in tracks if track[0] in (track_ids or [track[0]])]
        draw_track_bounding_boxes(vis_frame, tracks_in_track_ids, track_ids)

    # Draw frame index in the upper left corner
    frame_label = f"Frame: {frame_idx}"
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX,
                                                          font_scale, font_thickness)

    # Position text in upper left corner with padding
    padding = 10
    text_x = padding
    text_y = text_height + padding

    # Draw text background for better visibility
    cv2.rectangle(vis_frame, (text_x - 2, text_y - text_height - 2),
                  (text_x + text_width + 2, text_y + baseline + 2),
                  (0, 0, 0), -1)

    # Draw text in white
    cv2.putText(vis_frame, frame_label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return vis_frame


def draw_track_bounding_boxes(vis_frame: np.ndarray, tracks: list[list[float]],
                              track_ids: list[int] | None, detection_only: bool = False):
    """
    Draw bounding boxes and labels for all tracks on the visualization frame.

    Args:
        vis_frame (np.ndarray): Frame to draw on (modified in place)
        tracks (list[list[float]]): List of tracks for this frame
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, use green color for all boxes and hide track IDs
    """
    for track in tracks:
        assert len(track) >= 5, f"Track must have at least 5 elements: {track}"
        track_id, x1, y1, x2, y2 = track[:5]

        # Convert to integers for drawing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        color = (0, 255, 0) if detection_only else get_track_color(track_id, track_ids)

        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Draw track ID label only if not in detection-only mode
        if detection_only:
            continue

        label = str(track_id)
        font_scale = 0.6
        font_thickness = 2

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                font_scale, font_thickness)

        # Position text above the bounding box
        text_x = x1
        text_y = max(y1 - 10, text_height + 5)

        # Draw text background for better visibility
        cv2.rectangle(vis_frame, (text_x - 2, text_y - text_height - 2),
                        (text_x + text_width + 2, text_y + baseline + 2),
                        color, -1)

        # Draw text
        text_color = (254, 254, 254) if sum(color) > (255 * 2) else (1, 1, 1)
        cv2.putText(vis_frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)


def draw_trajectories(vis_frame: np.ndarray, trajectory_history: dict[int, list[tuple[int, int, int]]],
                      frame_idx: int, track_ids: list[int] | None):
    """
    Draw all trajectories with gradual fading on the visualization frame.

    Args:
        vis_frame (np.ndarray): Frame to draw on (modified in place)
        trajectory_history (dict[int, list[tuple[int, int, int]]]): History of track centers with frame timestamps
        frame_idx (int): Current frame index for fade calculations
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
    """
    ids_trajectories = sorted(trajectory_history.items(), key=lambda x: x[0] in (track_ids or []))
    for track_id, trajectory in ids_trajectories:
        if len(trajectory) > 1:
            color = get_track_color(track_id, track_ids)

            # Calculate fade parameters
            max_fade_frames = 30  # Number of frames for complete fade after track ends
            current_time = frame_idx

            # Check if track is still active (within last 5 frames)
            track_is_active = trajectory and current_time - trajectory[-1][2] <= 5

            # Calculate fade alpha for the entire trajectory
            if track_is_active:
                # Track is active - full opacity
                alpha = 1.0
            else:
                # Track has ended - calculate fade based on time since last detection
                time_since_end = current_time - trajectory[-1][2]
                if time_since_end >= max_fade_frames:
                    alpha = 0.0  # Completely faded
                else:
                    alpha = 1.0 - (time_since_end / max_fade_frames)

            # Only draw if trajectory is still visible
            if alpha > 0.01:
                # Apply alpha to color for the entire trajectory
                line_color = tuple(int(c * alpha) for c in color)
                point_color = tuple(int(c * alpha) for c in color)

                # Draw trajectory lines
                for i in range(1, len(trajectory)):
                    prev_center = trajectory[i-1]
                    curr_center = trajectory[i]

                    # Draw line
                    cv2.line(vis_frame, (prev_center[0], prev_center[1]),
                             (curr_center[0], curr_center[1]), line_color, 2)

                    # Draw trajectory points
                    point_radius = max(1, int(3 * alpha))
                    cv2.circle(vis_frame, (prev_center[0], prev_center[1]), point_radius, point_color, -1)

                # Draw final point
                final_center = trajectory[-1]
                cv2.circle(vis_frame, (final_center[0], final_center[1]), 3, point_color, -1)


def to_h264(input_path: str):
    """
    Convert video to H.264 codec with .h264 extension using FFMPEG.

    Args:
        input_path: Path to the input video file
    """
    # Create output path with .h264 extension
    # base_path = os.path.splitext(input_path)[0]
    output_path = f"{input_path[:-len('.mp4')]}.h264.mp4"

    # Run FFMPEG command to convert to H.264 (silent, optimized for small file size)
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-loglevel', 'quiet',  # silence FFMPEG output
        '-i', input_path,  # input file
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'fast',  # encoding preset
        '-crf', '28',  # constant rate factor (lower quality, smaller file)
        '-profile:v', 'baseline',  # baseline profile for better compatibility
        '-level', '3.0',  # H.264 level for broader device support
        '-movflags', '+faststart',  # optimize for streaming/web playback
        '-pix_fmt', 'yuv420p',  # pixel format for maximum compatibility
        '-tune', 'fastdecode',  # optimize for faster decoding
        output_path
    ]

    subprocess.run(cmd, capture_output=True, text=True, check=True)


def create_tracking_visualization(video_path: str, tracking_results: dict[int, list[list[float]]],
                                  output_path: str, speed_up: int, process_id: int, progress_queue=None,
                                  track_ids: list[int] | None = None, detection_only: bool = False):
    """
    Create a visualization video showing tracking results overlaid on the original video.

    Args:
        video_path (str): Path to the input video file
        tracking_results (dict[int, list[list[float]]]): Tracking results from load_tracking_results
        output_path (str): Path where the output visualization video will be saved
        speed_up (int): Speed up factor for visualization (process every Nth frame)
        process_id (int): Process ID for logging
        progress_queue: Queue for progress updates
        track_ids (list[int] | None): List of track IDs to color (others will be grey)
        detection_only (bool): If True, only show detections without trajectories, all boxes in green without track IDs
    """
    # print(f"Creating tracking visualization for video: {video_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize trajectory history for all tracks with frame timestamps
    trajectory_history: dict[int, list[tuple[int, int, int]]] = {}  # track_id -> [(x, y, frame_idx), ...]

    # Initialize frame_idx for exception handling
    frame_idx = 0

    # Open video and get properties
    with VideoCapture(video_path) as cap:
        # Get video properties
        frame_count = cap.frame_count
        fps = cap.fps
        width = cap.width
        height = cap.height

        # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")

        # Send initial progress update
        if progress_queue is not None:
            progress_queue.put((f'cuda:{process_id}', {
                'description': video_path[-min(20, len(video_path) - 1):],
                'completed': 0,
                'total': frame_count
            }))

        # Create video writer (writes H.264 directly)
        with VideoWriter(output_path, width, height, fps) as writer:
            # print(f"Creating visualization video with {frame_count} frames at {fps} FPS")

            # Process each frame
            for frame_idx in range(frame_count):
                # Read frame
                frame = cap.read()
                if frame is None:
                    break

                # Get tracking results for this frame
                tracks = tracking_results.get(frame_idx, [])

                # Create visualization frame with trajectory history
                vis_frame = create_visualization_frame(frame, tracks, frame_idx, trajectory_history,
                                                       speed_up, track_ids, detection_only)

                # Write frame to video
                if vis_frame is not None:
                    writer.write(vis_frame)

                # Send progress update
                if progress_queue is not None:
                    progress_queue.put((f'cuda:{process_id}', {'completed': frame_idx + 1}))


def get_overlapping_tiles(
    x1: float, y1: float, x2: float, y2: float,
    tile_size: int, grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    """
    Compute tile coordinate ranges overlapping a bounding box.

    Args:
        x1, y1, x2, y2: Bounding box coordinates
        tile_size: Size of each tile in pixels
        grid_h: Number of tile rows
        grid_w: Number of tile columns

    Returns:
        (row_start, row_end, col_start, col_end) inclusive tile ranges
    """
    col_start = max(0, int(x1) // tile_size)
    col_end = min(grid_w - 1, int(x2) // tile_size)
    row_start = max(0, int(y1) // tile_size)
    row_end = min(grid_h - 1, int(y2) // tile_size)
    return row_start, row_end, col_start, col_end


def mark_detections(
    detections: list[list[float]],
    width: int,
    height: int,
    tilesize: int,
    detection_slice: slice = slice(-4, None)
) -> np.ndarray:
    """
    Mark tiles as relevant based on groundtruth detections.
    This function creates a bitmap where 1 indicates a tile with detection and 0 indicates no detection.

    Args:
        detections (list[list[float]]): List of bounding boxes, each formatted as [tracking_id, x1, y1, x2, y2]
        width (int): Frame width
        height (int): Frame height
        tilesize (int): Size of each tile
        detection_slice (slice): Slice of the bounding box to use for marking detections

    Returns:
        np.ndarray: 2D array representing the grid of tiles, where 1 indicates relevant tiles
    """
    grid_h = height // tilesize
    grid_w = width // tilesize
    bitmap = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for bbox in detections:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox[detection_slice]

        # Get overlapping tile ranges and mark them
        row_start, row_end, col_start, col_end = get_overlapping_tiles(
            x1, y1, x2, y2, tilesize, grid_h, grid_w
        )
        bitmap[row_start:row_end+1, col_start:col_end+1] = 1

    return bitmap


def create_timer(file: typing.TextIO, meta: dict | None = None):
    row = []
    def timer(op: str) -> Timer:
        return Timer(op, row)
    def flush():
        nonlocal row
        if row:
            file.write(json.dumps(row) + '\n')
            row = []
    return timer, flush


class Timer:
    def __init__(self, op: str, row: list[dict]):
        self.start_time = time.time_ns() / 1e6
        self.op = op
        self.row = row

    def __enter__(self):
        self.start_time = time.time_ns() / 1e6

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time_ns() / 1e6
        elapsed_time = self.end_time - self.start_time
        self.row.append({'op': self.op, 'time': elapsed_time})


def progress_bars(command_queue: "mp.Queue", num_workers: int, num_tasks: int,
                  refresh_per_second: float = 1, script_name: str = ""):
    with progress.Progress(
        progress.TimeElapsedColumn(),
        progress.MofNCompleteColumn(),
        progress.BarColumn(),
        # "[progress.percentage]{task.percentage:>3.0f}%",
        # progress.TimeRemainingColumn(),
        "[progress.description]{task.description}",
        refresh_per_second=refresh_per_second,
    ) as p:
        bars: dict[str, progress.TaskID] = {}
        task_id = script_name.split('_')[0]
        task_name = script_name.split('_', 2)[2][:-len('.py')]
        task_str = f"[green]{task_id} {task_name}"
        overall_progress = p.add_task(task_str, total=num_tasks, completed=-num_workers)
        bars['overall'] = overall_progress
        for gpu_id in range(num_workers):
            bars[f'cuda:{gpu_id}'] = p.add_task("")

        while True:
            val = command_queue.get()
            if val is None:
                break
            progress_id, kwargs = val
            # if kwargs.get('remove', False):
            #     p.remove_task(bars[progress_id])
            #     # bars.pop(progress_id)
            # else:
            p.update(bars[progress_id], **kwargs)

        # remove all tasks
        for _, task_id in bars.items():
            p.remove_task(task_id)
        bars.clear()


class ProgressBar:
    """
    Context manager for handling progress bars with multiprocessing support.

    Usage:
        with ProgressBar(num_workers=4, num_tasks=100) as pb:
            # Use pb.command_queue to send progress updates
            # Use pb.worker_id_queue to manage worker IDs
            pass
    """

    def __init__(self, num_workers: int, num_tasks: int, refresh_per_second: float = 1, off: bool = False):
        """
        Initialize the progress bar manager.

        Args:
            num_workers (int): Number of worker processes/GPUs
            num_tasks (int): Total number of tasks to process
            refresh_per_second (float): Refresh rate for progress bars
        """
        off = True
        if off:
            num_workers = 1
        self.num_workers = min(num_workers, num_tasks)
        self.num_tasks = num_tasks
        self.refresh_per_second = refresh_per_second

        # Get the running script name for progress bar display
        script_path = sys.modules['__main__'].__file__
        self.script_name = os.path.basename(script_path) if script_path else ""

        # Initialize queues
        self.command_queue: "mp.Queue[tuple[str, dict] | None]" = mp.Queue()
        self.worker_id_queue: "mp.Queue[int]" = mp.Queue(maxsize=num_workers)
        self.progress_process: mp.Process | None = None
        # if not off:
        self.progress_process = mp.Process(
            target=progress_bars,
            args=(self.command_queue, self.num_workers,
                self.num_tasks, self.refresh_per_second, self.script_name),
            daemon=True
        )

    def __enter__(self):
        """Enter the context manager - set up queues and start progress process."""
        # Get the running script name and create log directory
        script_path = sys.modules['__main__'].__file__
        assert script_path is not None
        script_name = os.path.splitext(os.path.basename(script_path))[0]

        # Create log directory: {CACHE_DIR}/LOGS/{script_name}
        self.log_dir = os.path.join(CACHE_DIR, 'LOGS', script_name)
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Populate worker ID queue
        for worker_id in range(self.num_workers):
            self.worker_id_queue.put(worker_id)

        # Start progress bars process
        if self.progress_process is not None:
            self.progress_process.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager - clean up progress bars and terminate process."""
        # Signal progress bars to stop
        self.command_queue.put(None)

        if self.progress_process is not None:
            # Wait for progress process to finish and terminate it
            self.progress_process.join(timeout=5)  # Wait up to 5 seconds
            if self.progress_process.is_alive():
                self.progress_process.terminate()
                self.progress_process.join(timeout=2)  # Give it time to terminate

            # Force kill if still alive
            if self.progress_process.is_alive():
                self.progress_process.kill()
                self.progress_process.join()

    def update_overall_progress(self, advance: int = 1):
        """Update the overall progress bar."""
        self.command_queue.put(('overall', {'advance': advance}))

    def get_worker_id(self):
        """Get a worker ID from the worker ID queue."""
        return self.worker_id_queue.get()

    def run(self, func: typing.Callable[[int, mp.Queue], None]):
        """Run func in a new process with a worker ID."""
        worker_id = self.worker_id_queue.get()
        self.update_overall_progress(1)

        # Generate log file path if log_dir is set and no explicit stdout_file provided
        stdout_file = os.path.join(self.log_dir, f'worker_{worker_id}.stdout')
        stderr_file = os.path.join(self.log_dir, f'worker_{worker_id}.stderr')
        log_file = os.path.join(self.log_dir, f'worker_{worker_id}.log')
        process = mp.Process(target=ProgressBar.run_with_worker_id,
                             args=(func, worker_id, self.command_queue,
                                   self.worker_id_queue, stdout_file, stderr_file, log_file))
        process.start()
        return process

    def run_all(self, funcs: list[typing.Callable[[int, mp.Queue], None]]):
        """Run all funcs in a new process with a worker ID."""
        with self:
            processes: list[mp.Process] = []
            for func in funcs:
                processes.append(self.run(func))

            for _ in range(self.num_workers):
                worker_id = self.get_worker_id()
                self.update_overall_progress(1)
                self.command_queue.put((f'cuda:{worker_id}',
                                        {'remove': True}))

            for process in processes:
                process.join()
                process.terminate()


    @staticmethod
    def run_with_worker_id(func: typing.Callable[[int, mp.Queue], None], worker_id: int,
                           command_queue: mp.Queue, worker_id_queue: mp.Queue,
                           stdout_file: str, stderr_file: str, log_file: str):
        """Run func with a worker ID and command queue. Redirect stdout, stderr, and logging to files."""
        try:
            with contextlib.ExitStack() as stack:
                # Open file in append mode
                f_out = stack.enter_context(open(stdout_file, 'a', encoding='utf-8'))

                # Write function name and parameters before redirecting
                func_name = ProgressBar._get_function_name(func)
                func_params = ProgressBar._get_function_parameters(func)
                f_out.write(f"\n\n{'='*80}\n")
                f_out.write(f"Function: {func_name} ( {func_params} )\n")
                f_out.write(f"{'='*80}\n")
                f_out.flush()
                stack.enter_context(contextlib.redirect_stdout(f_out))

                # Open file in append mode
                f_err = stack.enter_context(open(stderr_file, 'a', encoding='utf-8'))
                stack.enter_context(contextlib.redirect_stderr(f_err))

                # Configure logging to write to log file
                # Open log file in append mode
                f_log = stack.enter_context(open(log_file, 'a', encoding='utf-8'))

                # Get root logger and remove all existing handlers
                root_logger = logging.getLogger()
                # Store original handlers to restore later
                original_handlers = root_logger.handlers[:]
                root_logger.handlers.clear()

                # Create file handler for logging
                file_handler = logging.StreamHandler(f_log)
                file_handler.setLevel(logging.DEBUG)
                # Use a format that includes timestamp, level, and message
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)

                # Add file handler to root logger
                root_logger.addHandler(file_handler)
                root_logger.setLevel(logging.DEBUG)

                # Ensure all loggers inherit from root
                logging.captureWarnings(True)

                try:
                    func(worker_id, command_queue)
                finally:
                    # Restore original handlers
                    root_logger.handlers.clear()
                    root_logger.handlers.extend(original_handlers)
        finally:
            kwargs = {'completed': 0, 'description': 'Done', 'total': 1}
            command_queue.put((f'cuda:{worker_id}', kwargs))
            worker_id_queue.put(worker_id)

    @staticmethod
    def _get_function_name(func: typing.Callable) -> str:
        """Get the name of a function, handling functools.partial."""
        if isinstance(func, functools.partial):
            return func.func.__name__
        return func.__name__

    @staticmethod
    def _get_function_parameters(func: typing.Callable) -> str:
        """Get the parameters of a function as a string, handling functools.partial."""
        assert isinstance(func, functools.partial)
        # Get the underlying function
        underlying_func = func.func
        # Get bound arguments
        bound_args = func.keywords.copy() if func.keywords else {}
        # Add positional arguments
        sig = inspect.signature(underlying_func)
        param_names = list(sig.parameters.keys())
        for i, arg in enumerate(func.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg

        # Format as string
        param_strs = [f"{k}={repr(v)}" for k, v in bound_args.items()]
        return ", ".join(param_strs)


def gcp_run(funcs: list[typing.Callable[[int, mp.Queue], None]]):
    """
    Run a list of functions in a GCP instance.

    Args:
        funcs (list[typing.Callable[[int, mp.Queue], None]]): List of functions to run
    """
    commands = []
    for func in funcs:
        assert isinstance(func, functools.partial)
        args: tuple = func.args
        func_name = func.func.__name__
        script: str = func.func.gcp  # type: ignore
        args_str = ' '.join(str(arg) for arg in args)
        command = f"python ./scripts/{script} {func_name} {args_str}"
        commands.append(command)

    command_funcs = [functools.partial(subprocess.run, command, shell=True, check=True, capture_output=True, text=True) for command in commands]
    processes = []
    for command_func in command_funcs:
        process = mp.Process(target=command_func)
        process.start()
        processes.append(process)

    for process in progress.track(processes):
        process.join()
        process.terminate()


def load_tradeoff_data(dataset: str):
    """
    Load the canonical split-level tradeoff table for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        pd.DataFrame: Split-level tradeoff rows for the dataset
    """
    # Construct the canonical split-level tradeoff directory path.
    tradeoff_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '090_tradeoff')
    # Resolve the canonical split-level tradeoff CSV.
    tradeoff_path = os.path.join(tradeoff_dir, 'tradeoff.csv')
    # Fail fast when the canonical tradeoff CSV is missing.
    assert os.path.exists(tradeoff_path), \
        f"Tradeoff data not found: {tradeoff_path}. " \
        "Please run p130_tradeoff_compute.py first."

    # Load the canonical split-level tradeoff CSV.
    import pandas as pd
    tradeoff = pd.read_csv(tradeoff_path)
    # Log the loaded split-level row count for traceability.
    print(f"Loaded split-level tradeoff data: {len(tradeoff)} rows from {tradeoff_path}")

    return tradeoff


def split_tradeoff_variants(tradeoff_df: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Split the canonical tradeoff table into Polytris and naive subsets.

    Args:
        tradeoff_df: Canonical split-level tradeoff table

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (polytris_rows, naive_rows)
    """
    # Work on a copy so callers keep their original frame unchanged.
    tradeoff_df = tradeoff_df.copy()
    # Backfill the legacy contract when variant is absent in older data.
    if 'variant' not in tradeoff_df.columns:
        tradeoff_df['variant'] = 'polytris'

    # Select the real Polytris search-space rows.
    polytris_df = tradeoff_df[tradeoff_df['variant'] == 'polytris'].copy()
    # Select the dedicated naive baseline rows.
    naive_df = tradeoff_df[tradeoff_df['variant'] == 'naive'].copy()

    return polytris_df, naive_df


def load_all_datasets_tradeoff_data(datasets: list[str], system_name: str | None = None):
    """
    Load tradeoff data from all datasets and combine into a single DataFrame.

    Args:
        datasets: list of dataset names
        system_name: Optional system name to add as a column (e.g., 'Polytris')

    Returns:
        pd.DataFrame: Combined split-level tradeoff rows from all datasets
    """
    all_tradeoff = []

    for dataset in datasets:
        # Load the canonical split-level tradeoff data for the current dataset.
        tradeoff = load_tradeoff_data(dataset)
        # Backfill the dataset column only when older data omitted it.
        if 'dataset' not in tradeoff.columns:
            tradeoff['dataset'] = dataset
        # Add the root dataset column for optional downstream diagnostics.
        tradeoff['dataset_root'] = dataset

        # Add the requested system label when the caller wants a shared schema.
        if system_name is not None:
            tradeoff['system'] = system_name

        all_tradeoff.append(tradeoff)

    # Combine the dataset-local tradeoff tables into one shared DataFrame.
    import pandas as pd
    combined_df = pd.concat(all_tradeoff, ignore_index=True)
    print(f"Combined tradeoff data from {len(datasets)} datasets: {len(combined_df)} total rows")

    return combined_df


def print_best_data_points(df_combined: "pd.DataFrame", metrics_list: list[str],
                          x_column: str, plot_suffix: str, include_system: bool = False):
    """
    Print the best data point (highest accuracy, faster than baseline) for each dataset and metric as tables.

    Args:
        df_combined: Combined DataFrame with data from all datasets (already merged with naive data)
        metrics_list: list of metrics to analyze
        x_column: Column name for x-axis data (runtime or throughput)
        plot_suffix: Suffix for the analysis type ('runtime' or 'throughput')
        include_system: Whether to include the 'system' column in output (default: False)
    """
    import pandas as pd

    print(f"\n=== Best Data Points Analysis ({plot_suffix.upper()}) ===")

    # Naive column is automatically created from merge with suffix '_naive'
    naive_column = f'{x_column}_naive'

    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'HOTA_HOTA'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'MOTA_MOTA'
            metric_name = 'MOTA'
        else:
            continue

        print(f"\n--- {metric_name} Analysis ---")

        # Collect results for this metric
        results = []

        for dataset in df_combined['dataset'].unique():
            dataset_data = df_combined[df_combined['dataset'] == dataset]

            # Filter data points that are faster than baseline for this dataset
            faster_than_baseline = dataset_data[dataset_data[x_column] < dataset_data[naive_column]]

            if len(faster_than_baseline) == 0:
                # If no points are faster than baseline, use the fastest point
                assert isinstance(dataset_data, pd.DataFrame), \
                    f"dataset_data should be a DataFrame, got {type(dataset_data)}"
                best_point = dataset_data.loc[dataset_data[x_column].idxmin()]
            else:
                # Find the point with highest accuracy among those faster than baseline
                assert isinstance(faster_than_baseline, pd.DataFrame), \
                    f"faster_than_baseline should be a DataFrame, got {type(faster_than_baseline)}"
                best_point = faster_than_baseline.loc[faster_than_baseline[accuracy_col].idxmax()]

            # Calculate speed improvement
            naive_runtime = best_point[naive_column]
            best_runtime = best_point[x_column]
            speedup = naive_runtime / best_runtime if best_runtime > 0 else 0

            result = {
                'Dataset': dataset,
            }

            if include_system:
                result['System'] = best_point['system']

            result[f'{metric_name} Score'] = f"{best_point[accuracy_col]:.2f}"
            result['Speedup'] = f"{speedup:.2f}"

            results.append(result)

        # Create and print table for this metric
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        else:
            print("No results found.")


def tradeoff_scatter_and_naive_baseline(base_chart: "alt.Chart", x_column: str, x_title: str,
                                        accuracy_col: str, metric_name: str,
                                        size_range: tuple[int, int] = (20, 200), scatter_opacity: float = 0.7,
                                        size: int | None = None, baseline_stroke_width: int = 2,
                                        shape_field: str = 'tilepadding',
                                        baseline_opacity: float = 0.8, size_field: str = 'tilesize') -> "tuple[alt.Chart, alt.LayerChart]":
    """
    Create both a scatter plot and naive baseline visualization with common styling.

    Args:
        base_chart: Base Altair chart
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        accuracy_col: Column name for accuracy data
        metric_name: Name of the metric (e.g., 'HOTA', 'MOTA')
        naive_column: Column name for naive baseline data
        size_range: Tuple of (min, max) for tile size scale
        scatter_opacity: Opacity for the scatter points
        size: Fixed size for scatter points (if None, uses size_field encoding)
        shape_field: Column name for shape encoding (default: 'tilepadding')
        baseline_stroke_width: Width of the baseline rule line
        baseline_opacity: Opacity of the baseline rule line
        size_field: Column name for size encoding (default: 'tilesize')

    Returns:
        tuple[alt.Chart, alt.Chart]: Tuple of (scatter_plot, naive_baseline)
    """
    import altair as alt
    # Create scatter plot
    scale = {'scale': alt.Scale(domain=[0, 1])} if metric_name != 'Count' else {}
    scatter = base_chart.mark_point(opacity=scatter_opacity).encode(
        x=alt.X(f'{x_column}:Q', title=x_title),
        y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score', **scale),  # type: ignore
        color=alt.Color('classifier:N', title='Classifier'),
        tooltip=['video', 'classifier', size_field, 'sample_rate', 'tilepadding', x_column, accuracy_col]
    ).properties(
        width=150,
        height=150
    )

    # Add size encoding only if no fixed size is provided
    if size is None:
        scatter = scatter.encode(size=alt.Size(f'{size_field}:O',
                                 title='Tile Size',
                                 scale=alt.Scale(range=size_range)))

    if shape_field is not None:
        scatter = scatter.encode(shape=alt.Shape(f'{shape_field}:O', title='Tile Padding'))

    # Create naive baseline as a point at 1.0 accuracy score
    baseline = base_chart.mark_point(
        color='red',
        fill='red',
        size=20,
        opacity=baseline_opacity,
    ).encode(
        x=f'{x_column}_naive:Q',
        y=alt.value(1.0)  # Fixed at 1.0 accuracy score
    )

    # Create annotation text for the baseline
    baseline_annotation = base_chart.mark_text(
        align='right',
        baseline='top',
        fontSize=12,
        fontWeight='bold',
        color='red',
        dy=3,
        dx=15,
        lineHeight=10
    ).encode(
        x=f'{x_column}_naive:Q',
        y=alt.value(1.0),  # Position at 1.0 accuracy score
        text=alt.value(['Without', 'Optimization'])
    )

    return scatter, baseline + baseline_annotation


STR_NA = '_NA_'
INT_NA = 0


OPTIMAL_PARAMS = {
    'jnc0': {
        'classifier': 'YoloN',
        'tilesize': 60,
        'tilepadding': 'unpadded',
    },
    'jnc2': {
        'classifier': 'YoloN',
        'tilesize': 60,
        'tilepadding': 'unpadded',
    },
    'jnc6': {
        'classifier': 'YoloN',
        'tilesize': 60,
        'tilepadding': 'unpadded',
    },
    'jnc7': {
        'classifier': 'YoloN',
        'tilesize': 60,
        'tilepadding': 'unpadded',
    },
    'caldot1': {
        'classifier': 'ShuffleNet05',
        'tilesize': 60,
        'tilepadding': 'padded',
    },
    'caldot2': {
        'classifier': 'MobileNetS',
        'tilesize': 60,
        'tilepadding': 'padded',
    },
}

CHOSEN_PARAMS = {
    'jnc0': [
        {'classifier': 'YoloN', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
    'jnc2': [
        {'classifier': 'YoloN', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
    'jnc6': [
        {'classifier': 'YoloN', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
    'jnc7': [
        {'classifier': 'YoloN', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'unpadded'},
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
    'caldot1': [
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'padded'},
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'padded'},
        {'classifier': 'YoloN', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
    'caldot2': [
        {'classifier': 'MobileNetS', 'tilesize': 60, 'tilepadding': 'padded'},
        {'classifier': 'ShuffleNet05', 'tilesize': 60, 'tilepadding': 'unpadded'},
    ],
}


VIDEO_SETS = ['train', 'valid', 'test']
PREFIX_TO_VIDEOSET = {
    'tr': 'train',
    'va': 'valid',
    'te': 'test',
}


PARAMS = [
    'classifier',
    'tilesize',
    'tilepadding',
]

TilePadding = typing.Literal['none', 'plus', 'tl', 'tr', 'bl', 'br', 'square']
TILEPADDING_MODES: "dict[TilePadding, int]" = {
    'none': 0,
    'plus': 1,
    # 'tl': 2,
    'tr': 3,
    'bl': 4,
    # 'br': 5,
    'square': 6,
}
TILEPADDING_MAPS: "dict[TilePadding, int]" = TILEPADDING_MODES

ParamTypes = tuple[str, int, str]


METRICS = [
    'HOTA',
    # 'CLEAR',
    # 'Identity',
    'Count',
]


DATASETS_TO_TEST = [
    'jnc0',
    'jnc2',
    'jnc6',
    'jnc7',
    # 'caldot1-yolov5',
    # 'caldot2-yolov5',
    'caldot1',
    'caldot2',
]


CLASSIFIERS_TO_TEST = [
    # 'SimpleCNN',
    'YoloN',
    # 'YoloS',
    # 'YoloM',
    # 'YoloL',
    # 'YoloX',
    'ShuffleNet05',
    # 'ShuffleNet20',
    # 'MobileNetL',
    'MobileNetS',
    # 'WideResNet50',
    # 'WideResNet101',
    # 'ResNet18',
    # 'ResNet101',
    # 'ResNet152',
    # 'EfficientNetS',
    # 'EfficientNetL',
]

CLASSIFIERS = CLASSIFIERS_TO_TEST + ['Perfect']

CLASSIFIERS_CHOICES = [
    # Cutsom CNNs
    'SimpleCNN',

    # YOLOv11 models
    'YoloN',
    'YoloS',
    'YoloM',
    'YoloL',
    'YoloX',

    # ShuffleNet models
    'ShuffleNet05',
    'ShuffleNet20',

    # MobileNet models
    'MobileNetL',
    'MobileNetS',

    # ResNet models
    'ResNet18',
    'ResNet101',
    'ResNet152',

    # WideResNet models
    'WideResNet50',
    'WideResNet101',

    # EfficientNet models
    'EfficientNetS',
    'EfficientNetL',
]


def get_classifier_from_name(classifier_name: str):
    """
    Get the classifier class based on the classifier name.

    Args:
        classifier_name (str): Name of the classifier to use

    Returns:
        The classifier class

    Raises:
        ValueError: If the classifier is not supported
    """
    if classifier_name == 'SimpleCNN':
        from polyis.models.classifier.simple_cnn import SimpleCNN
        return SimpleCNN
    elif classifier_name == 'YoloN':
        from polyis.models.classifier.yolo import YoloN
        return YoloN
    elif classifier_name == 'YoloS':
        from polyis.models.classifier.yolo import YoloS
        return YoloS
    elif classifier_name == 'YoloM':
        from polyis.models.classifier.yolo import YoloM
        return YoloM
    elif classifier_name == 'YoloL':
        from polyis.models.classifier.yolo import YoloL
        return YoloL
    elif classifier_name == 'YoloX':
        from polyis.models.classifier.yolo import YoloX
        return YoloX
    elif classifier_name == 'ShuffleNet05':
        from polyis.models.classifier.shufflenet import ShuffleNet05
        return ShuffleNet05
    elif classifier_name == 'ShuffleNet20':
        from polyis.models.classifier.shufflenet import ShuffleNet20
        return ShuffleNet20
    elif classifier_name == 'MobileNetL':
        from polyis.models.classifier.mobilenet import MobileNetL
        return MobileNetL
    elif classifier_name == 'MobileNetS':
        from polyis.models.classifier.mobilenet import MobileNetS
        return MobileNetS
    elif classifier_name == 'WideResNet50':
        from polyis.models.classifier.wide_resnet import WideResNet50
        return WideResNet50
    elif classifier_name == 'WideResNet101':
        from polyis.models.classifier.wide_resnet import WideResNet101
        return WideResNet101
    elif classifier_name == 'ResNet152':
        from polyis.models.classifier.resnet import ResNet152
        return ResNet152
    elif classifier_name == 'ResNet101':
        from polyis.models.classifier.resnet import ResNet101
        return ResNet101
    elif classifier_name == 'ResNet18':
        from polyis.models.classifier.resnet import ResNet18
        return ResNet18
    elif classifier_name == 'EfficientNetS':
        from polyis.models.classifier.efficientnet import EfficientNetS
        return EfficientNetS
    elif classifier_name == 'EfficientNetL':
        from polyis.models.classifier.efficientnet import EfficientNetL
        return EfficientNetL
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


class FakeQueue(queue.Queue):
    def __init__(self):
        pass

    def put(self, item, block: bool = True, timeout: float | None = None):
        pass


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


CONFIG_DIR = 'configs'


def get_config(config_name: str | None = None):
    config_name = config_name or os.environ.get('CONFIG', 'global.yaml')
    config_path = os.path.join(CONFIG_DIR, config_name)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
