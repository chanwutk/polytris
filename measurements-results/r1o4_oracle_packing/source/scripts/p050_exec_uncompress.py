#!/usr/local/bin/python

import argparse
import itertools
import json
import os
import shutil
import numpy as np
import cv2
import multiprocessing as mp
from functools import partial
from typing import Callable

from polyis.utilities import ProgressBar, create_timer, get_config, get_num_frames, build_param_str, TilePadding
from polyis.io import cache, store
from polyis.pareto import build_pareto_combo_filter


CONFIG = get_config()
DATASETS: list[str] = CONFIG['EXEC']['DATASETS']
CLASSIFIERS: list[str] = CONFIG['EXEC']['CLASSIFIERS']
TILE_SIZES: list[int] = CONFIG['EXEC']['TILE_SIZES']
SAMPLE_RATES: list[int] = CONFIG['EXEC']['SAMPLE_RATES']
TILEPADDING_MODES: list[TilePadding] = CONFIG['EXEC']['TILEPADDING_MODES']
CANVAS_SCALES: list[float] = CONFIG['EXEC']['CANVAS_SCALE']
TRACKERS: list[str] = CONFIG['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS: list[float] = CONFIG['EXEC']['TRACKING_ACCURACY_THRESHOLDS']
RELEVANCE_THRESHOLDS: list[float] = CONFIG['EXEC']['RELEVANCE_THRESHOLDS']


def parse_args():
    parser = argparse.ArgumentParser(description='Unpack compressed detections from 040_exec_detect.py')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    return parser.parse_args()


def load_mapping_file(index_map_path: str, offset_lookup_path: str):
    """
    Load mapping file that contains the index_map and offset_lookup for unpacking.

    Args:
        index_map_path (str): Path to the index map file
        offset_lookup_path (str): Path to the offset lookup file

    Returns:
        tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int], int]]]: Mapping information containing index_map, offset_lookup, etc.

    Raises:
        FileNotFoundError: If index map or offset lookup file doesn't exist
    """
    if not os.path.exists(index_map_path):
        raise FileNotFoundError(f"Index map file not found: {index_map_path}")
    if not os.path.exists(offset_lookup_path):
        raise FileNotFoundError(f"Offset lookup file not found: {offset_lookup_path}")

    index_map = np.load(index_map_path)
    with open(offset_lookup_path, 'r') as f:
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = [json.loads(line) for line in f]

    return index_map, offset_lookup


Det = list[float]
Dets = list[Det]
FrameIdToDets = dict[int, Dets]
UnpackedDets = tuple[FrameIdToDets, Dets, Dets]


def unpack_detections(detections: list[list[float]], index_map: np.ndarray,
                      offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
                      tilesize: int) -> UnpackedDets:
    """
    Unpack detections from compressed coordinates back to original frame coordinates.

    Args:
        detections (list[list[float]]): list of bounding boxes in compressed coordinates [x1, y1, x2, y2]
        index_map (np.ndarray): Index map from the mapping file (2D array with group_ids)
        offset_lookup (list[tuple[tuple[int, int], tuple[int, int], int]]): Offset lookup from the mapping file
        tilesize (int): Size of tiles used for compression

    Returns:
        tuple[
            dict[int, list[list[float]]],
            list[list[float]],
            list[list[float]],
        ]:
            dictionary mapping frame indices to lists of bounding boxes in original frame coordinates,
            list of detections that are not in any tile,
            list of detections that are in the center of a tile but not in any tile
    """
    # Initialize dictionary to store detections per frame
    frame_detections: dict[int, list[list[float]]] = {}
    not_in_any_tile_detections: list[list[float]] = []
    center_not_in_any_tile_detections: list[list[float]] = []

    # Process each detection
    for x1, y1, x2, y2, *_ in detections:
        # Get the center point of the detection in compressed coordinates
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # center, top-left, top-right, bottom-left, bottom-right
        xs = [center_x, x1, x2, x1, x2]
        ys = [center_y, y1, y1, y2, y2]

        group_id: int | None = None
        frame_idx: int | None = None
        center_in_any_tile: bool = True
        for x, y in zip(xs, ys):
            # Convert to tile coordinates in the compressed image
            tile_x = int(x // tilesize)
            tile_y = int(y // tilesize)

            # Ensure tile coordinates are within bounds
            if (tile_y < 0 or tile_y >= index_map.shape[0] or
                tile_x < 0 or tile_x >= index_map.shape[1]):
                continue

            # Get the group ID for this tile
            group_id_ = int(index_map[tile_y, tile_x])

            if group_id_ != 0:
                group_id = group_id_
                break
            center_in_any_tile = False

        if not center_in_any_tile:
            center_not_in_any_tile_detections.append([x1, y1, x2, y2, *_])

        if group_id is None:
            not_in_any_tile_detections.append([x1, y1, x2, y2, *_])
            continue

        # Convert group_id to 0-based index
        group_id -= 1

        # Get the offset information for this group
        assert 0 <= group_id < len(offset_lookup), f"Group {group_id} not found in offset lookup"
        (packed_y, packed_x), (original_offset_y, original_offset_x), frame_idx = offset_lookup[group_id]

        # Calculate the offset to convert from compressed to original coordinates
        # The offset represents how much the tile was moved during compression
        offset_x = (original_offset_x - packed_x) * tilesize
        offset_y = (original_offset_y - packed_y) * tilesize

        # Convert detection back to original frame coordinates
        original_det = [
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
            *_
        ]

        # Add to frame detections
        if frame_idx not in frame_detections:
            frame_detections[frame_idx] = []
        frame_detections[frame_idx].append(original_det)

    return frame_detections, not_in_any_tile_detections, center_not_in_any_tile_detections


def unpack(dataset: str, videoset: str, video: str, classifier: str, tilesize: int,
           sample_rate: int, tilepadding: str, canvas_scale: float, tracker: str | None,
           tracking_accuracy_threshold: float | None, relevance_threshold: float):
    """
    Process unpacking for a single video/classifier/tilesize combination.
    This function is designed to be called in parallel.

    Args:
        dataset (str): Name of the dataset
        video (str): Name of the video file
        classifier (str): Classifier name used for compression and detection
        tilesize (int): Tile size used for compression
        tilepadding (str): Whether padding was applied to classification results
        sample_rate (int): Sample rate for frame sampling
        canvas_scale (float): Canvas scale used for compression outputs
        tracker (str): Tracker name for upstream pruning
        tracking_accuracy_threshold (float | None): Accuracy threshold for pruning (None = no pruning)
    """
    # Build the shared key used by all 03x/04x/05x stage folders.
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker,
                                tracking_accuracy_threshold=tracking_accuracy_threshold,
                                relevance_threshold=relevance_threshold)

    # Check if compressed detections exist
    detections_file = cache.exec(dataset, 'comp-dets', video,
                                 param_str, 'detections.jsonl')
    assert os.path.exists(detections_file), f"Detections file not found: {detections_file}"

    # Check if compressed frames directory exists
    compressed_frames_dir = cache.exec(dataset, 'comp-frames', video,
                                       param_str)
    assert os.path.exists(compressed_frames_dir), f"Compressed frames directory not found: {compressed_frames_dir}"

    detections_file = cache.exec(dataset, 'comp-dets', video,
                                 param_str, 'detections.jsonl')

    unpacked_output_dir = cache.exec(dataset, 'ucomp-dets', video,
                                     param_str)
    if os.path.exists(unpacked_output_dir):
        shutil.rmtree(unpacked_output_dir)
    os.makedirs(unpacked_output_dir, exist_ok=True)
    print(f"Saving unpacked detections to {unpacked_output_dir}")
    runtime_file = os.path.join(unpacked_output_dir, 'runtime.jsonl')

    images_not_in_any_tile_dir = os.path.join(unpacked_output_dir, 'images_not_in_any_tile')
    os.makedirs(images_not_in_any_tile_dir, exist_ok=True)
    print(f"Saving images not in any tile to {images_not_in_any_tile_dir}")

    images_center_not_in_any_tile_dir = os.path.join(unpacked_output_dir, 'images_center_not_in_any_tile')
    os.makedirs(images_center_not_in_any_tile_dir, exist_ok=True)
    print(f"Saving images center not in any tile to {images_center_not_in_any_tile_dir}")

    # Get total number of frames from original video
    # This is important: we create entries for ALL frames (0 to num_frames-1)
    # Non-sampled frames will have empty bbox arrays
    num_frames = get_num_frames(store.dataset(dataset, videoset, video))

    # Create entries for ALL frames (0 to num_frames-1), not just sampled ones
    # Non-sampled frames (frame_idx % sample_rate != 0) will remain as empty arrays
    all_frame_detections: dict[int, list[list[float]]] = {
        i: [] for i in range(num_frames)
    }

    with open(detections_file, 'r') as f, open(runtime_file, 'w') as fr:
        # Process each detection file
        contents = f.readlines()
        timer, flush = create_timer(fr)
        for idx, line in enumerate(contents):
            content = json.loads(line)
            image_file: str = content['image_file']
            prefix = image_file.split('.')[0]

            # Construct paths
            index_map_path = os.path.join(compressed_frames_dir, 'index_maps', f'{prefix}.npy')
            offset_lookup_path = os.path.join(compressed_frames_dir, 'offset_lookups', f'{prefix}.jsonl')

            # Load detection results
            detections: list[list[float]] = content['bboxes']

            # Load corresponding mapping file
            index_map, offset_lookup = load_mapping_file(index_map_path, offset_lookup_path)

            with timer('unpack_detections'):
                # Unpack detections
                frame_detections, not_in_any_tile_detections, center_not_in_any_tile_detections \
                    = unpack_detections(detections, index_map, offset_lookup, tilesize)

            # save not_in_any_tile_detections and center_not_in_any_tile_detections
            if len(not_in_any_tile_detections) > 0:
                # load the image
                image_path = os.path.join(compressed_frames_dir, 'images', image_file)
                image = cv2.imread(image_path)
                assert image is not None, f"Image not found: {image_path}"

                # draw the detections
                for det in not_in_any_tile_detections:
                    x1, y1, x2, y2, *_ = det
                    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    image = cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                # save the image
                cv2.imwrite(os.path.join(images_not_in_any_tile_dir, image_file), image)

            if len(center_not_in_any_tile_detections) > 0:
                # load the image
                image_path = os.path.join(compressed_frames_dir, 'images', image_file)
                image = cv2.imread(image_path)
                assert image is not None, f"Image not found: {image_path}"

                # draw the detections
                for det in center_not_in_any_tile_detections:
                    x1, y1, x2, y2, *_ = det
                    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    image = cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                # save the image
                cv2.imwrite(os.path.join(images_center_not_in_any_tile_dir, image_file), image)

            # Note: Only sampled frames (frame_idx % sample_rate == 0) have detections from upstream
            # Non-sampled frames will remain as empty arrays
            # Merge with existing frame detections
            for frame_idx, bboxes in frame_detections.items():
                all_frame_detections[frame_idx].extend(bboxes)

        with timer('sort_frames'):
            # Save unpacked detections organized by frame
            # Sort frames by index
            sorted_frames = sorted(all_frame_detections.keys())
        flush()

    # Save each frame's detections
    with open(os.path.join(unpacked_output_dir, 'detections.jsonl'), 'w') as f:
        for frame_idx in sorted_frames:
            bboxes = all_frame_detections[frame_idx]
            f.write(json.dumps({ 'frame_idx': frame_idx, 'bboxes': bboxes }) + '\n')


def uncompress_all(dataset: str, videoset: str, videos: list[str], classifier: str, tilesize: int,
                   sample_rate: int, tilepadding: str, canvas_scale: float, tracker: str | None,
                   tracking_accuracy_threshold: float | None, relevance_threshold: float,
                   gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'
    # Build a human-readable description for the progress bar.
    param_str = build_param_str(classifier=classifier, tilesize=tilesize, sample_rate=sample_rate,
                                tilepadding=tilepadding, canvas_scale=canvas_scale, tracker=tracker,
                                tracking_accuracy_threshold=tracking_accuracy_threshold,
                                relevance_threshold=relevance_threshold)
    description = f"{dataset} {param_str}"
    # Report initial progress: 0 of N videos done.
    command_queue.put((device, {'completed': 0, 'total': len(videos), 'description': description}))
    # Iterate over all videos in the split for this parameter combination.
    for i, video in enumerate(videos):
        unpack(dataset, videoset, video, classifier, tilesize, sample_rate,
               tilepadding, canvas_scale, tracker, tracking_accuracy_threshold, relevance_threshold)
        # Advance the progress bar by one unit after each video completes.
        command_queue.put((device, {'completed': i + 1, 'total': len(videos), 'description': description}))


def main():
    """
    Main function that orchestrates the detection unpacking process using parallel processing.

    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize/tilepadding combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves unpacked detection results

    Note:
        - The script expects compressed detections from 040_exec_detect.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/040_compressed_detections/{classifier}_{tilesize}_{sample_rate}_{tilepadding}_s{scale_percent}/detections.jsonl
        - The script expects mapping files from 030_exec_compress.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/033_compressed_frames/{classifier}_{tilesize}_{sample_rate}_{tilepadding}_s{scale_percent}/
        - Unpacked detections are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/050_uncompressed_detections/{classifier}_{tilesize}_{sample_rate}_{tilepadding}_s{scale_percent}/detections.jsonl
        - Each line in the output JSONL file contains one bounding box [x1, y1, x2, y2] in original frame coordinates
        - All available video/classifier/tilesize/tilepadding combinations are processed
        - If no compressed detections are found for a video/tilesize/tilepadding combination, that combination is skipped
        - The number of processes equals the number of available GPUs
    """
    args = parse_args()

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
        collapse_tracker_when_no_threshold=True,
    )

    # mp.set_start_method('spawn', force=True)

    # Create tasks list with all video/classifier/tilesize combinations
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset, videoset in itertools.product(DATASETS, selected_videosets):
        videosets_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videosets_dir), f"Videoset directory {videosets_dir} does not exist"

        # Get all video files from the dataset directory
        videos = [f for f in os.listdir(videosets_dir) if f.endswith('.mp4')]
        print(f"Found {len(videos)} video files in dataset {dataset}/{videoset}")

        # Remove all existing ucomp-dets output for every video in this split
        # so stale results from previous runs do not persist.
        for video in videos:
            shutil.rmtree(cache.exec(dataset, 'ucomp-dets', video), ignore_errors=True)

        for classifier, tilesize, tilepadding, sample_rate, canvas_scale, acc_threshold, relevance_threshold in itertools.product(
            CLASSIFIERS, TILE_SIZES, TILEPADDING_MODES, SAMPLE_RATES, CANVAS_SCALES, TRACKING_ACCURACY_THRESHOLDS, RELEVANCE_THRESHOLDS):
            for tracker in [None] if acc_threshold is None else TRACKERS:
                # Skip parameter combos not on the Pareto front during the test pass.
                combo = (classifier, tilesize, sample_rate, tilepadding, canvas_scale, tracker, acc_threshold, relevance_threshold)
                if allowed_combos is not None and combo not in allowed_combos[dataset]:
                    continue
                funcs.append(partial(uncompress_all, dataset, videoset, sorted(videos), classifier,
                                     tilesize, sample_rate, tilepadding, canvas_scale, tracker, acc_threshold, relevance_threshold))

    print(f"Created {len(funcs)} tasks to process")
    ProgressBar(num_workers=int(mp.cpu_count() // 2), num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main()
