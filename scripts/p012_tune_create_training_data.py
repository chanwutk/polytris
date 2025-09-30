#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time

import cv2
import torch
from rich.progress import track

import polyis.images

from polyis.utilities import CACHE_DIR, DATA_DIR, mark_detections, overlap, DATASETS_TO_TEST

TILE_SIZES = [30, 60, 120]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def main(args):
    """
    Main function to create training data from video segments and detections.

    This function:
    1. Iterates through each dataset and video in the cache directory
    2. Creates training data directories for different tile sizes (30, 60, 120)
    3. Processes each frame in detection segments
    4. Splits frames into tiles of specified sizes
    5. Saves positive tiles (containing detections) and negative tiles (no detections)
    6. Records runtime performance metrics for each operation

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets: List of dataset names to process

    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/indexing/segment/detection/{video_file}.segments.jsonl (input segments)
        - CACHE_DIR/dataset_name/indexing/segment/detection/detections.jsonl (input detections)
        - DATA_DIR/dataset_name/{video_file} (original video file)

        Output structure created:
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/pos/ (positive tiles)
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/neg/ (negative tiles)
        - CACHE_DIR/dataset_name/indexing/training/runtime/tilesize_X/{video_file}_creating_training_data.jsonl (performance metrics)

        Tiles are saved as JPG images with naming format: {video_file}_{frame_idx}_y_x.jpg
        Performance metrics include timing for split and save operations.
    """
    datasets = args.datasets

    for dataset_name in datasets:
        cache_dir = os.path.join(CACHE_DIR, dataset_name)
        dataset_dir = os.path.join(DATA_DIR, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue

        # Get list of videos to process from segments files
        segments_dir = os.path.join(cache_dir, 'indexing', 'segment', 'detection')
        if not os.path.exists(segments_dir):
            print(f"Segments directory {segments_dir} does not exist, skipping dataset {dataset_name}")
            continue

        videos = [
            f[:-len('.segments.jsonl')]
            for f in os.listdir(segments_dir)
            if f.endswith('.segments.jsonl')
        ]

        if len(videos) == 0:
            print(f"No videos with segments found in {segments_dir}")
            continue

        print(f"Found {len(videos)} videos to process in dataset {dataset_name}")

        # Create training directories
        training_base_dir = os.path.join(cache_dir, 'indexing', 'training')
        if os.path.exists(training_base_dir):
            shutil.rmtree(training_base_dir)

        for tile_size in TILE_SIZES:
            runtime_path = os.path.join(training_base_dir, 'runtime', f'tilesize_{tile_size}')
            if os.path.exists(runtime_path):
                shutil.rmtree(runtime_path)
            os.makedirs(runtime_path, exist_ok=True)

            training_data_path = os.path.join(training_base_dir, 'data', f'tilesize_{tile_size}')
            if os.path.exists(training_data_path):
                shutil.rmtree(training_data_path)
            os.makedirs(training_data_path, exist_ok=True)
            os.makedirs(os.path.join(training_data_path, 'pos'), exist_ok=True)
            os.makedirs(os.path.join(training_data_path, 'neg'), exist_ok=True)

        for video_file in videos:
            print(f"Processing video {video_file} in dataset {dataset_name}")

            # Open runtime files for this video
            frs = {
                tile_size: open(os.path.join(training_base_dir,
                                             'runtime',
                                             f'tilesize_{tile_size}',
                                             f'{video_file}_creating_training_data.jsonl'), 'w')
                for tile_size in TILE_SIZES
            }

            # Read segments to get frame ranges
            segments_path = os.path.join(segments_dir, f'{video_file}.segments.jsonl')
            detections_path = os.path.join(segments_dir, f'{video_file}.detections.jsonl')

            if not os.path.exists(segments_path) or not os.path.exists(detections_path):
                print(f"Skipping {video_file} - missing segments or detections files")
                continue

            # Construct the path to the video file in the dataset directory
            dataset_video_path = os.path.join(dataset_dir, video_file)
            cap = cv2.VideoCapture(dataset_video_path)

            with open(segments_path, 'r') as segments_f, open(detections_path, 'r') as detections_f:
                segments_lines = [*segments_f.readlines()]
                # detections_lines = [*detections_f.readlines()]

                for segment in track(segments_lines):
                    segment_json = json.loads(segment)
                    segment_idx = segment_json['idx']
                    start = segment_json['start']
                    end = segment_json['end']

                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for frame_idx in range(start, end):
                        ret, frame = cap.read()
                        assert ret, f"Failed to read frame {frame_idx}"

                        frame_idx_, dets, segment_idx_, _ = json.loads(detections_f.readline())
                        assert frame_idx_ == frame_idx, f"Frame index mismatch: {frame_idx_} != {frame_idx}"
                        assert segment_idx_ == segment_idx, f"Segment index mismatch: {segment_idx_} != {segment_idx}"

                        for tile_size in TILE_SIZES:
                            split_start_time = time.time_ns() / 1e6
                            training_data_path = os.path.join(training_base_dir, 'data', f'tilesize_{tile_size}')

                            padded_frame = torch.from_numpy(frame).to('cuda')
                            assert polyis.images.isHWC(padded_frame), padded_frame.shape

                            patched = polyis.images.splitHWC(padded_frame, tile_size, tile_size)
                            patched = patched.cpu()
                            assert polyis.images.isGHWC(patched), patched.shape
                            patched = patched.contiguous().numpy()
                            split_time = (time.time_ns() / 1e6) - split_start_time
                            frs[tile_size].write(json.dumps({
                                'op': 'split',
                                'time': split_time,
                                'frame': frame_idx,
                                'tile_size': tile_size,
                                'frame_shape': frame.shape,
                                'patched_shape': patched.shape,
                            }) + '\n')

                            save_start_time = time.time_ns() / 1e6
                            relevancy_bitmap = mark_detections(dets, frame.shape[1], frame.shape[0], tile_size)
                            for y in range(patched.shape[0]):
                                for x in range(patched.shape[1]):
                                    filename = f'{video_file}_{frame_idx}_{y}_{x}.jpg'
                                    patch = patched[y, x]

                                    if relevancy_bitmap[y, x]:
                                        cv2.imwrite(os.path.join(training_data_path, 'pos', filename), patch)
                                    else:
                                        if patched[y, x].any():
                                            cv2.imwrite(os.path.join(training_data_path, 'neg', filename), patch)
                            save_time = (time.time_ns() / 1e6) - save_start_time
                            frs[tile_size].write(json.dumps({
                                'op': 'save',
                                'time': save_time,
                                'frame': frame_idx,
                                'tile_size': tile_size,
                                'frame_shape': frame.shape,
                                'patched_shape': patched.shape,
                            }) + '\n')

            cap.release()
            for fr in frs.values():
                fr.close()


if __name__ == '__main__':
    main(parse_args())
