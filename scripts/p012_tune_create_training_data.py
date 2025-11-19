#!/usr/local/bin/python

import json
import os
import shutil
import time
import multiprocessing as mp
from functools import partial
import cv2
import torch
import numpy as np

import polyis.images

from polyis.utilities import ProgressBar, mark_detections, get_config

config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
DATASETS_TO_TEST = config['EXEC']['DATASETS']
TILE_SIZES = config['EXEC']['TILE_SIZES']


def get_patched(frame: np.ndarray, tile_size: int) -> np.ndarray:
    padded_frame = torch.from_numpy(frame).to('cuda')
    assert polyis.images.isHWC(padded_frame), padded_frame.shape

    patched = polyis.images.splitHWC(padded_frame, tile_size, tile_size)
    patched = patched.cpu()
    assert polyis.images.isGHWC(patched), patched.shape
    return patched.contiguous().numpy()


def create_training_data(dataset_name: str, video_file: str, gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name, 'train')
    segments_dir = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'segment', 'detection')
    training_base_dir = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'training')

    # Open runtime files for this video
    frs = {
        tile_size: open(os.path.join(training_base_dir,
                                        'runtime',
                                        f'tilesize_{tile_size}',
                                        f'{video_file}_creating_training_data.jsonl'), 'w')
        for tile_size in TILE_SIZES
    }

    # Read segments to get frame ranges
    detections_path = os.path.join(segments_dir, f'{video_file}.detections.jsonl')

    assert os.path.exists(detections_path), f"Detections file {detections_path} does not exist"

    # Construct the path to the video file in the dataset directory
    dataset_video_path = os.path.join(dataset_dir, video_file)
    cap = cv2.VideoCapture(dataset_video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(detections_path, 'r') as detections_f:
        detections_lines = [*detections_f.readlines()]

        frame_idx = 0
        command_queue.put((device, {
            'description': f'{dataset_name} {video_file}',
            'completed': frame_idx,
            'total': frame_count,
        }))
        for detections_str in detections_lines:
            frame_idx, dets, _ = json.loads(detections_str)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))

            ret, prev_frame = cap.read()
            assert ret, f"Failed to read frame {frame_idx - 1} from {dataset_video_path}"
            ret, frame = cap.read()
            assert ret, f"Failed to read frame {frame_idx} from {dataset_video_path}"

            if frame_idx == 0:
                tmp = prev_frame
                prev_frame = frame
                frame = tmp

            for tile_size in TILE_SIZES:
                split_start_time = time.time_ns() / 1e6
                training_data_path = os.path.join(training_base_dir, 'data', f'tilesize_{tile_size}')
                training_diff_path = os.path.join(training_base_dir, 'diff', f'tilesize_{tile_size}')

                patched = get_patched(frame, tile_size)
                diffs = get_patched(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)).astype(np.uint8), tile_size)
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
                        diff = diffs[y, x]

                        if relevancy_bitmap[y, x]:
                            cv2.imwrite(os.path.join(training_data_path, 'pos', filename), patch)
                            cv2.imwrite(os.path.join(training_diff_path, 'pos', filename), diff)
                        else:
                            if patched[y, x].any():
                                cv2.imwrite(os.path.join(training_data_path, 'neg', filename), patch)
                                cv2.imwrite(os.path.join(training_diff_path, 'neg', filename), diff)
                save_time = (time.time_ns() / 1e6) - save_start_time
                frs[tile_size].write(json.dumps({
                    'op': 'save',
                    'time': save_time,
                    'frame': frame_idx,
                    'tile_size': tile_size,
                    'frame_shape': frame.shape,
                    'patched_shape': patched.shape,
                }) + '\n')

            command_queue.put((device, {
                'description': f'{dataset_name} {video_file}',
                'completed': frame_idx,
                'total': frame_count,
            }))

    cap.release()
    for fr in frs.values():
        fr.close()

def main():
    """
    Main function to create training data from video segments and detections.

    This function:
    1. Iterates through each dataset and video in the cache directory
    2. Creates training data directories for different tile sizes (30, 60, 120)
    3. Processes each frame in detection segments
    4. Splits frames into tiles of specified sizes
    5. Saves positive tiles (containing detections) and negative tiles (no detections)
    6. Records runtime performance metrics for each operation

    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/indexing/segment/detection/{video_file}.segments.jsonl (input segments)
        - CACHE_DIR/dataset_name/indexing/segment/detection/detections.jsonl (input detections)
        - DATASETS_DIR/dataset_name/{video_file} (original video file)

        Output structure created:
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/pos/ (positive tiles)
        - CACHE_DIR/dataset_name/indexing/training/data/tilesize_X/neg/ (negative tiles)
        - CACHE_DIR/dataset_name/indexing/training/runtime/tilesize_X/{video_file}_creating_training_data.jsonl (performance metrics)

        Tiles are saved as JPG images with naming format: {video_file}_{frame_idx}_y_x.jpg
        Performance metrics include timing for split and save operations.
    """
    funcs = []
    for dataset_name in DATASETS_TO_TEST:
        cache_dir = os.path.join(CACHE_DIR, dataset_name)
        dataset_dir = os.path.join(DATASETS_DIR, dataset_name, 'train')

        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"

        # Get list of videos to process from segments files
        segments_dir = os.path.join(cache_dir, 'indexing', 'segment', 'detection')
        assert os.path.exists(segments_dir), f"Segments directory {segments_dir} does not exist"

        videos = [
            f[:-len('.detections.jsonl')]
            for f in os.listdir(segments_dir)
            if f.endswith('.detections.jsonl')
        ]

        assert len(videos) > 0, f"No videos with segments found in {segments_dir}"

        # Create training directories
        training_base_dir = os.path.join(cache_dir, 'indexing', 'training')
        if os.path.exists(training_base_dir):
            shutil.rmtree(training_base_dir)

        for tile_size in TILE_SIZES:
            runtime_path = os.path.join(training_base_dir, 'runtime', f'tilesize_{tile_size}')
            if os.path.exists(runtime_path):
                shutil.rmtree(runtime_path)
            os.makedirs(runtime_path, exist_ok=True)

            for subpath in ['data', 'diff']:
                training_data_path = os.path.join(training_base_dir, subpath, f'tilesize_{tile_size}')
                if os.path.exists(training_data_path):
                    shutil.rmtree(training_data_path)
                os.makedirs(training_data_path, exist_ok=True)
                for label in ['pos', 'neg']:
                    os.makedirs(os.path.join(training_data_path, label), exist_ok=True)

        for video_file in videos:
            funcs.append(partial(create_training_data, dataset_name, video_file))
    
    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    ProgressBar(num_workers=20, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main()
