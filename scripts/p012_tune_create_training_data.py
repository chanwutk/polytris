#!/usr/local/bin/python

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description='Create training data from detections with FPS-based subsampling')
    parser.add_argument('--target_fps', type=int, default=5,
                        help='Target frames per second for subsampling (default: 5)')
    return parser.parse_args()


def get_patched(frame: np.ndarray, tile_size: int) -> np.ndarray:
    padded_frame = torch.from_numpy(frame).to('cuda')
    assert polyis.images.isHWC(padded_frame), padded_frame.shape

    patched = polyis.images.splitHWC(padded_frame, tile_size, tile_size)
    patched = patched.cpu()
    assert polyis.images.isGHWC(patched), patched.shape
    return patched.contiguous().numpy()


def create_training_data(dataset_name: str, video_file: str, target_fps: int, gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name, 'train')
    segments_dir = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'segment', 'detection')
    training_base_dir = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'training')
    always_relevant_tiles_path = os.path.join(CACHE_DIR, dataset_name, 'indexing', 'always_relevant')
    if os.path.exists(always_relevant_tiles_path):
        shutil.rmtree(always_relevant_tiles_path)
    os.makedirs(always_relevant_tiles_path, exist_ok=True)

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

    # Compute subsampled frame indices based on target_fps
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    selectivity = target_fps / original_fps
    frames_to_sample = max(1, int(frame_count * selectivity))
    frame_indices_set = set(np.linspace(0, frame_count - 1, frames_to_sample, dtype=int).tolist())

    with open(detections_path, 'r') as detections_f:
        detections_lines = [*detections_f.readlines()]

        frame_idx = 0
        command_queue.put((device, {
            'description': f'{dataset_name} {video_file}',
            'completed': 0,
            'total': frames_to_sample,
        }))
        processed_frames = 0
        always_relevant_tiles: "dict[int, np.ndarray]" = {}
        for detections_str in detections_lines:
            frame_idx, dets, _ = json.loads(detections_str)

            # Skip frames not in the subsampled set
            if frame_idx not in frame_indices_set:
                continue

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
                # Scale down resolution to fit tile size if not divisible
                target_h = (frame.shape[0] // tile_size) * tile_size
                target_w = (frame.shape[1] // tile_size) * tile_size
                if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
                    resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    resized_prev = cv2.resize(prev_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    # Scale detection coordinates (x1, y1, x2, y2) to match resized resolution
                    # Preserve any extra columns (e.g., confidence score) unchanged
                    scale_x = target_w / frame.shape[1]
                    scale_y = target_h / frame.shape[0]
                    resized_dets = [
                        [d[0] * scale_x, d[1] * scale_y, d[2] * scale_x, d[3] * scale_y] + d[4:]
                        for d in dets
                    ]
                else:
                    resized_frame = frame
                    resized_prev = prev_frame
                    resized_dets = dets

                split_start_time = time.time_ns() / 1e6
                training_data_path = os.path.join(training_base_dir, 'data', f'tilesize_{tile_size}')
                training_diff_path = os.path.join(training_base_dir, 'diff', f'tilesize_{tile_size}')

                patched = get_patched(resized_frame, tile_size)
                diffs = get_patched(np.abs(resized_frame.astype(np.int16) - resized_prev.astype(np.int16)).astype(np.uint8), tile_size)
                split_time = (time.time_ns() / 1e6) - split_start_time
                frs[tile_size].write(json.dumps({
                    'op': 'split',
                    'time': split_time,
                    'frame': frame_idx,
                    'tile_size': tile_size,
                    'frame_shape': resized_frame.shape,
                    'patched_shape': patched.shape,
                }) + '\n')

                save_start_time = time.time_ns() / 1e6
                # Use slice(0, 4) since bboxes have 5 values [x1, y1, x2, y2, score]
                relevancy_bitmap = mark_detections(resized_dets, resized_frame.shape[1], resized_frame.shape[0], tile_size, detection_slice=slice(0, 4))
                if tile_size not in always_relevant_tiles:
                    always_relevant_tiles[tile_size] = relevancy_bitmap
                always_relevant_tiles[tile_size] |= relevancy_bitmap
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
                    'frame_shape': resized_frame.shape,
                    'patched_shape': patched.shape,
                }) + '\n')

            processed_frames += 1
            command_queue.put((device, {
                'completed': processed_frames,
            }))
        for tile_size in TILE_SIZES:
            assert tile_size in always_relevant_tiles, f"Always relevant tiles is not found for tile size {tile_size} for {video_file}"
            np.save(os.path.join(always_relevant_tiles_path, f'{tile_size}_{video_file}.npy'), always_relevant_tiles[tile_size])

    cap.release()
    for fr in frs.values():
        fr.close()

def main(args):
    """
    Create training data from video detections with FPS-based subsampling.

    Reads full-frame detections from p011, subsamples based on --target_fps,
    splits frames into tiles, and saves positive/negative training images.
    """
    target_fps = args.target_fps
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
        print(f'Creating training directories for {dataset_name}')
        training_base_dir = os.path.join(cache_dir, 'indexing', 'training')
        if os.path.exists(training_base_dir):
            shutil.rmtree(training_base_dir)

        for tile_size in TILE_SIZES:
            runtime_path = os.path.join(training_base_dir, 'runtime', f'tilesize_{tile_size}')
            if os.path.exists(runtime_path):
                print(f'Removing existing runtime directory {runtime_path}')
                shutil.rmtree(runtime_path)
            os.makedirs(runtime_path, exist_ok=True)

            for subpath in ['data', 'diff']:
                training_data_path = os.path.join(training_base_dir, subpath, f'tilesize_{tile_size}')
                if os.path.exists(training_data_path):
                    print(f'Removing existing training data directory {training_data_path}')
                    shutil.rmtree(training_data_path)
                os.makedirs(training_data_path, exist_ok=True)
                for label in ['pos', 'neg']:
                    os.makedirs(os.path.join(training_data_path, label), exist_ok=True)

        for video_file in videos:
            funcs.append(partial(create_training_data, dataset_name, video_file, target_fps))
    
    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    ProgressBar(num_workers=20, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)
    print('Training data created')


if __name__ == '__main__':
    main(parse_args())
