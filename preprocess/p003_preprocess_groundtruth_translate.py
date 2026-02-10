#!/usr/local/bin/python

import argparse
import json
import os
import queue
from functools import partial

import cv2

from polyis.utilities import ProgressBar, get_config, save_tracking_results


CONFIG = get_config()
EXEC_DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']
OLD_CACHE_DIR = os.path.join(CACHE_DIR, 'ORIGINAL')

# FPS thresholds matching p000_preprocess_dataset.py
FPS_15_LO = 14.9
FPS_15_HI = 15.1
FPS_30_LO = 29.9
FPS_30_HI = 30.1


def parse_args():
    parser = argparse.ArgumentParser(description='Translate naive tracking results to groundtruth framerate')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def get_fps_step(fps: float) -> int:
    """
    Determine the frame sampling step based on video FPS.

    Args:
        fps (float): Video frames per second

    Returns:
        int: Step for frame sampling (1 for 15fps, 2 for 30fps)
    """
    if FPS_15_LO <= fps <= FPS_15_HI:
        return 1
    if FPS_30_LO <= fps <= FPS_30_HI:
        return 2
    raise AssertionError(
        f"Video FPS {fps} is not supported; must be 15 or 29.9-30"
    )


def translate(dataset: str, video_file: str, split: str,
              worker_id: int, command_queue: "queue.Queue[tuple[str, dict]]"):
    """
    Copy tracking results from OLD_CACHE_DIR/003_groundtruth to CACHE_DIR/003_groundtruth, sampling frames to match 15 FPS.

    Args:
        dataset (str): Dataset name
        video_file (str): Video filename (e.g., 'te01.mp4')
        split (str): Dataset split ('train', 'valid', 'test')
        worker_id (int): Worker ID for progress reporting
        command_queue (Queue): Queue for progress updates
    """
    # Build path to the video in DATASETS_DIR and read its FPS
    video_path = os.path.join(DATASETS_DIR, dataset, split, video_file)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Determine sampling step from FPS
    step = get_fps_step(fps)

    # Build input path for groundtruth tracking results from old cache
    input_path = os.path.join(OLD_CACHE_DIR, dataset, 'execution', video_file, '003_groundtruth', 'tracking.jsonl')
    assert os.path.exists(input_path), f"Tracking results not found: {input_path}"

    # Build output path for groundtruth tracking results
    output_path = os.path.join(CACHE_DIR, dataset, 'execution', video_file, '003_groundtruth', 'tracking.jsonl')

    # Load all frames from the input tracking file
    input_frames: list[dict] = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                input_frames.append(json.loads(line))

    # Send initial progress update
    command_queue.put((f'cuda:{worker_id}', {
        'description': f'{dataset}/{video_file}',
        'completed': 0,
        'total': len(input_frames)
    }))

    if len(input_frames) > 1500:
        step = 2
    else:
        step = 1

    # Sample frames according to step and assign new sequential indices
    frame_tracks: dict[int, list[list[float]]] = {}
    mod = max(1, int(len(input_frames) * 0.05))
    for i, frame_data in enumerate(input_frames):
        # Keep every step-th frame starting from frame 0
        frame_idx = frame_data['frame_idx']
        if frame_idx % step == 0:
            # Calculate new sequential index from original frame index
            new_frame_idx = frame_idx // step
            frame_tracks[new_frame_idx] = frame_data['tracks']

        # Send periodic progress updates
        if i % mod == 0:
            command_queue.put((f'cuda:{worker_id}', {'completed': i + 1}))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled tracking results to the groundtruth path
    save_tracking_results(frame_tracks, output_path)


def main():
    """
    Main function that translates naive tracking results to 15 FPS groundtruth.

    Reads tracking results from OLD_CACHE_DIR/003_groundtruth, checks video FPS, and writes
    frame-sampled results to CACHE_DIR/003_groundtruth.
    """
    args = parse_args()

    # Determine which splits to process based on arguments
    splits = []
    if args.test:
        splits.append('test')
    if args.train:
        splits.append('train')
    if args.valid:
        splits.append('valid')

    # Default to test if no splits specified
    if not splits:
        splits = ['test']

    # Collect translate tasks for all datasets and videos
    funcs = []
    for dataset in EXEC_DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"

        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            assert os.path.exists(split_dir), f"Split directory {split_dir} does not exist"

            # Collect video files from this split
            video_files = [f for f in os.listdir(split_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            for video_file in video_files:
                funcs.append(partial(translate, dataset, video_file, split))

    assert len(funcs) > 0, "No video files found to process"

    # Use ProgressBar with CPU-bound workers
    num_workers = min(8, len(funcs))
    print(f"Processing {len(funcs)} videos with {num_workers} workers")
    ProgressBar(num_workers=num_workers, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main()
