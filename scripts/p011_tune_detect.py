#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import time
from multiprocessing import Queue
from pathlib import Path

import cv2
import torch

import polyis.models.detector
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
DATASETS = config['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Execute object detection on all video frames')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for detection processing (default: 16)')
    return parser.parse_args()


def detect_objects(video: str, split: str, dataset: str, batch_size: int, gpu_id: int, command_queue: Queue):
    # New output path structure
    output_path = Path(CACHE_DIR) / dataset / 'indexing' / 'segment' / 'detection' / f'{video}.detections.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct the path to the video file in the dataset directory
    dataset_video_path = os.path.join(DATASETS_DIR, dataset, split, video)
    cap = cv2.VideoCapture(dataset_video_path)

    # Get total frame count from video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = 0

    # Send initial progress update
    command_queue.put((f'cuda:{gpu_id}', {
        'description': f'{dataset}/{video}',
        'completed': 0,
        'total': total_frames
    }))

    # Get detector with appropriate batch size
    detector = polyis.models.detector.get_detector(dataset, gpu_id, batch_size=batch_size,
                                                   num_images=total_frames)

    with open(output_path, 'w') as fd:
        # Process all frames in batches
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_size_actual = batch_end - batch_start
            batch_frames = []

            # Read frames sequentially
            read_times = []
            for _ in range(batch_size_actual):
                start_time = time.time_ns() / 1e6
                ret, frame = cap.read()
                frame = frame[:, :, ::-1]  # BGR to RGB
                end_time = time.time_ns() / 1e6
                read_times.append(end_time - start_time)

                assert ret, "Failed to read frame"
                batch_frames.append(frame)

            # Detect objects in the batch of frames
            start_time = time.time_ns() / 1e6
            batch_outputs = polyis.models.detector.detect_batch(batch_frames, detector)
            end_time = time.time_ns() / 1e6
            detect_time = end_time - start_time

            # Write detection results for each frame in the batch (5-col bboxes with confidence)
            for i in range(batch_size_actual):
                frame_idx = batch_start + i
                outputs = batch_outputs[i]
                formatted_time = format_time(read=read_times[i], detect=detect_time / batch_size_actual)
                fd.write(json.dumps([frame_idx, outputs[:, :5].tolist(), formatted_time]) + '\n')

            processed_frames += batch_size_actual

            # Send progress update
            command_queue.put((f'cuda:{gpu_id}', {'completed': processed_frames}))

    cap.release()


def main(args):
    """
    Main function to run object detection on all video frames.

    Processes each dataset's training videos, detecting objects on every frame.
    Detection results include 5-column bounding boxes (x1, y1, x2, y2, confidence).
    Results are saved in JSONL format to CACHE_DIR/dataset/indexing/segment/detection/.
    """
    batch_size = args.batch_size

    # Create task functions
    funcs = []
    for dataset in DATASETS:
        dataset_dir = Path(DATASETS_DIR) / dataset
        assert dataset_dir.exists(), f"Dataset directory {dataset_dir} does not exist"

        for split in ['train']:
            # Get list of videos to process from dataset directory
            videos = [
                v.name
                for v in (dataset_dir / split).iterdir()
                if v.is_file() and v.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
            ]

            if len(videos) == 0:
                print(f"No video files found in {dataset_dir}")
                continue

            print(f"Found {len(videos)} videos to process in dataset {dataset}")

            funcs.extend(
                partial(detect_objects, video, split, dataset, batch_size)
                for video in videos
            )

    if len(funcs) == 0:
        print("No videos found to process across all datasets")
        return

    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")

    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)

    print("All videos processed successfully!")


if __name__ == '__main__':
    main(parse_args())
