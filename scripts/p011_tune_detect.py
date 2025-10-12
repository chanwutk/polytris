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
from polyis.utilities import CACHE_DIR, DATA_DIR, format_time, ProgressBar, DATASETS_TO_TEST


def parse_args():
    parser = argparse.ArgumentParser(description='Execute object detection on video segments')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def detect_objects(video: str, dataset_name: str, gpu_id: int, command_queue: Queue):
    # Get detector based on detector_name parameter
    detector = polyis.models.detector.get_detector(dataset_name, gpu_id, batch_size=1)

    # New output path structure
    output_path = Path(CACHE_DIR) / dataset_name / 'indexing' / 'segment' / 'detection' / f'{video}.detections.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Input segments path
    segments_path = Path(CACHE_DIR) / dataset_name / 'indexing' / 'segment' / 'detection' / f'{video}.segments.jsonl'

    with (open(segments_path, 'r') as f, open(output_path, 'w') as fd):
        lines = [*f.readlines()]

        # Construct the path to the video file in the dataset directory
        dataset_video_path = os.path.join(DATA_DIR, dataset_name, video)
        cap = cv2.VideoCapture(dataset_video_path)

        # Calculate total frames across all segments for progress tracking
        total_frames = sum(json.loads(line)['end'] - json.loads(line)['start'] for line in lines)
        processed_frames = 0

        # Send initial progress update
        command_queue.put((f'cuda:{gpu_id}', {
            'description': f'{dataset_name}/{video}',
            'completed': 0,
            'total': total_frames
        }))

        for line in lines:
            snippet = json.loads(line)
            idx = snippet['idx']
            start = snippet['start']
            end = snippet['end']

            frame_idx = start
            # set cap to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            while cap.isOpened() and frame_idx < end:
                start_time = time.time_ns() / 1e6
                ret, frame = cap.read()
                end_time = time.time_ns() / 1e6
                read_time = end_time - start_time

                assert ret, "Failed to read frame"

                # Detect objects in the frame
                start_time = time.time_ns() / 1e6
                outputs = polyis.models.detector.detect(frame, detector)
                end_time = time.time_ns() / 1e6
                detect_time = end_time - start_time

                fd.write(json.dumps([frame_idx, outputs[:, :4].tolist(), idx, format_time(read=read_time, detect=detect_time)]) + '\n')

                frame_idx += 1
                processed_frames += 1

                # Send progress update
                command_queue.put((f'cuda:{gpu_id}', {'completed': processed_frames}))
        cap.release()


def main(args):
    """
    Main function to run object detection on video segments.

    This function:
    1. Processes multiple datasets in sequence
    2. For each dataset, sets up paths for cache and dataset directories
    3. Selects the appropriate detector (based on dataset name)
    4. Iterates through each video in the dataset directory
    5. Reads detection segments from the cache
    6. Processes each frame in each segment to detect objects using multiprocessing
    7. Saves detection results with timing information to detections.jsonl

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets: List of dataset names to process

    Raises:
        ValueError: If dataset is not found in configuration

    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/indexing/segments/detection/ (for processed segments and results)
        - DATA_DIR/dataset_name/ (for original video files)

        Detection results are saved in JSONL format with:
        - frame_idx: Current frame index
        - bounding_boxes: Detected object bounding boxes (first 4 columns of outputs)
        - segment_idx: Index of the current segment
        - timing: Dictionary with read and detection timing information
    """
    datasets = args.datasets

    # Show detector info
    print(f"Using detector: {datasets}")

    # Create task functions
    funcs = []
    for dataset_name in datasets:
        cache_dir = Path(CACHE_DIR) / dataset_name
        dataset_dir = Path(DATA_DIR) / dataset_name

        if not dataset_dir.exists():
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue

        # Get list of videos to process
        videos = [
            v.name[:-len('.segments.jsonl')]
            for v in (cache_dir / 'indexing' / 'segment' / 'detection').iterdir()
            if v.is_file()
        ]

        if len(videos) == 0:
            print(f"No videos with segments found in {cache_dir}")
            continue

        print(f"Found {len(videos)} videos to process in dataset {dataset_name}")

        funcs.extend(
            partial(detect_objects, video, dataset_name)
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
