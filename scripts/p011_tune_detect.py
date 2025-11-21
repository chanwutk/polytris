#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import time
from multiprocessing import Queue
from pathlib import Path

import cv2
import numpy as np
import torch

import polyis.models.detector
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
DATASETS = config['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Execute object detection on uniformly sampled video frames')
    parser.add_argument('--selectivity', type=float, default=0.1,
                        help='Fraction of frames to uniformly sample from video (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for detection processing (default: 64)')
    return parser.parse_args()


def detect_objects(video: str, split: str, dataset: str, selectivity: float, batch_size: int, gpu_id: int, command_queue: Queue):
    # New output path structure
    output_path = Path(CACHE_DIR) / dataset / 'indexing' / 'segment' / 'detection' / f'{video}.detections.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct the path to the video file in the dataset directory
    dataset_video_path = os.path.join(DATASETS_DIR, dataset, split, video)
    cap = cv2.VideoCapture(dataset_video_path)

    # Get total frame count from video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate number of frames to sample based on selectivity
    frames_to_sample = int(total_frames * selectivity)
    assert frames_to_sample > 1, "Number of frames to sample must be greater than 1"
    
    # Generate uniformly distributed frame indices
    frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
    
    processed_frames = 0

    # Send initial progress update
    command_queue.put((f'cuda:{gpu_id}', {
        'description': f'{dataset}/{video}',
        'completed': 0,
        'total': frames_to_sample
    }))

    # Get detector based on detector_name parameter with appropriate batch size
    detector = polyis.models.detector.get_detector(dataset, gpu_id, batch_size=batch_size,
                                                   num_images=len(frame_indices))

    with open(output_path, 'w') as fd:
        # Process frames in batches
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_indices))
            batch_indices = frame_indices[batch_start:batch_end]
            batch_frames = []
            
            # Read frames for current batch
            read_times = []
            for frame_idx in batch_indices:
                start_time = time.time_ns() / 1e6
                # Set cap to the target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                end_time = time.time_ns() / 1e6
                read_time = end_time - start_time
                read_times.append(read_time)

                assert ret, "Failed to read frame"
                batch_frames.append(frame)

            # Detect objects in the batch of frames
            start_time = time.time_ns() / 1e6
            batch_outputs = polyis.models.detector.detect_batch(batch_frames, detector)
            end_time = time.time_ns() / 1e6
            detect_time = end_time - start_time

            # Write detection results for each frame in the batch
            for i, frame_idx in enumerate(batch_indices):
                outputs = batch_outputs[i]
                formatted_time = format_time(read=read_times[i], detect=detect_time / len(batch_indices))
                fd.write(json.dumps([int(frame_idx), outputs[:, :4].tolist(), formatted_time]) + '\n')

            processed_frames += len(batch_indices)

            # Send progress update
            command_queue.put((f'cuda:{gpu_id}', {'completed': processed_frames}))
    
    cap.release()


def main(args):
    """
    Main function to run object detection on uniformly sampled video frames.

    This function:
    1. Processes multiple datasets in sequence
    2. For each dataset, sets up paths for cache and dataset directories
    3. Selects the appropriate detector (based on dataset name)
    4. Iterates through each video in the dataset directory
    5. Uniformly samples frames from each video based on selectivity parameter
    6. Processes sampled frames in batches to detect objects using multiprocessing
    7. Saves detection results with timing information to detections.jsonl

    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets: List of dataset names to process
            - selectivity: Fraction of frames to uniformly sample (default: 0.1)
            - batch_size: Number of frames to process in each batch (default: 8)

    Raises:
        ValueError: If dataset is not found in configuration

    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/indexing/segment/detection/ (for detection results)
        - DATASETS_DIR/dataset_name/ (for original video files)

        Detection results are saved in JSONL format with:
        - frame_idx: Current frame index
        - bounding_boxes: Detected object bounding boxes (first 4 columns of outputs)
        - segment_idx: Frame index (for compatibility)
        - timing: Dictionary with read and detection timing information
        
        Batch processing improves GPU utilization and overall performance.
    """
    selectivity = args.selectivity
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
                partial(detect_objects, video, split, dataset, selectivity, batch_size)
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
