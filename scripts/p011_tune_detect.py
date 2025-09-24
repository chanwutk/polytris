#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import time
from multiprocessing import Queue

import cv2
import torch

import polyis.models.detector
from polyis.utilities import CACHE_DIR, DATA_DIR, format_time, ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    return parser.parse_args()


def detect_objects(video_path: str, dataset_dir: str, video: str, dataset_name: str, gpu_id: int, command_queue: Queue):
    detector = polyis.models.detector.get_detector(dataset_name, gpu_id)

    with (open(os.path.join(video_path, 'segments', 'detection', 'segments.jsonl'), 'r') as f,
            open(os.path.join(video_path, 'segments', 'detection', 'detections.jsonl'), 'w') as fd):
        lines = [*f.readlines()]

        # Construct the path to the video file in the dataset directory
        dataset_video_path = os.path.join(dataset_dir, video)
        cap = cv2.VideoCapture(dataset_video_path)

        # Calculate total frames across all segments for progress tracking
        total_frames = sum(json.loads(line)['end'] - json.loads(line)['start'] for line in lines)
        processed_frames = 0

        # Send initial progress update
        command_queue.put((f'cuda:{gpu_id}', {
            'description': video,
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
    Main function to run object detection on video segments using auto-selected detector.
    
    This function:
    1. Sets up paths for cache and dataset directories based on command line arguments
    2. Automatically selects the appropriate detector based on the dataset name
    3. Iterates through each video in the dataset directory
    4. Reads detection segments from the cache
    5. Processes each frame in each segment to detect objects using multiprocessing
    6. Saves detection results with timing information to detections.jsonl
    7. Supports both RetinaNet and YOLOv3 detectors
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset: Name of the dataset to process (detector auto-selected)
            
    Raises:
        ValueError: If dataset is not found in configuration
        
    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/ (for processed segments and results)
        - DATA_DIR/dataset_name/ (for original video files)
        
        Detection results are saved in JSONL format with:
        - frame_idx: Current frame index
        - bounding_boxes: Detected object bounding boxes (first 4 columns of outputs)
        - segment_idx: Index of the current segment
        - timing: Dictionary with read and detection timing information
    """
    cache_dir = os.path.join(CACHE_DIR, args.dataset)
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    
    # Show detector info for this dataset
    detector_info = polyis.models.detector.get_detector_info(args.dataset)
    print(f"Using detector: {detector_info['detector']} ({detector_info['description']})")
    
    dataset_name = args.dataset
    
    # Get list of videos to process
    videos = [v for v in sorted(os.listdir(dataset_dir)) 
              if os.path.isdir(os.path.join(cache_dir, v))]
    
    assert len(videos) > 0, "No videos found to process"

    print(f"Found {len(videos)} videos to process")

    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(videos), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")
    
    # Create task functions
    funcs = []
    for video in videos:
        funcs.append(partial(detect_objects,
                     os.path.join(cache_dir, video), dataset_dir, video, dataset_name))
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=max_processes, num_tasks=len(videos)).run_all(funcs)
    
    print("All videos processed successfully!")


if __name__ == '__main__':
    main(parse_args())
