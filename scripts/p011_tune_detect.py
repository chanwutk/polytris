#!/usr/local/bin/python

import argparse
import json
import os
import time
import multiprocessing as mp
from multiprocessing import Queue

import cv2
import tqdm
import torch

import polyis.models.retinanet_b3d
from polyis.utilities import CACHE_DIR, DATA_DIR, format_time, progress_bars


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--detector', required=False,
                        default='retina',
                        help='Detector name')
    return parser.parse_args()


def _detect_retina(video_path: str, dataset_dir: str, video: str, gpu_id: int, progress_queue: Queue):
    detector = polyis.models.retinanet_b3d.get_detector(device=f'cuda:{gpu_id}')

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
        progress_queue.put((f'cuda:{gpu_id}', {
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
                outputs = polyis.models.retinanet_b3d.detect(frame, detector)
                end_time = time.time_ns() / 1e6
                detect_time = end_time - start_time

                fd.write(json.dumps([frame_idx, outputs[:, :4].tolist(), idx, format_time(read=read_time, detect=detect_time)]) + '\n')

                frame_idx += 1
                processed_frames += 1

                # Send progress update
                progress_queue.put((f'cuda:{gpu_id}', {'completed': processed_frames}))
        cap.release()


def detect_retina(cache_dir: str, dataset_dir: str):
    """
    Perform object detection on video segments using RetinaNet B3D model.
    
    This function:
    1. Loads a RetinaNet B3D detector on CUDA device
    2. Iterates through each video in the dataset directory
    3. Reads detection segments from the cache
    4. Processes each frame in each segment to detect objects
    5. Saves detection results with timing information to detections.jsonl
    
    Args:
        cache_dir (str): Path to the cache directory containing video segments
        dataset_dir (str): Path to the dataset directory containing original video files
        
    Note:
        The function expects the cache directory to have a specific structure:
        - cache_dir/video_name/segments/detection/segments.jsonl (input segments)
        - cache_dir/video_name/segments/detection/detections.jsonl (output detections)
        
        Detection results are saved in JSONL format with:
        - frame_idx: Current frame index
        - bounding_boxes: Detected object bounding boxes (first 4 columns of outputs)
        - segment_idx: Index of the current segment
        - timing: Dictionary with read and detection timing information
    """
    # Get list of videos to process
    videos = [v for v in sorted(os.listdir(dataset_dir)) 
              if os.path.isdir(os.path.join(cache_dir, v))]
    
    if not videos:
        print("No videos found to process")
        return

    print(f"Found {len(videos)} videos to process")

    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No CUDA GPUs available. Falling back to CPU processing.")
        num_gpus = min(mp.cpu_count(), 20)  # Use available CPUs but cap at 20
        gpu_ids = [None] * num_gpus  # Will use CPU
    else:
        gpu_ids = list(range(num_gpus))
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(videos), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")

    # Create progress queue and start progress display
    progress_queue = Queue()
    
    # Start progress display in a separate process
    progress_process = mp.Process(target=progress_bars, args=(progress_queue, max_processes, len(videos)))
    progress_process.start()
    
    # Create and start video processing processes
    processes: list[mp.Process] = []
    for i, video in enumerate(videos):
        video_path = os.path.join(cache_dir, video)
        gpu_id = gpu_ids[i % len(gpu_ids)]
        
        process = mp.Process(target=_detect_retina, 
                           args=(video_path, dataset_dir, video, gpu_id, progress_queue))
        process.start()
        processes.append(process)
        print(f"Started processing video: {video} on GPU {gpu_id}")
    
    # Wait for all video processing to complete
    for process in processes:
        process.join()
        process.terminate()
    
    # Signal progress display to stop and wait for it
    progress_queue.put(None)
    progress_process.join()
    
    print("All videos processed successfully!")


def main(args):
    """
    Main function to run object detection on video segments.
    
    This function:
    1. Sets up paths for cache and dataset directories based on command line arguments
    2. Routes to the appropriate detector based on the detector argument
    3. Currently supports 'retina' detector (RetinaNet B3D)
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset: Name of the dataset to process
            - detector: Type of detector to use (currently only 'retina' supported)
            
    Raises:
        ValueError: If an unknown detector is specified
        
    Note:
        The function expects the following directory structure:
        - CACHE_DIR/dataset_name/ (for processed segments and results)
        - DATA_DIR/dataset_name/ (for original video files)
    """
    cache_dir = os.path.join(CACHE_DIR, args.dataset)
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    detector = args.detector

    if detector == 'retina':
        detect_retina(cache_dir, dataset_dir)
    else:
        raise ValueError(f"Unknown detector: {detector}")


if __name__ == '__main__':
    main(parse_args())
