#!/usr/local/bin/python

import argparse
import json
import os
import time
import cv2
import multiprocessing as mp
from multiprocessing import Queue
import torch

import polyis.models.retinanet_b3d
from scripts.utilities import CACHE_DIR, DATA_DIR, format_time, progress_bars


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - detector (str): Detector name to use (default: 'retina')
    """
    parser = argparse.ArgumentParser(description='Execute object detection on preprocessed videos')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--detector', required=False,
                        default='retina',
                        help='Detector name')
    return parser.parse_args()


def detect_objects_in_video(video_path: str, detector_name: str, output_path: str,
                            gpu_id: int | None, progress_queue: Queue):
    """
    Execute object detection on a single video file and save results to JSONL.
    
    Args:
        video_path (str): Path to the input video file
        detector_name (str): Name of the detector being used
        output_path (str): Path where the output JSONL file will be saved
        gpu_id (int | None): GPU device ID to use for this process
        
    Note:
        - Video is processed frame by frame to minimize memory usage
        - Progress is displayed using a progress bar
        - Results are flushed to disk after each frame for safety
        - Each frame entry includes frame index, detections, and runtime measurements
        - Detections include bounding boxes and confidence scores
        
    Output Format:
        Each line in the JSONL file contains a JSON object with:
        - frame_idx (int): Zero-based frame index
        - detections (list): List of detection results from RetinaNet
        - runtime (dict): Runtime measurements for read and detect operations
    """
    print(f"Processing video: {video_path} on GPU {gpu_id}")
    
    # Load detector for this specific process and GPU
    detector = get_detector(detector_name, gpu_id)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        frame_idx = 0
        
        progress_queue.put(('cuda:' + str(gpu_id), {
            'description': os.path.basename(video_path),
            'completed': 0,
            'total': frame_count
        }))
        while cap.isOpened():
            # Measure frame reading time
            start_time = (time.time_ns() / 1e6)
            ret, frame = cap.read()
            end_time = (time.time_ns() / 1e6)
            read_time = end_time - start_time
            
            if not ret:
                break
            
            # Measure object detection time
            start_time = (time.time_ns() / 1e6)
            detections = polyis.models.retinanet_b3d.detect(frame, detector)
            end_time = (time.time_ns() / 1e6)
            detect_time = end_time - start_time
            
            # Create result entry for this frame
            frame_entry = {
                "frame_idx": frame_idx,
                "detections": detections.tolist() if detections is not None else [],
                "runtime": format_time(read=read_time, detect=detect_time)
            }
            
            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')
            f.flush()
            
            frame_idx += 1
            
            # Send progress update
            progress_queue.put(('cuda:' + str(gpu_id), { 'completed': frame_idx }))
    
    cap.release()
    print(f"GPU {gpu_id}: Completed processing {frame_idx} frames. Results saved to {output_path}")


def get_detector(detector_name: str, gpu_id: int | None = None):
    """
    Get the specified detector model.
    
    Args:
        detector_name (str): Name of the detector to load
        gpu_id (int | None): GPU device ID to use (default: None)
        
    Returns:
        The loaded detector model
        
    Raises:
        ValueError: If the detector name is not supported
    """
    if detector_name == 'retina':
        print(f"Loading RetinaNet detector model on GPU {gpu_id}...")
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
        detector = polyis.models.retinanet_b3d.get_detector(device=device)
        print(f"RetinaNet model loaded successfully on GPU {gpu_id}")
        return detector
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def main(args):
    """
    Main function that orchestrates the object detection process on preprocessed videos.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directory exists
    2. Determines the number of available GPUs
    3. Creates separate processes for each video, each using a different GPU
    4. Limits the total number of processes to the number of available GPUs
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - dataset (str): Name of the dataset to process
            - detector (str): Name of the detector to use
            
    Note:
        - The script expects preprocessed videos from 000_preprocess_dataset.py in:
          {DATA_DIR}/{dataset}/
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Object detection results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/detection.jsonl
        - The detector model is loaded based on the --detector argument
        - Runtime measurements include frame reading and object detection times
        - Processing is parallelized across available GPUs for improved performance
    """
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Using detector: {args.detector}")
    
    # Get all video files from the dataset directory
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Determine number of available GPUs
    num_processes = torch.cuda.device_count()
    print(f"Available GPUs: {num_processes}")
    
    if num_processes == 0:
        print("No CUDA GPUs available. Falling back to CPU processing.")
        num_processes = min(mp.cpu_count(), 20)  # Use available CPUs but cap at 20
        process_ids: list[int | None] = [None] * num_processes  # Will use CPU
    else:
        process_ids = list(range(num_processes))
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(video_files), num_processes)
    print(f"Using {max_processes} processes (limited by {num_processes} GPUs)")
    
    # Create a pool of workers
    print(f"Creating process pool with {max_processes} workers...")
    
    # Prepare arguments for each video
    video_args = []
    for i, video_file in enumerate(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)
        
        # Assign GPU ID (round-robin assignment)
        process_id = process_ids[i % len(process_ids)]
        
        # Create output path for detection results
        output_path = os.path.join(CACHE_DIR, args.dataset, video_file, 'groundtruth', 'detection.jsonl')
        
        video_args.append((video_file_path, args.detector, output_path, process_id))
        print(f"Prepared video: {video_file} for GPU {process_id}")
    
    # Create progress queue and start progress display
    progress_queue = Queue()
    num_gpus = max_processes
    
    # Start progress display in a separate process
    progress_process = mp.Process(target=progress_bars, args=(progress_queue, num_gpus, len(video_files)))
    progress_process.start()
    
    # Create and start video processing processes
    processes: list[mp.Process] = []
    for i, (video_file_path, detector_name, output_path, process_id) in enumerate(video_args):
        # TODO: use queue of gpu ids to ensure that one video is processed in one gpu at a time.
        process = mp.Process(target=detect_objects_in_video, 
                           args=(video_file_path, detector_name, output_path, process_id, progress_queue))
        process.start()
        processes.append(process)
    
    # Wait for all video processing to complete
    for process in processes:
        process.join()
        process.terminate()
    
    # Signal progress display to stop and wait for it
    progress_queue.put(None)
    progress_process.join()
    
    print("All videos processed successfully!")


if __name__ == '__main__':
    main(parse_args())
