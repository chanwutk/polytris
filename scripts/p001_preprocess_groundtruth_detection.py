#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import time
import cv2
import torch
from multiprocessing import Queue

import polyis.models.detector
from polyis.utilities import CACHE_DIR, DATA_DIR, format_time, ProgressBar


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - datasets (list): List of dataset names to process (default: ['caldot1', 'caldot2'])
    """
    parser = argparse.ArgumentParser(description='Execute object detection on preprocessed videos')
    parser.add_argument('--datasets', required=False,
                        default=['caldot1', 'caldot2'],
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def detect_objects(video_path: str, dataset_name: str, output_path: str,
                   gpu_id: int, command_queue: Queue):
    """
    Execute object detection on a single video file and save results to JSONL.
    
    Args:
        video_path (str): Path to the input video file
        dataset_name (str): Name of the dataset (used to auto-select detector)
        output_path (str): Path where the output JSONL file will be saved
        gpu_id (int): GPU device ID to use for this process
        command_queue (Queue): Queue for progress updates
        
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
    # print(f"Processing video: {video_path} on GPU {gpu_id}")
    
    # Load detector for this specific process and GPU (auto-selected based on dataset)
    detector = polyis.models.detector.get_detector(dataset_name, gpu_id)
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        frame_idx = 0
        
        command_queue.put(('cuda:' + str(gpu_id), {
            'description': os.path.basename(video_path).split('.')[0],
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
            detections = polyis.models.detector.detect(frame, detector)
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
            command_queue.put(('cuda:' + str(gpu_id), { 'completed': frame_idx }))
    
    cap.release()
    # print(f"GPU {gpu_id}: Completed processing {frame_idx} frames. Results saved to {output_path}")


def main(args):
    """
    Main function that orchestrates the object detection process on preprocessed videos.
    
    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Determines the number of available GPUs
    4. Creates separate processes for each video, each using a different GPU
    5. Limits the total number of processes to the number of available GPUs
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets (list): List of dataset names to process (detector auto-selected)
            
    Note:
        - The script expects preprocessed videos from 000_preprocess_dataset.py in:
          {DATA_DIR}/{dataset}/
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Object detection results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/detections.jsonl
        - The detector model is automatically selected based on the dataset name
        - Runtime measurements include frame reading and object detection times
        - Processing is parallelized across available GPUs for improved performance
    """
    datasets = args.datasets

    # Create task functions
    funcs = []
    for dataset in datasets:
        dataset_dir = os.path.join(DATA_DIR, dataset)
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"
        
        # # Show detector info for this dataset
        # detector_info = polyis.models.detector.get_detector_info(dataset)
        # print(f"Using detector: {detector_info['detector']} ({detector_info['description']})")
        
        # Get all video files from the dataset directory
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        assert len(video_files) > 0, f"No video files found in {dataset_dir}"
        
        for video_file in video_files:
            video_file_path = os.path.join(dataset_dir, video_file)
            output_path = os.path.join(CACHE_DIR, dataset, 'execution', video_file, 'groundtruth', 'detections.jsonl')
            funcs.append(partial(detect_objects, video_file_path, dataset, output_path))
            # detect_objects(video_file_path, dataset, output_path, 0, Queue())
    
    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
