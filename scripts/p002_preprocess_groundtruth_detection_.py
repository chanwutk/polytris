#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import time
import cv2
import torch
import queue
import shutil

from polyis.io.video_capture import VideoCapture
from polyis.utilities import format_time, ProgressBar, get_num_frames, get_config


CONFIG = get_config()
EXEC_DATASETS = CONFIG['EXEC']['DATASETS']
VIDEO_SETS = CONFIG['EXEC']['VIDEO_SETS']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']


def detect_objects(dataset: str, video_file: str, gpu_id: int, command_queue: queue.Queue):
    """
    Load groundtruth detection data from JSON files and save results to JSONL.
    
    Args:
        dataset (str): Name of the dataset
        video_file (str): Name of the video file (e.g., "test/0.mp4")
        gpu_id (int): GPU device ID to use for this process (unused, kept for compatibility)
        command_queue (Queue): Queue for progress updates
        
    Note:
        - Groundtruth data is loaded from /otif-dataset/dataset/{dataset}/{videoset}/yolov3-704x480/{videonumber}.json
        - Progress is displayed using a progress bar
        - Results are flushed to disk after each frame for safety
        - Each frame entry includes frame index and detections
        - Detections include bounding boxes in [x1, y1, x2, y2, score] format with score=1.0 for groundtruth
        
    Output Format:
        Each line in the JSONL file contains a JSON object with:
        - frame_idx (int): Zero-based frame index
        - detections (list): List of detection results in [x1, y1, x2, y2, score] format
    """

    # Extract videoset and video filename from video_file (e.g., "test/0.mp4")
    videoset = video_file.split('/')[0]
    video_filename = video_file.split('/')[1]
    # Extract video number from filename (e.g., "te0.mp4" -> "0")
    video_number = os.path.splitext(video_filename)[0][2:]
    
    # Construct path to groundtruth JSON file
    gt_json_path = os.path.join('/otif-dataset/dataset', dataset.split('-')[0], videoset, 'yolov3-704x480', f'{int(video_number)}.json')
    assert os.path.exists(gt_json_path), f"Groundtruth JSON file not found: {gt_json_path}"
    
    video_name = video_filename
    output_path = os.path.join(CACHE_DIR, dataset, 'execution', video_name, '000_groundtruth', 'detection_.jsonl')
    
    # Load groundtruth annotations from JSON file
    with open(gt_json_path, 'r') as f:
        annotations = json.load(f)
    
    # Get video path to determine frame count
    dataset_dir = os.path.join(DATASETS_DIR, dataset)
    video_path = os.path.join(dataset_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    with VideoCapture(video_path) as cap:
        frame_num = cap.frame_count
    assert frame_num == frame_count, f"Frame count mismatch: {frame_num} != {frame_count}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        assert len(annotations) == frame_count, f"Number of annotations ({len(annotations)}) does not match frame count ({frame_count})"
        
        # Process each frame
        for frame_idx in range(frame_count):
            # Get annotations for this frame (annotations is a list indexed by frame_idx)
            frame_annos = annotations[frame_idx]
            
            # Convert annotations to detection format [x1, y1, x2, y2, score]
            detections = []
            for obj in frame_annos:
                # Extract bounding box coordinates
                left = obj["left"]
                top = obj["top"]
                right = obj["right"]
                bottom = obj["bottom"]
                cls = obj["class"]
                assert cls == "car", f"Class {cls} is not supported"
                score = obj["score"]
                
                # Convert to [x1, y1, x2, y2, score] format
                detections.append([float(left), float(top), float(right), float(bottom), score])
            
            # Create result entry for this frame
            frame_entry = {
                "frame_idx": frame_idx,
                "detections": detections,
            }
            
            # Write to JSONL file
            f.write(json.dumps(frame_entry) + '\n')


def main():
    """
    Main function that orchestrates the groundtruth detection loading process.
    
    This function serves as the entry point for the script. It:
    1. Processes multiple datasets in sequence
    2. For each dataset, validates the dataset directory exists
    3. Determines the number of available GPUs
    4. Creates separate processes for each video
    5. Limits the total number of processes to the number of available GPUs
    
    Note:
        - The script expects groundtruth JSON files in:
          /otif-dataset/dataset/{dataset}/{videoset}/yolov3-704x480/{videonumber}.json
        - Videos are identified by common video file extensions (.mp4, .avi, .mov, .mkv)
        - Groundtruth detection results are saved to:
          {CACHE_DIR}/{dataset}/{video_file}/000_groundtruth/detection.jsonl
        - Groundtruth annotations are converted to [x1, y1, x2, y2, score] format with score=1.0
        - Processing is parallelized across available GPUs for improved performance
    """

    # Create task functions
    funcs = []
    for dataset in [dataset for dataset in EXEC_DATASETS if dataset.startswith('caldot')]:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist"
        
        # Get all video files from the dataset directory
        videos: list[str] = []
        for videoset in ['test']:
            videoset_dir = os.path.join(dataset_dir, videoset)
            assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"
            videos.extend([videoset + '/' + f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        assert len(videos) > 0, f"No video files found in {dataset_dir}"
        
        for video in videos:
            funcs.append(partial(detect_objects, dataset, video))
    
    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main()
