#!/usr/local/bin/python

import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Callable
import queue

import cv2
import torch

from polyis.utilities import create_tracker, format_time, ProgressBar, register_tracked_detections, get_config, save_tracking_results
from scripts.p020_exec_classify import load_model, process_frame_tiles


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']
CLASSIFIERS = CONFIG['EXEC']['CLASSIFIERS']
TILE_SIZES = CONFIG['EXEC']['TILE_SIZES']
TILEPADDING_MODES = CONFIG['EXEC']['TILEPADDING_MODES']


def classify(dataset: str, video: str, classifier: str, tilesize: int, tilepadding: str,
             gpu_id: int, idx_queue: queue.Queue[int], shared_frames: list[np.ndarray], shared_relevancy_grids: list[np.ndarray]):
    device = f'cuda:{gpu_id}'
    model = load_model(dataset, tilesize, classifier, device)
    model = model.to(device)

    # Pre-create normalization tensors for ImageNet normalization (1.9x speedup vs torchvision.Normalize)
    normalize_mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=device).view(1, 6, 1, 1)
    normalize_std = torch.tensor([0.229, 0.224, 0.225] * 2, device=device).view(1, 6, 1, 1)

    # Load always_relevant bitmap if available to filter out tiles that have never been relevant
    always_relevant_path = os.path.join(CACHE_DIR, dataset, 'indexing', 'always_relevant', f'{tilesize}_all.npy')
    assert os.path.exists(always_relevant_path), f"Always relevant bitmap not found for {dataset} {tilesize}"
    always_relevant_bitmap = np.load(always_relevant_path)
    # Flatten to match the tile processing order and convert to tensor
    always_relevant_mask = torch.from_numpy(always_relevant_bitmap.flatten()).to(device).to(torch.uint8)

    while True:
        idx = idx_queue.get()
        if idx is None:
            break
        frame = shared_frames[idx]
        previous_frame = shared_frames[idx - 1] if idx > 0 else shared_frames[1]
        relevancy_grid, _ = process_frame_tiles(frame, previous_frame, model, tilesize, device, 
                                                normalize_mean, normalize_std, always_relevant_mask)
        shared_relevancy_grids[idx] = relevancy_grid


def compress(shared_relevancy_grids: list[np.ndarray], shared_compressed_grids: list[np.ndarray]):


def e2d(dataset: str, video: str, classifier: str, tilesize: int, tilepadding: str,
          gpu_id: int, command_queue: mp.Queue):
    """
    Process E2D for a single video/classifier/tilesize combination.
    This function is designed to be called in parallel.
    
    Args:
        dataset (str): Name of the dataset
        video (str): Name of the video file to process
        classifier (str): Classifier name used for detections
        tilesize (int): Tile size used for detections
        tilepadding (str): Whether padding was applied to classification results
        gpu_id (int): GPU ID to use for processing
        command_queue (mp.Queue): Queue for progress updates
    """
    device = f'cuda:{gpu_id}'

    # load video into a list of numpy arrays
    video_path = os.path.join(DATASETS_DIR, dataset, 'test', video)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        assert ret, "Failed to read frame"
        frames.append(frame)
    cap.release()




def main():
    """
    Main function that orchestrates the object tracking process using parallel processing.
    
    This function serves as the entry point for the script. It:
    1. Validates the dataset directories exist
    2. Creates a list of all video/classifier/tilesize combinations to process
    3. Uses multiprocessing to process tasks in parallel across available GPUs
    4. Processes each video and saves tracking results
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects uncompressed detection results from 050_exec_uncompress.py in:
          {CACHE_DIR}/{dataset}/execution/{video_file}/050_uncompressed_detections/{classifier}_{tilesize}/detections.jsonl
        - Tracking results are saved to:
          {CACHE_DIR}/{dataset}/execution/{video_file}/060_uncompressed_tracks/{classifier}_{tilesize}/tracking.jsonl
        - Linear interpolation is optional and controlled by the --no_interpolate flag
        - Processing is parallelized for improved performance
        - The number of processes equals the number of available GPUs
    """
    mp.set_start_method('spawn', force=True)
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        videosets_dir = os.path.join(dataset_dir, 'test')
        
        # Find all videos with uncompressed detection results
        cache_dir = os.path.join(CACHE_DIR, dataset, 'execution')
        for video in os.listdir(videosets_dir):
            # uncompressed_tracking_dir = os.path.join(cache_dir, video, '060_uncompressed_tracks')
            # if os.path.exists(uncompressed_tracking_dir):
            #     shutil.rmtree(uncompressed_tracking_dir)

            for classifier in CLASSIFIERS:
                for tilesize in TILE_SIZES:
                    for tilepadding in TILEPADDING_MODES:
                        funcs.append(partial(track, dataset, video, classifier, tilesize, tilepadding))
    
    print(f"Created {len(funcs)} tasks to process")

    num_gpus = torch.cuda.device_count()
    
    # Set up multiprocessing with ProgressBar
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)
    print("All tasks completed!")


if __name__ == '__main__':
    main()
