#!/usr/local/bin/python

import argparse
import json
import os
import multiprocessing as mp
from functools import partial
from typing import Callable

import cv2
import torch

from polyis.utilities import CACHE_DIR, CLASSIFIERS_CHOICES, CLASSIFIERS_TO_TEST, ProgressBar, DATASETS_TO_TEST, TILE_SIZES, DATASETS_DIR
from polyis.train.benchmark_model_optimization import benchmark_model_optimization


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark optimization methods for trained classifier models')
    parser.add_argument('--classifiers', required=False,
                        default=CLASSIFIERS_TO_TEST,
                        choices=CLASSIFIERS_CHOICES,
                        nargs='+',
                        help='Model types to benchmark (can specify multiple)')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names to search for trained models (space-separated)')
    parser.add_argument('--iterations', type=int, default=128,
                        help='Number of iterations for benchmarking')
    return parser.parse_args()


def get_video_resolution(dataset_name: str) -> tuple[int, int]:
    """
    Get width and height of a video in the dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Width and height of the video
    """
    # Find a video file in the dataset
    videoset_dir = os.path.join(DATASETS_DIR, dataset_name, 'test')
    if not os.path.exists(videoset_dir):
        # Try 'train' directory if 'test' doesn't exist
        videoset_dir = os.path.join(DATASETS_DIR, dataset_name, 'train')
    
    if not os.path.exists(videoset_dir):
        # Fallback to default resolution if no video directory found
        return (720, 480)
    
    # Find first video file
    video_files = [f for f in os.listdir(videoset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        # Fallback to default resolution if no videos found
        return (720, 480)
    
    # Get resolution from first video
    video_path = os.path.join(videoset_dir, video_files[0])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback to default resolution if video can't be opened
        return 720, 480
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return width, height


def benchmark_classifier(datasets: list[str], width: int, height: int, classifier_name: str,
                         tile_size: int, iterations: int, gpu_id: int, _: mp.Queue):
    """
    Benchmark optimization methods for a specific classifier, tile size, and video resolution.
    
    Args:
        datasets: List of dataset names
        width: Width of the video
        height: Height of the video
        classifier_name: Name of the classifier
        tile_size: Tile size
        iterations: Number of iterations for benchmarking
        gpu_id: GPU ID to use
    """
    device = f'cuda:{gpu_id}'
    
    # Find the trained model for this classifier, tile size, and dataset
    model_path = os.path.join(
        CACHE_DIR, datasets[0], 'indexing', 'training', 'results',
        f'{classifier_name}_{tile_size}', 'model.pth'
    )
    
    if not os.path.exists(model_path):
        print(f"No trained model found for {classifier_name} (tile_size={tile_size}, dataset={datasets})")
        return

    batch_size = (width * height) // (tile_size * tile_size)
        
    # Load the model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Model loaded from {model_path}... done")
    model = model.to(device)
    model.eval()
    
    # Run benchmark
    results_sorted = benchmark_model_optimization(model, device, tile_size, batch_size, iterations)
    for dataset in datasets:
        # Create results directory per dataset
        results_dir = os.path.join(
            CACHE_DIR, dataset, 'indexing', 'training', 'results',
            f'{classifier_name}_{tile_size}'
        )
        # print(f"Saving results", results_sorted)
        # Ensure directory exists before writing
        os.makedirs(results_dir, exist_ok=True)
        # Save to JSONL file
        output_path = os.path.join(results_dir, 'model_compilation.jsonl')
        with open(output_path, 'w') as f:
            for result in results_sorted:
                f.write(json.dumps(result) + '\n')


def main(args):
    mp.set_start_method('spawn', force=True)

    resolutions: dict[tuple[int, int], list[str]] = {}
    for dataset in args.datasets:
        width, height = get_video_resolution(dataset)
        if (width, height) not in resolutions:
            resolutions[(width, height)] = []
        resolutions[(width, height)].append(dataset)

    
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for width, height in resolutions.keys():
        for classifier_name in args.classifiers:
            for tile_size in TILE_SIZES:
                func = partial(benchmark_classifier, resolutions[(width, height)], width,
                               height, classifier_name, tile_size, args.iterations)
                funcs.append(func)
    
    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    
    # ProgressBar(num_workers=1, num_tasks=len(funcs), off=True).run_all(funcs)
    for func in funcs:
        func(0, mp.Queue())


if __name__ == '__main__':
    main(parse_args())

