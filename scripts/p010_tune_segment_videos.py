#!/usr/local/bin/python

import argparse
import json
import os
import shutil

import cv2
import tqdm

from polyis.utilities import CACHE_DIR, DATA_DIR


SELECTIVITY = 0.05
CLASSIFIER_SIZES = [32, 64, 128]
PADDING_SIZES = [0, 1, 2]
DIFF_THRESHOLDS = [10, 20, 30]
DIFF_SCALE = [1, 2, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune parameters for the model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='b3d',
        help="The dataset name.",
    )
    parser.add_argument(
        "--selectivity",
        type=float,
        default=SELECTIVITY,
        help="Selectivity parameter for tuning.",
    )
    parser.add_argument(
        "--num_snippets",
        type=int,
        default=10,
        help="Number of snippets to extract from the video for tuning.",
    )
    parser.add_argument(
        "--tracking_selectivity_multiplier",
        type=int,
        default=4,
        help="Multiplier for tracking selectivity.",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing the dataset.",
    )
    return parser.parse_args()


def main(args):
    """
    Main function to process videos and create detection and tracking segments.
    
    This function:
    1. Iterates through all MP4 videos in the specified dataset directory
    2. Calculates snippet sizes for detection and tracking based on selectivity parameters
    3. Creates segment metadata files (segments.jsonl) for both detection and tracking
    4. Sets up directory structure in the cache directory
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets_dir: Directory containing the dataset
            - dataset: Name of the specific dataset to process
            - selectivity: Selectivity parameter for tuning
            - num_snippets: Number of snippets to extract
            - tracking_selectivity_multiplier: Multiplier for tracking snippet size
    
    Note:
        The function creates two types of segments:
        - Detection segments: Smaller snippets for object detection
        - Tracking segments: Larger snippets for object tracking (multiplied by tracking_selectivity_multiplier)
        
        Segment metadata is saved as JSONL files with frame start/end positions.
        Video files are not actually extracted (commented out) - only metadata is generated.
    """
    datasets_dir = args.datasets_dir
    dataset = args.dataset
    selectivity = args.selectivity

    for input_video in os.listdir(os.path.join(datasets_dir, dataset)):
        if not input_video.endswith(".mp4"):
            continue

        output_dir = os.path.join(dataset, input_video, 'segments')
        input_video = os.path.join(datasets_dir, dataset, input_video)
        print(input_video)

        cap = cv2.VideoCapture(input_video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(num_frames)

        num_snippets: int = args.num_snippets

        # Calculate the size of each snippet for detection.
        snippet_d_size = int(num_frames * selectivity / num_snippets)
        print(f"Detection snippet size: {snippet_d_size}")
        # Calculate the size of each snippet for tracking. Snippet size is longer than detection snippet size.
        snippet_t_size = int(num_frames * selectivity * args.tracking_selectivity_multiplier / num_snippets)
        print(f"Tracking snippet size: {snippet_t_size}")

        # Delect `num_snippets` snippets from the video. Each snippet is `snippet_d_size` frames long.
        starts_d = [s * (num_frames // num_snippets) for s in range(num_snippets)]
        ends_d = [s + snippet_d_size for s in starts_d]

        # Delect `num_snippets` snippets from the video. Each snippet is `snippet_t_size` frames long.
        starts_t = [s * (num_frames // num_snippets) for s in range(num_snippets)]
        ends_t = [s + snippet_t_size for s in starts_t]

        if os.path.exists(os.path.join(CACHE_DIR, output_dir)):
            shutil.rmtree(os.path.join(CACHE_DIR, output_dir))
        os.makedirs(os.path.join(CACHE_DIR, output_dir))

        if os.path.exists(os.path.join(CACHE_DIR, output_dir, 'detection')):
            shutil.rmtree(os.path.join(CACHE_DIR, output_dir, 'detection'))
        os.makedirs(os.path.join(CACHE_DIR, output_dir, 'detection'))
        if os.path.exists(os.path.join(CACHE_DIR, output_dir, 'tracking')):
            shutil.rmtree(os.path.join(CACHE_DIR, output_dir, 'tracking'))
        os.makedirs(os.path.join(CACHE_DIR, output_dir, 'tracking'))

        with (open(os.path.join(CACHE_DIR, output_dir, 'detection', 'segments.jsonl'), 'w') as fd,
              open(os.path.join(CACHE_DIR, output_dir, 'tracking', 'segments.jsonl'), 'w') as ft):
            for i, (start, end) in tqdm.tqdm(enumerate(zip(starts_d, ends_d)), total=num_snippets):
                fd.write(json.dumps({ 'idx': i, 'start': start, 'end': end }) + '\n')
            
            for i, (start, end) in tqdm.tqdm(enumerate(zip(starts_t, ends_t)), total=num_snippets):
                ft.write(json.dumps({ 'idx': i, 'start': start, 'end': end }) + '\n')


if __name__ == "__main__":
    main(parse_args())