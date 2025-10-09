#!/usr/local/bin/python

import argparse
import json
import shutil
from pathlib import Path

import cv2

from polyis.utilities import CACHE_DIR, DATA_DIR, DATASETS_TO_TEST


SELECTIVITY = 0.05
CLASSIFIER_SIZES = [32, 64, 128]
PADDING_SIZES = [0, 1, 2]
DIFF_THRESHOLDS = [10, 20, 30]
DIFF_SCALE = [1, 2, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune parameters for the model.")
    parser.add_argument("--datasets", type=str,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help="The dataset names (space-separated).")
    parser.add_argument("--selectivity", type=float,
                        default=SELECTIVITY,
                        help="Selectivity parameter for tuning.")
    parser.add_argument("--num_snippets", type=int,
                        default=10,
                        help="Number of snippets to extract from the video for tuning.")
    parser.add_argument("--tracking_selectivity_multiplier", type=int,
                        default=4,
                        help="Multiplier for tracking selectivity.")
    parser.add_argument("--datasets_dir", type=str,
                        default=DATA_DIR,
                        help="Directory containing the dataset.")
    return parser.parse_args()


def main(args):
    """
    Main function to process videos and create detection and tracking segments.
    
    This function:
    1. Iterates through multiple datasets
    2. For each dataset, iterates through all MP4 videos in the specified dataset directory
    3. Calculates snippet sizes for detection and tracking based on selectivity parameters
    4. Creates segment metadata files ({video_file}.segments.jsonl) for both detection and tracking
    5. Sets up directory structure in the cache directory
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - datasets_dir: Directory containing the datasets
            - datasets: List of dataset names to process
            - selectivity: Selectivity parameter for tuning
            - num_snippets: Number of snippets to extract
            - tracking_selectivity_multiplier: Multiplier for tracking snippet size
    
    Note:
        The function creates two types of segments:
        - Detection segments: Smaller snippets for object detection
        - Tracking segments: Larger snippets for object tracking (multiplied by tracking_selectivity_multiplier)
        
        Segment metadata is saved as JSONL files with frame start/end positions.
        Video files are not actually extracted (commented out) - only metadata is generated.
        Output structure: {dataset}/indexing/segment/{video_file}.segments.jsonl
    """
    datasets_dir = args.datasets_dir
    datasets = args.datasets
    selectivity = args.selectivity

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        dataset_path = Path(datasets_dir) / dataset
        output_dir = Path(dataset) / 'indexing' / 'segment'

        cache_output_dir = Path(CACHE_DIR) / output_dir
        if cache_output_dir.exists():
            shutil.rmtree(cache_output_dir)
        cache_output_dir.mkdir(parents=True)

        detection_dir = cache_output_dir / 'detection'
        if detection_dir.exists():
            shutil.rmtree(detection_dir)
        detection_dir.mkdir(parents=True)
        
        tracking_dir = cache_output_dir / 'tracking'
        if tracking_dir.exists():
            shutil.rmtree(tracking_dir)
        tracking_dir.mkdir(parents=True)

        for input_video in dataset_path.iterdir():
            if not input_video.name.endswith(".mp4"):
                continue

            # Remove .mp4 extension for output filename
            video_name = str(input_video).split('/')[-1]
            input_video_path = input_video
            print(input_video_path)

            cap = cv2.VideoCapture(input_video_path)
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

            detection_file = detection_dir / f'{video_name}.segments.jsonl'
            tracking_file = tracking_dir / f'{video_name}.segments.jsonl'
            with (open(detection_file, 'w') as fd, open(tracking_file, 'w') as ft):
                print(f"Writing detection segments to {detection_file}")
                for i, (start, end) in enumerate(zip(starts_d, ends_d)):
                    fd.write(json.dumps({ 'idx': i, 'start': start, 'end': end }) + '\n')
                
                print(f"Writing tracking segments to {tracking_file}")
                for i, (start, end) in enumerate(zip(starts_t, ends_t)):
                    ft.write(json.dumps({ 'idx': i, 'start': start, 'end': end }) + '\n')


if __name__ == "__main__":
    main(parse_args())