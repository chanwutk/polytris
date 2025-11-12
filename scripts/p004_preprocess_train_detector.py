#!/usr/local/bin/python

"""
Create COCO format dataset from tracking results for detector training.
Extracts frames from videos and generates COCO JSON annotations.
The output can be used with any COCO-compatible training framework like PyTorch vision.
"""

import argparse
import datetime
import json
import os
from typing import Any

import cv2
import numpy as np

from polyis.utilities import (
    CACHE_DIR,
    DATASETS_DIR,
    VIDEO_SETS,
    DATASETS_TO_TEST,
    load_tracking_results,
    ProgressBar,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create COCO format dataset from tracking results"
    )
    parser.add_argument(
        "--datasets",
        required=False,
        default=DATASETS_TO_TEST,
        nargs="+",
        help="Dataset names (space-separated)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Sample every Nth frame from videos (1 = use all frames)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="/polyis-data/coco_datasets",
        help="Base directory for COCO dataset output",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for extracted frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (1-100, only used when --image-format=jpg)",
    )
    return parser.parse_args()


def create_dataset_samples(
    datasets: list[str], sample_rate: int = 1
) -> list[dict[str, Any]]:
    """
    Create dataset samples from tracking results.

    Args:
        datasets: List of dataset names to process
        sample_rate: Sample every Nth frame (1 = use all frames)

    Returns:
        List of samples where each sample contains video_path, frame_idx, and boxes
    """
    all_samples = []

    print("Creating dataset from tracking results...")

    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Find all videos with tracking results
        dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
        execution_dir = os.path.join(dataset_cache_dir, "execution")

        video_dirs = []
        # Iterate through videos in execution directory
        for item in os.listdir(execution_dir):
            item_path = os.path.join(execution_dir, item)
            # Check if it's a directory with tracking results
            if os.path.isdir(item_path):
                tracking_path = os.path.join(
                    item_path, "000_groundtruth", "tracking.jsonl"
                )
                # Add to list if tracking file exists
                if os.path.exists(tracking_path):
                    video_dirs.append(item)

        print(f"  Found {len(video_dirs)} videos with tracking results")

        # Process each video
        for video_file in video_dirs:
            # Load tracking results
            frame_tracks = load_tracking_results(CACHE_DIR, dataset, video_file)

            # Find video file path
            video_path = None
            # Search for video in all video sets
            for videoset in VIDEO_SETS:
                candidate_path = os.path.join(
                    DATASETS_DIR, dataset, videoset, video_file
                )
                # Use first found path
                if os.path.exists(candidate_path):
                    video_path = candidate_path
                    break

            # Skip if video not found
            if video_path is None:
                print(f"    Warning: Video not found for {video_file}, skipping")
                continue

            # Create samples for each frame with detections
            frame_count = 0
            for frame_idx, tracks in frame_tracks.items():
                # Skip frames with no detections
                if len(tracks) == 0:
                    continue

                # Apply sampling rate
                if frame_idx % sample_rate != 0:
                    continue

                # Create sample entry
                sample = {
                    "video_path": video_path,
                    "video_name": video_file,
                    "frame_idx": frame_idx,
                    "boxes": tracks,
                    "dataset": dataset,
                }
                all_samples.append(sample)
                frame_count += 1

            print(f"    {video_file}: {frame_count} frames")

    print(f"\nTotal samples: {len(all_samples)}")
    return all_samples


def extract_and_save_frame(
    video_path: str,
    frame_idx: int,
    output_path: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> tuple[int, int]:
    """
    Extract a frame from video and save as image file.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        output_path: Path to save the extracted frame
        image_format: Image format ('jpg' or 'png')
        jpeg_quality: JPEG quality (1-100, only for jpg)

    Returns:
        Tuple of (height, width) of the extracted frame
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    # Read frame
    ret, frame = cap.read()
    cap.release()

    # Check if frame was read successfully
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save frame based on format
    if image_format == "jpg":
        # Save as JPEG with specified quality
        cv2.imwrite(
            output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
    else:
        # Save as PNG (lossless)
        cv2.imwrite(output_path, frame)

    return height, width


def create_coco_json(
    samples: list[dict[str, Any]],
    output_dir: str,
    split_name: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> None:
    """
    Create COCO format JSON annotations from samples.

    Args:
        samples: List of sample dictionaries with video_path, frame_idx, boxes
        output_dir: Output directory for COCO dataset
        split_name: Name of the split ('train' or 'val')
        image_format: Image format ('jpg' or 'png')
        jpeg_quality: JPEG quality for jpg format
    """
    print(f"\nGenerating COCO annotations for {split_name} split...")
    print(f"  Total samples: {len(samples)}")

    # Initialize COCO JSON structure
    coco_data = {
        "info": {
            "description": f"Vehicle detection dataset - {split_name} split",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Polyis",
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "vehicle",
                "supercategory": "vehicle",
            }
        ],
    }

    # Create image output directory
    image_dir = os.path.join(output_dir, split_name)
    os.makedirs(image_dir, exist_ok=True)

    # Counters for IDs
    image_id = 1
    annotation_id = 1

    print(f"  Extracting frames to {image_dir}/")

    # Process each sample
    for idx, sample in enumerate(samples):
        # Create unique image filename
        # Format: {dataset}_{video_name}_{frame_idx:06d}.{ext}
        image_filename = f"{sample['dataset']}_{sample['video_name']}_{sample['frame_idx']:06d}.{image_format}"
        image_path = os.path.join(image_dir, image_filename)

        # Extract and save frame
        try:
            height, width = extract_and_save_frame(
                sample["video_path"],
                sample["frame_idx"],
                image_path,
                image_format,
                jpeg_quality,
            )
        except Exception as e:
            print(f"    Warning: Failed to extract frame for {image_filename}: {e}")
            continue

        # Add image entry
        image_entry = {
            "id": image_id,
            "file_name": image_filename,
            "height": height,
            "width": width,
            "license": 1,
        }
        coco_data["images"].append(image_entry)

        # Process annotations (bounding boxes)
        for track in sample["boxes"]:
            # tracks format: [track_id, x1, y1, x2, y2]
            # Extract coordinates (skip track_id)
            x1, y1, x2, y2 = track[1:5]

            # Convert from [x1, y1, x2, y2] to COCO format [x, y, width, height]
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)

            # Calculate area
            area = w * h

            # Skip invalid boxes
            if area <= 0 or w <= 0 or h <= 0:
                continue

            # Create annotation entry
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # vehicle category
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_entry)
            annotation_id += 1

        # Increment image ID
        image_id += 1

        # Progress update
        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            print(f"    Processed {idx + 1}/{len(samples)} samples")

    # Create annotations directory
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Save COCO JSON
    output_json_path = os.path.join(annotations_dir, f"instances_{split_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"  Saved annotations to {output_json_path}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")


def generate_coco_dataset(dataset: str, args) -> None:
    """
    Generate COCO format dataset for a single dataset.

    Args:
        dataset: Dataset name to process
        args: Parsed command line arguments
    """
    print("\n" + "=" * 80)
    print(f"Generating COCO dataset for: {dataset}")
    print("=" * 80)

    # Create output directory
    output_dir = os.path.join(args.output_base_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load tracking results and create samples
    print(f"\nLoading tracking results...")
    all_samples = create_dataset_samples([dataset], args.sample_rate)

    # Check if we have samples
    if len(all_samples) == 0:
        print(f"  Warning: No samples found for {dataset}, skipping")
        return

    # Split into train and validation
    num_train = int(len(all_samples) * args.train_split)
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:]

    print(f"\nSplit configuration:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")

    # Generate training set
    if len(train_samples) > 0:
        create_coco_json(
            train_samples,
            output_dir,
            "train",
            args.image_format,
            args.jpeg_quality,
        )
    else:
        print("\n  Warning: No training samples, skipping train split")

    # Generate validation set
    if len(val_samples) > 0:
        create_coco_json(
            val_samples,
            output_dir,
            "val",
            args.image_format,
            args.jpeg_quality,
        )
    else:
        print("\n  Warning: No validation samples, skipping val split")

    print("\n" + "=" * 80)
    print(f"COCO dataset generation complete for {dataset}!")
    print(f"Dataset location: {output_dir}")
    print("\nTo use with PyTorch vision training:")
    print(f"  python train.py --data-path {output_dir} --dataset coco \\")
    print(f"    --model fasterrcnn_resnet50_fpn --epochs 26")
    print("=" * 80)


def main(args):
    """
    Main function to generate COCO format datasets.

    This function:
    1. Loads tracking results from p003_preprocess_groundtruth_tracking.py
    2. Extracts frames from videos and saves them as images
    3. Creates COCO format JSON annotations (instances_train.json, instances_val.json)
    4. Outputs to /polyis-data/coco_datasets/{dataset}/

    The output can be used with any COCO-compatible training framework, including:
    - PyTorch vision reference implementation (train.py)
    - Detectron2
    - MMDetection
    - YOLOv5/v8 (with conversion)

    Args:
        args: Parsed command line arguments
    """
    print("=" * 80)
    print("COCO Dataset Generator")
    print("=" * 80)
    print(f"Datasets to process: {args.datasets}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Train/val split: {args.train_split:.2f}")
    print(f"Image format: {args.image_format}")
    if args.image_format == "jpg":
        print(f"JPEG quality: {args.jpeg_quality}")
    print(f"Output base directory: {args.output_base_dir}")

    # Process each dataset
    for dataset in args.datasets:
        generate_coco_dataset(dataset, args)

    print("\n" + "=" * 80)
    print("All datasets processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
