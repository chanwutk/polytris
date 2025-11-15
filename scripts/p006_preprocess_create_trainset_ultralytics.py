#!/usr/local/bin/python

"""
Create Ultralytics training dataset from CalDOT video annotations.
Extracts frames from videos and converts annotations to Ultralytics format.
"""

import argparse
import json
from pathlib import Path
import shutil

import cv2
from tqdm import tqdm

from polyis.train.data import (
    adjust_val_frames_for_prefix,
    collect_valid_frames,
    discover_videos_in_subsets,
    find_highest_resolution_annotations,
    get_dataset_subsets,
    get_video_annotation_path,
    split_frames_train_val,
    get_adjusted_frame_stride,
)
from polyis.train.data.ultralytics import create_data_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Ultralytics dataset from CalDOT videos and annotations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="caldot1",
        help="Dataset name (default: caldot1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Ultralytics dataset (default: /polyis-data/training/ultralytics/{dataset}/training-data)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of frames to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Extract every Nth frame (default: 5)",
    )
    return parser.parse_args()


def extract_frames_and_annotations(video_path, anno_path, train_image_dir, train_label_dir, 
                                   val_image_dir, val_label_dir, video_id, frame_stride, val_frames):
    """
    Extract frames from video and save Ultralytics format annotations.

    Args:
        video_path: Path to video file
        anno_path: Path to annotation JSON file
        train_image_dir: Directory to save training images
        train_label_dir: Directory to save training labels
        val_image_dir: Directory to save validation images
        val_label_dir: Directory to save validation labels
        video_id: Video identifier for naming files
        frame_stride: Extract every Nth frame (adjusted based on video FPS)
        val_frames: Set of frame_idx that should go to validation.
                    If None, all frames go to train directories.

    Returns:
        Tuple of (train_frames_extracted, val_frames_extracted)
    """
    # Load annotations from JSON file
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))

    # Get adjusted stride based on video FPS
    actual_stride = get_adjusted_frame_stride(video_path, frame_stride)

    train_frames_extracted = 0
    val_frames_extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on adjusted stride
        if frame_idx % actual_stride != 0:
            frame_idx += 1
            continue

        # Check if we have annotations for this frame
        if frame_idx >= len(annotations):
            break

        frame_annos = annotations[frame_idx]

        # Include all frames, even those without annotations
        # Frames without annotations will have empty label files (negative examples)

        # Determine if this frame should go to validation
        is_val = False
        if val_frames is not None:
            is_val = frame_idx in val_frames

        # Select target directories based on split
        if is_val:
            target_image_dir = val_image_dir
            target_label_dir = val_label_dir
        else:
            target_image_dir = train_image_dir
            target_label_dir = train_label_dir

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Generate filenames
        image_filename = f"{video_id}_{frame_idx:06d}.jpg"
        label_filename = f"{video_id}_{frame_idx:06d}.txt"

        # Save frame image to appropriate directory
        image_path = target_image_dir / image_filename
        cv2.imwrite(str(image_path), frame)

        # Convert annotations to Ultralytics format and save
        # Create label file even if empty (for negative examples)
        label_path = target_label_dir / label_filename
        with open(label_path, 'w') as label_file:
            # Only write annotations if frame has objects
            if frame_annos:
                for obj in frame_annos:
                    # Extract bounding box coordinates
                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    right = obj.get("right", 0)
                    bottom = obj.get("bottom", 0)

                    # Convert to Ultralytics format: class_id center_x center_y width height (normalized 0-1)
                    bbox_width = right - left
                    bbox_height = bottom - top

                    # Skip invalid boxes
                    if bbox_width <= 0 or bbox_height <= 0:
                        raise Exception(f"Invalid box: {left}, {top}, {right}, {bottom}")

                    # Calculate normalized center coordinates and dimensions
                    center_x = (left + bbox_width / 2.0) / width
                    center_y = (top + bbox_height / 2.0) / height
                    norm_width = bbox_width / width
                    norm_height = bbox_height / height

                    # Class ID (0 for "car" in CalDOT dataset)
                    class_id = 0

                    # Write Ultralytics format line
                    label_file.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            # If frame_annos is empty, label file will be empty (negative example)

        if is_val:
            val_frames_extracted += 1
        else:
            train_frames_extracted += 1
        frame_idx += 1

    cap.release()
    return (train_frames_extracted, val_frames_extracted)


def setup_output_directories(output_dir):
    """
    Create output directory structure for Ultralytics dataset.

    Args:
        output_dir: Root output directory

    Returns:
        Tuple of (train_image_dir, val_image_dir, train_label_dir, val_label_dir)
    """
    # Remove existing directory if present
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists and will be overwritten.")
        shutil.rmtree(output_dir)

    # Create directory structure for Ultralytics format
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image directories
    train_image_dir = output_dir / "images" / "train"
    val_image_dir = output_dir / "images" / "val"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)

    # Label directories
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    return train_image_dir, val_image_dir, train_label_dir, val_label_dir


def process_video_file(video_file, anno_dir, val_frames,
                       train_image_dir, val_image_dir,
                       train_label_dir, val_label_dir,
                       frame_stride, id_prefix: str = ""):
    """
    Process a single video file and extract frames with annotations.

    Args:
        video_file: Path to video file
        anno_dir: Directory containing annotation files
        val_frames: Set of frame_idx for validation frames
        train_image_dir: Directory for training images
        val_image_dir: Directory for validation images
        train_label_dir: Directory for training labels
        val_label_dir: Directory for validation labels
        frame_stride: Extract every Nth frame
        id_prefix: Optional prefix to ensure unique image file names across subsets

    Returns:
        Tuple of (train_frames_extracted, val_frames_extracted)
    """
    # Get video ID and annotation file path
    video_id, anno_file = get_video_annotation_path(video_file, anno_dir)

    # Use prefixed video id to avoid filename collisions across subsets
    prefixed_video_id = f"{id_prefix}{video_id}" if id_prefix else video_id

    # Extract frames and annotations
    train_count, val_count = extract_frames_and_annotations(
        video_file, anno_file, train_image_dir, train_label_dir,
        val_image_dir, val_label_dir, prefixed_video_id, frame_stride,
        val_frames
    )

    return (train_count, val_count)


def main():
    args = parse_args()

    # Setup paths for CalDOT dataset structure
    base_root = Path("/otif-dataset/dataset")
    dataset = args.dataset

    output_dir = Path(args.output_dir or f"/polyis-data/training/ultralytics/{dataset}/training-data")

    # Create output directory structure
    train_image_dir, val_image_dir, train_label_dir, val_label_dir = setup_output_directories(output_dir)

    print("=" * 80)
    print(f"Creating Ultralytics Dataset for {dataset}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {args.val_split}")
    print(f"Frame stride: {args.frame_stride}")
    print()

    # Track statistics
    subsets = get_dataset_subsets(base_root, dataset)
    total_counts = {"train": 0, "valid": 0, "test": 0}
    total_frames = {"train": 0, "val": 0}

    # Discover all videos in subsets
    video_info = discover_videos_in_subsets(base_root, dataset)

    # Count videos per subset for statistics
    for subset_name, _, _ in video_info:
        if subset_name in total_counts:
            total_counts[subset_name] += 1

    # Print subset information
    for subset_name, subset_root in subsets:
        if subset_root.exists():
            video_dir = subset_root / "video"
            if video_dir.exists():
                anno_dir = find_highest_resolution_annotations(subset_root)
                print(f"Using annotations for '{subset_name}' from: {anno_dir.name}")
                print(f"Found {total_counts[subset_name]} videos in '{subset_name}'")

    # First pass: Collect all valid frames from all videos
    print("\nCollecting valid frames from all videos...")
    all_valid_frames = []

    for subset_name, video_file, anno_dir in tqdm(video_info, desc="Scanning videos"):
        video_id, anno_file = get_video_annotation_path(video_file, anno_dir)

        # Collect valid frames (using original video_id without prefix for splitting)
        valid_frames = collect_valid_frames(video_file, anno_file, video_id, args.frame_stride)
        all_valid_frames.extend((subset_name, *frame) for frame in valid_frames)

    print(f"\nTotal frames collected: {len(all_valid_frames)} (including frames without annotations)")

    # Split frames into train/val with seed for reproducibility
    print("\nSplitting frames into train/val sets...")
    print(f"Using val_split={args.val_split} (expecting {int(len(all_valid_frames) * args.val_split)} validation frames)")
    val_frames = split_frames_train_val(all_valid_frames, args.val_split)
    print(f"Split complete: {len(val_frames)} frames marked for validation")

    val_frames_map = {}
    for subset_name, video_id, frame_idx in val_frames:
        if (subset_name, video_id) not in val_frames_map:
            val_frames_map[(subset_name, video_id)] = []
        val_frames_map[(subset_name, video_id)].append(frame_idx)

    # Second pass: Extract frames and save to appropriate directories
    print("\nExtracting frames and saving to dataset...")
    for subset_name, video_file, anno_dir in tqdm(video_info, desc="Processing all videos"):
        video_id, _ = get_video_annotation_path(video_file, anno_dir)
        train_count, val_count = process_video_file(
            video_file, anno_dir, val_frames_map[(subset_name, video_id)],
            train_image_dir, val_image_dir,
            train_label_dir, val_label_dir,
            args.frame_stride, id_prefix=f"{subset_name}_"
        )

        # Track frame counts
        total_frames["train"] += train_count
        total_frames["val"] += val_count

    # Create data.yaml configuration file
    create_data_yaml(output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)
    print(f"Subsets discovered: train={total_counts['train']}, valid={total_counts['valid']}, test={total_counts['test']}")
    print(f"Train frames extracted: {total_frames['train']}")
    print(f"Val frames extracted: {total_frames['val']}")
    total_extracted = total_frames['train'] + total_frames['val']
    if total_extracted > 0:
        print(f"Actual split: {total_frames['val']/total_extracted*100:.1f}% validation")
    print(f"\nDataset saved to: {output_dir}")
    print(f"  Images: {output_dir}/images/train/ and {output_dir}/images/val/")
    print(f"  Labels: {output_dir}/labels/train/ and {output_dir}/labels/val/")
    print(f"  Config: {output_dir}/data.yaml")


if __name__ == "__main__":
    main()
