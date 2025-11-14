#!/usr/local/bin/python

"""
Create YOLOv5 training dataset from CalDOT video annotations.
Extracts frames from videos and converts annotations to YOLO format.
"""

import argparse
import json
from pathlib import Path
import random
import shutil

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create YOLOv5 dataset from CalDOT videos and annotations"
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
        help="Output directory for YOLOv5 dataset",
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
                                   val_image_dir, val_label_dir, video_id, frame_stride=1, val_frames=None):
    """
    Extract frames from video and save YOLO format annotations.

    Args:
        video_path: Path to video file
        anno_path: Path to annotation JSON file
        train_image_dir: Directory to save training images
        train_label_dir: Directory to save training labels
        val_image_dir: Directory to save validation images
        val_label_dir: Directory to save validation labels
        video_id: Video identifier for naming files
        frame_stride: Extract every Nth frame (adjusted based on video FPS)
        val_frames: Set of (video_id, frame_idx) tuples that should go to validation.
                    If None, all frames go to train directories.

    Returns:
        Tuple of (train_frames_extracted, val_frames_extracted)
    """
    # Load annotations from JSON file
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return (0, 0)

    # Get video FPS and adjust stride accordingly
    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_stride = frame_stride
    if fps > 20:  # 30 fps video
        actual_stride = frame_stride * 2
    # else: 15 fps video, keep original stride

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
            frame_key = (video_id, frame_idx)
            is_val = frame_key in val_frames

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

        # Convert annotations to YOLO format and save
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

                    # Convert to YOLO format: class_id center_x center_y width height (normalized 0-1)
                    bbox_width = right - left
                    bbox_height = bottom - top

                    # Skip invalid boxes
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue

                    # Calculate normalized center coordinates and dimensions
                    center_x = (left + bbox_width / 2.0) / width
                    center_y = (top + bbox_height / 2.0) / height
                    norm_width = bbox_width / width
                    norm_height = bbox_height / height

                    # Class ID (0 for "car" in CalDOT dataset)
                    class_id = 0

                    # Write YOLO format line
                    label_file.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            # If frame_annos is empty, label file will be empty (negative example)

        if is_val:
            val_frames_extracted += 1
        else:
            train_frames_extracted += 1
        frame_idx += 1

    cap.release()
    return (train_frames_extracted, val_frames_extracted)


def find_highest_resolution_annotations(dataset_root):
    """
    Find annotation directory with highest resolution.

    Args:
        dataset_root: Root directory of the dataset

    Returns:
        Path to annotation directory with highest resolution
    """
    # Look for yolov3-* annotation directories
    anno_dirs = sorted(dataset_root.glob("yolov3-*"))
    if not anno_dirs:
        raise FileNotFoundError(f"No annotation directories found in {dataset_root}")

    def get_width(path):
        # Extract width from "yolov3-WIDTHxHEIGHT"
        dims = path.name.split('-')[1]
        width = int(dims.split('x')[0])
        return width

    # Return directory with highest width
    return max(anno_dirs, key=get_width)


def setup_output_directories(output_dir):
    """
    Create output directory structure for YOLOv5 dataset.

    Args:
        output_dir: Root output directory

    Returns:
        Tuple of (train_image_dir, val_image_dir, train_label_dir, val_label_dir)
    """
    # Remove existing directory if present
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists and will be overwritten.")
        shutil.rmtree(output_dir)

    # Create directory structure for YOLOv5/Ultralytics format
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


def collect_valid_frames(video_file, anno_path, video_id, frame_stride=1):
    """
    Collect frame indices from a video (including frames without annotations).

    Args:
        video_file: Path to video file
        anno_path: Path to annotation JSON file
        video_id: Video identifier
        frame_stride: Extract every Nth frame (adjusted based on video FPS)

    Returns:
        List of (video_id, frame_idx) tuples for all frames (with or without annotations)
    """
    # Load annotations from JSON file
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Open video capture to get FPS
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return []

    # Get video FPS and adjust stride accordingly
    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_stride = frame_stride
    if fps > 20:  # 30 fps video
        actual_stride = frame_stride * 2
    # else: 15 fps video, keep original stride

    cap.release()

    # Collect all frame indices (including frames without annotations)
    # This provides negative examples for training
    valid_frames = []
    for frame_idx in range(len(annotations)):
        # Skip frames based on adjusted stride
        if frame_idx % actual_stride != 0:
            continue

        # Include all frames, even those without annotations
        # Frames without annotations will have empty label files
        valid_frames.append((video_id, frame_idx))

    return valid_frames


def split_frames_train_val(all_frames, val_split):
    """
    Split frame identifiers into training and validation sets with reproducible random shuffle.

    Args:
        all_frames: List of (video_id, frame_idx) tuples
        val_split: Fraction of frames for validation (0.0-1.0)
        seed: Random seed for reproducible split

    Returns:
        Set of (video_id, frame_idx) tuples for validation frames
    """
    # Create a copy and shuffle with seed for reproducible split
    frames_shuffled = list(all_frames)
    random.seed(42)
    random.shuffle(frames_shuffled)

    # Use last val_split fraction of shuffled frames for validation
    num_val = int(len(frames_shuffled) * val_split)
    val_frames = set(frames_shuffled[-num_val:])

    print(f"Train frames: {len(frames_shuffled) - num_val}")
    print(f"Val frames: {num_val}")

    return val_frames


def process_video_file(video_file, anno_dir, val_frames,
                       train_image_dir, val_image_dir,
                       train_label_dir, val_label_dir,
                       frame_stride, id_prefix: str = ""):
    """
    Process a single video file and extract frames with annotations.

    Args:
        video_file: Path to video file
        anno_dir: Directory containing annotation files
        val_frames: Set of (video_id, frame_idx) tuples for validation frames
        train_image_dir: Directory for training images
        val_image_dir: Directory for validation images
        train_label_dir: Directory for training labels
        val_label_dir: Directory for validation labels
        frame_stride: Extract every Nth frame
        id_prefix: Optional prefix to ensure unique image file names across subsets

    Returns:
        Tuple of (train_frames_extracted, val_frames_extracted)
    """
    # Get video ID from filename (without extension)
    video_id = video_file.stem
    anno_file = anno_dir / f"{video_id}.json"

    # Check if annotation file exists
    if not anno_file.exists():
        print(f"Warning: No annotations found for video {video_id}")
        return (0, 0)

    # Use prefixed video id to avoid filename collisions across subsets
    prefixed_video_id = f"{id_prefix}{video_id}" if id_prefix else video_id

    # Create adjusted val_frames set with prefixed video_id for lookup
    adjusted_val_frames = set()
    for orig_video_id, frame_idx in val_frames:
        # Match frames by checking if the original video_id matches
        # (accounting for prefix that will be added)
        if orig_video_id == video_id:
            adjusted_val_frames.add((prefixed_video_id, frame_idx))

    # Extract frames and annotations
    train_count, val_count = extract_frames_and_annotations(
        video_file, anno_file, train_image_dir, train_label_dir,
        val_image_dir, val_label_dir, prefixed_video_id, frame_stride,
        adjusted_val_frames
    )

    return (train_count, val_count)


def create_data_yaml(output_dir, dataset_name):
    """
    Create data.yaml configuration file for YOLOv5 training.

    Args:
        output_dir: Dataset root directory
        dataset_name: Name of the dataset
    """
    # Create YAML content with relative paths
    yaml_content = f"""# YOLOv5 dataset configuration for CalDOT {dataset_name}
# Auto-generated by p007_preprocess_create_trainset_yolo5.py

path: {output_dir}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names:
  0: car
"""

    # Write to data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated data.yaml: {yaml_path}")


def main():
    args = parse_args()

    # Setup paths for CalDOT dataset structure
    base_root = Path("/otif-dataset/dataset") / args.dataset

    # We'll include videos from train, valid, and test splits (if present)
    subsets = [
        ("train", base_root / "train"),
        ("valid", base_root / "valid"),
        ("test", base_root / "test"),
    ]

    # Set output directory
    output_dir = Path(args.output_dir or f"/polyis-data/yolo5/{args.dataset}/training-data")

    # Create output directory structure
    train_image_dir, val_image_dir, train_label_dir, val_label_dir = setup_output_directories(output_dir)

    print("=" * 80)
    print(f"Creating YOLOv5 Dataset for {args.dataset}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {args.val_split}")
    print(f"Frame stride: {args.frame_stride}")
    print()

    # Track statistics
    total_counts = {"train": 0, "valid": 0, "test": 0}
    total_frames = {"train": 0, "val": 0}

    # First pass: Collect all valid frames from all videos
    print("Collecting valid frames from all videos...")
    all_valid_frames = []
    video_info = []  # Store (subset_name, video_file, anno_dir) for second pass

    for subset_name, subset_root in subsets:
        # Skip subsets that don't exist
        if not subset_root.exists():
            continue

        video_dir = subset_root / "video"
        if not video_dir.exists():
            print(f"Warning: Missing video directory for subset '{subset_name}': {video_dir}")
            continue

        # Find annotations directory with highest resolution for this subset
        anno_dir = find_highest_resolution_annotations(subset_root)
        print(f"Using annotations for '{subset_name}' from: {anno_dir.name}")

        # Gather videos for this subset
        video_files = sorted([f for f in video_dir.glob("*.mp4")])
        total_counts[subset_name] = len(video_files)
        print(f"Found {len(video_files)} videos in '{subset_name}'")

        # Collect valid frames from each video
        for video_file in tqdm(video_files, desc=f"Scanning {subset_name} videos"):
            video_id = video_file.stem
            anno_file = anno_dir / f"{video_id}.json"

            # Check if annotation file exists
            if not anno_file.exists():
                continue

            # Collect valid frames (using original video_id without prefix for splitting)
            valid_frames = collect_valid_frames(video_file, anno_file, video_id, args.frame_stride)
            all_valid_frames.extend(valid_frames)

            # Store video info for second pass
            video_info.append((subset_name, video_file, anno_dir))

    print(f"\nTotal frames collected: {len(all_valid_frames)} (including frames without annotations)")

    # Split frames into train/val with seed for reproducibility
    print("\nSplitting frames into train/val sets...")
    val_frames = split_frames_train_val(all_valid_frames, args.val_split)

    # Second pass: Extract frames and save to appropriate directories
    print("\nExtracting frames and saving to dataset...")
    for subset_name, video_file, anno_dir in tqdm(video_info, desc="Processing all videos"):
        train_count, val_count = process_video_file(
            video_file, anno_dir, val_frames,
            train_image_dir, val_image_dir,
            train_label_dir, val_label_dir,
            args.frame_stride, id_prefix=f"{subset_name}_"
        )

        # Track frame counts
        total_frames["train"] += train_count
        total_frames["val"] += val_count

    # Create data.yaml configuration file
    create_data_yaml(output_dir, args.dataset)

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)
    print(f"Subsets discovered: train={total_counts['train']}, valid={total_counts['valid']}, test={total_counts['test']}")
    print(f"Train frames extracted: {total_frames['train']}")
    print(f"Val frames extracted: {total_frames['val']}")
    print(f"\nDataset saved to: {output_dir}")
    print(f"  Images: {output_dir}/images/train/ and {output_dir}/images/val/")
    print(f"  Labels: {output_dir}/labels/train/ and {output_dir}/labels/val/")
    print(f"  Config: {output_dir}/data.yaml")


if __name__ == "__main__":
    main()
