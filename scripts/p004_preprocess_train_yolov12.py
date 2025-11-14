#!/usr/local/bin/python

"""
Convert video dataset with JSON annotations to COCO format and train YOLOv12x.
Extracts frames from videos with FPS-based sampling and generates COCO JSON annotations.
Trains YOLOv12x model using Ultralytics training API.
"""

import argparse
import datetime
import json
import os
from typing import Any

import cv2
import numpy as np
import ultralytics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert video dataset to COCO format and train YOLOv12x"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/otif-dataset/dataset/caldot2/train/video",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="/otif-dataset/dataset/caldot2/train/yolov3-704x480",
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/polyis-data/coco-datasets/caldot2",
        help="Output directory for COCO dataset",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
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
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only create COCO dataset",
    )
    parser.add_argument(
        "--skip-dataset-creation",
        action="store_true",
        help="Skip dataset creation and only run training (assumes dataset already exists)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use for training (e.g., '0,1,2'). If not specified, uses all available GPUs or single GPU.",
    )
    return parser.parse_args()


def get_video_fps(video_path: str) -> float:
    """
    Get the FPS of a video file.

    Args:
        video_path: Path to video file

    Returns:
        FPS value as float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def determine_sample_rate(fps: float) -> int:
    """
    Determine frame sampling rate based on video FPS.

    Args:
        fps: Video FPS value

    Returns:
        Sample rate (16 for 30fps, 8 for 15fps, or closest match)
    """
    # Round FPS to nearest common value
    if abs(fps - 30.0) < abs(fps - 15.0):
        return 16  # Sample every 16 frames for 30fps
    else:
        return 8  # Sample every 8 frames for 15fps


def extract_frames_from_video(
    video_path: str,
    annotation_path: str,
    output_dir: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> list[dict[str, Any]]:
    """
    Determine which frames to extract from video with FPS-based sampling and load annotations.
    Does not decode video frames - only determines frame indices and loads annotations.

    Args:
        video_path: Path to video file
        annotation_path: Path to annotation JSON file
        output_dir: Base output directory for frames (unused, kept for compatibility)
        image_format: Image format ('jpg' or 'png') (unused, kept for compatibility)
        jpeg_quality: JPEG quality (1-100, only for jpg) (unused, kept for compatibility)

    Returns:
        List of sample dictionaries with video_path, frame_idx, and boxes
    """
    # Get video FPS to determine sample rate
    fps = get_video_fps(video_path)
    sample_rate = determine_sample_rate(fps)

    # Load annotation file
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # Open video capture only to get metadata (no frame decoding)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    # Get video properties without decoding frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    samples = []

    # Calculate frame indices to sample without decoding frames
    # Generate list of frame indices based on sample rate
    frame_indices = list(range(0, frame_count, sample_rate))

    # Process each sampled frame index
    for frame_idx in frame_indices:
        # Get annotations for this frame
        frame_annotations = []
        if frame_idx < len(annotations):
            frame_annotations = annotations[frame_idx]

        # Only create sample if there are annotations
        if len(frame_annotations) > 0:
            # Create sample entry
            sample = {
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "frame_idx": frame_idx,
                "boxes": frame_annotations,
                "width": width,
                "height": height,
            }
            samples.append(sample)

    return samples


def convert_bbox_to_coco(left: float, top: float, right: float, bottom: float) -> tuple[float, float, float, float]:
    """
    Convert bounding box from [left, top, right, bottom] to COCO format [x, y, width, height].

    Args:
        left: Left coordinate
        top: Top coordinate
        right: Right coordinate
        bottom: Bottom coordinate

    Returns:
        Tuple of (x, y, width, height) in COCO format
    """
    x = float(left)
    y = float(top)
    w = float(right - left)
    h = float(bottom - top)
    return x, y, w, h


def convert_coco_to_yolo_format(
    coco_json_path: str,
    image_dir: str,
    labels_dir: str,
) -> None:
    """
    Convert COCO format annotations to YOLO format (txt files).

    Args:
        coco_json_path: Path to COCO format JSON file
        image_dir: Directory containing images
        labels_dir: Directory to save YOLO format label files
    """
    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create labels directory
    os.makedirs(labels_dir, exist_ok=True)

    # Create mapping from image_id to image info
    image_id_to_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Convert each image's annotations to YOLO format
    for image_id, image_info in image_id_to_info.items():
        # Get image filename without extension
        image_filename = image_info["file_name"]
        image_name = os.path.splitext(image_filename)[0]
        label_filename = f"{image_name}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        # Get image dimensions
        img_width = float(image_info["width"])
        img_height = float(image_info["height"])

        # Get annotations for this image
        annotations = annotations_by_image.get(image_id, [])

        # Write YOLO format labels
        with open(label_path, "w") as f:
            for ann in annotations:
                # COCO bbox format: [x, y, width, height] (absolute coordinates)
                bbox = ann["bbox"]
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                # Convert to YOLO format: [class_id, center_x, center_y, width, height] (normalized 0-1)
                # COCO uses top-left corner, YOLO uses center
                center_x = (x + w / 2.0) / img_width
                center_y = (y + h / 2.0) / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                # Category ID (COCO uses 1-indexed, YOLO uses 0-indexed)
                # Our dataset has category_id=1 for "car", so we use class_id=0
                class_id = ann["category_id"] - 1  # Convert to 0-indexed

                # Write YOLO format line: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    print(f"  Converted COCO annotations to YOLO format: {len(annotations_by_image)} images with labels")


def create_coco_dataset(
    samples: list[dict[str, Any]],
    output_dir: str,
    split_name: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> None:
    """
    Create COCO format dataset from samples.

    Args:
        samples: List of sample dictionaries with video_path, frame_idx, and boxes
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
            "description": f"caldot2 car detection dataset - {split_name} split",
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
                "name": "car",
                "supercategory": "vehicle",
            }
        ],
    }

    # Create image output directory following Ultralytics structure: images/train/ and images/val/
    image_dir = os.path.join(output_dir, "images", split_name)
    os.makedirs(image_dir, exist_ok=True)

    # Counters for IDs
    image_id = 1
    annotation_id = 1

    print(f"  Extracting frames to {image_dir}/")

    # Group samples by video_path for efficient sequential reading
    samples_by_video: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        video_path = sample["video_path"]
        if video_path not in samples_by_video:
            samples_by_video[video_path] = []
        samples_by_video[video_path].append(sample)

    # Process each video once, reading frames sequentially
    total_processed = 0
    for video_path, video_samples in samples_by_video.items():
        # Sort samples by frame_idx for sequential reading
        video_samples.sort(key=lambda x: x["frame_idx"])

        # Create a set of frame indices we need from this video
        needed_frames = {sample["frame_idx"] for sample in video_samples}
        max_frame_idx = max(needed_frames)

        # Open video once for this video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Warning: Could not open video {video_path}, skipping")
            continue

        # Track which samples we've processed
        sample_idx = 0
        current_frame_idx = 0

        # Read frames sequentially
        while current_frame_idx <= max_frame_idx and sample_idx < len(video_samples):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if this frame is needed
            if current_frame_idx in needed_frames:
                # Process all samples for this frame index
                while sample_idx < len(video_samples) and video_samples[sample_idx]["frame_idx"] == current_frame_idx:
                    sample = video_samples[sample_idx]

                    # Create unique image filename
                    video_name = os.path.splitext(sample["video_name"])[0]
                    image_filename = f"{video_name}_{sample['frame_idx']:06d}.{image_format}"
                    image_path = os.path.join(image_dir, image_filename)

                    try:
                        # Get frame dimensions
                        height, width = frame.shape[:2]

                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)

                        # Save frame based on format
                        if image_format == "jpg":
                            cv2.imwrite(
                                image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                            )
                        else:
                            cv2.imwrite(image_path, frame)

                    except Exception as e:
                        print(f"    Warning: Failed to save frame for {image_filename}: {e}")
                        sample_idx += 1
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
                    for box in sample["boxes"]:
                        # Box format: {"left": x1, "top": y1, "right": x2, "bottom": y2, "class": "car", "score": conf}
                        if not isinstance(box, dict):
                            continue

                        # Extract coordinates
                        left = box.get("left", 0)
                        top = box.get("top", 0)
                        right = box.get("right", 0)
                        bottom = box.get("bottom", 0)

                        # Convert to COCO format [x, y, width, height]
                        x, y, w, h = convert_bbox_to_coco(left, top, right, bottom)

                        # Calculate area
                        area = w * h

                        # Skip invalid boxes
                        if area <= 0 or w <= 0 or h <= 0:
                            continue

                        # Create annotation entry
                        annotation_entry = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,  # car category
                            "bbox": [x, y, w, h],
                            "area": area,
                            "iscrowd": 0,
                        }
                        coco_data["annotations"].append(annotation_entry)
                        annotation_id += 1

                    # Increment image ID and sample index
                    image_id += 1
                    sample_idx += 1
                    total_processed += 1

            # Move to next frame
            current_frame_idx += 1

        cap.release()

        # Progress update
        if total_processed % 100 == 0 or total_processed == len(samples):
            print(f"    Processed {total_processed}/{len(samples)} samples")

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

    # Convert COCO format to YOLO format for Ultralytics compatibility
    # Ultralytics requires YOLO format (txt files) in labels/train/ and labels/val/
    labels_dir = os.path.join(output_dir, "labels", split_name)
    convert_coco_to_yolo_format(output_json_path, image_dir, labels_dir)


def load_all_samples(
    video_dir: str,
    annotation_dir: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> list[dict[str, Any]]:
    """
    Load all samples from video and annotation directories.

    Args:
        video_dir: Directory containing video files
        annotation_dir: Directory containing annotation JSON files
        image_format: Image format ('jpg' or 'png')
        jpeg_quality: JPEG quality (1-100, only for jpg)

    Returns:
        List of all samples
    """
    all_samples = []

    # Find all video files
    video_files = []
    if os.path.exists(video_dir):
        for filename in os.listdir(video_dir):
            if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(filename)

    print(f"Found {len(video_files)} video files")

    # Process each video
    for video_filename in sorted(video_files):
        video_path = os.path.join(video_dir, video_filename)
        # Get video number (filename without extension)
        video_num = os.path.splitext(video_filename)[0]
        annotation_path = os.path.join(annotation_dir, f"{video_num}.json")

        # Check if annotation file exists
        if not os.path.exists(annotation_path):
            print(f"  Warning: Annotation file not found for {video_filename}, skipping")
            continue

        try:
            # Extract frames and load annotations
            samples = extract_frames_from_video(
                video_path, annotation_path, "", image_format, jpeg_quality
            )
            all_samples.extend(samples)
            print(f"  {video_filename}: {len(samples)} frames with annotations")
        except Exception as e:
            print(f"  Warning: Failed to process {video_filename}: {e}")
            continue

    print(f"\nTotal samples: {len(all_samples)}")
    return all_samples


def train_yolov12x(coco_dataset_dir: str, gpus: str | None = None) -> None:
    """
    Train YOLOv12x model using Ultralytics API on COCO dataset.

    Args:
        coco_dataset_dir: Directory containing COCO dataset (with annotations/ subdirectory)
        gpus: Comma-separated list of GPU IDs (e.g., "0,1,2") or None for auto-detection
    """
    print("\n" + "=" * 80)
    print("Training YOLOv12x model")
    print("=" * 80)

    # Check if COCO dataset exists
    train_json = os.path.join(coco_dataset_dir, "annotations", "instances_train.json")
    val_json = os.path.join(coco_dataset_dir, "annotations", "instances_val.json")

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation annotations not found: {val_json}")

    # Verify COCO JSON files are valid and contain annotations
    with open(train_json, "r") as f:
        train_data = json.load(f)
        print(f"  Train JSON: {len(train_data.get('images', []))} images, {len(train_data.get('annotations', []))} annotations")
    
    with open(val_json, "r") as f:
        val_data = json.load(f)
        print(f"  Val JSON: {len(val_data.get('images', []))} images, {len(val_data.get('annotations', []))} annotations")

    # Clear cache files to force Ultralytics to regenerate with correct annotations
    train_cache = os.path.join(coco_dataset_dir, "train.cache")
    val_cache = os.path.join(coco_dataset_dir, "val.cache")
    if os.path.exists(train_cache):
        os.remove(train_cache)
        print(f"  Cleared train cache: {train_cache}")
    if os.path.exists(val_cache):
        os.remove(val_cache)
        print(f"  Cleared val cache: {val_cache}")

    # Load YOLOv12x model
    # Use model name without extension - Ultralytics will download automatically if needed
    # Note: When training with nc: 1 in data.yaml, Ultralytics automatically modifies
    # the model's last layer to output only 1 class instead of the pre-trained 80 classes
    print("Loading YOLOv12x model...")
    model = ultralytics.YOLO("yolo11x.pt")  # type: ignore

    # Train the model
    print("Starting training...")
    print(f"  Training annotations: {train_json}")
    print(f"  Validation annotations: {val_json}")

    # Ultralytics can train directly on COCO format
    # The data parameter should point to a YAML file or we can use the COCO JSON directly
    # For COCO format, we need to create a data.yaml file
    data_yaml_path = os.path.join(coco_dataset_dir, "data.yaml")
    create_data_yaml(data_yaml_path, coco_dataset_dir)

    # Prepare device parameter for multi-GPU training
    device = None
    if gpus:
        # Parse comma-separated GPU IDs and convert to list of integers
        device = [int(gpu_id.strip()) for gpu_id in gpus.split(",")]
        print(f"Using GPUs: {device}")
    else:
        print("Using default device (auto-detect GPUs)")

    # Train with default parameters
    # Ultralytics should auto-detect COCO format when it finds annotations/instances_*.json
    # in the dataset directory. The YAML file should point to image directories.
    # Set workers=0 to avoid shared memory issues in Docker/containers
    # Set batch size to 8 to avoid GPU out of memory errors
    train_kwargs = {
        "data": data_yaml_path,
        "workers": 0,  # Disable multiprocessing to avoid shared memory issues
        "batch": 8,  # Reduced batch size to avoid GPU out of memory errors
    }
    
    # Add device parameter if specified
    if device is not None:
        train_kwargs["device"] = device
    
    model.train(**train_kwargs)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


def create_data_yaml(yaml_path: str, dataset_dir: str) -> None:
    """
    Create YAML data configuration file for Ultralytics training.

    Args:
        yaml_path: Path to save YAML file
        dataset_dir: Directory containing COCO dataset
    """
    # Ultralytics expects images/train/ and images/val/ structure
    # See: https://docs.ultralytics.com/datasets/#steps-to-contribute-a-new-dataset
    train_images = os.path.join(dataset_dir, "images", "train")
    val_images = os.path.join(dataset_dir, "images", "val")
    train_json = os.path.join(dataset_dir, "annotations", "instances_train.json")
    val_json = os.path.join(dataset_dir, "annotations", "instances_val.json")

    yaml_content = f"""# YOLO dataset configuration for YOLOv12x training
# Following Ultralytics dataset structure: https://docs.ultralytics.com/datasets/#steps-to-contribute-a-new-dataset
# Structure: dataset/images/train/, dataset/images/val/, dataset/labels/train/, dataset/labels/val/
path: {dataset_dir}
train: {train_images}
val: {val_images}
test: {val_images}

# Number of classes
nc: 1

# Class names
names:
  0: car
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created data configuration: {yaml_path}")
    print(f"  Train images: {train_images}")
    print(f"  Val images: {val_images}")
    print(f"  Train annotations: {train_json}")
    print(f"  Val annotations: {val_json}")


def main(args):
    """
    Main function to convert dataset to COCO format and train YOLOv12x.

    This function:
    1. Loads videos and annotations from specified directories
    2. Extracts frames with FPS-based sampling (16 for 30fps, 8 for 15fps)
    3. Converts annotations to COCO format
    4. Creates COCO dataset with train/val split (0.8 ratio)
    5. Trains YOLOv12x model using Ultralytics API

    Args:
        args: Parsed command line arguments
    """
    print("=" * 80)
    print("YOLOv12x Training Script for caldot2 Dataset")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")

    # Skip dataset creation if flag is set
    if args.skip_dataset_creation:
        print("Skipping dataset creation (--skip-dataset-creation flag set)")
        print("Assuming dataset already exists at:", args.output_dir)
    else:
        print(f"Video directory: {args.video_dir}")
        print(f"Annotation directory: {args.annotation_dir}")
        print(f"Train/val split: {args.train_split:.2f}")
        print(f"Image format: {args.image_format}")
        if args.image_format == "jpg":
            print(f"JPEG quality: {args.jpeg_quality}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load all samples
        print("\nLoading videos and annotations...")
        all_samples = load_all_samples(
            args.video_dir, args.annotation_dir, args.image_format, args.jpeg_quality
        )

        # Check if we have samples
        if len(all_samples) == 0:
            print("Error: No samples found, exiting")
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
            create_coco_dataset(
                train_samples,
                args.output_dir,
                "train",
                args.image_format,
                args.jpeg_quality,
            )
        else:
            print("\n  Warning: No training samples, skipping train split")

        # Generate validation set
        if len(val_samples) > 0:
            create_coco_dataset(
                val_samples,
                args.output_dir,
                "val",
                args.image_format,
                args.jpeg_quality,
            )
        else:
            print("\n  Warning: No validation samples, skipping val split")

        print("\n" + "=" * 80)
        print("COCO dataset generation complete!")
        print(f"Dataset location: {args.output_dir}")
        print("=" * 80)

    # Train YOLOv12x if not skipped
    if not args.skip_training:
        train_yolov12x(args.output_dir, args.gpus)
    else:
        print("\nSkipping training (--skip-training flag set)")


if __name__ == "__main__":
    main(parse_args())

