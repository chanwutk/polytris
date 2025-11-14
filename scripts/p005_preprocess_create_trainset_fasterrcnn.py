#!/usr/local/bin/python

"""
Create COCO dataset from CalDOT video annotations for Faster R-CNN training.
Extracts frames from videos and converts YOLOv3 annotations to COCO format.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create COCO dataset from CalDOT videos and annotations"
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
        help="Output directory for COCO dataset",
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


def extract_frames_and_annotations(video_path, anno_path, train_image_dir, val_image_dir,
                                   video_id, frame_stride=1, val_frames=None):
    """
    Extract frames from video and return frame info with annotations.
    
    Args:
        video_path: Path to video file
        anno_path: Path to annotation JSON file
        train_image_dir: Directory to save training images
        val_image_dir: Directory to save validation images
        video_id: Video identifier for naming files
        frame_stride: Extract every Nth frame (adjusted based on video FPS)
        val_frames: Set of (video_id, frame_idx) tuples that should go to validation.
                    If None, all frames go to train directories.
    
    Returns:
        Tuple of (train_frame_data, val_frame_data) where each is a list of (image_info, annotations) tuples
    """
    # Load annotations
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Get adjusted stride based on video FPS
    actual_stride = get_adjusted_frame_stride(video_path, frame_stride)
    
    train_frame_data = []
    val_frame_data = []
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
        # Frames without annotations will have empty annotation lists (negative examples)
        
        # Determine if this frame should go to validation
        is_val = False
        if val_frames is not None:
            frame_key = (video_id, frame_idx)
            is_val = frame_key in val_frames
        
        # Select target directory based on split
        if is_val:
            target_image_dir = val_image_dir
        else:
            target_image_dir = train_image_dir
        
        # Save frame
        height, width = frame.shape[:2]
        image_filename = f"{video_id}_{frame_idx:06d}.jpg"
        image_path = target_image_dir / image_filename
        cv2.imwrite(str(image_path), frame)
        
        # Create image info
        image_info = {
            "id": f"{video_id}_{frame_idx:06d}",
            "file_name": image_filename,
            "width": width,
            "height": height,
            "video_id": video_id,
            "frame_idx": frame_idx
        }
        
        # Add to appropriate list
        if is_val:
            val_frame_data.append((image_info, frame_annos))
        else:
            train_frame_data.append((image_info, frame_annos))
        
        frame_idx += 1
    
    cap.release()
    return (train_frame_data, val_frame_data)




def setup_output_directories(output_dir):
    """
    Create output directory structure for COCO dataset.
    
    Args:
        output_dir: Root output directory
    
    Returns:
        Tuple of (train_image_dir, val_image_dir, anno_output_dir)
    """
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists and will be overwritten.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_image_dir = output_dir / "train2017"
    val_image_dir = output_dir / "val2017"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    anno_output_dir = output_dir / "annotations"
    anno_output_dir.mkdir(parents=True, exist_ok=True)
    
    return train_image_dir, val_image_dir, anno_output_dir




def initialize_coco_structure(categories):
    """
    Initialize empty COCO dataset structures for train and validation.
    
    Args:
        categories: List of category dictionaries
    
    Returns:
        Tuple of (train_coco, val_coco) dictionaries
    """
    train_coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {
            "description": "CalDOT Dataset - Training Split",
            "version": "1.0",
            "year": 2025
        }
    }
    
    val_coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {
            "description": "CalDOT Dataset - Validation Split",
            "version": "1.0",
            "year": 2025
        }
    }
    
    return train_coco, val_coco


def convert_bbox_annotation_to_coco(obj, image_id, annotation_id, category_name_to_id):
    """
    Convert a single bounding box annotation to COCO format.
    
    Args:
        obj: Annotation object with left, top, right, bottom, class fields
        image_id: ID of the image this annotation belongs to
        annotation_id: Unique annotation ID
        category_name_to_id: Mapping from class name to category ID
    
    Returns:
        COCO annotation dictionary
    """
    # Convert box format: [left, top, right, bottom] -> [x, y, width, height]
    left, top, right, bottom = obj["left"], obj["top"], obj["right"], obj["bottom"]
    width = right - left
    height = bottom - top
    
    # Skip invalid boxes (must have positive dimensions and reasonable coordinates)
    if width <= 0 or height <= 0:
        raise Exception(f"Invalid box: {left}, {top}, {right}, {bottom}")
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        raise Exception(f"Invalid box: {left}, {top}, {right}, {bottom}")
    if width > 10000 or height > 10000:  # Unreasonably large boxes
        raise Exception(f"Invalid box: {left}, {top}, {right}, {bottom}")
    
    # Get category ID
    class_name = obj["class"]
    if class_name not in category_name_to_id:
        raise Exception(f"Invalid class: {class_name}")
    
    category_id = category_name_to_id[class_name]
    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(left), float(top), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [[
            float(left), float(top),
            float(right), float(top),
            float(right), float(bottom),
            float(left), float(bottom)
        ]]
    }
    
    return annotation


def add_frame_to_coco_dataset(image_info, frame_annos, image_id, annotation_id, target_coco, category_name_to_id):
    """
    Add a single frame and its annotations to COCO dataset.
    
    Args:
        image_info: Image metadata dictionary
        frame_annos: List of annotations for this frame
        image_id: ID to assign to this image
        annotation_id: Starting annotation ID
        target_coco: COCO dataset dictionary to add to
        category_name_to_id: Mapping from class name to category ID
    
    Returns:
        Updated annotation_id after adding all annotations
    """
    # Add image
    image_info["id"] = image_id
    target_coco["images"].append(image_info)
    
    # Add annotations for this frame
    for obj in frame_annos:
        annotation = convert_bbox_annotation_to_coco(obj, image_id, annotation_id, category_name_to_id)
        assert annotation is not None, f"Invalid annotation: {obj}"
        target_coco["annotations"].append(annotation)
        annotation_id += 1
    
    return annotation_id


def process_video_file(video_file, anno_dir, val_frames,
                       train_coco, val_coco,
                       train_image_dir, val_image_dir,
                       category_name_to_id,
                       image_id, annotation_id,
                       frame_stride, id_prefix: str = ""):
    """
    Process a single video file and add its frames to COCO dataset.
    
    Args:
        video_file: Path to video file
        anno_dir: Directory containing annotation files
        val_frames: Set of (video_id, frame_idx) tuples for validation frames
        train_coco: Training COCO dataset dictionary
        val_coco: Validation COCO dataset dictionary
        train_image_dir: Directory for training images
        val_image_dir: Directory for validation images
        category_name_to_id: Mapping from class name to category ID
        image_id: Starting image ID
        annotation_id: Starting annotation ID
        frame_stride: Extract every Nth frame
        id_prefix: Optional prefix to ensure unique image file names across subsets
    
    Returns:
        Tuple of (updated image_id, updated annotation_id)
    """
    # Get video ID and annotation file path
    video_id, anno_file = get_video_annotation_path(video_file, anno_dir)
    
    # Use prefixed video id to avoid filename collisions across subsets
    prefixed_video_id = f"{id_prefix}{video_id}" if id_prefix else video_id

    # Create adjusted val_frames set with prefixed video_id for lookup
    adjusted_val_frames = adjust_val_frames_for_prefix(val_frames, video_id, id_prefix)

    # Extract frames and annotations
    train_frame_data, val_frame_data = extract_frames_and_annotations(
        video_file, anno_file, train_image_dir, val_image_dir,
        prefixed_video_id, frame_stride, adjusted_val_frames
    )
    
    # Add training frames to COCO dataset
    for image_info, frame_annos in train_frame_data:
        annotation_id = add_frame_to_coco_dataset(
            image_info, frame_annos, image_id, annotation_id, train_coco, category_name_to_id
        )
        image_id += 1
    
    # Add validation frames to COCO dataset
    for image_info, frame_annos in val_frame_data:
        annotation_id = add_frame_to_coco_dataset(
            image_info, frame_annos, image_id, annotation_id, val_coco, category_name_to_id
        )
        image_id += 1
    
    return image_id, annotation_id


def save_coco_json_files(train_coco, val_coco, anno_output_dir):
    """
    Save COCO dataset dictionaries to JSON files.
    
    Args:
        train_coco: Training COCO dataset dictionary
        val_coco: Validation COCO dataset dictionary
        anno_output_dir: Directory to save annotation files
    
    Returns:
        Tuple of (train_json_path, val_json_path)
    """
    train_json_path = anno_output_dir / "instances_train2017.json"
    val_json_path = anno_output_dir / "instances_val2017.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    return train_json_path, val_json_path


def main():
    args = parse_args()

    # Setup paths
    base_root = Path("/otif-dataset/dataset")
    dataset = args.dataset
    
    output_dir = Path(args.output_dir or f"/polyis-data/fasterrcnn/{dataset}/coco-dataset")
    train_image_dir, val_image_dir, anno_output_dir = setup_output_directories(output_dir)
    
    print("=" * 80)
    print(f"Creating COCO Dataset for {dataset}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {args.val_split}")
    print(f"Frame stride: {args.frame_stride}")
    print()
    
    # Initialize counters and accumulators
    subsets = get_dataset_subsets(base_root, dataset)
    total_counts = {"train": 0, "valid": 0, "test": 0}
    
    # Setup category mapping (CalDOT dataset only has 'car' class)
    categories = [{"id": 1, "name": "car"}]
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    # Initialize COCO structures
    train_coco, val_coco = initialize_coco_structure(categories)
    
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
    
    # First pass: Collect all valid frames from all videos (including negative samples)
    print("\nCollecting valid frames from all videos...")
    all_valid_frames = []

    for subset_name, video_file, anno_dir in tqdm(video_info, desc="Scanning videos"):
        video_id, anno_file = get_video_annotation_path(video_file, anno_dir)

        # Collect valid frames (using original video_id without prefix for splitting)
        valid_frames = collect_valid_frames(video_file, anno_file, video_id, args.frame_stride)
        all_valid_frames.extend(valid_frames)

    print(f"\nTotal frames collected: {len(all_valid_frames)} (including frames without annotations for negative examples)")

    # Split frames into train/val with seed for reproducibility
    print("\nSplitting frames into train/val sets...")
    val_frames = split_frames_train_val(all_valid_frames, args.val_split)

    # Second pass: Extract frames and save to appropriate directories
    print("\nExtracting frames and saving to dataset...")
    image_id = 1
    annotation_id = 1

    for subset_name, video_file, anno_dir in tqdm(video_info, desc="Processing all videos"):
        image_id, annotation_id = process_video_file(
            video_file, anno_dir, val_frames,
            train_coco, val_coco,
            train_image_dir, val_image_dir,
            category_name_to_id,
            image_id, annotation_id,
            args.frame_stride, id_prefix=f"{subset_name}_"
        )
    
    # Save COCO JSON files
    train_json_path, val_json_path = save_coco_json_files(train_coco, val_coco, anno_output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)
    print(f"Subsets discovered: train={total_counts['train']}, valid={total_counts['valid']}, test={total_counts['test']}")
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val images: {len(val_coco['images'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")
    print(f"\nDataset saved to: {output_dir}")
    print(f"Train JSON: {train_json_path}")
    print(f"Val JSON: {val_json_path}")


if __name__ == "__main__":
    main()
