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
        help="Fraction of videos to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Extract every Nth frame (default: 5)",
    )
    return parser.parse_args()


def extract_frames_and_annotations(video_path, anno_path, output_image_dir, video_id, frame_stride=1):
    """
    Extract frames from video and return frame info with annotations.
    
    Args:
        video_path: Path to video file
        anno_path: Path to annotation JSON file
        output_image_dir: Directory to save extracted frames
        video_id: Video identifier
        frame_stride: Extract every Nth frame (adjusted based on video FPS)
    
    Returns:
        List of (image_info, annotations) tuples
    """
    # Load annotations
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return []
    
    # Get video FPS and adjust stride
    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_stride = frame_stride
    if fps > 20:  # 30 fps video
        actual_stride = frame_stride * 2
    # else: 15 fps video, keep original stride
    
    frame_data = []
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
        
        # Skip frames with no annotations
        if not frame_annos:
            frame_idx += 1
            continue
        
        # Save frame
        height, width = frame.shape[:2]
        image_filename = f"{video_id}_{frame_idx:06d}.jpg"
        image_path = output_image_dir / image_filename
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
        
        frame_data.append((image_info, frame_annos))
        frame_idx += 1
    
    cap.release()
    return frame_data


def find_highest_resolution_annotations(dataset_root):
    """
    Find annotation directory with highest resolution.
    
    Args:
        dataset_root: Root directory of the dataset
    
    Returns:
        Path to annotation directory with highest resolution
    """
    anno_dirs = sorted(dataset_root.glob("yolov3-*"))
    if not anno_dirs:
        raise FileNotFoundError(f"No annotation directories found in {dataset_root}")
    
    def get_width(path):
        # Extract width from "yolov3-WIDTHxHEIGHT"
        dims = path.name.split('-')[1]
        width = int(dims.split('x')[0])
        return width
    
    return max(anno_dirs, key=get_width)


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


def split_videos_train_val(video_files, val_split):
    """
    Split video files into training and validation sets.
    
    Args:
        video_files: List of video file paths
        val_split: Fraction of videos for validation (0.0-1.0)
    
    Returns:
        Set of validation video paths
    """
    num_val = int(len(video_files) * val_split)
    val_videos = set(video_files[-num_val:])
    
    print(f"Train videos: {len(video_files) - num_val}")
    print(f"Val videos: {num_val}")
    
    return val_videos


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
        COCO annotation dictionary, or None if invalid
    """
    # Convert box format: [left, top, right, bottom] -> [x, y, width, height]
    left, top, right, bottom = obj["left"], obj["top"], obj["right"], obj["bottom"]
    width = right - left
    height = bottom - top
    
    # Skip invalid boxes (must have positive dimensions and reasonable coordinates)
    if width <= 0 or height <= 0:
        return None
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        return None
    if width > 10000 or height > 10000:  # Unreasonably large boxes
        return None
    
    # Get category ID
    class_name = obj["class"]
    if class_name not in category_name_to_id:
        return None
    
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
        if annotation is not None:
            target_coco["annotations"].append(annotation)
            annotation_id += 1
    
    return annotation_id


def process_video_file(video_file, anno_dir, val_videos, train_coco, val_coco, 
                       train_image_dir, val_image_dir, category_name_to_id, 
                       image_id, annotation_id, frame_stride):
    """
    Process a single video file and add its frames to COCO dataset.
    
    Args:
        video_file: Path to video file
        anno_dir: Directory containing annotation files
        val_videos: Set of validation video paths
        train_coco: Training COCO dataset dictionary
        val_coco: Validation COCO dataset dictionary
        train_image_dir: Directory for training images
        val_image_dir: Directory for validation images
        category_name_to_id: Mapping from class name to category ID
        image_id: Starting image ID
        annotation_id: Starting annotation ID
        frame_stride: Extract every Nth frame
    
    Returns:
        Tuple of (updated image_id, updated annotation_id)
    """
    video_id = video_file.stem
    anno_file = anno_dir / f"{video_id}.json"
    
    if not anno_file.exists():
        print(f"Warning: No annotations found for video {video_id}")
        return image_id, annotation_id
    
    # Determine if this is train or val
    is_val = video_file in val_videos
    target_coco = val_coco if is_val else train_coco
    target_image_dir = val_image_dir if is_val else train_image_dir
    
    # Extract frames and annotations
    frame_data = extract_frames_and_annotations(
        video_file, anno_file, target_image_dir, video_id, frame_stride
    )
    
    # Add each frame to COCO dataset
    for image_info, frame_annos in frame_data:
        annotation_id = add_frame_to_coco_dataset(
            image_info, frame_annos, image_id, annotation_id, target_coco, category_name_to_id
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
    dataset_root = Path("/otif-dataset/dataset") / args.dataset / "train"
    video_dir = dataset_root / "video"
    anno_dir = find_highest_resolution_annotations(dataset_root)
    print(f"Using annotations from: {anno_dir.name}")
    
    output_dir = Path(args.output_dir or f"/polyis-data/fasterrcnn/{args.dataset}/coco-dataset")
    train_image_dir, val_image_dir, anno_output_dir = setup_output_directories(output_dir)
    
    # Get and split videos
    video_files = sorted([f for f in video_dir.glob("*.mp4")])
    print(f"Found {len(video_files)} videos")
    val_videos = split_videos_train_val(video_files, args.val_split)
    
    # Setup category mapping (CalDOT dataset only has 'car' class)
    categories = [{"id": 1, "name": "car"}]
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    # Initialize COCO structures
    train_coco, val_coco = initialize_coco_structure(categories)
    
    # Process all videos
    image_id = 1
    annotation_id = 1
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        image_id, annotation_id = process_video_file(
            video_file, anno_dir, val_videos, train_coco, val_coco,
            train_image_dir, val_image_dir, category_name_to_id,
            image_id, annotation_id, args.frame_stride
        )
    
    # Save COCO JSON files
    train_json_path, val_json_path = save_coco_json_files(train_coco, val_coco, anno_output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val images: {len(val_coco['images'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")
    print(f"\nDataset saved to: {output_dir}")
    print(f"Train JSON: {train_json_path}")
    print(f"Val JSON: {val_json_path}")


if __name__ == "__main__":
    main()
