"""
Utilities for creating fine-tuning datasets from compressed/packed frames.

This module provides functions for:
- Loading offset lookup and index map files
- Transforming bounding boxes from original to compressed frame coordinates
- Creating intermediate dataset representations
- Converting to training formats (Ultralytics YOLO, COCO)
"""

import json
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CompressedImageAnnotation:
    """Annotation entry for intermediate dataset."""
    image_path: str          # Path to compressed image
    image_width: int
    image_height: int
    dataset: str
    video: str
    classifier: str
    tilesize: int
    tilepadding: str
    collage_idx: int
    start_frame: int
    end_frame: int
    annotations: list[dict]  # List of {"bbox": [x1,y1,x2,y2], "class": str, "frame_idx": int, "track_id": int}


def load_offset_lookup(offset_lookup_path: Path) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """
    Load offset lookup JSONL file.
    
    Each line in the file contains: [[packed_y, packed_x], [original_y, original_x], frame_idx]
    
    Args:
        offset_lookup_path: Path to the offset lookup JSONL file
        
    Returns:
        List of ((packed_y, packed_x), (original_y, original_x), frame_idx)
    """
    # Load and parse the JSONL file
    with open(offset_lookup_path, 'r') as f:
        offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = [
            json.loads(line) for line in f
        ]
    return offset_lookup


def load_index_map(index_map_path: Path) -> np.ndarray:
    """
    Load numpy array containing group IDs for each tile position.
    
    Args:
        index_map_path: Path to the numpy file containing the index map
        
    Returns:
        2D numpy array where each element is a group ID (0 for empty tiles)
    """
    # Load the numpy array
    return np.load(index_map_path)


def get_annotations_for_compressed_image(
    start_frame: int,
    end_frame: int,
    tracking_results: dict[int, list[list[float]]],
    offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]],
    index_map: np.ndarray,
    tilesize: int,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Get transformed annotations where bbox center is in valid tile.
    
    For each track in each frame within the range, this function:
    1. Checks if the center of the bbox falls within a valid tile
    2. Transforms the bbox coordinates to the compressed frame
    3. Clips the bbox to image bounds
    
    Args:
        start_frame: First frame index in the compressed image
        end_frame: Last frame index in the compressed image
        tracking_results: Dictionary mapping frame indices to lists of tracks
        offset_lookup: List of ((packed_y, packed_x), (original_y, original_x), frame_idx)
        index_map: 2D array where each element is a group ID
        tilesize: Size of each tile in pixels
        image_width: Width of the compressed image
        image_height: Height of the compressed image
        
    Returns:
        List of annotation dictionaries with transformed bbox coordinates
    """
    annotations = []
    
    # Build frame_idx -> group_info mapping from offset_lookup
    # offset_lookup format: [((packed_y, packed_x), (original_y, original_x), frame_idx), ...]
    # Group IDs in index_map are 1-indexed (0 means empty)
    frame_to_group: dict[int, list[dict]] = {}
    for group_id, (packed_pos, original_pos, frame_idx) in enumerate(offset_lookup, start=1):
        if frame_idx not in frame_to_group:
            frame_to_group[frame_idx] = []
        frame_to_group[frame_idx].append({
            'group_id': group_id,
            'packed_y': packed_pos[0],
            'packed_x': packed_pos[1],
            'original_y': original_pos[0],
            'original_x': original_pos[1],
        })
    
    # Process each frame in range
    for frame_idx in range(start_frame, end_frame + 1):
        # Skip if no tracks for this frame
        if frame_idx not in tracking_results:
            continue
        
        tracks = tracking_results[frame_idx]
        
        # Skip if this frame has no polyominoes in the compressed image
        if frame_idx not in frame_to_group:
            continue
        
        for track in tracks:
            # Track format: [track_id, x1, y1, x2, y2]
            track_id, x1, y1, x2, y2 = track[:5]
            
            # Calculate center of bbox in original frame coordinates
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Find tile containing center in original frame coordinates
            tile_x = int(center_x // tilesize)
            tile_y = int(center_y // tilesize)
            
            # Check each polyomino group for this frame to find which one contains the center
            transformed_bbox = None
            for group_info in frame_to_group[frame_idx]:
                group_id = group_info['group_id']
                packed_y = group_info['packed_y']
                packed_x = group_info['packed_x']
                original_y = group_info['original_y']
                original_x = group_info['original_x']
                
                # Calculate relative tile position within the polyomino
                rel_tile_x = tile_x - original_x
                rel_tile_y = tile_y - original_y
                
                # Calculate packed tile position
                packed_tile_x = packed_x + rel_tile_x
                packed_tile_y = packed_y + rel_tile_y
                
                # Check if this tile exists in index_map and belongs to this group
                if (0 <= packed_tile_y < index_map.shape[0] and
                    0 <= packed_tile_x < index_map.shape[1] and
                    index_map[packed_tile_y, packed_tile_x] == group_id):
                    
                    # Transform bbox coordinates by applying the offset
                    offset_x = (packed_x - original_x) * tilesize
                    offset_y = (packed_y - original_y) * tilesize
                    
                    transformed_bbox = [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y,
                    ]
                    break
            
            # Only add if center was in valid tile
            if transformed_bbox is not None:
                # Clip to image bounds
                tx1 = max(0, min(transformed_bbox[0], image_width))
                ty1 = max(0, min(transformed_bbox[1], image_height))
                tx2 = max(0, min(transformed_bbox[2], image_width))
                ty2 = max(0, min(transformed_bbox[3], image_height))
                
                # Skip if bbox becomes invalid after clipping
                if tx2 > tx1 and ty2 > ty1:
                    annotations.append({
                        'bbox': [tx1, ty1, tx2, ty2],
                        'class': 'car',  # Single class for this dataset
                        'frame_idx': frame_idx,
                        'track_id': int(track_id),
                    })
    
    return annotations


def save_intermediate_dataset(entries: list[CompressedImageAnnotation], output_path: Path):
    """
    Save intermediate dataset as JSONL.
    
    Args:
        entries: List of CompressedImageAnnotation entries
        output_path: Path to save the JSONL file
    """
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write each entry as a JSON line
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + '\n')


def load_intermediate_dataset(input_path: Path) -> list[CompressedImageAnnotation]:
    """
    Load intermediate dataset from JSONL.
    
    Args:
        input_path: Path to the JSONL file
        
    Returns:
        List of CompressedImageAnnotation entries
    """
    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(CompressedImageAnnotation(**data))
    return entries


def split_dataset(
    dataset: list[CompressedImageAnnotation],
    val_split: float,
    seed: int = 42,
) -> tuple[list[CompressedImageAnnotation], list[CompressedImageAnnotation]]:
    """
    Split dataset into train and val sets with reproducible shuffle.
    
    Args:
        dataset: List of CompressedImageAnnotation entries
        val_split: Fraction of data for validation (0.0 to 1.0)
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_entries, val_entries)
    """
    # Create a copy and shuffle with fixed seed
    shuffled = dataset.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    
    # Calculate split point
    val_count = int(len(shuffled) * val_split)
    
    # Split into train and val
    val_entries = shuffled[:val_count]
    train_entries = shuffled[val_count:]
    
    return train_entries, val_entries


def convert_to_ultralytics(
    intermediate_dataset: list[CompressedImageAnnotation],
    output_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Convert intermediate dataset to Ultralytics YOLO format.
    
    Creates:
        - {output_dir}/images/train/*.jpg (symlinks)
        - {output_dir}/images/val/*.jpg (symlinks)
        - {output_dir}/labels/train/*.txt
        - {output_dir}/labels/val/*.txt
        - {output_dir}/data.yaml
        
    Args:
        intermediate_dataset: List of CompressedImageAnnotation entries
        output_dir: Directory to write the Ultralytics dataset
        val_split: Fraction of data for validation
        seed: Random seed for reproducible splits
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    train_entries, val_entries = split_dataset(intermediate_dataset, val_split, seed)
    
    # Process each split
    for split_name, entries in [("train", train_entries), ("val", val_entries)]:
        for entry in entries:
            # Create unique filename based on the source image path
            source_path = Path(entry.image_path)
            unique_name = f"{entry.dataset}_{entry.video}_{entry.classifier}_{source_path.stem}"
            
            # Create symlink for image (fallback to copy on Windows)
            image_dest = output_dir / "images" / split_name / f"{unique_name}.jpg"
            try:
                # Remove existing symlink/file if present
                if image_dest.exists() or image_dest.is_symlink():
                    image_dest.unlink()
                image_dest.symlink_to(source_path.resolve())
            except OSError:
                # Fallback to copy on systems that don't support symlinks
                shutil.copy2(source_path, image_dest)
            
            # Create label file with YOLO format annotations
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            label_dest = output_dir / "labels" / split_name / f"{unique_name}.txt"
            with open(label_dest, 'w') as f:
                for ann in entry.annotations:
                    x1, y1, x2, y2 = ann['bbox']
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((x1 + x2) / 2) / entry.image_width
                    y_center = ((y1 + y2) / 2) / entry.image_height
                    width = (x2 - x1) / entry.image_width
                    height = (y2 - y1) / entry.image_height
                    
                    # Class 0 for 'car'
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create data.yaml
    yaml_content = f"""# Auto-generated finetune dataset for Ultralytics YOLO
path: {output_dir.resolve()}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names:
  0: car
"""
    with open(output_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Created Ultralytics dataset at {output_dir}")
    print(f"  Train images: {len(train_entries)}")
    print(f"  Val images: {len(val_entries)}")


def convert_to_coco(
    intermediate_dataset: list[CompressedImageAnnotation],
    output_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Convert intermediate dataset to COCO format.
    
    Creates:
        - {output_dir}/train2017/*.jpg (symlinks)
        - {output_dir}/val2017/*.jpg (symlinks)
        - {output_dir}/annotations/instances_train2017.json
        - {output_dir}/annotations/instances_val2017.json
        
    Args:
        intermediate_dataset: List of CompressedImageAnnotation entries
        output_dir: Directory to write the COCO dataset
        val_split: Fraction of data for validation
        seed: Random seed for reproducible splits
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "val2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    train_entries, val_entries = split_dataset(intermediate_dataset, val_split, seed)
    
    # COCO category definition
    categories = [{"id": 1, "name": "car", "supercategory": "vehicle"}]
    
    # Process each split
    for split_name, entries in [("train2017", train_entries), ("val2017", val_entries)]:
        images = []
        annotations = []
        annotation_id = 1
        
        for image_id, entry in enumerate(entries, start=1):
            # Create unique filename
            source_path = Path(entry.image_path)
            unique_name = f"{entry.dataset}_{entry.video}_{entry.classifier}_{source_path.stem}.jpg"
            
            # Create symlink for image (fallback to copy)
            image_dest = output_dir / split_name / unique_name
            try:
                if image_dest.exists() or image_dest.is_symlink():
                    image_dest.unlink()
                image_dest.symlink_to(source_path.resolve())
            except OSError:
                shutil.copy2(source_path, image_dest)
            
            # Add image info
            images.append({
                "id": image_id,
                "file_name": unique_name,
                "width": entry.image_width,
                "height": entry.image_height,
            })
            
            # Add annotations
            for ann in entry.annotations:
                x1, y1, x2, y2 = ann['bbox']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # car
                    "bbox": [x1, y1, width, height],  # COCO format: [x, y, width, height]
                    "area": area,
                    "iscrowd": 0,
                })
                annotation_id += 1
        
        # Create COCO annotation file
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        
        annotation_file = output_dir / "annotations" / f"instances_{split_name}.json"
        with open(annotation_file, 'w') as f:
            json.dump(coco_data, f)
    
    print(f"Created COCO dataset at {output_dir}")
    print(f"  Train images: {len(train_entries)}")
    print(f"  Val images: {len(val_entries)}")


def convert_to_darknet(
    intermediate_dataset: list[CompressedImageAnnotation],
    output_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Convert intermediate dataset to Darknet YOLOv3 format.
    
    Creates:
        - {output_dir}/images/*.jpg (symlinks)
        - {output_dir}/labels/*.txt (YOLO format labels)
        - {output_dir}/train.txt (list of training image paths)
        - {output_dir}/val.txt (list of validation image paths)
        - {output_dir}/obj.names (class names)
        - {output_dir}/obj.data (data configuration)
        
    Args:
        intermediate_dataset: List of CompressedImageAnnotation entries
        output_dir: Directory to write the Darknet dataset
        val_split: Fraction of data for validation
        seed: Random seed for reproducible splits
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "backup").mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    train_entries, val_entries = split_dataset(intermediate_dataset, val_split, seed)
    
    # Track image paths for train.txt and val.txt
    train_paths: list[str] = []
    val_paths: list[str] = []
    
    # Process all entries
    for split_name, entries, paths_list in [
        ("train", train_entries, train_paths),
        ("val", val_entries, val_paths)
    ]:
        for entry in entries:
            # Create unique filename based on the source image path
            source_path = Path(entry.image_path)
            unique_name = f"{entry.dataset}_{entry.video}_{entry.classifier}_{source_path.stem}"
            
            # Create symlink for image (fallback to copy)
            image_dest = output_dir / "images" / f"{unique_name}.jpg"
            try:
                if image_dest.exists() or image_dest.is_symlink():
                    image_dest.unlink()
                image_dest.symlink_to(source_path.resolve())
            except OSError:
                shutil.copy2(source_path, image_dest)
            
            # Add to paths list (use absolute path for Darknet)
            paths_list.append(str(image_dest.resolve()))
            
            # Create label file with YOLO format annotations
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            label_dest = output_dir / "labels" / f"{unique_name}.txt"
            with open(label_dest, 'w') as f:
                for ann in entry.annotations:
                    x1, y1, x2, y2 = ann['bbox']
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((x1 + x2) / 2) / entry.image_width
                    y_center = ((y1 + y2) / 2) / entry.image_height
                    width = (x2 - x1) / entry.image_width
                    height = (y2 - y1) / entry.image_height
                    
                    # Class 0 for 'car'
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write train.txt
    train_txt = output_dir / "train.txt"
    with open(train_txt, 'w') as f:
        for path in train_paths:
            f.write(path + '\n')
    
    # Write val.txt
    val_txt = output_dir / "val.txt"
    with open(val_txt, 'w') as f:
        for path in val_paths:
            f.write(path + '\n')
    
    # Write obj.names
    names_file = output_dir / "obj.names"
    with open(names_file, 'w') as f:
        f.write("car\n")
    
    # Write obj.data
    data_file = output_dir / "obj.data"
    with open(data_file, 'w') as f:
        f.write(f"classes = 1\n")
        f.write(f"train = {train_txt.resolve()}\n")
        f.write(f"valid = {val_txt.resolve()}\n")
        f.write(f"names = {names_file.resolve()}\n")
        f.write(f"backup = {(output_dir / 'backup').resolve()}\n")
    
    print(f"Created Darknet dataset at {output_dir}")
    print(f"  Train images: {len(train_entries)}")
    print(f"  Val images: {len(val_entries)}")
    print(f"  obj.data: {data_file}")
    print(f"  obj.names: {names_file}")


def visualize_annotations(
    image_path: str | Path,
    annotations: list[dict],
    output_path: str | Path,
) -> None:
    """
    Visualize bounding boxes on an image with class labels.
    
    Simplified visualization showing only bounding boxes and class names,
    without track IDs or frame indices.
    
    Args:
        image_path: Path to the image file
        annotations: List of annotation dicts with keys 'bbox' [x1, y1, x2, y2] and 'class'
        output_path: Path to save the visualized image
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Color for bounding boxes (green in BGR)
    box_color = (0, 255, 0)
    box_thickness = 2
    
    # Draw each bounding box
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann.get('class', 'car')
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Draw class label
        label = class_name
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Position text above the bounding box
        text_x = x1
        text_y = max(y1 - 10, text_height + 5)
        
        # Draw text background for better visibility
        cv2.rectangle(
            img,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            box_color,
            -1
        )
        
        # Draw text in white
        cv2.putText(
            img, label, (text_x, text_y),
            font, font_scale, (255, 255, 255), font_thickness
        )
    
    # Save visualized image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def read_ultralytics_labels(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Read Ultralytics YOLO format labels and convert to absolute coordinates.
    
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    
    Args:
        label_path: Path to the label file (.txt)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        List of annotation dicts with 'bbox' [x1, y1, x2, y2] and 'class'
    """
    annotations = []
    
    if not Path(label_path).exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized coordinates to absolute pixel coordinates
            x_center_abs = x_center * image_width
            y_center_abs = y_center * image_height
            width_abs = width * image_width
            height_abs = height * image_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x_center_abs - width_abs / 2
            y1 = y_center_abs - height_abs / 2
            x2 = x_center_abs + width_abs / 2
            y2 = y_center_abs + height_abs / 2
            
            # Class name (assuming class 0 is 'car')
            class_name = 'car' if class_id == 0 else f'class_{class_id}'
            
            annotations.append({
                'bbox': [x1, y1, x2, y2],
                'class': class_name,
            })
    
    return annotations


def read_coco_annotations(
    annotation_file: str | Path,
    image_filename: str,
) -> list[dict]:
    """
    Read COCO format annotations for a specific image.
    
    COCO format: bbox is [x, y, width, height] in absolute pixels
    
    Args:
        annotation_file: Path to the COCO annotation JSON file
        image_filename: Filename of the image to get annotations for
        
    Returns:
        List of annotation dicts with 'bbox' [x1, y1, x2, y2] and 'class'
    """
    annotations = []
    
    if not Path(annotation_file).exists():
        return annotations
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find image ID for this filename
    image_id = None
    for img_info in coco_data.get('images', []):
        if img_info['file_name'] == image_filename:
            image_id = img_info['id']
            break
    
    if image_id is None:
        return annotations
    
    # Get category mapping
    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Find all annotations for this image
    for ann in coco_data.get('annotations', []):
        if ann['image_id'] == image_id:
            # COCO format: [x, y, width, height]
            x, y, width, height = ann['bbox']
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            
            # Get class name
            category_id = ann['category_id']
            class_name = category_map.get(category_id, f'class_{category_id}')
            
            annotations.append({
                'bbox': [x1, y1, x2, y2],
                'class': class_name,
            })
    
    return annotations


def read_darknet_labels(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Read Darknet YOLO format labels and convert to absolute coordinates.
    
    Darknet format is the same as Ultralytics: class_id x_center y_center width height (all normalized 0-1)
    
    Args:
        label_path: Path to the label file (.txt)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        List of annotation dicts with 'bbox' [x1, y1, x2, y2] and 'class'
    """
    # Darknet uses the same format as Ultralytics
    return read_ultralytics_labels(label_path, image_width, image_height)
