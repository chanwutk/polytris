"""
Convert COCO format annotations to YOLO format.

Converts COCO format JSON annotations to YOLO format text files.
"""

import json
import os
from typing import Any


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
                f.write(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                )

    print(f"  Converted COCO annotations to YOLO format: {len(annotations_by_image)} images with labels")

