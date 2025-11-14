"""
Create COCO format dataset from samples.

Converts sample dictionaries with image and annotation data into COCO format JSON files.
"""

import json
from pathlib import Path
from typing import Any

from polyis.train.data import convert_bbox_to_coco
from .initialize_coco_structure import initialize_coco_structure


def create_coco_dataset_from_samples(
    samples: list[dict[str, Any]],
    output_dir: Path,
    split_name: str,
    annotations_dir: Path,
    dataset_name: str,
) -> None:
    """
    Create COCO format dataset from samples.

    Args:
        samples: List of sample dictionaries with image_filename, image_path, boxes, width, height
        output_dir: Output directory for COCO dataset
        split_name: Name of the split ('train' or 'val')
        annotations_dir: Directory to save COCO JSON files
        dataset_name: Name of the dataset
    """
    print(f"\nGenerating COCO annotations for {split_name} split...")
    print(f"  Total samples: {len(samples)}")

    # Initialize COCO JSON structure
    coco_data = initialize_coco_structure()

    # Update info with split name
    coco_data["info"]["description"] = f"{dataset_name} car detection dataset - {split_name} split"

    # Counters for IDs
    image_id = 1
    annotation_id = 1

    # Process each sample
    for sample in samples:
        # Add image entry
        image_entry = {
            "id": image_id,
            "file_name": sample["image_filename"],
            "height": sample["height"],
            "width": sample["width"],
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

        # Increment image ID
        image_id += 1

    # Save COCO JSON
    output_json_path = annotations_dir / f"instances_{split_name}.json"
    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"  Saved annotations to {output_json_path}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")

