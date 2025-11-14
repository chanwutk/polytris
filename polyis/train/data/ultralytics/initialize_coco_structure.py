"""
Initialize COCO JSON structure.

Creates a COCO format data structure with default or custom categories.
"""

import datetime
from typing import Any


def initialize_coco_structure(
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Initialize COCO JSON structure.

    Args:
        categories: List of category dictionaries. If None, uses default car category.

    Returns:
        COCO data structure dictionary
    """
    if categories is None:
        categories = [
            {
                "id": 1,
                "name": "car",
                "supercategory": "vehicle",
            }
        ]

    coco_data = {
        "info": {
            "description": "CalDOT car detection dataset",
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
        "categories": categories,
    }

    return coco_data

