"""
Find annotation directory with highest resolution.

Looks for directories matching "yolov3-*" pattern and returns the one
with the highest width dimension.
"""

from pathlib import Path


def find_highest_resolution_annotations(dataset_root: Path) -> Path:
    """
    Find annotation directory with highest resolution.

    Looks for directories matching "yolov3-*" pattern and returns the one
    with the highest width dimension.

    Args:
        dataset_root: Root directory of the dataset

    Returns:
        Path to annotation directory with highest resolution

    Raises:
        FileNotFoundError: If no annotation directories are found
    """
    # Look for yolov3-* annotation directories
    anno_dirs = sorted(dataset_root.glob("yolov3-*"))
    if not anno_dirs:
        raise FileNotFoundError(f"No annotation directories found in {dataset_root}")

    def get_width(path: Path) -> int:
        # Extract width from "yolov3-WIDTHxHEIGHT"
        dims = path.name.split('-')[1]
        width = int(dims.split('x')[0])
        return width

    # Return directory with highest width
    return max(anno_dirs, key=get_width)

