"""
Convert bounding box to COCO format.

Converts bounding box from [left, top, right, bottom] to COCO format [x, y, width, height].
"""


def convert_bbox_to_coco(
    left: float, top: float, right: float, bottom: float
) -> tuple[float, float, float, float]:
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

