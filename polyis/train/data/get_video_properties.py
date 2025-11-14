"""
Extract video properties (FPS, frame count, width, height) from video file.

Common utility for getting video metadata without processing frames.
"""

from pathlib import Path

import cv2


def get_video_properties(video_path: Path) -> dict[str, float]:
    """
    Extract video properties from video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with keys: 'fps', 'frame_count', 'width', 'height'
        Returns None values if video cannot be opened

    Raises:
        RuntimeError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }

