"""
Calculate adjusted frame stride based on video FPS.

For videos with FPS > 20 (typically 30 fps), the stride is doubled
to maintain similar temporal sampling rate.
"""

from pathlib import Path

import cv2


def get_adjusted_frame_stride(video_path: Path, frame_stride: int) -> int:
    """
    Calculate adjusted frame stride based on video FPS.

    For videos with FPS > 20 (typically 30 fps), the stride is doubled
    to maintain similar temporal sampling rate.

    Args:
        video_path: Path to video file
        frame_stride: Base frame stride (extract every Nth frame)

    Returns:
        Adjusted frame stride based on video FPS
    """
    # Open video capture to get FPS
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # If we can't open the video, return original stride
        return frame_stride

    # Get video FPS and adjust stride accordingly
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    actual_stride = frame_stride
    if fps > 20:  # 30 fps video
        actual_stride = frame_stride * 2
    # else: 15 fps video, keep original stride

    return actual_stride

