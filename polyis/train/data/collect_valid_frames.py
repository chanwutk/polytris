"""
Collect frame indices from a video.

Supports including or excluding frames without annotations based on use case.
"""

import json
from pathlib import Path

from .get_adjusted_frame_stride import get_adjusted_frame_stride


def collect_valid_frames(
    video_file: Path,
    anno_path: Path,
    video_id: str,
    frame_stride: int = 1,
    include_empty: bool = True,
) -> list[tuple[str, int]]:
    """
    Collect frame indices from a video.

    Args:
        video_file: Path to video file
        anno_path: Path to annotation JSON file
        video_id: Video identifier
        frame_stride: Extract every Nth frame (adjusted based on video FPS)
        include_empty: If True, include frames without annotations (for negative examples).
                      If False, only include frames with annotations.

    Returns:
        List of (video_id, frame_idx) tuples for valid frames
    """
    # Load annotations from JSON file
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Get adjusted stride based on video FPS
    actual_stride = get_adjusted_frame_stride(video_file, frame_stride)

    # Collect frame indices
    valid_frames = []
    for frame_idx in range(len(annotations)):
        # Skip frames based on adjusted stride
        if frame_idx % actual_stride != 0:
            continue

        # Check if we should include this frame
        if include_empty:
            # Include all frames, even those without annotations
            valid_frames.append((video_id, frame_idx))
        else:
            # Only include frames with annotations
            frame_annos = annotations[frame_idx]
            if frame_annos:
                valid_frames.append((video_id, frame_idx))

    return valid_frames

