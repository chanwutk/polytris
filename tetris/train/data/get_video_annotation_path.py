"""
Get video ID from filename and construct annotation file path.

Common pattern for extracting video identifier and finding corresponding
annotation JSON file.
"""

from pathlib import Path


def get_video_annotation_path(video_file: Path, anno_dir: Path) -> tuple[str, Path]:
    """
    Extract video ID from filename and construct annotation file path.

    Args:
        video_file: Path to video file
        anno_dir: Directory containing annotation files

    Returns:
        Tuple of (video_id, anno_file_path)
    """
    # Get video ID from filename (without extension)
    video_id = video_file.stem
    anno_file = anno_dir / f"{video_id}.json"

    return video_id, anno_file

