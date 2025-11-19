"""
Discover video files in dataset subsets.

Common pattern for finding video files across train/valid/test subsets
and their corresponding annotation directories.
"""

from pathlib import Path

from .find_highest_resolution_annotations import find_highest_resolution_annotations
from .get_dataset_subsets import get_dataset_subsets


def discover_videos_in_subsets(
    base_root: Path,
    dataset: str,
) -> list[tuple[str, Path, Path]]:
    """
    Discover video files in dataset subsets with their annotation directories.

    Args:
        base_root: Base root directory (e.g., Path("/otif-dataset/dataset"))
        dataset: Dataset name

    Returns:
        List of (subset_name, video_file, anno_dir) tuples
    """
    subsets = get_dataset_subsets(base_root, dataset)
    video_info = []

    for subset_name, subset_root in subsets:
        # Skip subsets that don't exist
        if not subset_root.exists():
            continue

        video_dir = subset_root / "video"
        if not video_dir.exists():
            continue

        # Find annotations directory with highest resolution for this subset
        anno_dir = find_highest_resolution_annotations(subset_root)

        # Gather videos for this subset
        video_files = sorted([f for f in video_dir.glob("*.mp4")])

        # Add video info for each video file
        for video_file in video_files:
            video_info.append((subset_name, video_file, anno_dir))

    return video_info

