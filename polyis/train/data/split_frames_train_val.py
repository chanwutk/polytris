"""
Split frame identifiers into training and validation sets with reproducible random shuffle.
"""

import random


def split_frames_train_val(
    all_frames: list[tuple[str, int]],
    val_split: float,
    seed: int = 42,
) -> set[tuple[str, int]]:
    """
    Split frame identifiers into training and validation sets with reproducible random shuffle.

    Args:
        all_frames: List of (video_id, frame_idx) tuples
        val_split: Fraction of frames for validation (0.0-1.0)
        seed: Random seed for reproducible split (default: 42)

    Returns:
        Set of (video_id, frame_idx) tuples for validation frames
    """
    # Create a copy and shuffle with seed for reproducible split
    frames_shuffled = list(all_frames)
    random.seed(seed)
    random.shuffle(frames_shuffled)

    # Use last val_split fraction of shuffled frames for validation
    num_val = int(len(frames_shuffled) * val_split)
    val_frames = set(frames_shuffled[-num_val:])

    print(f"Train frames: {len(frames_shuffled) - num_val}")
    print(f"Val frames: {num_val}")

    return val_frames

