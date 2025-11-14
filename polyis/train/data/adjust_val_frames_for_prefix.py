"""
Adjust validation frame set for prefixed video IDs.

Common pattern for handling video ID prefixes to avoid filename collisions
across dataset subsets when splitting frames.
"""


def adjust_val_frames_for_prefix(
    val_frames: set[tuple[str, int]],
    video_id: str,
    id_prefix: str,
) -> set[tuple[str, int]]:
    """
    Create adjusted val_frames set with prefixed video_id for lookup.

    When processing videos with prefixes (e.g., "train_", "valid_") to avoid
    filename collisions, this function adjusts the validation frame set to
    match the prefixed video IDs.

    Args:
        val_frames: Set of (video_id, frame_idx) tuples for validation frames
        video_id: Original video ID (without prefix)
        id_prefix: Prefix to add to video_id (e.g., "train_", "valid_")

    Returns:
        Set of (prefixed_video_id, frame_idx) tuples for frames matching this video
    """
    # Use prefixed video id to avoid filename collisions across subsets
    prefixed_video_id = f"{id_prefix}{video_id}" if id_prefix else video_id

    # Create adjusted val_frames set with prefixed video_id for lookup
    adjusted_val_frames = set()
    for orig_video_id, frame_idx in val_frames:
        # Match frames by checking if the original video_id matches
        # (accounting for prefix that will be added)
        if orig_video_id == video_id:
            adjusted_val_frames.add((prefixed_video_id, frame_idx))

    return adjusted_val_frames

