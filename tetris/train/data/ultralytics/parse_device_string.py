"""
Parse device string for Ultralytics training.

Converts device string (e.g., "0" or "0,1,2") to the format expected by Ultralytics.
"""


def parse_device_string(device: str | None) -> int | list[int] | None:
    """
    Parse device string for Ultralytics training.

    Converts device string to int (single GPU) or list of ints (multi-GPU).

    Args:
        device: Device string (e.g., "0" or "0,1,2") or None for auto-detect

    Returns:
        int for single GPU, list of ints for multi-GPU, or None for auto-detect
    """
    if device is None:
        return None

    # Parse device string: "0" or "0,1,2" -> list of ints or single int
    if "," in device:
        return [int(d.strip()) for d in device.split(",")]
    else:
        return int(device)

