"""
Get dataset subset directory structure.

Common pattern for discovering train/valid/test subsets in CalDOT dataset structure.
"""

from pathlib import Path


def get_dataset_subsets(base_root: Path, dataset: str) -> list[tuple[str, Path]]:
    """
    Get list of dataset subsets (train, valid, test) with their root paths.

    Args:
        base_root: Base root directory (e.g., Path("/otif-dataset/dataset"))
        dataset: Dataset name

    Returns:
        List of (subset_name, subset_root) tuples
    """
    dataset_root = base_root / dataset

    # We'll include videos from train, valid, and test splits (if present)
    subsets = [
        ("train", dataset_root / "train"),
        ("valid", dataset_root / "valid"),
        ("test", dataset_root / "test"),
    ]

    return subsets

