"""
Verify Ultralytics dataset structure.

Checks that a dataset directory contains all required files and directories
for Ultralytics YOLO training.
"""

import os
from pathlib import Path


def verify_dataset(data_dir: str | Path) -> None:
    """
    Verify that the dataset directory exists and contains required files.

    Args:
        data_dir: Path to dataset directory

    Raises:
        FileNotFoundError: If dataset or required files don't exist
        ValueError: If dataset is empty
    """
    data_dir = Path(data_dir)

    # Check if dataset directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Please create the dataset first using the appropriate dataset creation script."
        )

    # Check for data.yaml
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Check for image directories
    train_images = data_dir / "images" / "train"
    val_images = data_dir / "images" / "val"

    if not train_images.exists():
        raise FileNotFoundError(f"Training images directory not found: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images}")

    # Check for label directories
    train_labels = data_dir / "labels" / "train"
    val_labels = data_dir / "labels" / "val"

    if not train_labels.exists():
        raise FileNotFoundError(f"Training labels directory not found: {train_labels}")
    if not val_labels.exists():
        raise FileNotFoundError(f"Validation labels directory not found: {val_labels}")

    # Count files for verification
    num_train_images = len(
        [f for f in train_images.iterdir() if f.suffix.lower() in (".jpg", ".png")]
    )
    num_val_images = len(
        [f for f in val_images.iterdir() if f.suffix.lower() in (".jpg", ".png")]
    )
    num_train_labels = len([f for f in train_labels.iterdir() if f.suffix == ".txt"])
    num_val_labels = len([f for f in val_labels.iterdir() if f.suffix == ".txt"])

    print(f"Dataset verification:")
    print(f"  Train images: {num_train_images}")
    print(f"  Train labels: {num_train_labels}")
    print(f"  Val images: {num_val_images}")
    print(f"  Val labels: {num_val_labels}")
    print()

    # Warn if image/label counts don't match
    if num_train_images != num_train_labels:
        print(
            f"Warning: Train image count ({num_train_images}) doesn't match label count ({num_train_labels})"
        )
    if num_val_images != num_val_labels:
        print(
            f"Warning: Val image count ({num_val_images}) doesn't match label count ({num_val_labels})"
        )

    # Check if dataset is empty
    if num_train_images == 0:
        raise ValueError("Training dataset is empty!")
    if num_val_images == 0:
        raise ValueError("Validation dataset is empty!")

