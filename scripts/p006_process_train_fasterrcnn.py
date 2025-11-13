#!/usr/local/bin/python

"""
Train Faster R-CNN using PyTorch Vision's detection training script.
Uses the same dataset created by p005_preprocess_train_yolov12.py.
Fixes COCO JSON file_name paths for PyTorch Vision compatibility and trains the model.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN using PyTorch Vision on dataset from p005"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/polyis-data/coco-datasets/caldot1",
        help="Directory containing COCO dataset (created by p005_preprocess_train_yolov12.py)",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=None,
        help="Path to PyTorch Vision train.py script. If not specified, will try to find it or use torchvision's built-in training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn_resnet50_fpn",
        help="Model architecture (default: fasterrcnn_resnet50_fpn)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=26,
        help="Number of training epochs (default: 26)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for trained model and logs",
    )
    parser.add_argument(
        "--skip-fix-paths",
        action="store_true",
        help="Skip fixing COCO JSON file_name paths (use if already fixed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda)",
    )
    return parser.parse_args()


def fix_coco_json_paths(dataset_dir: str) -> None:
    """
    Fix COCO JSON file_name paths to include images/train/ or images/val/ prefix.
    PyTorch Vision expects file_name to be relative to dataset root.
    
    Note: This modifies the COCO JSON files, but YOLOv11 training uses YOLO format
    txt files (not COCO JSON), so this won't affect YOLOv11 training.

    Args:
        dataset_dir: Directory containing COCO dataset
    """
    train_json = os.path.join(dataset_dir, "annotations", "instances_train.json")
    val_json = os.path.join(dataset_dir, "annotations", "instances_val.json")

    # Fix training JSON
    if os.path.exists(train_json):
        print(f"Fixing file_name paths in {train_json}...")
        # Create backup before modifying
        backup_path = train_json + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(train_json, backup_path)
            print(f"  Created backup: {backup_path}")
        
        with open(train_json, "r") as f:
            train_data = json.load(f)

        # Update file_name paths to include images/train/ prefix
        updated_count = 0
        for image_entry in train_data["images"]:
            filename = image_entry["file_name"]
            # Only add prefix if not already present
            if not filename.startswith("images/train/"):
                image_entry["file_name"] = os.path.join("images", "train", filename)
                updated_count += 1

        with open(train_json, "w") as f:
            json.dump(train_data, f, indent=2)
        print(f"  Updated {updated_count} image paths in train JSON")

    # Fix validation JSON
    if os.path.exists(val_json):
        print(f"Fixing file_name paths in {val_json}...")
        # Create backup before modifying
        backup_path = val_json + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(val_json, backup_path)
            print(f"  Created backup: {backup_path}")
        
        with open(val_json, "r") as f:
            val_data = json.load(f)

        # Update file_name paths to include images/val/ prefix
        updated_count = 0
        for image_entry in val_data["images"]:
            filename = image_entry["file_name"]
            # Only add prefix if not already present
            if not filename.startswith("images/val/"):
                image_entry["file_name"] = os.path.join("images", "val", filename)
                updated_count += 1

        with open(val_json, "w") as f:
            json.dump(val_data, f, indent=2)
        print(f"  Updated {updated_count} image paths in val JSON")


def find_pytorch_vision_train_script() -> str | None:
    """
    Try to find PyTorch Vision's train.py script.

    Returns:
        Path to train.py script or None if not found
    """
    # Common locations for PyTorch Vision references
    possible_paths = [
        "/usr/local/lib/python3.13/site-packages/torchvision/references/detection/train.py",
        "/usr/local/lib/python3.13/site-packages/torchvision/references/detection/engine.py",
        os.path.expanduser("~/torchvision/references/detection/train.py"),
    ]

    # Try to find torchvision package location
    try:
        import torchvision
        torchvision_path = Path(torchvision.__file__).parent
        possible_paths.insert(0, str(torchvision_path / "references" / "detection" / "train.py"))
    except ImportError:
        pass

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def main(args):
    """
    Train Faster R-CNN using PyTorch Vision's training script.

    Args:
        args: Parsed command line arguments
    """
    print("=" * 80)
    print("Faster R-CNN Training Script")
    print("=" * 80)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Check if dataset exists
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    train_json = os.path.join(args.dataset_dir, "annotations", "instances_train.json")
    val_json = os.path.join(args.dataset_dir, "annotations", "instances_val.json")

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation annotations not found: {val_json}")

    # Fix COCO JSON file_name paths if needed
    if not args.skip_fix_paths:
        print("\nFixing COCO JSON file_name paths for PyTorch Vision compatibility...")
        fix_coco_json_paths(args.dataset_dir)
    else:
        print("\nSkipping path fixing (--skip-fix-paths flag set)")

    # Find or use provided training script
    train_script = args.train_script
    if train_script is None:
        train_script = find_pytorch_vision_train_script()
    assert train_script is not None

    # Build command to run train.py
    cmd = [
        sys.executable,
        train_script,
        "--data-path", args.dataset_dir,
        "--dataset", "coco",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--momentum", str(args.momentum),
        "--output-dir", args.output_dir,
    ]

    if args.device:
        cmd.extend(["--device", args.device])

    print("\nRunning training command:")
    print(" ".join(cmd))
    print()

    # Run training
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
