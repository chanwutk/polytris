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


def train_faster_rcnn(args) -> None:
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

    if train_script is None or not os.path.exists(train_script):
        # If train.py not found, use torchvision's built-in training API
        print("\nPyTorch Vision train.py not found, using torchvision API directly...")
        train_with_torchvision_api(args)
    else:
        # Use the train.py script
        print(f"\nUsing PyTorch Vision training script: {train_script}")
        train_with_script(train_script, args)


def train_with_script(train_script: str, args) -> None:
    """
    Train using PyTorch Vision's train.py script.

    Args:
        train_script: Path to train.py script
        args: Parsed command line arguments
    """
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


def train_with_torchvision_api(args) -> None:
    """
    Train using torchvision API directly (fallback if train.py not found).

    Args:
        args: Parsed command line arguments
    """
    try:
        import torch
        import torchvision
        from torchvision import transforms
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torch.utils.data import DataLoader
        from torchvision.datasets import CocoDetection
        from torch.optim import SGD
        import torchvision.transforms.functional as F
    except ImportError as e:
        raise ImportError(f"Required packages not found: {e}. Please install torch and torchvision.")

    print("\nSetting up Faster R-CNN model...")

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the classifier head for 1 class (car) + background = 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 1 class + background

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded on {device}")
    print(f"Number of classes: 2 (1 object class + background)")

    # Create dataset and dataloader
    print("\nLoading dataset...")
    
    # PyTorch Vision's CocoDetection expects:
    # - root: directory containing images (or parent directory if file_name includes subdirs)
    # - annFile: path to COCO JSON file
    # - file_name in JSON: relative to root
    # - transform: only transforms the image, not the target
    # Since we fixed file_name to include images/train/ or images/val/, we use dataset root
    train_dataset = CocoDetection(
        root=args.dataset_dir,  # Root directory (file_name includes images/train/)
        annFile=os.path.join(args.dataset_dir, "annotations", "instances_train.json"),
        transform=transforms.ToTensor(),
    )

    val_dataset = CocoDetection(
        root=args.dataset_dir,  # Root directory (file_name includes images/val/)
        annFile=os.path.join(args.dataset_dir, "annotations", "instances_val.json"),
        transform=transforms.ToTensor(),
    )
    
    # Define function to convert COCO format targets to model format
    def convert_coco_target_to_model_format(target):
        """
        Convert COCO format target to model format.
        COCO format: list of dicts with 'bbox' [x, y, width, height] and 'category_id'
        Model format: dict with 'boxes' tensor [x1, y1, x2, y2] and 'labels' tensor
        """
        boxes = []
        labels = []
        if isinstance(target, list):
            for ann in target:
                if isinstance(ann, dict):
                    # COCO format: bbox is [x, y, width, height]
                    bbox = ann.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        # Convert to [x1, y1, x2, y2]
                        boxes.append([x, y, x + w, y + h])
                        # COCO uses 1-indexed category_id, model uses 0-indexed
                        # Our dataset has category_id=1 for "car", so we use label=0
                        category_id = ann.get('category_id', 1)
                        labels.append(category_id - 1)  # Convert to 0-indexed
        
        # Create target dict
        target_dict = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }
        return target_dict

    # Set workers=0 to avoid shared memory issues in Docker/containers
    # This disables multiprocessing for data loading
    if args.workers > 0:
        print(f"  Note: Setting workers=0 to avoid shared memory issues (requested: {args.workers})")
    
    # Custom collate function that converts COCO targets to model format
    def collate_fn(batch):
        """
        Collate function that converts COCO format targets to model format.
        """
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(convert_coco_target_to_model_format(target))
        return images, targets
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid shared memory issues
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid shared memory issues
        collate_fn=collate_fn,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    model.train()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move images and targets to device
            images = [img.to(device) for img in images]
            # Targets are already in dict format from our transform, just move to device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            # Sum all losses into a single tensor
            losses = sum(loss for loss in loss_dict.values())
            # Extract loss value for logging
            if isinstance(losses, torch.Tensor):
                loss_value = losses.item()
            else:
                loss_value = float(losses)

            # Backward pass
            optimizer.zero_grad()
            losses.backward()  # type: ignore
            optimizer.step()

            running_loss += loss_value

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss_value:.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"fasterrcnn_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Final model saved to: {os.path.join(args.output_dir, 'fasterrcnn_epoch_{args.epochs}.pth')}")
    print("=" * 80)


def main(args):
    """
    Main function to train Faster R-CNN.

    Args:
        args: Parsed command line arguments
    """
    train_faster_rcnn(args)


if __name__ == "__main__":
    main(parse_args())
