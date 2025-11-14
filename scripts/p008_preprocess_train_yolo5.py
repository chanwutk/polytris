#!/usr/local/bin/python

"""
Train YOLOv5x6 model on CalDOT dataset.
Uses Ultralytics API to train YOLOv5 on dataset created by p007.
"""

import argparse
import os
import shutil
import sys

import ultralytics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv5x6 on CalDOT dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="caldot1",
        help="Dataset name (default: caldot1)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory containing data.yaml (default: /polyis-data/yolo5/{dataset}/training-data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=720,
        help="Image size for training (default: 720)",
    )
    parser.add_argument(
        # Note: batch size = 4 in freddie
        "--batch",
        type=int,
        default=-1,
        help="Batch size (default: -1 for auto-batch)",
    )
    parser.add_argument(
        # Note: one device in freddie
        "--device",
        type=str,
        default=None,
        help="Device(s) to use for training (e.g., '0' or '0,1,2' for multi-GPU, default: auto-detect)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0 for Docker compatibility)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project directory for saving results (default: /polyis-data/yolo5/{dataset})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name (default: train)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="yolov5x6.pt",
        help="Pretrained weights to use (default: yolov5x6.pt)",
    )
    return parser.parse_args()


def verify_dataset(data_dir):
    """
    Verify that the dataset directory exists and contains required files.

    Args:
        data_dir: Path to dataset directory

    Raises:
        FileNotFoundError: If dataset or required files don't exist
    """
    # Check if dataset directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Please run p007_preprocess_create_trainset_yolo5.py first to create the dataset."
        )

    # Check for data.yaml
    data_yaml = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Check for image directories
    train_images = os.path.join(data_dir, "images", "train")
    val_images = os.path.join(data_dir, "images", "val")

    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Training images directory not found: {train_images}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images directory not found: {val_images}")

    # Check for label directories
    train_labels = os.path.join(data_dir, "labels", "train")
    val_labels = os.path.join(data_dir, "labels", "val")

    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels directory not found: {train_labels}")
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels directory not found: {val_labels}")

    # Count files for verification
    num_train_images = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png'))])
    num_val_images = len([f for f in os.listdir(val_images) if f.endswith(('.jpg', '.png'))])
    num_train_labels = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
    num_val_labels = len([f for f in os.listdir(val_labels) if f.endswith('.txt')])

    print(f"Dataset verification:")
    print(f"  Train images: {num_train_images}")
    print(f"  Train labels: {num_train_labels}")
    print(f"  Val images: {num_val_images}")
    print(f"  Val labels: {num_val_labels}")
    print()

    # Warn if image/label counts don't match
    if num_train_images != num_train_labels:
        print(f"Warning: Train image count ({num_train_images}) doesn't match label count ({num_train_labels})")
    if num_val_images != num_val_labels:
        print(f"Warning: Val image count ({num_val_images}) doesn't match label count ({num_val_labels})")

    # Check if dataset is empty
    if num_train_images == 0:
        raise ValueError("Training dataset is empty!")
    if num_val_images == 0:
        raise ValueError("Validation dataset is empty!")


def train_yolov5(args):
    """
    Train YOLOv5x6 model using Ultralytics API.

    Args:
        args: Parsed command line arguments
    """
    # Set up paths
    data_dir = args.data_dir or f"/polyis-data/yolo5/{args.dataset}/training-data"
    project_dir = args.project or f"/polyis-data/yolo5/{args.dataset}"
    weights_dir = f"/polyis-data/yolo5/{args.dataset}/weights"
    data_yaml = os.path.join(data_dir, "data.yaml")

    print("=" * 80)
    print("YOLOv5x6 Training on CalDOT Dataset")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Data config: {data_yaml}")
    print(f"Project directory: {project_dir}")
    print(f"Experiment name: {args.name}")
    print(f"Model: {args.pretrained}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch if args.batch > 0 else 'auto'}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print(f"Workers: {args.workers}")
    print()

    # Verify dataset exists and is valid
    verify_dataset(data_dir)

    # Load YOLOv5x6 pretrained model
    print(f"Loading YOLOv5 model: {args.pretrained}...")
    model = ultralytics.YOLO(args.pretrained)  # type: ignore

    print("Model loaded successfully!")
    print()

    # Prepare training arguments
    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": project_dir,
        "name": args.name,
        "workers": args.workers,
        "seed": 42,
        "exist_ok": True,
        "rect": False,
        "dropout": 0.2,
        "plots": True,
        "classes": [0],
    }

    # Add batch size if specified
    if args.batch > 0:
        train_kwargs["batch"] = args.batch

    # Add device if specified
    if args.device:
        # Parse device string: "0" or "0,1,2" -> list of ints or single int
        if "," in args.device:
            train_kwargs["device"] = [int(d.strip()) for d in args.device.split(",")]
        else:
            train_kwargs["device"] = int(args.device)

    # Start training
    print("Starting training...")
    print(f"Training arguments: {train_kwargs}")
    print()

    results = model.train(**train_kwargs)
    
    # Copy weights to the desired location
    source_weights_dir = os.path.join(project_dir, args.name, "weights")
    if os.path.exists(source_weights_dir):
        # Create destination directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)
        # Copy weight files
        for weight_file in ["best.pt", "last.pt"]:
            source_path = os.path.join(source_weights_dir, weight_file)
            dest_path = os.path.join(weights_dir, weight_file)
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"Copied {weight_file} to {dest_path}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Results saved to: {project_dir}/{args.name}")
    print(f"Best model weights: {weights_dir}/best.pt")
    print(f"Last model weights: {weights_dir}/last.pt")
    print("=" * 80)


def main():
    args = parse_args()
    train_yolov5(args)


if __name__ == "__main__":
    main()
