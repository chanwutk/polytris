#!/usr/local/bin/python

"""
Train YOLOv5x6 or YOLOv11x model on CalDOT dataset.
Uses Ultralytics API to train YOLO models on dataset created by p007 or p004.
"""

import argparse
import os
import shutil

import ultralytics

from polyis.train.data.ultralytics import parse_device_string, verify_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv5x6 or YOLOv11x on CalDOT dataset"
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
        help="Dataset directory containing data.yaml (default: /polyis-data/training/ultralytics/{dataset}/training-data)",
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
        default=1280,
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
        "--model",
        type=str,
        default="5x6",
        choices=["5x6", "11x"],
        help="Model to use (default: 5x6)",
    )
    return parser.parse_args()


def main():
    """
    Train YOLOv5x6 or YOLOv11x model using Ultralytics API.
    """
    args = parse_args()

    # Map model option to pretrained weights filename
    model_to_weights = {
        "5x6": "yolov5x6.pt",
        "11x": "yolo11x.pt",
    }
    pretrained_weights = model_to_weights[args.model]
    # Extract model name without extension for weights directory
    model_name = os.path.splitext(pretrained_weights)[0]
    
    # Set up paths
    data_dir = args.data_dir or f"/polyis-data/training/ultralytics/{args.dataset}/training-data"
    weights_dir = f"/polyis-data/training/ultralytics/{args.dataset}/weights/{model_name}"
    # Set project to parent directory and name to model_name so Ultralytics saves to weights_dir/weights/
    project_dir = os.path.dirname(weights_dir)
    experiment_name = model_name
    data_yaml = os.path.join(data_dir, "data.yaml")

    print("=" * 80)
    print(f"YOLO ({model_name}) Training on CalDOT Dataset")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Data config: {data_yaml}")
    print(f"Project directory: {project_dir}")
    print(f"Experiment name: {experiment_name}")
    print(f"Model: {args.model} ({pretrained_weights})")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch if args.batch > 0 else 'auto'}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print(f"Workers: {args.workers}")
    print()

    # Verify dataset exists and is valid
    verify_dataset(data_dir)

    # Clear cache files to force Ultralytics to regenerate with correct annotations
    train_cache = os.path.join(data_dir, "train.cache")
    val_cache = os.path.join(data_dir, "val.cache")
    if os.path.exists(train_cache):
        os.remove(train_cache)
        print(f"  Cleared train cache: {train_cache}")
    if os.path.exists(val_cache):
        os.remove(val_cache)
        print(f"  Cleared val cache: {val_cache}")

    # Clean up weights directory before training
    experiment_dir = os.path.join(project_dir, experiment_name)
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
        print(f"  Cleared previous training results: {experiment_dir}")

    # Load pretrained model
    print(f"Loading model: {pretrained_weights}...")
    model = ultralytics.YOLO(pretrained_weights)  # type: ignore

    print("Model loaded successfully!")
    print()

    # Prepare training arguments
    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": project_dir,
        "name": experiment_name,
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
    device = parse_device_string(args.device)
    if device is not None:
        train_kwargs["device"] = device

    # Start training
    print("Starting training...")
    print(f"Training arguments: {train_kwargs}")
    print()

    results = model.train(**train_kwargs)
    
    # Weights are saved directly to {project_dir}/{experiment_name}/weights/
    # which equals {weights_dir}/weights/ when using default paths
    weights_output_dir = os.path.join(project_dir, experiment_name, "weights")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Results saved to: {project_dir}/{experiment_name}")
    print(f"Best model weights: {os.path.join(weights_output_dir, 'best.pt')}")
    print(f"Last model weights: {os.path.join(weights_output_dir, 'last.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
