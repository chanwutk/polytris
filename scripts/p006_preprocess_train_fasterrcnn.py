#!/usr/local/bin/python

"""
Train Faster R-CNN on CalDOT COCO dataset.
Simple wrapper around PyTorch Vision's training script.
"""

import argparse
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on CalDOT COCO dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="caldot1",
        help="Dataset name (default: caldot1)",
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
        default=1,
        help="Batch size per GPU (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0025,
        help="Learning rate (default: 0.0025, scaled for single GPU)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes including background (default: 2 for CalDOT: background + car)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=5.0,
        help="Max norm for gradient clipping (default: 5.0, use 0 to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda)",
    )
    parser.add_argument(
        "--track-best-loss",
        action="store_true",
        help="Track and save the model with lowest validation loss",
    )
    return parser.parse_args()


def main(args):
    """
    Train Faster R-CNN using PyTorch Vision's training script.
    
    Args:
        args: Parsed command line arguments
    """
    # Derive paths from dataset name
    dataset_dir = f"/polyis-data/training/torchvision/{args.dataset}/training-data"
    output_dir = f"/polyis-data/training/torchvision/{args.dataset}/weights/{args.model}"
    
    print("=" * 80)
    print("Faster R-CNN Training on CalDOT Dataset")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Gradient clipping: {args.clip_grad_norm if args.clip_grad_norm > 0 else 'disabled'}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please run p007_create_coco_from_caldot.py --dataset {args.dataset} first to create the dataset."
        )
    
    train_json = os.path.join(dataset_dir, "annotations", "instances_train2017.json")
    val_json = os.path.join(dataset_dir, "annotations", "instances_val2017.json")
    
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation annotations not found: {val_json}")
    
    # Create output directory
    if os.path.exists(output_dir):
        print(f"Warning: Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to training script
    train_script = "/polyis/modules/vision_references/references/detection/train.py"
    
    # Build command
    cmd = [
        sys.executable,
        train_script,
        "--data-path", dataset_dir,
        "--dataset", "coco",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--lr", str(args.lr),
        "--weight-decay", "0.0001",
        "--momentum", "0.9",
        "--weights-backbone", "ResNet50_Weights.IMAGENET1K_V1",
        "--lr-steps", "16", "22",
        "--aspect-ratio-group-factor", "3",
        "--output-dir", output_dir,
        "--device", args.device,
        "--num-classes", str(args.num_classes),
        "--clip-grad-norm", str(args.clip_grad_norm),
    ]
    
    if args.track_best_loss:
        cmd.append("--track-best-loss")
    
    print("Running training command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    if args.track_best_loss:
        best_model_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Best model (lowest validation loss): {best_model_path}")
        else:
            print("Warning: Best model file not found.")
    print(f"All weights saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
