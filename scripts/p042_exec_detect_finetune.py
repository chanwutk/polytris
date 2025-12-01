#!/usr/local/bin/python
"""
Train object detection models on fine-tuning datasets created by p041.

This script trains YOLO (Ultralytics), Detectron2, or Darknet YOLOv3 models
on the compressed frame datasets to improve detection accuracy on packed/collaged images.

Supported model types:
- Ultralytics YOLO: yolov5x6, yolo11x
- Detectron2: Faster R-CNN, RetinaNet
- Darknet: YOLOv3
"""

import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train detector models on fine-tuning datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (required)')
    parser.add_argument('--tilesize', type=int, required=True,
                        help='Tile size (required)')
    parser.add_argument('--tilepadding', type=str, required=True,
                        help='Tile padding mode (required)')
    parser.add_argument('--model', type=str, default='5x6',
                        choices=['5x6', '11x', 'detectron2', 'darknet'],
                        help='Model type: 5x6 (yolov5x6), 11x (yolo11x), detectron2, darknet (yolov3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size for training (default: 1280)')
    parser.add_argument('--batch', type=int, default=-1,
                        help='Batch size (-1 for auto, default: -1)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "0" or "0,1,2")')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory path')
    parser.add_argument('--darknet-cfg', type=str, default=None,
                        help='Path to Darknet config file (.cfg) for YOLOv3 training')
    parser.add_argument('--darknet-weights', type=str, default=None,
                        help='Path to pretrained Darknet weights for transfer learning')
    return parser.parse_args()


def train_ultralytics(args, data_dir: Path):
    """
    Train using Ultralytics YOLO.
    
    Args:
        args: Parsed command line arguments
        data_dir: Path to the dataset directory
    """
    import ultralytics
    from polyis.train.data.ultralytics import verify_dataset, parse_device_string
    
    # Paths
    ultralytics_dir = data_dir / "ultralytics"
    data_yaml = ultralytics_dir / "data.yaml"
    weights_dir = data_dir / "weights" / f"yolov{args.model}"
    
    # Verify dataset exists
    print(f"Verifying dataset at {ultralytics_dir}")
    verify_dataset(str(ultralytics_dir))
    
    # Load model with pretrained weights
    model_weights = {
        "5x6": "yolov5x6.pt",
        "11x": "yolo11x.pt"
    }[args.model]
    
    print(f"Loading model: {model_weights}")
    model = ultralytics.YOLO(model_weights)
    
    # Training configuration
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": str(weights_dir.parent),
        "name": weights_dir.name,
        "workers": 0,
        "seed": 42,
        "exist_ok": True,
        "plots": True,
    }
    
    # Set batch size if specified
    if args.batch > 0:
        train_kwargs["batch"] = args.batch
    
    # Set device if specified
    device = parse_device_string(args.device)
    if device is not None:
        train_kwargs["device"] = device
    
    # Print configuration
    print(f"\nTraining configuration:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")
    
    # Start training
    print(f"\nStarting training...")
    model.train(**train_kwargs)
    
    print(f"\nTraining complete!")
    print(f"Weights saved to: {weights_dir}")


def train_detectron2(args, data_dir: Path):
    """
    Train using Detectron2.
    
    Args:
        args: Parsed command line arguments
        data_dir: Path to the dataset directory
    """
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    
    # Paths
    coco_dir = data_dir / "coco"
    weights_dir = data_dir / "weights" / "detectron2"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify COCO dataset exists
    train_json = coco_dir / "annotations" / "instances_train2017.json"
    val_json = coco_dir / "annotations" / "instances_val2017.json"
    
    if not train_json.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    if not val_json.exists():
        raise FileNotFoundError(f"Validation annotations not found: {val_json}")
    
    # Create unique dataset names
    dataset_name = f"finetune_{args.dataset}_{args.tilesize}_{args.tilepadding}"
    
    # Register datasets
    print(f"Registering COCO datasets...")
    register_coco_instances(
        f"{dataset_name}_train", {},
        str(train_json),
        str(coco_dir / "train2017")
    )
    register_coco_instances(
        f"{dataset_name}_val", {},
        str(val_json),
        str(coco_dir / "val2017")
    )
    
    # Configure Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class (car)
    
    # Solver configuration
    cfg.SOLVER.MAX_ITER = args.epochs * 1000  # Rough conversion from epochs
    cfg.SOLVER.IMS_PER_BATCH = args.batch if args.batch > 0 else 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []  # No LR decay
    
    # Output configuration
    cfg.OUTPUT_DIR = str(weights_dir)
    
    # Device configuration
    if args.device is not None:
        cfg.MODEL.DEVICE = f"cuda:{args.device}"
    
    # Print configuration
    print(f"\nTraining configuration:")
    print(f"  Model: Faster R-CNN R50-FPN")
    print(f"  Train dataset: {dataset_name}_train")
    print(f"  Val dataset: {dataset_name}_val")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Output dir: {cfg.OUTPUT_DIR}")
    
    # Start training
    print(f"\nStarting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Weights saved to: {weights_dir}")


def train_darknet(args, data_dir: Path):
    """
    Train using Darknet YOLOv3.
    
    Args:
        args: Parsed command line arguments
        data_dir: Path to the dataset directory
    """
    # Paths
    darknet_dir = data_dir / "darknet"
    data_file = darknet_dir / "obj.data"
    weights_dir = data_dir / "weights" / "darknet"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset exists
    if not data_file.exists():
        raise FileNotFoundError(
            f"Darknet data file not found: {data_file}\n"
            f"Please run p041_exec_detect_finetune_dataset.py with --format darknet or --format all first."
        )
    
    # Darknet binary path
    darknet_bin = "/polyis/modules/darknet/darknet"
    if not os.path.exists(darknet_bin):
        raise FileNotFoundError(
            f"Darknet binary not found: {darknet_bin}\n"
            f"Please build Darknet first."
        )
    
    # Config file - use provided or default
    if args.darknet_cfg:
        cfg_file = Path(args.darknet_cfg)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
    else:
        # Use default YOLOv3 config from Darknet
        cfg_file = Path("/polyis/modules/darknet/cfg/yolov3.cfg")
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"Default YOLOv3 config not found: {cfg_file}\n"
                f"Please provide a config file with --darknet-cfg"
            )
    
    # Pretrained weights - use provided or default
    if args.darknet_weights:
        weights_file = Path(args.darknet_weights)
        if not weights_file.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_file}")
        pretrained_weights = str(weights_file)
    else:
        # Use darknet53 conv weights for transfer learning (common practice)
        darknet53_weights = Path("/polyis/modules/darknet/darknet53.conv.74")
        if darknet53_weights.exists():
            pretrained_weights = str(darknet53_weights)
        else:
            # No pretrained weights - train from scratch
            pretrained_weights = ""
            print("Warning: No pretrained weights found. Training from scratch.")
    
    # Set GPU device if specified
    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    # Build Darknet training command
    cmd = [
        darknet_bin,
        "detector",
        "train",
        str(data_file),
        str(cfg_file),
    ]
    
    # Add pretrained weights if available
    if pretrained_weights:
        cmd.append(pretrained_weights)
    
    # Add flags
    cmd.extend([
        "-dont_show",  # Don't show visualization window
        "-map",        # Calculate mAP during training
    ])
    
    # Print configuration
    print(f"\nTraining configuration:")
    print(f"  Darknet binary: {darknet_bin}")
    print(f"  Data file: {data_file}")
    print(f"  Config file: {cfg_file}")
    print(f"  Pretrained weights: {pretrained_weights or 'None (training from scratch)'}")
    print(f"  Output dir: {weights_dir}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Update obj.data to save weights to our directory
    # Read current obj.data
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # Update backup path
    updated_lines = []
    for line in lines:
        if line.strip().startswith('backup'):
            updated_lines.append(f"backup = {weights_dir}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(data_file, 'w') as f:
        f.writelines(updated_lines)
    
    # Start training
    print(f"\nStarting Darknet training...")
    print(f"Note: Training will save weights to {weights_dir}")
    print(f"Note: Best weights will be saved as {cfg_file.stem}_best.weights")
    
    # Run Darknet training
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(darknet_dir),
    )
    
    if result.returncode != 0:
        print(f"\nWarning: Darknet training exited with code {result.returncode}")
    else:
        print(f"\nTraining complete!")
        print(f"Weights saved to: {weights_dir}")


def main():
    """Main function to train detection models."""
    args = parse_args()
    
    # Derive data directory path
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(f"/polyis-data/training/finetune/{args.dataset}_{args.tilesize}_{args.tilepadding}")
    
    print(f"Dataset: {args.dataset}")
    print(f"Tile size: {args.tilesize}")
    print(f"Tile padding: {args.tilepadding}")
    print(f"Model: {args.model}")
    print(f"Data directory: {data_dir}")
    
    # Verify data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run p041_exec_detect_finetune_dataset.py first to create the dataset."
        )
    
    # Train based on model type
    if args.model in ['5x6', '11x']:
        train_ultralytics(args, data_dir)
    elif args.model == 'detectron2':
        train_detectron2(args, data_dir)
    elif args.model == 'darknet':
        train_darknet(args, data_dir)


if __name__ == '__main__':
    main()
