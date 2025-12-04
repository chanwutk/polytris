#!/usr/local/bin/python
"""
Train object detection models on fine-tuning datasets created by p041.

This script trains a single dataset/tilesize/tilepadding combination.
Model types, image sizes, and pretrained weights are determined
automatically from dataset configurations.

Supported model types:
- Ultralytics YOLO: yolov5x6, yolo11x
- Detectron2: Faster R-CNN, RetinaNet
- Darknet: YOLOv3
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2

from polyis.models.detector import get_detector_info
from polyis.utilities import get_config


CONFIG = get_config()
CACHE_DIR = Path(CONFIG['DATA']['CACHE_DIR'])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train detector models on fine-tuning datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (required)')
    parser.add_argument('--tilesize', type=int, required=True,
                        help='Tile size (required)')
    parser.add_argument('--tilepadding', type=str, required=True,
                        help='Tile padding mode (required)')
    parser.add_argument('--epochs', type=int, default=13,
                        help='Training epochs (default: 13)')
    parser.add_argument('--batch', type=int, default=-1,
                        help='Batch size (-1 for auto, default: -1)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "0" or "0,1,2")')
    return parser.parse_args()


def get_model_type_from_dataset(dataset: str) -> tuple[str, dict]:
    """
    Determine model type and get detector configuration from dataset name.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Tuple of (model_type, detector_config)
        model_type: 'ultralytics', 'darknet', or 'detectron2'
        
    Raises:
        ValueError: If dataset is not found in detector configuration
    """
    try:
        detector_config = get_detector_info(dataset)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Dataset '{dataset}' not found in detector configuration: {e}")
    
    detector_type = detector_config['detector']
    
    # Map detector types to training model types
    if detector_type == 'ultralytics':
        return 'ultralytics', detector_config
    elif detector_type == 'yolov3':
        return 'darknet', detector_config
    elif detector_type in ['retina', 'torchvision']:
        return 'detectron2', detector_config
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def get_ultralytics_model_variant(model_path: str) -> str:
    """
    Determine Ultralytics model variant (5x6 or 11x) from model path.
    
    Args:
        model_path: Path to model weights file
        
    Returns:
        '5x6' or '11x'
    """
    model_path_lower = model_path.lower()
    if 'yolov5x6' in model_path_lower or 'yolo5' in model_path_lower:
        return '5x6'
    elif 'yolo11x' in model_path_lower or 'yolo11' in model_path_lower:
        return '11x'
    else:
        # Default to 5x6 if unclear
        return '5x6'


def get_image_size_from_dataset(data_dir: Path, format: str) -> int:
    """
    Detect image size from training dataset images.
    
    Args:
        data_dir: Path to dataset directory
        format: Dataset format ('ultralytics', 'darknet', or 'coco')
        
    Returns:
        Image size (assumes square images, returns width/height)
        
    Raises:
        FileNotFoundError: If no images found
        ValueError: If images have inconsistent sizes
    """
    image_paths = []
    
    if format == 'ultralytics':
        train_images_dir = data_dir / "ultralytics" / "images" / "train"
        if train_images_dir.exists():
            image_paths = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
    elif format == 'darknet':
        images_dir = data_dir / "darknet" / "images"
        if images_dir.exists():
            image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    elif format == 'coco':
        train_images_dir = data_dir / "coco" / "train2017"
        if train_images_dir.exists():
            image_paths = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir} for format {format}")
    
    # Read first image to get size
    img = cv2.imread(str(image_paths[0]))
    if img is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    
    height, width = img.shape[:2]
    
    # Verify a few more images have the same size
    for img_path in image_paths[1:min(5, len(image_paths))]:
        test_img = cv2.imread(str(img_path))
        if test_img is None:
            continue
        test_h, test_w = test_img.shape[:2]
        if test_h != height or test_w != width:
            raise ValueError(
                f"Inconsistent image sizes: first image {width}x{height}, "
                f"but {img_path.name} is {test_w}x{test_h}"
            )
    
    # Return the larger dimension (assuming square or near-square)
    return max(width, height)


def load_detectron2_config(dataset: str, detector_config: dict) -> dict:
    """
    Load Detectron2 configuration for training.
    
    For RetinaNet (b3d dataset), loads from configs/retinanet.json.
    For other models, uses detector config or defaults.
    
    Args:
        dataset: Dataset name
        detector_config: Detector configuration dict
        
    Returns:
        Dict with config file path, weights path, and training parameters
    """
    detector_type = detector_config['detector']
    
    # For RetinaNet (b3d), load from JSON config
    if detector_type == 'retina':
        retinanet_config_path = Path("/polyis/configs/retinanet.json")
        if not retinanet_config_path.exists():
            raise FileNotFoundError(f"RetinaNet config not found: {retinanet_config_path}")
        
        with open(retinanet_config_path) as f:
            retinanet_config = json.load(f)
        
        return {
            'config_file': retinanet_config['config'],
            'weights': retinanet_config['weights'],
            'num_classes': retinanet_config['num_classes'],
            'base_lr': retinanet_config.get('base_lr', 1e-4),
            'max_iter': retinanet_config.get('max_iter', 100000),
            'ims_per_batch': retinanet_config.get('ims_per_batch', 2),
        }
    else:
        # For other Detectron2 models (torchvision), use defaults
        return {
            'config_file': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'weights': None,  # Will use model_zoo checkpoint
            'num_classes': 1,
            'base_lr': 0.00025,
            'max_iter': None,  # Will be set from epochs
            'ims_per_batch': 2,
        }


def train_ultralytics(
    dataset: str,
    tilesize: int,
    tilepadding: str,
    data_dir: Path,
    detector_config: dict,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | None
):
    """
    Train using Ultralytics YOLO.
    
    Args:
        dataset: Dataset name
        tilesize: Tile size
        tilepadding: Tile padding mode
        data_dir: Path to the dataset directory
        detector_config: Detector configuration dict
        imgsz: Image size for training
        epochs: Number of training epochs
        batch: Batch size (-1 for auto)
        device: Device string (e.g., "0")
    """
    import ultralytics
    from polyis.train.data.ultralytics import verify_dataset, parse_device_string
    
    # Paths
    ultralytics_dir = data_dir / "ultralytics"
    data_yaml = ultralytics_dir / "data.yaml"
    
    # Determine model variant from detector config
    model_path = detector_config['model_path']
    model_variant = get_ultralytics_model_variant(model_path)
    weights_dir = data_dir / "weights" / f"yolov{model_variant}"
    
    # Verify dataset exists
    print(f"[{dataset}/{tilesize}/{tilepadding}] Verifying dataset at {ultralytics_dir}")
    verify_dataset(str(ultralytics_dir))
    
    # Load model with pretrained weights from detector config
    print(f"[{dataset}/{tilesize}/{tilepadding}] Loading model: {model_path}")
    model = ultralytics.YOLO(model_path)  # type: ignore
    
    # Training configuration
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "project": str(weights_dir.parent),
        "name": weights_dir.name,
        "workers": 0,
        "seed": 42,
        "exist_ok": True,
        "plots": True,
    }
    
    # Set batch size if specified
    if batch > 0:
        train_kwargs["batch"] = batch
    
    # Set device if specified
    parsed_device = parse_device_string(device)
    if parsed_device is not None:
        train_kwargs["device"] = parsed_device
    
    # Print configuration
    print(f"[{dataset}/{tilesize}/{tilepadding}] Training configuration:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")
    
    # Start training
    print(f"[{dataset}/{tilesize}/{tilepadding}] Starting training...")
    model.train(**train_kwargs)
    
    print(f"[{dataset}/{tilesize}/{tilepadding}] Training complete!")
    print(f"[{dataset}/{tilesize}/{tilepadding}] Weights saved to: {weights_dir}")


def train_detectron2(
    dataset: str,
    tilesize: int,
    tilepadding: str,
    data_dir: Path,
    detector_config: dict,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | None
):
    """
    Train using Detectron2.
    
    Args:
        dataset: Dataset name
        tilesize: Tile size
        tilepadding: Tile padding mode
        data_dir: Path to the dataset directory
        detector_config: Detector configuration dict
        imgsz: Image size for training (not used for Detectron2, but kept for consistency)
        epochs: Number of training epochs
        batch: Batch size (-1 for auto)
        device: Device string (e.g., "0")
        gpu_id: GPU ID for this worker
        command_queue: Progress queue for reporting
    """
    from detectron2 import model_zoo  # type: ignore
    from detectron2.config import get_cfg  # type: ignore
    from detectron2.data.datasets import register_coco_instances  # type: ignore
    from detectron2.engine import DefaultTrainer  # type: ignore
    
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
    dataset_name = f"finetune_{dataset}_{tilesize}_{tilepadding}"
    
    # Register datasets
    print(f"[{dataset}/{tilesize}/{tilepadding}] Registering COCO datasets...")
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
    
    # Load Detectron2 config
    d2_config = load_detectron2_config(dataset, detector_config)
    
    # Configure Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(d2_config['config_file'])
    )
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    
    # Model configuration
    if d2_config['weights']:
        # Use weights from config (for RetinaNet)
        cfg.MODEL.WEIGHTS = d2_config['weights']
    else:
        # Use model zoo checkpoint (for other models)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_config['config_file'])
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = d2_config['num_classes']
    
    # Solver configuration
    if d2_config['max_iter']:
        cfg.SOLVER.MAX_ITER = d2_config['max_iter']
    else:
        cfg.SOLVER.MAX_ITER = epochs * 1000  # Rough conversion from epochs
    
    cfg.SOLVER.IMS_PER_BATCH = batch if batch > 0 else d2_config['ims_per_batch']
    cfg.SOLVER.BASE_LR = d2_config['base_lr']
    cfg.SOLVER.STEPS = []  # No LR decay
    
    # Output configuration
    cfg.OUTPUT_DIR = str(weights_dir)
    
    # Device configuration
    if device is not None:
        cfg.MODEL.DEVICE = f"cuda:{device}"
    
    # Print configuration
    print(f"[{dataset}/{tilesize}/{tilepadding}] Training configuration:")
    print(f"  Model: {d2_config['config_file']}")
    print(f"  Train dataset: {dataset_name}_train")
    print(f"  Val dataset: {dataset_name}_val")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Output dir: {cfg.OUTPUT_DIR}")
    
    # Start training
    print(f"[{dataset}/{tilesize}/{tilepadding}] Starting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"[{dataset}/{tilesize}/{tilepadding}] Training complete!")
    print(f"[{dataset}/{tilesize}/{tilepadding}] Weights saved to: {weights_dir}")


def train_darknet(
    dataset: str,
    tilesize: int,
    tilepadding: str,
    data_dir: Path,
    detector_config: dict,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | None
):
    """
    Train using Darknet YOLOv3.
    
    Args:
        dataset: Dataset name
        tilesize: Tile size
        tilepadding: Tile padding mode
        data_dir: Path to the dataset directory
        detector_config: Detector configuration dict (must contain config_path and model_path)
        imgsz: Image size for training (not used for Darknet, but kept for consistency)
        epochs: Number of training epochs (not used for Darknet, but kept for consistency)
        batch: Batch size (not used for Darknet, but kept for consistency)
        device: Device string (e.g., "0")
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
    
    # Get config and weights from detector config
    if 'config_path' not in detector_config:
        raise ValueError(f"config_path not found in detector config for dataset {dataset}")
    if 'model_path' not in detector_config:
        raise ValueError(f"model_path not found in detector config for dataset {dataset}")
    
    cfg_file = Path(detector_config['config_path'])
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    
    # Read and validate Darknet config file, create temporary copy with corrected batch/subdivisions/width/height if needed
    batch_size = None
    subdivisions = None
    config_width = None
    config_height = None
    config_random = None  # Track random resize setting
    config_lines = []
    
    # Get actual image dimensions from dataset
    images_dir = darknet_dir / "images"
    actual_width = None
    actual_height = None
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if image_files:
            # Read first image to get dimensions
            test_img = cv2.imread(str(image_files[0]))
            if test_img is not None:
                actual_height, actual_width = test_img.shape[:2]
                print(f"[{dataset}/{tilesize}/{tilepadding}] Detected image dimensions: {actual_width}x{actual_height}")
    
    with open(cfg_file, 'r') as f:
        for line in f:
            original_line = line
            # Remove comments and whitespace for parsing
            line_stripped = line.strip()
            # Skip empty lines and comments for parsing
            if line_stripped and not line_stripped.startswith('#'):
                # Handle lines with comments at the end
                line_parse = line_stripped
                if '#' in line_parse:
                    line_parse = line_parse.split('#')[0].strip()
                # Parse batch, subdivisions, width, and height
                if '=' in line_parse:
                    key, value = line_parse.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'batch':
                        try:
                            batch_size = int(value)
                        except ValueError:
                            pass
                    elif key == 'subdivisions':
                        try:
                            subdivisions = int(value)
                        except ValueError:
                            pass
                    elif key == 'width':
                        try:
                            config_width = int(value)
                        except ValueError:
                            pass
                    elif key == 'height':
                        try:
                            config_height = int(value)
                        except ValueError:
                            pass
                    elif key == 'random':
                        try:
                            config_random = float(value)
                        except ValueError:
                            pass
            config_lines.append(original_line)
    
    if batch_size is None or subdivisions is None:
        raise ValueError(
            f"Config file {cfg_file} is missing 'batch' or 'subdivisions' setting. "
            f"Found batch={batch_size}, subdivisions={subdivisions}"
        )
    
    # In Darknet: mini_batch = batch / subdivisions
    # The mini_batch is what gets processed at once (memory usage)
    # The total batch size for gradient update is 'batch'
    mini_batch_size = batch_size // subdivisions if subdivisions > 0 else batch_size
    needs_fix = False
    fixed_batch = batch_size
    fixed_subdivisions = subdivisions
    fixed_width = config_width
    fixed_height = config_height
    fixed_random = config_random  # Disable random resize to prevent CUDA errors
    temp_cfg_path = None
    
    # Check if batch/subdivisions need fixing
    # In Darknet: mini_batch = batch / subdivisions (this is what uses GPU memory)
    if mini_batch_size == 1 or mini_batch_size == 0:
        # Fix to reasonable values with subdivisions to avoid memory issues
        # Use batch=64 with subdivisions=8: mini_batch = 64/8 = 8 images at once
        fixed_batch = 64
        fixed_subdivisions = 8  # mini_batch = 64/8 = 8, reducing memory usage
        needs_fix = True
        print(
            f"Warning: Darknet config has invalid mini_batch={mini_batch_size} (batch={batch_size} / subdivisions={subdivisions}). "
            f"Creating temporary config with batch={fixed_batch}, subdivisions={fixed_subdivisions} (mini_batch={fixed_batch // fixed_subdivisions})."
        )
    elif mini_batch_size > 16:
        # If mini_batch is too large, increase subdivisions to reduce memory usage
        # Try to keep mini_batch around 8-16
        if mini_batch_size <= 32:
            fixed_batch = batch_size
            fixed_subdivisions = max(2, batch_size // 16)  # Aim for mini_batch ~16
        elif mini_batch_size <= 64:
            fixed_batch = batch_size
            fixed_subdivisions = max(4, batch_size // 16)  # Aim for mini_batch ~16
        else:
            # For very large mini_batch, cap at reasonable size
            fixed_batch = 64
            fixed_subdivisions = 8  # mini_batch = 8
        needs_fix = True
        print(
            f"Warning: Darknet config has large mini_batch={mini_batch_size} (batch={batch_size} / subdivisions={subdivisions}), "
            f"which may cause memory issues. Creating temporary config with batch={fixed_batch}, subdivisions={fixed_subdivisions} (mini_batch={fixed_batch // fixed_subdivisions})."
        )
    elif mini_batch_size < 2:
        # Fix to minimum acceptable values
        fixed_batch = 16
        fixed_subdivisions = 2  # mini_batch = 16/2 = 8
        needs_fix = True
        print(
            f"Warning: Darknet config has small mini_batch={mini_batch_size} (batch={batch_size} / subdivisions={subdivisions}). "
            f"Creating temporary config with batch={fixed_batch}, subdivisions={fixed_subdivisions} (mini_batch={fixed_batch // fixed_subdivisions})."
        )
    
    # Check if width/height need updating to match actual image dimensions
    if actual_width is not None and actual_height is not None:
        # Round to nearest multiple of 32 (Darknet requirement)
        def round_to_multiple(value: int, multiple: int = 32) -> int:
            return int(round(value / multiple) * multiple)
        
        rounded_width = round_to_multiple(actual_width)
        rounded_height = round_to_multiple(actual_height)
        
        # Update if config dimensions don't match or are missing
        if config_width is None or config_height is None:
            needs_fix = True
            fixed_width = rounded_width
            fixed_height = rounded_height
            print(
                f"Warning: Darknet config missing width/height settings. "
                f"Setting to width={fixed_width}, height={fixed_height} based on image dimensions {actual_width}x{actual_height} (rounded to multiple of 32)."
            )
        elif config_width != rounded_width or config_height != rounded_height:
            needs_fix = True
            fixed_width = rounded_width
            fixed_height = rounded_height
            print(
                f"Warning: Darknet config has width={config_width}, height={config_height}, "
                f"but images are {actual_width}x{actual_height}. "
                f"Updating config to width={fixed_width}, height={fixed_height} (rounded to multiple of 32)."
            )
    
    # Disable random resize to prevent CUDA errors from large random dimensions
    # Random resize can cause Darknet to try dimensions like 1024x704 which may exceed GPU memory
    if config_random is not None and config_random != 0:
        needs_fix = True
        fixed_random = 0
        print(
            f"Warning: Disabling random resize (was {config_random}) to prevent CUDA errors. "
            f"Training will use fixed dimensions {fixed_width}x{fixed_height}."
        )
    
    # Create temporary config file if needed
    if needs_fix:
        import tempfile
        temp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        temp_cfg_path = Path(temp_cfg.name)
        
        for line in config_lines:
            line_stripped = line.strip()
            # Check if this line needs to be modified
            if line_stripped and not line_stripped.startswith('#'):
                line_parse = line_stripped
                if '#' in line_parse:
                    comment_part = '#' + line_parse.split('#', 1)[1]
                    line_parse = line_parse.split('#')[0].strip()
                else:
                    comment_part = ''
                
                if '=' in line_parse:
                    key, value = line_parse.split('=', 1)
                    key = key.strip()
                    if key == 'batch':
                        # Replace batch value
                        temp_cfg.write(f"batch={fixed_batch}{comment_part}\n")
                        continue
                    elif key == 'subdivisions':
                        # Replace subdivisions value
                        temp_cfg.write(f"subdivisions={fixed_subdivisions}{comment_part}\n")
                        continue
                    elif key == 'width' and fixed_width is not None:
                        # Replace width value
                        temp_cfg.write(f"width={fixed_width}{comment_part}\n")
                        continue
                    elif key == 'height' and fixed_height is not None:
                        # Replace height value
                        temp_cfg.write(f"height={fixed_height}{comment_part}\n")
                        continue
                    elif key == 'random' and fixed_random is not None:
                        # Replace random value (disable random resize)
                        temp_cfg.write(f"random={fixed_random}{comment_part}\n")
                        continue
            
            # Write original line
            temp_cfg.write(line)
        
        temp_cfg.close()
        cfg_file = temp_cfg_path
        batch_size = fixed_batch
        subdivisions = fixed_subdivisions
        mini_batch_size = batch_size // subdivisions
        print(f"Created temporary config file: {cfg_file}")
    
    weights_file = Path(detector_config['model_path'])
    if not weights_file.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_file}")
    
    pretrained_weights = str(weights_file)
    
    # Set GPU device
    # Set CUDA_VISIBLE_DEVICES to restrict visible GPUs, then use GPU 0 from Darknet's perspective
    env = os.environ.copy()
    
    # Check if CUDA_VISIBLE_DEVICES is already set externally
    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    
    if cuda_visible_devices is not None:
        # If CUDA_VISIBLE_DEVICES is set externally, always use GPU 0 (first visible GPU)
        gpu_index = 0
        print(f"[{dataset}/{tilesize}/{tilepadding}] CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected, using GPU index 0")
    elif device is not None:
        # Extract first GPU ID from device string (e.g., "0" from "0" or "0,1,2")
        device_id = device.split(',')[0].strip()
        try:
            device_num = int(device_id)
            # Set CUDA_VISIBLE_DEVICES to make only this GPU visible
            env["CUDA_VISIBLE_DEVICES"] = str(device_num)
            # After setting CUDA_VISIBLE_DEVICES, the first visible GPU becomes GPU 0
            gpu_index = 0
            print(f"[{dataset}/{tilesize}/{tilepadding}] Setting CUDA_VISIBLE_DEVICES={device_num}, using GPU index 0")
        except ValueError:
            print(f"[{dataset}/{tilesize}/{tilepadding}] Warning: Invalid device '{device}', using GPU 0")
            gpu_index = 0
    else:
        # If no device specified, use default GPU 0
        gpu_index = 0
    
    # Build Darknet training command
    cmd = [
        darknet_bin,
        "detector",
        "train",
        str(data_file),
        str(cfg_file),
        pretrained_weights,
    ]
    
    # Add flags
    cmd.extend([
        "-i", str(gpu_index),  # Specify GPU index directly
        "-dont_show",  # Don't show visualization window
        # Note: -map is disabled to avoid CUDA context issues during mAP network initialization
        # You can calculate mAP separately after training if needed
    ])
    
    # Print configuration
    print(f"[{dataset}/{tilesize}/{tilepadding}] Training configuration:")
    print(f"  Darknet binary: {darknet_bin}")
    print(f"  Data file: {data_file}")
    print(f"  Config file: {cfg_file}")
    mini_batch = batch_size // subdivisions if subdivisions > 0 else batch_size
    print(f"  Config batch: {batch_size}, subdivisions: {subdivisions}, mini_batch: {mini_batch}")
    print(f"  Pretrained weights: {pretrained_weights}")
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
    
    # Fix label file paths: Darknet resolves symlinks and looks for labels at the resolved path
    # Since images are symlinked to original locations, we need to create label files there
    images_dir = darknet_dir / "images"
    labels_dir = darknet_dir / "labels"
    
    if images_dir.exists() and labels_dir.exists():
        print(f"[{dataset}/{tilesize}/{tilepadding}] Creating label files at original image locations...")
        label_count = 0
        missing_labels = []
        
        for image_file in images_dir.glob("*.jpg"):
            if image_file.is_symlink():
                # Resolve symlink to get original image path
                original_image_path = image_file.resolve()
                # Get label name from symlink (in Darknet dataset labels directory)
                label_name_in_dataset = image_file.stem + ".txt"
                label_file = labels_dir / label_name_in_dataset
                
                if label_file.exists():
                    # Create label file next to original image (where symlink points)
                    # Use original image's stem (not the symlink's stem) for the label filename
                    # Darknet resolves symlinks and expects labels to match the original image name
                    original_label_name = original_image_path.stem + ".txt"
                    original_label_path = original_image_path.parent / original_label_name
                    
                    # Create parent directory if it doesn't exist
                    original_label_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy label file to original location with the correct name
                    if original_label_path.exists():
                        original_label_path.unlink()
                    import shutil
                    shutil.copy2(label_file, original_label_path)
                    label_count += 1
                else:
                    missing_labels.append(image_file.name)
        
        if missing_labels:
            print(f"[{dataset}/{tilesize}/{tilepadding}] Warning: {len(missing_labels)} images missing corresponding labels")
        
        print(f"[{dataset}/{tilesize}/{tilepadding}] Created {label_count} label files at original image locations")
    
    # Start training
    print(f"[{dataset}/{tilesize}/{tilepadding}] Starting Darknet training...")
    print(f"[{dataset}/{tilesize}/{tilepadding}] Note: Training will save weights to {weights_dir}")
    print(f"[{dataset}/{tilesize}/{tilepadding}] Note: Best weights will be saved as {cfg_file.stem}_best.weights")
    
    # Run Darknet training
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(darknet_dir),
    )
    
    if result.returncode != 0:
        print(f"[{dataset}/{tilesize}/{tilepadding}] Warning: Darknet training exited with code {result.returncode}")
    else:
        print(f"[{dataset}/{tilesize}/{tilepadding}] Training complete!")
        print(f"[{dataset}/{tilesize}/{tilepadding}] Weights saved to: {weights_dir}")
    
    # Clean up temporary config file if created
    if temp_cfg_path is not None and temp_cfg_path.exists():
        try:
            temp_cfg_path.unlink()
            print(f"[{dataset}/{tilesize}/{tilepadding}] Cleaned up temporary config file")
        except Exception as e:
            print(f"[{dataset}/{tilesize}/{tilepadding}] Warning: Failed to clean up temporary config file: {e}")




def main():
    """Main function to train detection models."""
    args = parse_args()
    
    # Derive data directory path
    data_dir = CACHE_DIR / args.dataset / "finetune" / f"{args.tilesize}_{args.tilepadding}"
    
    print(f"Dataset: {args.dataset}")
    print(f"Tile size: {args.tilesize}")
    print(f"Tile padding: {args.tilepadding}")
    print(f"Data directory: {data_dir}")
    
    # Verify data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run p041_exec_detect_finetune_dataset.py first to create the dataset."
        )
    
    # Get model type and detector config
    try:
        model_type, detector_config = get_model_type_from_dataset(args.dataset)
        print(f"Detected model type: {model_type}")
    except ValueError as e:
        raise ValueError(f"Could not determine model type for dataset '{args.dataset}': {e}")
    
    # Determine required format based on model type
    if model_type == 'ultralytics':
        required_format = 'ultralytics'
    elif model_type == 'darknet':
        required_format = 'darknet'
    else:  # detectron2
        required_format = 'coco'
    
    # Check if required format directory exists
    format_dir = data_dir / required_format
    if not format_dir.exists():
        raise FileNotFoundError(
            f"Required format directory not found: {format_dir}\n"
            f"Please run p041_exec_detect_finetune_dataset.py with --format {required_format} first."
        )
    
    # Get image size from dataset
    try:
        imgsz = get_image_size_from_dataset(data_dir, required_format)
        print(f"Detected image size: {imgsz}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not detect image size: {e}, using default 1280")
        imgsz = 1280
    
    # Train based on model type
    if model_type == 'ultralytics':
        train_ultralytics(
            args.dataset, args.tilesize, args.tilepadding, data_dir, detector_config,
            imgsz, args.epochs, args.batch, args.device
        )
    elif model_type == 'detectron2':
        train_detectron2(
            args.dataset, args.tilesize, args.tilepadding, data_dir, detector_config,
            imgsz, args.epochs, args.batch, args.device
        )
    elif model_type == 'darknet':
        train_darknet(
            args.dataset, args.tilesize, args.tilepadding, data_dir, detector_config,
            imgsz, args.epochs, args.batch, args.device
        )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
