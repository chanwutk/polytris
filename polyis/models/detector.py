"""
Dataset-based detector selection utilities.

This module provides automatic detector selection based on dataset names,
allowing the system to automatically choose the appropriate detector
(RetinaNet or YOLOv3) for each dataset.
"""

from typing import Any
import ultralytics

import polyis.models.retinanet_b3d
import polyis.models.yolov3
import polyis.models.ultralytics
import polyis.models.torch_vision
import polyis.dtypes
from polyis.utilities import get_config


CONFIG = get_config('detectors.yaml')
DATASET_NAME_MAPPING = CONFIG['dataset_name_mapping']
DATASET_DETECTOR_CONFIG = CONFIG['dataset_detector_mapping']


def get_detector(dataset_name: str, gpu_id, batch_size: int = 16, num_images: int = 0):
    """
    Get the appropriate detector for a given dataset name.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'b3d', 'caldot', 'shibuya')
        gpu_id: GPU device ID to use (optional)
        
    Returns:
        Loaded detector instance (RetinaNet or YOLOv3)
        
    Raises:
        ValueError: If dataset is not found in configuration
        FileNotFoundError: If required model files are not found
    """
    # Get dataset configuration
    dataset_config = DATASET_DETECTOR_CONFIG[DATASET_NAME_MAPPING[dataset_name]]
    detector_type = dataset_config['detector']
    device = f'cuda:{gpu_id}'
    
    if detector_type == 'retina':
        return polyis.models.retinanet_b3d.get_detector(device=device)
    
    elif detector_type == 'yolov3':
        return polyis.models.yolov3.get_detector(
            gpu_id=gpu_id,
            config_path=dataset_config['config_path'],
            model_path=dataset_config['model_path'],
            detector_label=dataset_config['detector_label'],
            width=dataset_config['width'],
            height=dataset_config['height'],
            threshold=0.25,
            nms_threshold=0.45,
            batch_size=batch_size,
            num_images=num_images,
        )
    
    elif detector_type == 'ultralytics':
        return polyis.models.ultralytics.get_detector(device=device, model_path=dataset_config['model_path'])
    
    elif detector_type == 'torchvision':
        return polyis.models.torch_vision.get_detector(device=device, model_path=dataset_config['model_path'], model_type=dataset_config['model_type'])
    
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def detect(image, detector: Any, threshold: float = 0.25):
    """
    Generic object detection function that works with any pre-loaded detector.
    
    Args:
        image: Input image as numpy array
        detector: Pre-loaded detector instance (RetinaNet, YOLOv3, or YOLOv5)
        threshold: Detection confidence threshold
        
    Returns:
        np.ndarray: Detection results as array of shape (N, 5) where each row is [x1, y1, x2, y2, confidence]
                   Only returns detections for car, truck, and van classes
    """
    # Use the appropriate detect function based on detector type
    if hasattr(detector, 'net'):  # YOLOv3 detector
        return polyis.models.yolov3.detect(image, detector, threshold)
    elif isinstance(detector, polyis.models.retinanet_b3d.DefaultPredictor):
        return polyis.models.retinanet_b3d.detect(image, detector, threshold)
    
    else:
        return polyis.models.ultralytics.detect(image, detector)


def detect_batch(
    images: list[polyis.dtypes.NPImage],
    detector: Any,
    threshold: float = 0.25
) -> "list[polyis.dtypes.DetArray]":
    """
    Detect vehicles in a batch of images using the appropriate detector.
    """
    if hasattr(detector, 'net'):  # YOLOv3 detector
        return polyis.models.yolov3.detect_batch(images, detector, threshold)
    elif isinstance(detector, polyis.models.retinanet_b3d.DefaultPredictor):
        return polyis.models.retinanet_b3d.detect_batch(images, detector, threshold)
    elif isinstance(detector, ultralytics.YOLO):  # type: ignore
        return polyis.models.ultralytics.detect_batch(images, detector)
    else:
        return polyis.models.torch_vision.detect_batch(images, detector)


def get_detector_info(dataset_name: str) -> dict:
    """
    Get information about the detector configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing detector information
    """
    dataset_config = DATASET_DETECTOR_CONFIG[DATASET_NAME_MAPPING[dataset_name]]
    
    return {
        'detector': dataset_config['detector'],
        'description': f"Using {dataset_config['detector']} detector for dataset '{dataset_name}'",
        **dataset_config
    }
