"""
Dataset-based detector selection utilities.

This module provides automatic detector selection based on dataset names,
allowing the system to automatically choose the appropriate detector
(RetinaNet or YOLOv3) for each dataset.
"""

from typing import Any

import polyis.models.retinanet_b3d
import polyis.models.yolov3_otif
import polyis.models.yolov5


DATASET_NAME_MAPPING = {
    "caldot1": "caldot",
    "caldot2": "caldot",
    "caldot1-yolov5": "caldot-yolov5",
    "caldot2-yolov5": "caldot-yolov5",
    "b3d-jnc00": "b3d",
    "b3d-jnc02": "b3d",
    "b3d-jnc06": "b3d",
    "b3d-jnc07": "b3d",
}


# Dataset-to-detector mapping configuration
DATASET_DETECTOR_CONFIG = {
    "dataset_detector_mapping": {
        "b3d": {
            "detector": "retina",
            "model_path": None,
            "config_path": None
        },
        "caldot": {
            "detector": "yolov3",
            "detector_label": "caldot",
            "width": 704,
            "height": 480,
            "model_path": "/otif-dataset/yolov3/caldot/yolov3-704x480.best",
            "config_path": "/otif-dataset/yolov3/caldot/yolov3-704x480-test.cfg"
        },
        "caldot-yolov5": {
            "detector": "yolov5",
            "detector_label": "caldot-yolov5",
        },
    },
    "default_detector": "retina",
    "fallback_config": {
        "yolov3": {
            "threshold": 0.25,
            "nms_threshold": 0.45,
            "batch_size": 1
        }
    }
}


def get_detector(dataset_name: str, gpu_id):
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
    config = DATASET_DETECTOR_CONFIG
    
    # Get dataset configuration
    dataset_config = config.get('dataset_detector_mapping', {}).get(DATASET_NAME_MAPPING[dataset_name])
    if dataset_config is None:
        # Fall back to default detector
        default_detector = config.get('default_detector', 'retina')
        print(f"Dataset '{dataset_name}' not found in configuration, using default detector: {default_detector}")
        dataset_config = {'detector': default_detector}
    
    detector_type = dataset_config['detector']
    device = f'cuda:{gpu_id}'
    
    if detector_type == 'retina':
        print(f"Loading RetinaNet detector for dataset '{dataset_name}' on {device}")
        return polyis.models.retinanet_b3d.get_detector(device=device)
    
    elif detector_type == 'yolov3':
        print(f"Loading YOLOv3 detector for dataset '{dataset_name}' on {device}")
        
        # Extract YOLOv3-specific parameters
        detector_label = dataset_config.get('detector_label', 'caldot')
        width = dataset_config.get('width', 704)
        height = dataset_config.get('height', 480)
        model_path = dataset_config.get('model_path')
        config_path_yolo = dataset_config.get('config_path')
        
        # Get fallback configuration
        fallback_config = config.get('fallback_config', {}).get('yolov3', {})
        threshold = fallback_config.get('threshold', 0.25)
        nms_threshold = fallback_config.get('nms_threshold', 0.45)
        
        return polyis.models.yolov3_otif.get_detector(
            gpu_id=gpu_id,
            config_path=config_path_yolo,
            model_path=model_path,
            detector_label=detector_label,
            width=width,
            height=height,
            threshold=threshold,
            nms_threshold=nms_threshold
        )
    
    elif detector_type == 'yolov5':
        print(f"Loading YOLOv5 detector for dataset '{dataset_name}' on {device}")
        return polyis.models.yolov5.get_detector(device=device)
    
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
        return polyis.models.yolov3_otif.detect(image, detector, threshold)
    elif isinstance(detector, polyis.models.retinanet_b3d.DefaultPredictor):
        return polyis.models.retinanet_b3d.detect(image, detector, threshold)
    else:
        return polyis.models.yolov5.detect(image, detector)


def get_detector_info(dataset_name: str) -> dict:
    """
    Get information about the detector configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing detector information
    """
    config = DATASET_DETECTOR_CONFIG
    dataset_config = config.get('dataset_detector_mapping', {}).get(DATASET_NAME_MAPPING[dataset_name])
    
    if dataset_config is None:
        default_detector = config.get('default_detector', 'retina')
        return {
            'detector': default_detector,
            'is_default': True,
            'description': f"Using default detector '{default_detector}' for unknown dataset"
        }
    
    return {
        'detector': dataset_config['detector'],
        'is_default': False,
        'description': f"Using {dataset_config['detector']} detector for dataset '{dataset_name}'",
        **dataset_config
    }
