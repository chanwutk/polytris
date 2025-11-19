import numpy as np
import torch
import torchvision.models
from torchvision.models import ResNet50_Weights

import polyis.dtypes


# COCO class indices: car=2, motorcycle=3, bus=5, truck=7
# Note: Vans are typically classified as cars (2) in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car (includes vans), motorcycle, bus, truck


def get_detector(device: str, model_path: str, model_type: str = "fasterrcnn"):
    """
    Get torchvision detection model from retrained checkpoint.

    Args:
        device: Device to run the model on ('cpu' or 'cuda')
        model_path: Path to the checkpoint file.
        model_type: Type of model, either 'fasterrcnn' or 'retinanet' (default: 'fasterrcnn')

    Returns:
        torch.nn.Module: Configured detection model for vehicle detection
    """
    # Map model type to torchvision model name
    model_name_map = {
        "fasterrcnn": "fasterrcnn_resnet50_fpn_v2",
        "retinanet": "retinanet_resnet50_fpn_v2",
    }
    
    if model_type not in model_name_map:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(model_name_map.keys())}")
    
    model_name = model_name_map[model_type]
    
    # Load trained model from checkpoint
    # Note: num_classes=2 matches training config (background + vehicle)
    # Use get_model to match training script configuration
    model = torchvision.models.get_model(
        model_name,
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        num_classes=2,
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()
    return model


def detect(image: np.ndarray, model: torch.nn.Module) -> polyis.dtypes.DetArray:
    """
    Detect vehicles in an image using torchvision detection model.

    Args:
        image: Input image as numpy array (H, W, C) in BGR format
        model: Detection model instance (Faster R-CNN or RetinaNet)

    Returns:
        np.ndarray: Detection results as array of shape (N, 5)
                    where each row is [x1, y1, x2, y2, confidence]
    """
    # Get device first
    device = next(model.parameters()).device
    
    # Convert to tensor and move to device first (transfer smaller uint8)
    tensor = torch.from_numpy(image).to(device)
    # Convert BGR to RGB on GPU (reverse channels), then permute, convert to float, and normalize
    tensor = torch.flip(tensor, dims=[2]).permute(2, 0, 1).float() / 255.0

    # Run inference
    with torch.no_grad():
        outputs = model([tensor])[0]

    # Extract boxes and scores
    boxes = outputs["boxes"].detach().cpu().numpy()  # [x1, y1, x2, y2]
    scores = outputs["scores"].detach().cpu().numpy()  # confidence scores

    if len(boxes) == 0:
        res = np.empty((0, 5))
        assert polyis.dtypes.is_det_array(res)
        return res

    # Create output array with shape (N, 5): [x1, y1, x2, y2, confidence]
    detections = np.zeros((len(boxes), 5))
    detections[:, :4] = boxes
    detections[:, 4] = scores

    assert polyis.dtypes.is_det_array(detections)
    return detections


def detect_batch(
    images: list[polyis.dtypes.NPImage],
    model: torch.nn.Module,
) -> list[polyis.dtypes.DetArray]:
    """
    Detect vehicles in a batch of images using torchvision detection model.

    Args:
        images: Input images as list of numpy arrays (H, W, C) in BGR format
        model: Detection model instance (Faster R-CNN or RetinaNet)

    Returns:
        list[np.ndarray]: Detection results as list of arrays of shape (N, 5)
                    where each row is [x1, y1, x2, y2, confidence]
    """
    # Get device first to avoid repeated calls
    device = next(model.parameters()).device
    
    # Convert BGR to RGB and prepare tensors
    tensors = []
    for image in images:
        # Convert to tensor and move to device first (transfer smaller uint8)
        tensor = torch.from_numpy(image).to(device)
        # Convert BGR to RGB on GPU (reverse channels), then permute, convert to float, and normalize
        tensor = torch.flip(tensor, dims=[2]).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    # Run batch inference
    outputs = model(tensors)

    # Process each output
    all_detections: list[polyis.dtypes.DetArray] = []
    for output in outputs:
        # Extract boxes and scores
        boxes = output["boxes"].detach().cpu().numpy()  # [x1, y1, x2, y2]
        scores = output["scores"].detach().cpu().numpy()  # confidence scores

        if len(boxes) == 0:
            res = np.empty((0, 5))
            assert polyis.dtypes.is_det_array(res)
            all_detections.append(res)
            continue

        # Create output array with shape (N, 5): [x1, y1, x2, y2, confidence]
        detections = np.zeros((len(boxes), 5))
        detections[:, :4] = boxes
        detections[:, 4] = scores

        assert polyis.dtypes.is_det_array(detections)
        all_detections.append(detections)

    return all_detections

