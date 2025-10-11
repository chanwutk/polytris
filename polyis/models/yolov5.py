import numpy as np
import ultralytics
# import torch

import polyis.dtypes


MODEL_PATH = "yolov5x6u.pt"
# ultralytics.YOLO(MODEL_PATH)


# COCO class indices: car=2, bus=5, truck=7
# Note: Vans are typically classified as cars (2) in COCO dataset
VEHICLE_CLASSES = [2, 5, 7]  # car (includes vans), bus, truck


def get_detector(device: str, model_path: str | None = None):
    """
    Get YOLOv5 detector model configured to detect only vehicles.

    Args:
        device: Device to run the model on ('cpu' or 'cuda')
        model_path: Path to the YOLOv5 model file

    Returns:
        torch.nn.Module: Configured YOLOv5 model for vehicle detection
    """
    # print(f"Loading YOLOv5 detector for dataset on {device}")
    model = ultralytics.YOLO(model_path or MODEL_PATH)
    # print(f"Loaded YOLOv5 detector for dataset on {device}")
    model.fuse()
    # print(f"Fused YOLOv5 detector for dataset on {device}")
    model.to(device)
    # print(f"Moved YOLOv5 detector for dataset to {device}")
    model.eval()
    # print(f"Evaluated YOLOv5 detector for dataset on {device}")
    return model


def detect(image: np.ndarray, model: ultralytics.YOLO) -> polyis.dtypes.DetArray:
    """
    Detect vehicles in an image using YOLOv5.

    Args:
        image: Input image as numpy array (H, W, C)
        model: YOLOv5 model instance

    Returns:
        np.ndarray: Detection results as array of shape (N, 5)
                    where each row is [x1, y1, x2, y2, confidence]
    """

    # Filter classes during prediction for better performance
    results = model(image, classes=VEHICLE_CLASSES, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        res = np.empty((0, 5))
        assert polyis.dtypes.is_det_array(res)
        return res

    # Get bounding boxes and confidence scores
    bboxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = results.boxes.conf.cpu().numpy()  # confidence scores

    # Create output array with shape (N, 5): [x1, y1, x2, y2, confidence]
    detections = np.zeros((len(bboxes), 5))
    detections[:, :4] = bboxes
    detections[:, 4] = confidences

    assert polyis.dtypes.is_det_array(detections)
    return detections


def detect_batch(
    images: list[polyis.dtypes.NPImage],
    model: ultralytics.YOLO
) -> list[polyis.dtypes.DetArray]:
    """
    Detect vehicles in a batch of images using YOLOv5.

    Args:
        images: Input images as numpy array (H, W, C)
        model: YOLOv5 model instance

    Returns:
        list[np.ndarray]: Detection results as array of shape (N, 5)
                    where each row is [x1, y1, x2, y2, confidence]
    """

    all_results = model(np.stack(images), classes=VEHICLE_CLASSES, verbose=False)
    all_detections: list[polyis.dtypes.DetArray] = []
    for results in all_results:
        if results.boxes is None or len(results.boxes) == 0:
            res = np.empty((0, 5))
            assert polyis.dtypes.is_det_array(res)
            all_detections.append(res)
            continue

        # Get bounding boxes and confidence scores
        bboxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = results.boxes.conf.cpu().numpy()  # confidence scores

        # Create output array with shape (N, 5): [x1, y1, x2, y2, confidence]
        detections = np.zeros((len(bboxes), 5))
        detections[:, :4] = bboxes
        detections[:, 4] = confidences

        assert polyis.dtypes.is_det_array(detections)
        all_detections.append(detections)

    return all_detections
