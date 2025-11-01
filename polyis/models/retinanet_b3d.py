import json
import os
import typing

import numpy as np
import torch
from torch.nn.functional import interpolate

import polyis.dtypes

if typing.TYPE_CHECKING:
    from detectron2.modeling.meta_arch.retinanet import RetinaNet


from polyis.b3d import nms
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


MODULES = '/polyis/modules'
POLYIS = '/polyis'
CONFIG = os.path.join(POLYIS, 'configs/retinanet.json')
DETECTRON_CONFIG_DIR = os.path.join(POLYIS, 'configs/detectron2')


def get_detector(device: str, configfile: str | None = None) -> "DefaultPredictor":
    with open(configfile or CONFIG) as fp:
        config = json.load(fp)

    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(DETECTRON_CONFIG_DIR, config['config']))
    cfg.MODEL.WEIGHTS = config['weights']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
    cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
    cfg.MODEL.DEVICE = device

    return DefaultPredictor(cfg)


def detect(
    image: np.ndarray,
    detector: "DefaultPredictor",
    nms_threshold: float = 0.5
) -> polyis.dtypes.DetArray:
    """
    Detect vehicles in an image using RetinaNet.
    
    Args:
        image: Input image as numpy array (H, W, C)
        detector: RetinaNet detector instance
        nms_threshold: NMS threshold
        
    Returns:
        np.ndarray: Detection results as array of shape (N, 5)
                    where each row is [x1, y1, x2, y2, confidence]
    """
    detections = detect_batch([image], detector, nms_threshold)[0]
    return detections


def detect_batch(
    images: list[np.ndarray],
    detector: "DefaultPredictor",
    nms_threshold: float = 0.5
) -> list[polyis.dtypes.DetArray]:
    """
    Detect vehicles in a batch of images using RetinaNet.
    """
    model: "RetinaNet" = detector.model
    image = images[0]
    height, width = image.shape[:2]
    transform = detector.aug.get_transform(image)
    new_h: int = transform.new_h  # type: ignore
    new_w: int = transform.new_w  # type: ignore
    images_stack = torch.from_numpy(np.stack(images)).to(device=detector.cfg.MODEL.DEVICE)

    assert str(model.pixel_mean.device) == str(detector.cfg.MODEL.DEVICE), \
        f"Model pixel mean device {model.pixel_mean.device} does " \
        f"not match detector device {detector.cfg.MODEL.DEVICE}"
    
    all_detections = []

    with torch.no_grad():
        # Apply pre-processing to image.
        if detector.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            images_stack = images_stack[:, :, :, ::-1]

        images_stack = images_stack.permute(0, 3, 1, 2)  # NCHW -> NCHW
        images_stack = images_stack.to(dtype=torch.float32)
        images_stack = interpolate(images_stack,
                                   size=(new_h, new_w),
                                   mode="bilinear",
                                   align_corners=False)

        inputs = [
            {"image": img, "height": height, "width": width}
            for img in images_stack
        ]
        predictions = model(inputs)

        for outputs in predictions:
            instances = outputs['instances'].to('cpu')
            bboxes = instances.pred_boxes.tensor
            scores = instances.scores

            nms_bboxes, nms_scores = nms.nms(bboxes, scores, nms_threshold)
            detections = np.zeros((len(nms_bboxes), 5))
            assert polyis.dtypes.is_det_array(detections)
            if len(nms_bboxes) > 0:
                detections[:, 0:4] = nms_bboxes
                detections[:, 4] = nms_scores
            all_detections.append(detections)

    return all_detections