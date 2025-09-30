import json
import os
import sys

import numpy as np

import polyis.dtypes

sys.path.append('/polyis/modules/detectron2')
sys.path.append('/polyis/modules/b3d')

from b3d.external import nms
from b3d.utils import parse_outputs
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


MODULES = '/polyis/modules'
CONFIG = os.path.join(MODULES, 'b3d/b3d/configs/config_refined.json')
DETECTRON_CONFIG_DIR = os.path.join(MODULES, 'detectron2/configs')


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
    _outputs = detector(image)
    bboxes, scores, _ = parse_outputs(_outputs, (0, 0))
    nms_bboxes, nms_scores = nms.nms(bboxes, scores, nms_threshold)
    detections = np.zeros((len(nms_bboxes), 5))
    assert polyis.dtypes.is_det_array(detections)
    if len(nms_bboxes) == 0:
        return detections

    detections[:, 0:4] = nms_bboxes
    detections[:, 4] = nms_scores

    return detections
