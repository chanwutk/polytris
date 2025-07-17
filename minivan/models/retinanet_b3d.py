import json
import os
import sys

import numpy as np

sys.path.append('/minivan/modules/detectron2')
sys.path.append('/minivan/modules/b3d')

from b3d.external import nms
from b3d.utils import parse_outputs
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


MODULES = '/minivan/modules'
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


def detect(image, detector, nms_threshold: float = 0.5):
    _outputs = detector(image)
    bboxes, scores, _ = parse_outputs(_outputs, (0, 0))
    nms_bboxes, nms_scores = nms.nms(bboxes, scores, nms_threshold)
    detections = np.zeros((len(nms_bboxes), 5))
    if len(nms_bboxes) == 0:
        return detections

    detections[:, 0:4] = nms_bboxes
    detections[:, 4] = nms_scores

    return detections
