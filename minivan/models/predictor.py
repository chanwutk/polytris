import json
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# DETECTRON_CONFIG_DIR = './modules/detectron2/configs'
# CONFIG = './modules/b3d/configs/config_refined.json'
CONFIG = os.path.join('/data/chanwutk/projects/minivan/modules', 'b3d/b3d/configs/config_refined.json')
DETECTRON_CONFIG_DIR = os.path.join('/data/chanwutk/projects/minivan/modules', 'detectron2/configs')


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