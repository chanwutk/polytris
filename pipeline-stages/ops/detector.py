import sys, pathlib

MODULES_PATH = pathlib.Path().absolute().parent / 'modules'
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'b3d'))
sys.path.append(str(MODULES_PATH / 'detectron2'))

import json
import os

import numpy as np

from b3d.external.nms import nms
from b3d.utils import parse_outputs, regionize_image
from minivan.dtypes import S5, Array, DetArray, InPipe, NPImage, OutPipe
from minivan.models.predictor import get_detector


CONFIG = os.path.join('/data/chanwutk/projects/minivan/modules', 'b3d/b3d/configs/config_refined.json')
predictor = get_detector('cuda:0')


def detect(imgQueue: "InPipe[NPImage]", bboxQueue: "OutPipe[DetArray]", device: str = 'cuda:0'):
    with open(CONFIG) as fp:
        config = json.load(fp)

    flog = open('detector.py.log', 'w')
    count = 0

    while True:
        frame = imgQueue.get()
        if frame is None:
            break

        flog.write(f"Detecting {frame.shape}...\n")
        flog.flush()

        image_regions = regionize_image(frame)
        assert len(image_regions) == 1, len(image_regions)

        _image, _offset = image_regions[0]
        assert _offset == (0, 0), _offset
        # TODO: new predictor with GPU image transformation
        _outputs = predictor(_image)
        bboxes, scores, _ = parse_outputs(_outputs, _offset)

        nms_threshold = config['nms_threshold']
        nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
        detections = np.zeros((len(nms_bboxes), 5))
        if len(nms_bboxes) > 0:
            detections[:, 0:4] = nms_bboxes
            detections[:, 4] = nms_scores

        bboxQueue.put(detections)
        count += 1
    bboxQueue.put(None)

    flog.write("Detector finished.\n")
    flog.close()

    with open(f'detector_count.log', 'w') as f:
        f.write(f"Total frames detected: {count}\n")
        f.flush()


def detectIdx(imgQueue: "InPipe[tuple[int, NPImage]]", bboxQueue: "OutPipe[tuple[int, DetArray]]", device: str = 'cuda:0'):
    with open(CONFIG) as fp:
        config = json.load(fp)

    flog = open('detector.py.log', 'w')
    count = 0

    while True:
        frame = imgQueue.get()
        if frame is None:
            break
        idx, frame = frame

        flog.write(f"Detecting {frame.shape}...\n")
        flog.flush()

        image_regions = regionize_image(frame)
        assert len(image_regions) == 1, len(image_regions)

        _image, _offset = image_regions[0]
        assert _offset == (0, 0), _offset
        # TODO: new predictor with GPU image transformation
        _outputs = predictor(_image)
        bboxes, scores, _ = parse_outputs(_outputs, _offset)

        nms_threshold = config['nms_threshold']
        nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
        detections = np.zeros((len(nms_bboxes), 5))
        detections[:, 0:4] = nms_bboxes
        detections[:, 4] = nms_scores

        bboxQueue.put((idx, detections))
        count += 1
    bboxQueue.put(None)

    flog.write("Detector finished.\n")
    flog.close()

    with open(f'detector_count.log', 'w') as f:
        f.write(f"Total frames detected: {count}\n")
        f.flush()