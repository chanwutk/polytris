import json
from queue import Queue

import numpy as np
import numpy.typing as npt

from b3d.external.nms import nms
from b3d.utils import parse_outputs, regionize_image
from minivan.models.predictor import get_detector


CONFIG = './modules/b3d/configs/config_refined.json'


def detect(imgQueue: "Queue[npt.NDArray]", bboxQueue: "Queue[npt.NDArray | None]", device: str):
    with open(CONFIG) as fp:
        config = json.load(fp)
    predictor = get_detector(device)

    while True:
        frame = imgQueue.get()
        if frame is None:
            break


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

        bboxQueue.put(detections)
    bboxQueue.put(None)