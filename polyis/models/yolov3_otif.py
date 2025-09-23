import os
from pathlib import Path
import shutil

import numpy as np


cwd = os.getcwd()
os.chdir('/polyis/modules/darknet')
import darknet
os.chdir(cwd)


data_root = Path('/otif-dataset')
batch_size = 1
width = 704
height = 480
param_width = 704
param_height = 480
threshold = 0.25
classes = None
label = 'caldot'


detector_label = label
if detector_label.startswith('caldot'):
    detector_label = 'caldot'
if detector_label in ['amsterdam', 'jackson']:
    detector_label = 'generic'

detector_label_path = data_root / 'yolov3' / detector_label

config_path = detector_label_path / f'yolov3-{param_width}x{param_height}-test.cfg'
meta_path = detector_label_path / 'obj.data'
names_path = detector_label_path / 'obj.names'

if detector_label == 'caldot':
    weight_path = detector_label_path / f'yolov3-{param_width}x{param_height}.best'
else:
    weight_path = detector_label_path / 'yolov3.best'

# ensure width/height in config file
with open(config_path, 'r') as f:
    tmp_config_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('width='):
            line = f'width={width}'
        if line.startswith('height='):
            line = f'height={height}'
        tmp_config_buf += line + "\n"
tmp_config_path = f'/tmp/yolov3-{os.getpid()}.cfg'
with open(tmp_config_path, 'w') as f:
    f.write(tmp_config_buf)

# Write out our own obj.data which has direct path to obj.names.
tmp_obj_names = f'/tmp/obj-{os.getpid()}.names'
shutil.copy(names_path, tmp_obj_names)

with open(meta_path, 'r') as f:
    tmp_meta_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('names='):
            line = f'names={tmp_obj_names}'
        tmp_meta_buf += line + "\n"
tmp_obj_meta = f'/tmp/obj-{os.getpid()}.data'
with open(tmp_obj_meta, 'w') as f:
    f.write(tmp_meta_buf)

# Finally we can load YOLOv3.
net, class_names, _ = darknet.load_network(tmp_config_path, tmp_obj_meta, weight_path, batch_size=batch_size)
assert len(class_names) == 1, f'Only detect 1 class, but got {len(class_names)}'
os.remove(tmp_config_path)
os.remove(tmp_obj_names)
os.remove(tmp_obj_meta)


def get_detector(weight_path: str, threshold: float = 0.25):
    return net


def detect(image, detector):
    # Prepare image for darknet
    arr = np.array([image], dtype=np.uint8)
    arr = arr.transpose((0, 3, 1, 2))
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(width, height, 3, darknet_images)

    # Detect
    raw_detections = darknet.network_predict_batch(detector, darknet_images,
                                                   batch_size, width, height,
                                                   threshold, 0.5, None, 0, 0)

    # Process detections
    num = raw_detections[0].num
    raw_dlist = raw_detections[0].dets
    darknet.do_nms_obj(raw_dlist, num, len(class_names), 0.45)
    predictions: list[tuple[float, float, float, float, float]] = []
    for det in raw_dlist:
        if det.prob[0] > 0:
            bbox = det.bbox
            cx, cy, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            if int(w) == 0 or int(h) == 0:
                print(f"Zero detected ({det.prob[0]}): {cx}, {cy}, {w}, {h}")
                continue
            predictions.append((int(cx-w/2), int(cy-h/2), int(cx+w/2),
                                int(cy+h/2), det.prob[0]))
    detections = np.array(predictions) if predictions else np.empty((0, 5))
    darknet.free_batch_detections(raw_detections, batch_size)

    return detections
