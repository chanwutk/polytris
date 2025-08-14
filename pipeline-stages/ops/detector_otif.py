import math
import os
import pathlib
import shutil
import sys
import io

MODULES_PATH = pathlib.Path().absolute().parent / 'modules'
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'b3d'))

sys.path.append('/data/chanwutk/projects/polyis/modules/darknet')
cwd = os.getcwd()
os.chdir('/data/chanwutk/projects/polyis/modules/darknet')
import darknet
os.chdir(cwd)

import numpy as np

from polyis.dtypes import DetArray, InPipe, NPImage, OutPipe


data_root = '/data/chanwutk/data/otif-dataset'
batch_size = 1
width = 704
height = 480
param_width = 704
param_height = 480
threshold = 0.25
classes = ''
label = 'caldot'

if classes != '':
    classes = {cls.strip(): True for cls in classes.split(',')}
else:
    classes = None

detector_label = label
if detector_label.startswith('caldot'):
    detector_label = 'caldot'
if detector_label in ['amsterdam', 'jackson']:
    detector_label = 'generic'

config_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}-test.cfg'.format(param_width, param_height))
meta_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.data')
names_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.names')

if detector_label == 'generic':
    weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3.best')
else:
    weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}.best'.format(param_width, param_height))

# ensure width/height in config file
with open(config_path, 'r') as f:
    tmp_config_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('width='):
            line = 'width={}'.format(width)
        if line.startswith('height='):
            line = 'height={}'.format(height)
        tmp_config_buf += line + "\n"
tmp_config_path = '/tmp/yolov3-{}.cfg'.format(os.getpid())
with open(tmp_config_path, 'w') as f:
    f.write(tmp_config_buf)

# Write out our own obj.data which has direct path to obj.names.
tmp_obj_names = '/tmp/obj-{}.names'.format(os.getpid())
shutil.copy(names_path, tmp_obj_names)

with open(meta_path, 'r') as f:
    tmp_meta_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('names='):
            line = 'names={}'.format(tmp_obj_names)
        tmp_meta_buf += line + "\n"
tmp_obj_meta = '/tmp/obj-{}.data'.format(os.getpid())
with open(tmp_obj_meta, 'w') as f:
    f.write(tmp_meta_buf)

# Finally we can load YOLOv3.
net, class_names, _ = darknet.load_network(tmp_config_path, tmp_obj_meta, weight_path, batch_size=batch_size)
os.remove(tmp_config_path)
os.remove(tmp_obj_names)
os.remove(tmp_obj_meta)


def _detect(flog: io.TextIOWrapper[io._WrappedBuffer], frame: NPImage):
    flog.write(f"Detecting {frame.shape}...\n")
    flog.flush()

    assert frame.shape == (480, 704, 3), frame.shape
    height, width = frame.shape[:2]
    arr = np.array([frame], dtype='uint8')

    arr = arr.transpose((0, 3, 1, 2))
    arr = np.ascontiguousarray(arr.flat, dtype='float32')/255.0
    darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(width, height, 3, darknet_images)
    raw_detections = darknet.network_predict_batch(net, darknet_images, batch_size, width, height, threshold, 0.5, None, 0, 0)
    num = raw_detections[0].num
    raw_dlist = raw_detections[0].dets
    darknet.do_nms_obj(raw_dlist, num, len(class_names), 0.45)
    raw_dlist = darknet.remove_negatives(raw_dlist, class_names, num)
    detections = np.zeros((len(raw_dlist), 5))
    if len(raw_dlist) > 0:
        for idx, (cls, score, (cx, cy, w, h)) in enumerate(raw_dlist):
            if classes is not None and cls not in classes:
                continue
            if int(w) == 0 or int(h) == 0:
                print(f"Zero detected ({cls}, {score}): {cx}, {cy}, {w}, {h}")
            detections[idx, 0:4] = (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))
            detections[idx, 4] = score
    darknet.free_batch_detections(raw_detections, batch_size)

    return detections


def detect(imgQueue: "InPipe[NPImage]", bboxQueue: "OutPipe[DetArray]", device: str = 'cuda:0'):
    flog = open('detector.py.log', 'w')
    count = 0

    while True:
        frame = imgQueue.get()
        if frame is None:
            break
            
        detections = _detect(flog, frame)
        bboxQueue.put(detections)
        count += 1
    bboxQueue.put(None)

    flog.write("Detector finished.\n")
    flog.close()

    with open(f'detector_count.log', 'w') as f:
        f.write(f"Total frames detected: {count}\n")
        f.flush()


def detectIdx(imgQueue: "InPipe[tuple[int, NPImage]]", bboxQueue: "OutPipe[tuple[int, DetArray]]", device: str = 'cuda:0'):
    flog = open('detector.py.log', 'w')
    count = 0

    while True:
        frame = imgQueue.get()
        if frame is None:
            break
        idx, frame = frame

        detections = _detect(flog, frame)
        bboxQueue.put((idx, detections))
        count += 1
    bboxQueue.put(None)

    flog.write("Detector finished.\n")
    flog.close()

    with open(f'detector_count.log', 'w') as f:
        f.write(f"Total frames detected: {count}\n")
        f.flush()