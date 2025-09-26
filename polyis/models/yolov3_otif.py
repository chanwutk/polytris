import os
import ctypes
import tempfile
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import subprocess

import polyis.dtypes

sys.path.append('/polyis/modules/darknet')
cwd = os.getcwd()
os.chdir('/polyis/modules/darknet')
import darknet
os.chdir(cwd)


# Default configuration values
DEFAULT_DATA_ROOT = "/otif-dataset"
DEFAULT_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_BATCH_SIZE = 1
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 480
DEFAULT_DETECTOR_LABEL = "caldot"


class YOLOv3Detector:
    """YOLOv3 detector wrapper for consistent interface with other detectors."""
    
    def __init__(self, net, class_names: list, width: int, height: int, 
                 batch_size: int = 1, threshold: float = 0.25, nms_threshold: float = 0.45):
        self.net = net
        self.class_names = class_names
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.threshold = threshold
        self.nms_threshold = nms_threshold


def get_detector(gpu_id: int, config_path: str | None = None, model_path: str | None = None,
                 data_root: str | None = None, detector_label: str | None = None,
                 width: int | None = None, height: int | None = None,
                 threshold: float = 0.25, nms_threshold: float = 0.45):
    """
    Load YOLOv3 detector with configurable parameters.
    
    Args:
        gpu_id: GPU device ID to use (e.g., 0, 1, 2, 3)
        config_path: Path to YOLOv3 config file (.cfg)
        model_path: Path to YOLOv3 weights file (.best)
        data_root: Root directory for YOLOv3 models (default: '/otif-dataset')
        detector_label: Label for detector (e.g., 'caldot', 'shibuya', 'uav', 'warsaw')
        width: Model input width (default: 704)
        height: Model input height (default: 480)
        threshold: Detection confidence threshold (default: 0.25)
        nms_threshold: NMS threshold (default: 0.45)
        
    Returns:
        YOLOv3Detector: Configured detector instance
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If configuration is invalid
    """

    # Set defaults
    data_root = data_root or DEFAULT_DATA_ROOT
    detector_label = detector_label or DEFAULT_DETECTOR_LABEL
    width = width or DEFAULT_WIDTH
    height = height or DEFAULT_HEIGHT
    
    # Normalize detector label
    if detector_label.startswith('caldot'):
        detector_label = 'caldot'
    if detector_label in ['amsterdam', 'jackson']:
        detector_label = 'generic'
    
    detector_label_path = Path(data_root) / 'yolov3' / detector_label
    
    # Set up file paths
    if config_path is None:
        config_path = str(detector_label_path / f'yolov3-{width}x{height}-test.cfg')
    else:
        config_path = str(Path(config_path))
        
    if model_path is None:
        if detector_label == 'caldot':
            model_path = str(detector_label_path / f'yolov3-{width}x{height}.best')
        else:
            model_path = str(detector_label_path / 'yolov3.best')
    else:
        model_path = str(Path(model_path))
    
    meta_path = str(detector_label_path / 'obj.data')
    names_path = str(detector_label_path / 'obj.names')
    
    # Validate files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Names file not found: {names_path}")
    
    # Create temporary files for processing
    temp_files = []
    try:
        # Create temporary config file with correct width/height
        with open(config_path, 'r') as f:
            tmp_config_buf = ''
            for line in f.readlines():
                line = line.strip()
                if line.startswith('width='):
                    line = f'width={width}'
                if line.startswith('height='):
                    line = f'height={height}'
                tmp_config_buf += line + "\n"
        
        tmp_config_path = tempfile.mktemp(suffix='.cfg')
        with open(tmp_config_path, 'w') as f:
            f.write(tmp_config_buf)
        temp_files.append(tmp_config_path)
        
        # Create temporary obj.names file
        tmp_obj_names = tempfile.mktemp(suffix='.names')
        shutil.copy(names_path, tmp_obj_names)
        temp_files.append(tmp_obj_names)
        
        # Create temporary obj.data file
        with open(meta_path, 'r') as f:
            tmp_meta_buf = ''
            for line in f.readlines():
                line = line.strip()
                if line.startswith('names='):
                    line = f'names={tmp_obj_names}'
                tmp_meta_buf += line + "\n"
        
        tmp_obj_meta = tempfile.mktemp(suffix='.data')
        with open(tmp_obj_meta, 'w') as f:
            f.write(tmp_meta_buf)
        temp_files.append(tmp_obj_meta)
        
        # Load YOLOv3 network
        batch_size = DEFAULT_BATCH_SIZE
        darknet.set_gpu(gpu_id)
        # Suppress architecture output from load_network
        # Use file descriptor redirection for C-level output
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            net, class_names, _ = darknet.load_network(
                tmp_config_path, tmp_obj_meta, model_path, batch_size=batch_size
            )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(devnull)
        
        if len(class_names) != 1:
            raise ValueError(f'Expected 1 class, but got {len(class_names)}')
        
        return YOLOv3Detector(
            net=net,
            class_names=class_names,
            width=width,
            height=height,
            batch_size=batch_size,
            threshold=threshold or DEFAULT_THRESHOLD,
            nms_threshold=nms_threshold or DEFAULT_NMS_THRESHOLD
        )
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def detect(image: np.ndarray, detector: YOLOv3Detector, threshold: float | None = None):
    """
    Detect objects in an image using YOLOv3.
    
    Args:
        image: Input image as numpy array (H, W, C)
        detector: YOLOv3Detector instance
        threshold: Override detection threshold (optional)
        
    Returns:
        np.ndarray: Detection results as array of shape (N, 5) where each row is [x1, y1, x2, y2, confidence]
    """
    if threshold is None:
        threshold = detector.threshold
    
    # Prepare image for darknet
    oheight, owidth = image.shape[:2]
    image = cv2.resize(image, (detector.width, detector.height),
                       interpolation=cv2.INTER_LINEAR)
    # image = image[:detector.height, :detector.width, :]
    arr = np.array([image], dtype=np.uint8)
    arr = arr.transpose((0, 3, 1, 2))
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(detector.width, detector.height, 3, darknet_images)

    # Detect
    raw_detections = darknet.network_predict_batch(
        detector.net, darknet_images, detector.batch_size, 
        detector.width, detector.height, threshold, 0.5, None, 0, 0
    )

    # Process detections
    num = raw_detections[0].num
    
    raw_dlist = raw_detections[0].dets
    darknet.do_nms_obj(raw_dlist, num, len(detector.class_names), detector.nms_threshold)
    # raw_dlist = darknet.remove_negatives(raw_dlist, detector.class_names, num)
    
    predictions: list[tuple[float, float, float, float, float]] = []
    for i in range(num):
        det = raw_dlist[i]
        if det.prob[0] > 0:
            bbox = det.bbox
            cx, cy, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            if int(w) == 0 or int(h) == 0:
                print(f"Zero detected ({det.prob[0]}): {cx}, {cy}, {w}, {h}")
                continue
            predictions.append((int(cx-w/2), int(cy-h/2), int(cx+w/2),
                                int(cy+h/2), det.prob[0]))
    # predictions = []
    # for _cls, score, (cx, cy, w, h) in raw_dlist:
    #     predictions.append((int(cx-w/2), int(cy-h/2), int(cx+w/2),
    #                         int(cy+h/2), score))
    
    detections = np.array(predictions) if len(predictions) > 0 else np.empty((0, 5))
    darknet.free_batch_detections(raw_detections, detector.batch_size)

    detections[:, [0, 2]] = detections[:, [0, 2]] * owidth / detector.width
    detections[:, [1, 3]] = detections[:, [1, 3]] * oheight / detector.height

    assert polyis.dtypes.is_det_array(detections)
    return detections
