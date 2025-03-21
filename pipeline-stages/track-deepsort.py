import sys
import pathlib
# ROOT = pathlib.Path().absolute().parent
MODULES_PATH = pathlib.Path().absolute() / 'modules'
# sys.path.append(str(ROOT))
# sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'b3d'))
sys.path.append(str(MODULES_PATH / 'boxmot'))
sys.path.append(str(MODULES_PATH / 'detectron2'))

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


import argparse
import cv2
import json
import numpy as np
import os
from xml.etree import ElementTree
import torch
from typing import Literal, NamedTuple
import multiprocessing as mp
import shutil
import time
from PIL import Image

import torch.nn.functional as F

import numpy.typing as npt
from fvcore.transforms.transform import Transform, NoOpTransform

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge

from b3d.external.nms import nms
from b3d.external.sort import Sort
from b3d.utils import parse_outputs, regionize_image

# from minivan.utils import get_mask
from boxmot.tracker_zoo import get_tracker_config, create_tracker
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.utils import WEIGHTS


CONFIG = './modules/b3d/configs/config_refined.json'
MASK = './masks.xml'

PIPELINE_DIR = 'pipeline-stages'
TRACK_RESULTS_DIR = os.path.join(PIPELINE_DIR, 'track-results')
VIDEO_DIR = os.path.join(PIPELINE_DIR, 'video-masked')



import numpy as np

from matplotlib.path import Path
from xml.etree.ElementTree import Element


def get_mask(mask: Element, width: int, height: int):
    domain = mask.find('.//polygon[@label="domain"]')
    assert isinstance(domain, Element)
    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    # width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))
    # bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br



class Input(NamedTuple):
    video: str
    gpu: str
    skip: int


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def get_transform(
    image: npt.NDArray | torch.Tensor,
    short_edge_length: tuple[int, int],
    max_size: int = sys.maxsize,
    sample_style: Literal['range', 'choice'] = "range",
    interp: Image.Resampling = Image.Resampling.BILINEAR
):
    is_range = sample_style == "range"

    if isinstance(short_edge_length, int):
        short_edge_length = (short_edge_length, short_edge_length)
    if is_range:
        assert len(short_edge_length) == 2, (
            "short_edge_length must be two values using 'range' sample style."
            f" Got {short_edge_length}!"
        )

    h, w = image.shape[:2]
    if is_range:
        size = np.random.randint(short_edge_length[0], short_edge_length[1] + 1)
    else:
        size = np.random.choice(short_edge_length)
    if size == 0:
        return NoOpTransform()

    newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, max_size)
    return h, w, newh, neww, interp


def apply_image(img, h: int, w: int, new_h: int, new_w: int, interp: Image.Resampling):
    assert img.shape[:2] == (h, w)
    assert len(img.shape) <= 4
    interp_method = interp if interp is not None else self.interp

    if img.dtype == np.uint8:
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((new_w, new_h), interp_method)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
    else:
        # PIL only supports uint8
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        _PIL_RESIZE_TO_INTERPOLATE_MODE = {
            Image.Resampling.NEAREST: "nearest",
            Image.Resampling.BILINEAR: "bilinear",
            Image.Resampling.BICUBIC: "bicubic",
        }
        mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img, (new_h, new_w), mode=mode, align_corners=align_corners
        )
        shape[:2] = (new_h, new_w)
        assert isinstance(img, torch.Tensor)
        ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        description='Example detection and tracking script')
    parser.add_argument('-v', '--video', required=True,
                        help='Input video')
    # parser.add_argument('-c', '--config', required=True,
    #                     help='Detection model configuration')
    parser.add_argument('-g', '--gpu', required=True,
                        help='GPU device')
    # parser.add_argument('-m', '--mask', required=True,
    #                     help='Mask for the video')
    return parser.parse_args()


def main(args):
    videofilename = args.video.split('/')[-1]
    device = f'cuda:{args.gpu}'

    with torch.no_grad():
        fpd = open(os.path.join(TRACK_RESULTS_DIR + '-0', f'{videofilename}.d.jsonl'), 'r')
        fpr = open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.r.{args.skip}.jsonl'), 'w')
        fpp = open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.p.{args.skip}.jsonl'), 'w')
        try:
            # tracker = Sort(max_age=5)
            tracker: "DeepOcSort" = create_tracker(
                'deepocsort',
                tracker_config=get_tracker_config('deepocsort'),
                reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                device=device
            )
            tracker.cmc_off = True
            cap = cv2.VideoCapture(os.path.expanduser(args.video))
            trajectories = {}
            frame_index = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_sep = frame_count // 5


            def log(action: str, start: int, end: int, **kwargs):
                start //= 1000
                end //= 1000
                fpp.write(json.dumps({
                    'fid': frame_index,
                    'action': action,
                    'time': end - start,
                    'start': start,
                    'end': end,
                    **kwargs
                }) + '\n')


            inference_time = []
            for frame_index in range(3000):
                print(f'{videofilename} -- GPU {device} -- skip {args.skip} ' + 'Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
                start = time.time_ns()
                success, frame = cap.read()
                if not success:
                    break
                end = time.time_ns()
                log('read', start, end)

                start = time.time_ns()
                frame_masked = frame.copy()
                end = time.time_ns()
                log('mask', start, end)

                if frame_index % frame_sep == 0:
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{frame_index}.0.jpg'), frame_masked)

                start = time.time_ns()
                detections = json.loads(fpd.readline())
                assert detections[0] == frame_index, (detections[0], frame_index)
                detections = np.array(detections[1])
                detections = np.hstack([detections, np.zeros((len(detections), 1))])
                end = time.time_ns()
                inference_time.append(end-start)
                log('detect', start, end, num_detections=len(detections))

                # fpd.write(json.dumps([frame_index, detections.tolist()]) + '\n')

                if frame_index % args.skip != 0:
                    continue

                start = time.time_ns()
                # tracked_objects = tracker.update(detections)
                tracked_objects = tracker.update(detections, frame_masked)
                end = time.time_ns()
                log('track', start, end, num_tracks=len(tracked_objects))

                rendering = []
                for tracked_object in tracked_objects:
                    tl = (int(tracked_object[0]), int(tracked_object[1]))
                    br = (int(tracked_object[2]), int(tracked_object[3]))
                    object_index = int(tracked_object[4])
                    if object_index not in trajectories:
                        trajectories[object_index] = []
                    trajectories[object_index].append([
                        frame_index, tl[0], tl[1], br[0], br[1]])
                    rendering.append([
                        object_index, tl[0], tl[1], br[0], br[1]])
                fpr.write(json.dumps([frame_index, rendering]) + '\n')

                if frame_index % 50 == 0:
                    # fpd.flush()
                    fpr.flush()
                    fpp.flush()
                
                frame_masked_r = frame_masked.copy()
                frame_masked_d = frame_masked.copy()
                if frame_index % frame_sep == 0:
                    for object_index, tlx, tly, brx, bry in rendering:
                        cv2.rectangle(frame_masked_r, (tlx, tly), (brx, bry), colors[object_index % len(colors)], 2)
                        cv2.putText(frame_masked_r, str(object_index), (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[object_index % len(colors)], 2)
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{frame_index}.r.jpg'), frame_masked_r)
                    for bbox in detections:
                        tlx, tly, brx, bry, *_ = map(int, bbox)
                        cv2.rectangle(frame_masked_d, (tlx, tly), (brx, bry), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{frame_index}.d.jpg'), frame_masked_d)
            cap.release()
            cv2.destroyAllWindows()

            with open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.t.{args.skip}.json'), 'w') as fp:
                json.dump(trajectories, fp)
            with open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{args.skip}.done'), 'w') as fp:
                fp.write('done')
            print(f'{videofilename} -- GPU {device} -- Inference time: {sum(inference_time) / 1_000_000 / 512:.2f} ms')
        finally:
            fpd.close()
            fpr.close()
            fpp.close()


if __name__ == '__main__':
    # main(parse_args())

    if os.path.exists(TRACK_RESULTS_DIR):
        shutil.rmtree(TRACK_RESULTS_DIR)
    os.makedirs(TRACK_RESULTS_DIR)


    for skip in 2 ** np.arange(8):
        processes = []
        try:
            for idx, videofile in enumerate(os.listdir(VIDEO_DIR)):
                assert videofile.endswith('.mp4')
                if videofile.endswith('.x264.mp4'):
                    continue
                if not videofile.startswith('jnc'):
                    continue

                input = Input(
                    os.path.join(VIDEO_DIR, videofile),
                    str(idx % torch.cuda.device_count()),
                    int(skip),
                )
                # main(input)
                p = mp.Process(target=main, args=(input,))
                p.start()
                processes.append(p)
                print(f'Processed {videofile}')

            for p in processes:
                p.join()
                print(f'Joined {p}')
            
        finally:
            for p in processes:
                p.terminate()
                print(f'Terminated {p}')
