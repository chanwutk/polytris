import sys
import pathlib
MODULES_PATH = pathlib.Path().absolute() / 'modules'
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'detectron2'))


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

from minivan.utils import get_mask


CONFIG = './modules/b3d/configs/config_refined.json'
MASK = './masks.xml'


class Input(NamedTuple):
    video: str
    gpu: str


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
        fpd = open(os.path.join('track-results', f'{videofilename}.d.jsonl'), 'w')
        fpr = open(os.path.join('track-results', f'{videofilename}.r.jsonl'), 'w')
        fpp = open(os.path.join('track-results', f'{videofilename}.p.jsonl'), 'w')
        try:
            with open(CONFIG) as fp:
                config = json.load(fp)
            cfg = get_cfg()
            cfg.merge_from_file(os.path.join('./modules/detectron2/configs', config['config']))
            cfg.MODEL.WEIGHTS = config['weights']
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
            cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
            cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
            cfg.MODEL.DEVICE = device
            predictor = DefaultPredictor(cfg)
            tree = ElementTree.parse(MASK)
            mask = tree.getroot()
            mask = mask.find(f'.//image[@name="{videofilename[:-len('.mp4')]}.jpg"]')
            assert isinstance(mask, ElementTree.Element)

            tracker = Sort(max_age=5)
            cap = cv2.VideoCapture(os.path.expanduser(args.video))
            trajectories = {}
            frame_index = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_sep = frame_count // 5

            bmmask, btl, bbr = get_mask(mask, width, height)
            bmmask = bmmask[btl[0]:bbr[0], btl[1]:bbr[1], :]
            bmmask = torch.from_numpy(bmmask).to(device)


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
            while cap.isOpened():
                if frame_index > 512:
                    break
                print(f'{videofilename} -- GPU {device}' + 'Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
                start = time.time_ns()
                success, frame = cap.read()
                if not success:
                    break
                end = time.time_ns()
                log('read', start, end)

                start = time.time_ns()
                # frame_masked = mask_frame(frame, mask)
                frame_masked = frame[btl[0]:bbr[0], btl[1]:bbr[1], :]
                frame_masked = torch.from_numpy(frame_masked).to(device) * bmmask
                frame_masked = frame_masked.detach().cpu().numpy()
                end = time.time_ns()
                log('mask', start, end)

                if frame_index % frame_sep == 0:
                    cv2.imwrite(os.path.join('track-results', f'{videofilename}.{frame_index}.0.jpg'), frame_masked)

                start = time.time_ns()
                image_regions = regionize_image(frame_masked)
                bboxes = []
                scores = []
                resolutions = []
                for _image, _offset in image_regions:
                    resolutions.append(_image.shape)
                    _outputs = predictor(_image)  # TODO: new predictor with GPU image transformation
                    _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
                    bboxes += _bboxes
                    scores += _scores
                    break
                nms_threshold = config['nms_threshold']
                nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
                detections = np.zeros((len(nms_bboxes), 5))
                detections[:, 0:4] = nms_bboxes
                detections[:, 4] = nms_scores
                end = time.time_ns()
                inference_time.append(end-start)
                log('detect', start, end, resolutions=resolutions, num_detections=len(detections), num_regions=len(image_regions))

                fpd.write(json.dumps([frame_index, detections.tolist()]) + '\n')

                start = time.time_ns()
                tracked_objects = tracker.update(detections)
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
                    fpd.flush()
                    fpr.flush()
                    fpp.flush()
                
                frame_masked_r = frame_masked.copy()
                frame_masked_d = frame_masked.copy()
                if frame_index % frame_sep == 0:
                    for object_index, tlx, tly, brx, bry in rendering:
                        cv2.rectangle(frame_masked_r, (tlx, tly), (brx, bry), colors[object_index % len(colors)], 2)
                        cv2.putText(frame_masked_r, str(object_index), (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[object_index % len(colors)], 2)
                    cv2.imwrite(os.path.join('track-results', f'{videofilename}.{frame_index}.r.jpg'), frame_masked_r)
                    for bbox in nms_bboxes:
                        tlx, tly, brx, bry = map(int, bbox)
                        cv2.rectangle(frame_masked_d, (tlx, tly), (brx, bry), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join('track-results', f'{videofilename}.{frame_index}.d.jpg'), frame_masked_d)

                frame_index = frame_index + 1
            cap.release()
            cv2.destroyAllWindows()

            with open(os.path.join('track-results', f'{videofilename}.t.json'), 'w') as fp:
                json.dump(trajectories, fp)
            with open(os.path.join('track-results', f'{videofilename}.done'), 'w') as fp:
                fp.write('done')
            print(f'{videofilename} -- GPU {device} -- Inference time: {sum(inference_time) / 1_000_000 / 512:.2f} ms')
        finally:
            fpd.close()
            fpr.close()
            fpp.close()


if __name__ == '__main__':
    # main(parse_args())

    if os.path.exists('track-results'):
        shutil.rmtree('track-results')
    os.makedirs('track-results')


    processes = []
    try:
        for idx, videofile in enumerate(os.listdir('videos')):
            assert videofile.endswith('.mp4')
            if videofile != 'jnc00.mp4':
                continue

            input = Input(os.path.join('videos', videofile), str(idx % torch.cuda.device_count()))
            # main(input)
            p = mp.Process(target=main, args=(input,))
            p.start()
            processes.append(p)
            print(f'Processed {videofile}')
    finally:
        for p in processes:
            p.join()
            print(f'Joined {p}')
        
        for p in processes:
            p.terminate()
            print(f'Terminated {p}')
