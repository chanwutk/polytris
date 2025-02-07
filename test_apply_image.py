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
    # if size == 0:
    #     return NoOpTransform()
    assert size != 0

    newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, max_size)
    return h, w, newh, neww, interp


def apply_image(img, h: int, w: int, new_h: int, new_w: int, interp: Image.Resampling, device: str):
    assert img.shape[:2] == (h, w)
    assert len(img.shape) <= 4
    # interp_method = interp  # if interp is not None else self.interp

    # if img.dtype == np.uint8:
    #     print(1)
    #     if len(img.shape) > 2 and img.shape[2] == 1:
    #         pil_image = Image.fromarray(img[:, :, 0], mode="L")
    #     else:
    #         pil_image = Image.fromarray(img)
    #     pil_image = pil_image.resize((new_w, new_h), interp_method)
    #     ret = np.asarray(pil_image)
    #     if len(img.shape) > 2 and img.shape[2] == 1:
    #         ret = np.expand_dims(ret, -1)
    # else:
    # print(2)
    # PIL only supports uint8
    if any(x < 0 for x in img.strides):
        img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).to(torch.float32)
    shape = list(img.shape)
    shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
    img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
    # _PIL_RESIZE_TO_INTERPOLATE_MODE = {
    #     Image.Resampling.NEAREST: "nearest",
    #     Image.Resampling.BILINEAR: "bilinear",
    #     Image.Resampling.BICUBIC: "bicubic",
    # }
    # mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
    mode = "bilinear"
    # align_corners = None if mode == "nearest" else False
    align_corners = False
    img = F.interpolate(
        img, (new_h, new_w), mode=mode, align_corners=align_corners
    )
    shape[:2] = (new_h, new_w)
    assert isinstance(img, torch.Tensor)
    ret = img.permute(2, 3, 0, 1).view(shape).to(torch.uint8).cpu().numpy()  # nchw -> hw(c)

    return ret


def apply_image2(img, h: int, w: int, new_h: int, new_w: int, interp: Image.Resampling):
    assert img.shape[:2] == (h, w)
    assert len(img.shape) <= 4
    interp_method = interp  # if interp is not None else self.interp

    # if img.dtype == np.uint8:
    # print(1)
    if len(img.shape) > 2 and img.shape[2] == 1:
        pil_image = Image.fromarray(img[:, :, 0], mode="L")
    else:
        pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((new_w, new_h), interp_method)
    ret = np.asarray(pil_image)
    if len(img.shape) > 2 and img.shape[2] == 1:
        ret = np.expand_dims(ret, -1)
    # # else:
    # # print(2)
    # # PIL only supports uint8
    # if any(x < 0 for x in img.strides):
    #     img = np.ascontiguousarray(img)
    # img = torch.from_numpy(img).to('cuda:0').to(torch.float32)
    # shape = list(img.shape)
    # shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
    # img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
    # # _PIL_RESIZE_TO_INTERPOLATE_MODE = {
    # #     Image.Resampling.NEAREST: "nearest",
    # #     Image.Resampling.BILINEAR: "bilinear",
    # #     Image.Resampling.BICUBIC: "bicubic",
    # # }
    # # mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
    # mode = "bilinear"
    # # align_corners = None if mode == "nearest" else False
    # align_corners = False
    # img = F.interpolate(
    #     img, (new_h, new_w), mode=mode, align_corners=align_corners
    # )
    # shape[:2] = (new_h, new_w)
    # assert isinstance(img, torch.Tensor)
    # ret = img.permute(2, 3, 0, 1).view(shape).to(torch.uint8).cpu().numpy()  # nchw -> hw(c)

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

        cap = cv2.VideoCapture(os.path.expanduser(args.video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bmmask, btl, bbr = get_mask(mask, width, height)
        bmmask = bmmask[btl[0]:bbr[0], btl[1]:bbr[1], :]
        bmmask = torch.from_numpy(bmmask).to(device)


        transform: None | tuple[int, int, int, int, Image.Resampling] = None
        idx = 0
        while cap.isOpened():
            start = time.time_ns()
            success, frame = cap.read()
            if not success:
                break
            end = time.time_ns()

            start = time.time_ns()
            frame_masked = frame[btl[0]:bbr[0], btl[1]:bbr[1], :]
            frame_masked = torch.from_numpy(frame_masked).to(device) * bmmask
            frame_masked = frame_masked.detach().cpu().numpy()
            end = time.time_ns()

            start = time.time_ns()
            image_regions = regionize_image(frame_masked)
            resolutions = []
            for _image, _offset in image_regions:
                resolutions.append(_image.shape)

                if transform is None:
                    transform = get_transform(_image, (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST), cfg.INPUT.MAX_SIZE_TEST)

                if predictor.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    _image = _image[:, :, ::-1]
                height, width = _image.shape[:2]
                start = time.time()
                image = apply_image(_image, *transform)
                end = time.time()
                print(f'Apply image 1: {(end - start) * 1000:.2f} ms')
                cv2.imwrite(os.path.join('test_apply_image_output', f'apply{idx}.1.jpg'), image)
                start = time.time()
                image2 = apply_image2(_image, *transform)
                end = time.time()
                cv2.imwrite(os.path.join('test_apply_image_output', f'apply{idx}.2.jpg'), image2)
                diff = np.abs(image.astype(np.int32) - image2.astype(np.int32))
                cv2.imwrite(os.path.join('test_apply_image_output', f'apply{idx}.d.jpg'), diff)
                print(f'Apply image 2: {(end - start) * 1000:.2f} ms')
                print('diff', diff.max(), diff.mean(), diff.std())
                cv2.imwrite(os.path.join('test_apply_image_output', f'apply{idx}.t.jpg'), (diff > 15).astype(np.uint8) * 255)

                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                image.to(cfg.MODEL.DEVICE)

                # inputs = {"image": image, "height": height, "width": width}

                # predictions = self.model([inputs])[0]
                # return predictions
            idx += 1


if __name__ == '__main__':
    # main(parse_args())

    # if os.path.exists('track-results'):
    #     shutil.rmtree('track-results')
    # os.makedirs('track-results')


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
