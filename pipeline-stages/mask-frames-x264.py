import sys, pathlib
MODULES_PATH = pathlib.Path().absolute() / 'modules'
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'detectron2'))


import json
import os
import queue
import shutil
import time
from xml.etree import ElementTree

import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import numpy.typing as npt
import torch

from b3d.utils import parse_outputs
from b3d.external.nms import nms
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import minivan.images




def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


CONFIG = './modules/b3d/configs/config_refined.json'
LIMIT = 512
CHUNK_SIZE = 128



def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

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

    bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


DIR = 'video-masked'


def logger(queue):
    with open('crop.jsonl', 'w') as fc:
        while True:
            val = queue.get()
            if val == None:
                return
            fc.write(val + '\n')
            fc.flush()


def process(file: str):
    filename = os.path.join(DIR, file)
    output_filename = f"{filename[:-len('.mp4')]}.x264.mp4"

    if os.path.exists(output_filename):
        os.system(output_filename)

    command = (
        "docker run --rm -v $(pwd):/config linuxserver/ffmpeg " +
        "-i {input_file} ".format(input_file=os.path.join('/config', filename)) +
        "-vcodec libx264 " +
        "{output_file}".format(output_file=os.path.join('/config', output_filename))
    )
    print(command)
    os.system(command)


def main():
    ps = []
    for file in os.listdir(DIR):
        p = torch.multiprocessing.Process(target=process, args=(file,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()