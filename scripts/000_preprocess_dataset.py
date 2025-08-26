#!/usr/local/bin/python

import argparse
import os
from xml.etree import ElementTree
import multiprocessing as mp

import cv2
import numpy as np
import torch
from matplotlib.path import Path

from scripts.utilities import DATA_RAW_DIR, DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('-i', '--input', required=False,
                        default=DATA_RAW_DIR,
                        help='Video Dataset directory')
    parser.add_argument('-o', '--output', required=False,
                        default=DATA_DIR,
                        help='Processed Dataset directory')
    parser.add_argument('-b', '--batch_size', required=False,
                        default=256,
                        type=int,
                        help='Batch size')
    parser.add_argument('-c', '--chunk_size', required=False,
                        default=2048,
                        type=int,
                        help='Chunk size')
    parser.add_argument('-d', '--datasets', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--isr', default=1, type=int)
    return parser.parse_args()


def process_video(file, videodir, outputdir, mask, gpuIdx, batch_size, chunk_size, isr):
    WIDTH = 1152
    HEIGHT = 768

    root = ElementTree.parse(mask).getroot()
    img = root.find(f'.//image[@name="{file.replace(".mp4", ".jpg")}"]')
    assert img is not None

    box = img.find('.//box[@label="crop"]')
    assert box is not None

    left = round(float(box.attrib['xtl']))
    top = round(float(box.attrib['ytl']))
    right = round(float(box.attrib['xbr']))
    bottom = round(float(box.attrib['ybr']))

    video_path = os.path.join(videodir, file)
    cap = cv2.VideoCapture(video_path)
    iwidth, iheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) // isr
    owidth, oheight = WIDTH, HEIGHT

    domains = img.findall('.//polygon[@label="domain"]')
    bitmaps = []
    for domain in domains:
        assert isinstance(domain, ElementTree.Element)
        domain = domain.attrib['points']
        domain = domain.replace(';', ',')
        domain = np.array([float(pt) for pt in domain.split(',')]).reshape((-1, 2))
        domain_poly = Path(domain)
        x, y = np.meshgrid(np.arange(iwidth), np.arange(iheight))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((x, y)).T
        bitmap = domain_poly.contains_points(pixel_points)
        bitmap = bitmap.reshape((1, iheight, iwidth, 1))
        bitmaps.append(bitmap)
    bitmap = bitmaps[0]
    for b in bitmaps[1:]:
        bitmap |= b
    bitmap = bitmap.astype(np.uint8) * 255
    bitmap = bitmap[:, top:bottom, left:right, :]
    bitmask = torch.from_numpy(bitmap).to(f'cuda:{gpuIdx}').to(torch.bool)

    iheight, iwidth = bitmap.shape[1:3]
    if iwidth < iheight:
        oheight, owidth = owidth, oheight

    out_filename = os.path.join(outputdir, file)
    writer = cv2.VideoWriter(out_filename, cv2.VideoWriter.fourcc(*'mp4v'), fps, (owidth, oheight))

    fidx = 0
    done = False
    while cap.isOpened() and not done:
        frames = []
        print('start', fidx)
        for i in range(batch_size):
            ret, frame = cap.read()
            fidx += 1
            if not ret:
                done = True
                break
            if fidx % isr == 0:
                frame = frame[top:bottom, left:right, :]
                frames.append(frame)

        print('mask', fidx)
        frames_gpu = torch.from_numpy(np.array(frames)).to(f'cuda:{gpuIdx}')
        frames_gpu = frames_gpu * bitmask

        print('scale', fidx)
        frames_gpu = frames_gpu.permute(0, 3, 1, 2)
        frames_gpu = torch.nn.functional.interpolate(frames_gpu.float(), size=(oheight, owidth), mode='bilinear', align_corners=False)
        assert isinstance(frames_gpu, torch.Tensor)
        frames_gpu = frames_gpu.permute(0, 2, 3, 1)
        frames = frames_gpu.detach().to(torch.uint8).cpu().numpy()

        print('write', fidx)
        for frame in frames:
            writer.write(np.ascontiguousarray(frame))
        print('write done', fidx)

    cap.release()
    writer.release()


def process_b3d(args: argparse.Namespace):
    videodir = os.path.join(args.input, 'b3d')
    outputdir = os.path.join(args.output, 'b3d')
    mask = os.path.join(args.input, 'b3d', 'masks.xml')
    isr = args.isr

    root = ElementTree.parse(mask).getroot()
    assert root is not None

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    processes: list[mp.Process] = []
    count = 0
    for file in os.listdir(videodir):
        if not file.endswith('.mp4'):
            continue

        img = root.find(f'.//image[@name="{file.replace('.mp4', '.jpg')}"]')
        if img is None:
            continue

        domain = img.find('.//polygon[@label="domain"]')
        if domain is None:
            continue

        print(f'Processing {file}...')
        process = mp.Process(target=process_video, args=(file, videodir, outputdir, mask, count % torch.cuda.device_count(), args.batch_size, args.chunk_size, isr))
        process.start()
        processes.append(process)
        count += 1
    
    for process in processes:
        process.join()
        process.terminate()


def main(args):
    datasets = args.datasets

    if datasets == 'b3d':
        process_b3d(args)
    else:
        raise ValueError(f'Unknown dataset: {datasets}')


if __name__ == '__main__':
    main(parse_args())