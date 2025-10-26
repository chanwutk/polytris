#!/usr/local/bin/python

import argparse
from functools import partial
import os
import shutil
import subprocess
import json
from xml.etree import ElementTree
from multiprocessing import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.path import Path

from polyis.utilities import DATASETS_DIR, SOURCE_DIR, ProgressBar, DATASETS_TO_TEST


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('-i', '--input', required=False,
                        default=SOURCE_DIR,
                        help='Video Dataset directory')
    parser.add_argument('-o', '--output', required=False,
                        default=DATASETS_DIR,
                        help='Processed Dataset directory')
    parser.add_argument('-b', '--batch_size', required=False,
                        default=128,
                        type=int,
                        help='Batch size')
    parser.add_argument('-d', '--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--segment_size', type=int, default=60,
                        help='Video segment size in seconds')
    parser.add_argument('--num_segments', type=int, default=18,
                        help='Number of segments to split the video into')
    
    args = parser.parse_args()
    
    return args


def process_b3d_video(file: str, videodir: str, outputdir: str, mask: str, batch_size: int,
                      num_segments: int, gpuIdx: int, command_queue: Queue):

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
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

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

    segment_size_frames = total_frames / num_segments
    # print(duration, (duration // 60, duration % 60), num_segments, segment_size_frames, fps)

    command_queue.put(('cuda:' + str(gpuIdx), {
        'completed': 0,
        'total': total_frames
    }))
    processed_frames = 0
    for i in [*range(num_segments)][::-1]:
        start_frame = int(i * segment_size_frames)
        end_frame = min(int((i + 1) * segment_size_frames), total_frames)

        segment_filename = f"{i:02d}.mp4"
        segment_path = os.path.join(outputdir, segment_filename)

        command_queue.put(('cuda:' + str(gpuIdx), { 'description': f'{file} {(num_segments - i):02d}/{num_segments}',}))
        processed_frames = process_b3d_segment(file, videodir, segment_path, batch_size,
                            start_frame, end_frame, top, bottom, left, right, bitmask,
                            processed_frames, gpuIdx, command_queue)


def process_b3d_segment(file: str, videodir: str, outputfile: str, batch_size: int,
                        start_frame: int, end_frame: int,
                        top: int, bottom: int, left: int, right: int, bitmask: torch.Tensor,
                        processed_frames: int, gpuIdx: int, command_queue: Queue):
    WIDTH = 1080
    HEIGHT = 720

    video_path = os.path.join(videodir, file)
    cap = cv2.VideoCapture(video_path)
    iwidth, iheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate output fps and frame sampling based on whether isr or target_fps was provided
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    owidth, oheight = WIDTH, HEIGHT

    masked_height = bottom - top
    masked_width = right - left
    is_vertical = masked_height > masked_width

    writer = cv2.VideoWriter(outputfile, cv2.VideoWriter.fourcc(*'mp4v'), fps, (owidth, oheight))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fidx = start_frame
    done = False
    with torch.no_grad():
        while cap.isOpened() and not done:
            frames = []
            # print('start', fidx)
            for i in range(batch_size):
                ret, frame = cap.read()
                fidx += 1
                if not ret or fidx > end_frame:
                    done = True
                    break

                frame = frame[top:bottom, left:right, :]
                frames.append(frame)

                processed_frames += 1
                # Send progress update
                command_queue.put(('cuda:' + str(gpuIdx), { 'completed': processed_frames }))

            # print('mask', fidx)
            frames_gpu = torch.from_numpy(np.array(frames)).to(f'cuda:{gpuIdx}')
            frames_gpu = frames_gpu * bitmask

            if is_vertical:
                # Rotate frames: (batch, height, width, channels) -> (batch, width, height, channels)
                frames_gpu = torch.rot90(frames_gpu, k=-1, dims=(1, 2))  # k=-1 for clockwise

            # print('scale', fidx)
            frames_gpu = frames_gpu.permute(0, 3, 1, 2)
            frames_gpu = F.interpolate(frames_gpu.float(), size=(oheight, owidth),
                                       mode='bilinear', align_corners=False)
            assert isinstance(frames_gpu, torch.Tensor)
            frames_gpu = frames_gpu.permute(0, 2, 3, 1)
            frames = frames_gpu.detach().to(torch.uint8).cpu().numpy()

            # print('write', fidx)
            for frame in frames:
                writer.write(np.ascontiguousarray(frame))

    cap.release()
    writer.release()

    return processed_frames


def process_b3d(args: argparse.Namespace, dataset: str):
    funcs = []

    videodir = os.path.join(args.input, dataset)
    outputdir = os.path.join(args.output, dataset)
    mask = os.path.join(args.input, 'b3d', 'annotations.xml')
    batch_size = args.batch_size
    assert batch_size is not None and batch_size > 0
    segment_size = args.segment_size
    assert segment_size is not None and segment_size > 0
    num_segments = args.num_segments
    assert num_segments is not None and num_segments > 0

    root = ElementTree.parse(mask).getroot()
    assert root is not None

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # Collect all valid video files first
    for file in os.listdir(videodir):
        if not file.endswith('.mp4'):
            continue

        img = root.find(f'.//image[@name="{file.replace('.mp4', '.jpg')}"]')
        if img is None:
            continue

        domain = img.find('.//polygon[@label="domain"]')
        if domain is None:
            continue
        
        funcs.append(partial(process_b3d_video, file, videodir, outputdir, mask, batch_size, num_segments))
    return funcs


def process_caldot_video(video_file: str, videodir: str, outputdir: str,
                         worker_id: int, command_queue: Queue):
    video_path = os.path.join(videodir, video_file)

    video_file = f"{int(video_file.split('.')[0]):02d}.mp4"
    
    # Send initial progress update
    command_queue.put(('cuda:' + str(worker_id), {
        'description': f'{video_file}',
        'completed': 0,
        'total': 1
    }))

    os.makedirs(outputdir, exist_ok=True)

    # Use ffmpeg to extract the segment with scale to 720x480 (no audio)
    cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        "-hide_banner", "-loglevel", "warning", "-threads", "4",
        '-i', video_path,  # Input file
        "-vf", 'scale=720:480,setsar=1',  # Scale with square pixels per ffmpeg filter syntax
        '-c:v', 'libx264',  # Re-encode video with H.264
        '-an',  # Disable audio
        os.path.join(outputdir, video_file)
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    command_queue.put(('cuda:' + str(worker_id), { 'completed': 1 }))
    

def process_caldot(args: argparse.Namespace, dataset: str):
    video_dataset_dir = os.path.join(args.input, dataset)
    output_dataset_dir = os.path.join(args.output, dataset)

    if os.path.exists(output_dataset_dir):
        shutil.rmtree(output_dataset_dir)
    os.makedirs(output_dataset_dir, exist_ok=True)

    funcs: list[partial] = []
    videosets = ['train', 'test', 'valid']
    for videoset in videosets:
        videoset_dir = os.path.join(video_dataset_dir, videoset, 'video')
        assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

        video_files = os.listdir(videoset_dir)
        assert len(video_files) > 0

        output_dir = os.path.join(output_dataset_dir, videoset)

        for video_file in video_files:
            funcs.append(partial(process_caldot_video, video_file, videoset_dir, output_dir))

    return funcs

def main(args):
    datasets = args.datasets

    funcs = []
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        if dataset.startswith('b3dJnc'):
            funcs.extend(process_b3d(args, dataset))
        elif dataset.startswith('caldot'):
            funcs.extend(process_caldot(args, dataset))
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
    
    assert len(funcs) > 0
        
    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=10, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())