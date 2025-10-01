#!/usr/local/bin/python

import argparse
from functools import partial
import os
import shutil
from xml.etree import ElementTree
from multiprocessing import Queue, cpu_count

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.path import Path

from polyis.utilities import DATA_RAW_DIR, DATA_DIR, ProgressBar, DATASETS_TO_TEST, to_h264


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('-i', '--input', required=False,
                        default=DATA_RAW_DIR,
                        help='Video Dataset directory')
    parser.add_argument('-o', '--output', required=False,
                        default=DATA_DIR,
                        help='Processed Dataset directory')
    parser.add_argument('-b', '--batch_size', required=False,
                        default=128,
                        type=int,
                        help='Batch size')
    parser.add_argument('-d', '--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    
    # Create mutually exclusive group for isr and fps
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument('--isr', default=2, type=int,
                            help='Frame sampling rate (every nth frame)')
    frame_group.add_argument('--fps', type=int,
                            help='Target FPS for output video')
    
    parser.add_argument('--portion', type=str,
                        help='Portion of video to process in format <start-percent>'
                             ':<end-percent> (e.g., "10:90" or "0:50" or "25:")')
    
    args = parser.parse_args()
    
    # Validate that at least one of isr or fps is provided
    if args.isr is None and args.fps is None:
        parser.error("Either --isr or --fps must be specified")
    
    # Parse portion argument
    if args.portion:
        try:
            if ':' in args.portion:
                start_str, end_str = args.portion.split(':', 1)
                start_percent = float(start_str) if start_str else 0.0
                end_percent = float(end_str) if end_str else 100.0
            else:
                # If no colon, treat as end percentage
                start_percent = 0.0
                end_percent = float(args.portion)
            
            if not (0 <= start_percent <= 100 and 0 <= end_percent <= 100):
                parser.error("Portion percentages must be between 0 and 100")
            if start_percent >= end_percent:
                parser.error("Start percentage must be less than end percentage")
            
            args.start_percent = start_percent
            args.end_percent = end_percent
        except ValueError:
            parser.error("Portion must be in format <start-percent>:<end-percent> (e.g., '10:90')")
    else:
        args.start_percent = 0.0
        args.end_percent = 100.0
    
    return args


def process_b3d_video(file: str, videodir: str, outputdir: str, mask: str, batch_size: int,
                      isr: int, target_fps: int | None, start_percent: float, end_percent: float,
                      gpuIdx: int, command_queue: Queue):
    WIDTH = 1080
    HEIGHT = 720

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
    
    # Calculate output fps and frame sampling based on whether isr or target_fps was provided
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if target_fps is None:
        fps = original_fps // isr
    else:
        fps = target_fps
        # Calculate isr based on target fps
        isr = max(1, round(original_fps / target_fps))
    
    owidth, oheight = WIDTH, HEIGHT
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    # Calculate start and end frame indices based on percentage
    start_frame = int(total_frames * start_percent / 100.0)
    end_frame = int(total_frames * end_percent / 100.0)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
    is_vertical = iwidth < iheight

    out_filename = os.path.join(outputdir, file)
    writer = cv2.VideoWriter(out_filename, cv2.VideoWriter.fourcc(*'mp4v'), fps, (owidth, oheight))

    fidx = start_frame
    done = False
    with torch.no_grad():
        command_queue.put(('cuda:' + str(gpuIdx), {
            'description': f'{file}',
            'completed': 0,
            'total': end_frame - start_frame
        }))
        while cap.isOpened() and not done:
            frames = []
            # print('start', fidx)
            for i in range(batch_size):
                ret, frame = cap.read()
                fidx += 1
                if not ret or fidx > end_frame:
                    done = True
                    break
                if fidx % isr == 0:
                    frame = frame[top:bottom, left:right, :]
                    frames.append(frame)

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
                processed_frames += 1
                
                # Send progress update
                command_queue.put(('cuda:' + str(gpuIdx), { 'completed': processed_frames }))
            print('write done', fidx)

    cap.release()
    writer.release()
    
    # Convert to H.264 using FFMPEG
    to_h264(out_filename)


def process_b3d(args: argparse.Namespace):
    videodir = os.path.join(args.input, 'b3d')
    outputdir = os.path.join(args.output, 'b3d')
    mask = os.path.join(args.input, 'b3d', 'annotations.xml')
    isr = args.isr
    target_fps = args.fps

    root = ElementTree.parse(mask).getroot()
    assert root is not None

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # Collect all valid video files first
    funcs = []
    for file in os.listdir(videodir):
        if not file.endswith('.mp4'):
            continue

        img = root.find(f'.//image[@name="{file.replace('.mp4', '.jpg')}"]')
        if img is None:
            continue

        domain = img.find('.//polygon[@label="domain"]')
        if domain is None:
            continue
        
        funcs.append(partial(process_b3d_video, file, videodir,
                             outputdir, mask, args.batch_size, isr, target_fps,
                             args.start_percent, args.end_percent))
    
    assert len(funcs) > 0
    
    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=max_processes, num_tasks=len(funcs)).run_all(funcs)


def process_caldot_video(video_file: str, videodir: str, outputdir: str, isr: int, target_fps: int | None,
                         start_percent: float, end_percent: float, worker_id: int, command_queue: Queue):
    video_path = os.path.join(videodir, video_file)
    cap = cv2.VideoCapture(video_path)
    iwidth, iheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate output fps and frame sampling based on whether isr or target_fps was provided
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if target_fps is None:
        fps = original_fps // isr
    else:
        fps = target_fps
        # Calculate isr based on target fps
        isr = max(1, round(original_fps / target_fps))
    assert isr is not None

    out_filename = os.path.join(outputdir, video_file)
    writer = cv2.VideoWriter(out_filename, cv2.VideoWriter.fourcc(*'mp4v'), fps, (iwidth, iheight))

    # Calculate start and end frame indices based on percentage
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * start_percent / 100.0)
    end_frame = int(total_frames * end_percent / 100.0)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fidx = start_frame
    done = False
    with torch.no_grad():
        command_queue.put(('cuda:' + str(worker_id), {
            'description': f'{video_file}',
            'completed': 0,
            'total': end_frame - start_frame
        }))
        while cap.isOpened() and not done:
            ret, frame = cap.read()
            fidx += 1
            if not ret or fidx > end_frame:
                break
            if fidx % isr != 0:
                continue

            writer.write(np.ascontiguousarray(frame))
            command_queue.put(('cuda:' + str(worker_id), { 'completed': fidx - start_frame }))

    cap.release()
    writer.release()
    
    # Convert to H.264 using FFMPEG
    to_h264(out_filename)


def process_caldot(args: argparse.Namespace, dataset: str):
    videodir = os.path.join(args.input, dataset)
    outputdir = os.path.join(args.output, dataset)
    isr = args.isr
    target_fps = args.fps

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    video_files = os.listdir(videodir)
    assert len(video_files) > 0

    num_workers = min(int(cpu_count() * 0.8), len(video_files), 20)

    funcs = []
    for video_file in video_files:
        funcs.append(partial(process_caldot_video, video_file, videodir, outputdir, isr, target_fps,
                             args.start_percent, args.end_percent))

    ProgressBar(num_workers=num_workers, num_tasks=len(video_files)).run_all(funcs)


def main(args):
    datasets = args.datasets

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        if dataset.startswith('b3d'):
            process_b3d(args)
        elif dataset.startswith('caldot'):
            process_caldot(args, dataset)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')


if __name__ == '__main__':
    main(parse_args())