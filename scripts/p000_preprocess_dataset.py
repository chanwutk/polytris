#!/usr/local/bin/python

import argparse
from functools import partial
import os
import shutil
import subprocess
import json
from xml.etree import ElementTree
from multiprocessing import Queue, cpu_count

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.path import Path

from polyis.utilities import DATA_RAW_DIR, DATA_DIR, ProgressBar, DATASETS_TO_TEST


def get_video_info(video_path: str) -> dict:
    """
    Get video information using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        dict: Video information including width, height, fps, duration, frame_count, codec
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")
        
        # Extract video properties
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        codec_name = video_stream.get('codec_name', 'h264')
        
        # Calculate fps from frame rate
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        # Get duration and calculate frame count
        duration = float(data['format']['duration'])
        frame_count = int(duration * fps)
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count,
            'codec': codec_name
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")


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
    parser.add_argument('--segment_size', type=int, default=60,
                        help='Video segment size in seconds')
    
    args = parser.parse_args()
    
    return args


def process_b3d_video(file: str, videodir: str, outputdir: str, mask: str, batch_size: int,
                      segment_size: int, gpuIdx: int, command_queue: Queue):

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
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    duration = fps * total_frames
    num_segments = duration // segment_size
    segment_size_frames = segment_size * fps

    command_queue.put(('cuda:' + str(gpuIdx), {
        'description': f'{file}',
        'completed': 0,
        'total': total_frames
    }))
    processed_frames = 0
    for i in range(num_segments):
        start_frame = i * segment_size_frames
        end_frame = (i + 1) * segment_size_frames

        segment_filename = f"{file.replace('.mp4', '')}.{i:03d}.mp4"
        segment_path = os.path.join(outputdir, segment_filename)

        processed_frames = process_b3d_segment(file, videodir, segment_path, batch_size,
                            start_frame, end_frame, top, bottom, left, right, bitmask,
                            processed_frames, gpuIdx, command_queue)


def process_b3d_segment(file: str, videodir: str, outputdir: str, batch_size: int,
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

    is_vertical = iwidth < iheight

    out_filename = os.path.join(outputdir, file.replace('.mp4', f'.{start_frame:03d}.mp4'))
    writer = cv2.VideoWriter(out_filename, cv2.VideoWriter.fourcc(*'mp4v'), fps, (owidth, oheight))

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
                # processed_frames += 1
                
                # # Send progress update
                # command_queue.put(('cuda:' + str(gpuIdx), { 'completed': processed_frames }))

    cap.release()
    writer.release()

    return processed_frames


def process_b3d(args: argparse.Namespace):
    funcs = []

    for dataset in os.listdir(args.input):
        if not dataset.startswith('b3d-'):
            continue

        videodir = os.path.join(args.input, dataset)
        outputdir = os.path.join(args.output, dataset)
        mask = os.path.join(args.input, 'b3d', 'annotations.xml')
        batch_size = args.batch_size
        assert batch_size is not None and batch_size > 0
        segment_size = args.segment_size
        assert segment_size is not None and segment_size > 0

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
            
            funcs.append(partial(process_b3d_video, file, videodir, outputdir, mask, batch_size, segment_size))
        
        assert len(funcs) > 0
        
    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Limit the number of processes to the number of available GPUs
    max_processes = min(len(funcs), num_gpus)
    print(f"Using {max_processes} processes (limited by {num_gpus} GPUs)")
    
    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=max_processes, num_tasks=len(funcs)).run_all(funcs)


def process_caldot_video(video_file: str, videodir: str, outputdir: str,
                         segment_size: int, worker_id: int, command_queue: Queue):
    video_path = os.path.join(videodir, video_file)
    
    # Get video information for progress tracking
    video_info = get_video_info(video_path)
    duration = video_info['duration']
    
    # Send initial progress update
    command_queue.put(('cuda:' + str(worker_id), {
        'description': f'{video_file}',
        'completed': 0,
        'total': duration
    }))
    
    # Get video information
    video_info = get_video_info(video_path)
    duration = video_info['duration']
    
    # Calculate number of segments needed
    num_segments = duration // segment_size
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for i in range(num_segments):
        start_time = i * segment_size
        end_time = (i + 1) * segment_size
        
        # Create output filename for this segment
        segment_filename = f"{base_name}.{i:03d}.mp4"
        segment_path = os.path.join(outputdir, segment_filename)
        
        # Use ffmpeg to extract the segment with crop to 720x480 (no audio)
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', video_path,  # Input file
            '-ss', str(start_time),  # Start time
            '-t', str(end_time - start_time),  # Duration
            '-vf', 'crop=720:480:0:0',  # Crop frames to 720x480 from upper left
            '-c:v', 'libx264',  # Re-encode video with H.264
            '-an',  # Disable audio
            segment_path
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        command_queue.put(('cuda:' + str(worker_id), { 'completed': i + 1 }))
    

def process_caldot(args: argparse.Namespace, dataset: str):
    videodir = os.path.join(args.input, dataset)
    outputdir = os.path.join(args.output, dataset)
    segment_size = args.segment_size

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    video_files = os.listdir(videodir)
    assert len(video_files) > 0

    num_workers = min(int(cpu_count() * 0.8), len(video_files), 20)

    funcs = []
    for video_file in video_files:
        funcs.append(partial(process_caldot_video, video_file, videodir, outputdir, segment_size))

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