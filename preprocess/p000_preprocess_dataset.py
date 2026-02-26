#!/usr/local/bin/python

import argparse
from functools import partial
import os
import shutil
import subprocess
from xml.etree import ElementTree
from multiprocessing import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from polyis.utilities import (
    ProgressBar,
    build_b3d_mask_and_crop,
    dedupe_datasets_by_root,
    get_segment_frame_range,
    get_config,
    resolve_otif_dataset_name,
)

CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
SOURCE_DIR = CONFIG['DATA']['SOURCE_DIR']
OTIF_DATASET = CONFIG['DATA']['OTIF_DATASET']


def parse_args():
    preprocess_ops = CONFIG.get('OPS', {}).get('preprocess_dataset', {})
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('-b', '--batch_size', required=False,
                        default=preprocess_ops.get('batch_size', 128),
                        type=int,
                        help='Batch size')
    parser.add_argument('--num_segments', type=int,
                        default=preprocess_ops.get('num_segments', 18),
                        help='Number of segments to split the video into')

    args = parser.parse_args()

    return args


OUTPUT_FPS = 15
FPS_15_LO = 14.9
FPS_15_HI = 15.1
FPS_30_LO = 29.0
FPS_30_HI = 31.0
FPS_60_LO = 59.0
FPS_60_HI = 60.1


def assert_fps_and_get_step(fps: float) -> int:
    if FPS_15_LO <= fps <= FPS_15_HI:
        return 1
    if FPS_30_LO <= fps <= FPS_30_HI:
        return 2
    if FPS_60_LO <= fps <= FPS_60_HI:
        return 4
    raise AssertionError(
        f"Video FPS {fps} is not allowed; must be 15, 29–31, or 59–60"
    )


def process_b3d_video(file: str, videodir: str, outputdir: str, mask: str, batch_size: int,
                      num_segments: int, gpuIdx: int, command_queue: Queue):
    video_path = os.path.join(videodir, file)
    cap = cv2.VideoCapture(video_path)
    iwidth, iheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    step = assert_fps_and_get_step(input_fps)
    cap.release()
    output_frame_count = (total_frames + step - 1) // step

    top, bottom, left, right, bitmap = build_b3d_mask_and_crop(
        file_name=file,
        annotations_path=mask,
        frame_width=iwidth,
        frame_height=iheight,
    )
    bitmask = torch.from_numpy(bitmap).to(f'cuda:{gpuIdx}').to(torch.bool)

    command_queue.put(('cuda:' + str(gpuIdx), {
        'completed': 0,
        'total': output_frame_count
    }))
    processed_frames = 0
    for i in [*range(num_segments)][::-1]:
        start_frame, end_frame = get_segment_frame_range(total_frames, i, num_segments)

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
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    step = assert_fps_and_get_step(input_fps)

    owidth, oheight = WIDTH, HEIGHT
    masked_height = bottom - top
    masked_width = right - left
    is_vertical = masked_height > masked_width

    writer = cv2.VideoWriter(
        outputfile, cv2.VideoWriter.fourcc(*'mp4v'), OUTPUT_FPS, (owidth, oheight)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fidx = start_frame
    done = False
    with torch.no_grad():
        while cap.isOpened() and not done:
            frames = []
            while len(frames) < batch_size:
                ret, frame = cap.read()
                fidx += 1
                if not ret or fidx > end_frame:
                    done = True
                    break
                if (fidx - 1) % step != 0:
                    continue
                frame = frame[top:bottom, left:right, :]
                frames.append(frame)
                processed_frames += 1
                command_queue.put(('cuda:' + str(gpuIdx), {'completed': processed_frames}))

            if not frames:
                break

            frames_gpu = torch.from_numpy(np.array(frames)).to(f'cuda:{gpuIdx}')
            frames_gpu = frames_gpu * bitmask

            if is_vertical:
                frames_gpu = torch.rot90(frames_gpu, k=-1, dims=(1, 2))

            frames_gpu = frames_gpu.permute(0, 3, 1, 2)
            frames_gpu = F.interpolate(frames_gpu.float(), size=(oheight, owidth),
                                       mode='bilinear', align_corners=False)
            assert isinstance(frames_gpu, torch.Tensor)
            frames_gpu = frames_gpu.permute(0, 2, 3, 1)
            frames = frames_gpu.detach().to(torch.uint8).cpu().numpy()

            for frame in frames:
                writer.write(np.ascontiguousarray(frame))

    cap.release()
    writer.release()

    return processed_frames


def process_b3d(args: argparse.Namespace, dataset: str):
    funcs = []

    videodir = os.path.join(SOURCE_DIR, dataset)
    outputdir = os.path.join(DATASETS_DIR, dataset)
    mask = os.path.join(SOURCE_DIR, 'b3d', 'annotations.xml')
    batch_size = args.batch_size
    assert batch_size is not None and batch_size > 0
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
                         dataset: str, worker_id: int, command_queue: Queue):
    video_path = os.path.join(videodir, video_file)

    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert_fps_and_get_step(input_fps)

    video_file = f"{int(video_file.split('.')[0]):02d}.mp4"

    command_queue.put(('cuda:' + str(worker_id), {
        'description': f'{video_file}',
        'completed': 0,
        'total': 1
    }))

    os.makedirs(outputdir, exist_ok=True)

    # Load detector config from detectors.yaml to get resolution
    with open('configs/detectors.yaml', 'r') as f:
        detector_configs = yaml.safe_load(f)

    dataset_mapping = detector_configs['dataset_name_mapping']
    dataset_detector_mapping = detector_configs['dataset_detector_mapping']

    # Map dataset name to canonical name
    canonical_dataset = dataset_mapping[dataset]
    detector_info = dataset_detector_mapping[canonical_dataset]

    # Get width and height from detector config, with fallback to defaults
    width = detector_info['width']
    height = detector_info['height']

    cmd = [
        'ffmpeg', '-y',
        "-hide_banner", "-loglevel", "warning", "-threads", "4",
        '-i', video_path,
        "-vf", f'scale={width}:{height},setsar=1,fps={OUTPUT_FPS}',
        '-c:v', 'libx264',
        '-an',
        os.path.join(outputdir, video_file)
    ]

    subprocess.run(cmd, capture_output=True, text=True, check=True)
    command_queue.put(('cuda:' + str(worker_id), {'completed': 1}))


def process_caldot(dataset: str):
    # Map dataset key to the corresponding OTIF dataset directory name.
    otif_dataset = resolve_otif_dataset_name(dataset)
    # Build source and output dataset paths.
    video_dataset_dir = os.path.join(OTIF_DATASET, otif_dataset)
    output_dataset_dir = os.path.join(DATASETS_DIR, dataset)

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
            funcs.append(partial(process_caldot_video, video_file, videoset_dir, output_dir, dataset))

    return funcs

def main(args):
    funcs = []
    # Resolve configured datasets to unique preprocessing roots.
    datasets_to_process = dedupe_datasets_by_root(DATASETS)
    print(f"Resolved datasets for preprocessing: {datasets_to_process}")

    # Build preprocessing jobs for each resolved dataset.
    for dataset in datasets_to_process:
        print(f"Processing dataset: {dataset}")
        if dataset.startswith('jnc'):
            funcs.extend(process_b3d(args, dataset))
        elif dataset.startswith('caldot') or dataset.startswith('ams'):
            funcs.extend(process_caldot(dataset))
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

    assert len(funcs) > 0

    # Determine number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Use ProgressBar for parallel processing
    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
