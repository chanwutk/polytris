#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import queue
import re
from xml.etree import ElementTree

import cv2
import numpy as np
import torch

import polyis.models.detector
from polyis.b3d.nms import nms
from polyis.utilities import (
    ProgressBar,
    build_b3d_mask_and_crop,
    dedupe_datasets_by_root,
    get_config,
    get_segment_frame_range,
)
from polyis.utils import intersects_polygon


CONFIG = get_config()
EXEC_DATASETS = CONFIG['EXEC']['DATASETS']
CACHE_DIR = CONFIG['DATA']['CACHE_DIR']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
SOURCE_DIR = CONFIG['DATA']['SOURCE_DIR']
OTIF_DATASET = CONFIG['DATA']['OTIF_DATASET']
NUM_SEGMENTS = CONFIG.get('OPS', {}).get('preprocess_dataset', {}).get('num_segments', 18)
EXCLUDE_AREA: dict[str, str] = CONFIG['DATA'].get('EXCLUDE_AREA', {}) or {}

TARGET_WIDTH = 1080
TARGET_HEIGHT = 720
NMS_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess groundtruth detection data')
    parser.add_argument('--test', action='store_true', help='Process test videoset')
    parser.add_argument('--train', action='store_true', help='Process train videoset')
    parser.add_argument('--valid', action='store_true', help='Process valid videoset')
    return parser.parse_args()


def dataset_root_name(dataset: str) -> str:
    if dataset.startswith('ams'):
        return 'ams'
    return dataset.split('-')[0]


def resolve_otif_dataset_name(dataset: str) -> str:
    root = dataset_root_name(dataset)
    if root == 'ams':
        return 'amsterdam'
    return root


def get_mask_key(dataset: str) -> str:
    if dataset.startswith('caldot1'):
        return 'caldot1'
    if dataset.startswith('caldot2'):
        return 'caldot2'
    if dataset.startswith('ams'):
        return 'ams'
    return dataset_root_name(dataset)


def parse_video_parts(video_file: str) -> tuple[str, str]:
    parts = video_file.split('/')
    assert len(parts) == 2, f'Invalid video file path: {video_file}'
    return parts[0], parts[1]


def parse_segment_index(video_name: str) -> int:
    stem = os.path.splitext(video_name)[0]
    match = re.match(r'^[a-z]{2}(\d+)$', stem)
    assert match is not None, f'Invalid split video filename: {video_name}'
    return int(match.group(1))


def build_output_path(dataset: str, video_name: str) -> str:
    return os.path.join(CACHE_DIR, dataset, 'execution', video_name, '003_groundtruth', 'detection.jsonl')


def load_include_rect(dataset: str) -> tuple[float, float, float, float] | None:
    key = get_mask_key(dataset)
    include_path = os.path.join('data', 'masks', 'include', f'{key}.xml')
    if not os.path.exists(include_path):
        return None

    root = ElementTree.parse(include_path).getroot()
    box = root.find('.//box[@label="crop-area"]')
    if box is None:
        box = root.find('.//box[@label="crop"]')
    if box is None:
        box = root.find('.//box')
    assert box is not None, f'No include box found in {include_path}'

    left = float(box.attrib['xtl'])
    top = float(box.attrib['ytl'])
    right = float(box.attrib['xbr'])
    bottom = float(box.attrib['ybr'])
    return left, top, right, bottom


def load_exclude_polygon_xml(dataset: str) -> str | None:
    key = get_mask_key(dataset)
    exclude_path = os.path.join('data', 'masks', 'exclude', f'{key}.xml')
    if os.path.exists(exclude_path):
        with open(exclude_path, 'r') as f:
            return f.read().strip()

    legacy_path = EXCLUDE_AREA.get(dataset_root_name(dataset))
    if legacy_path:
        assert os.path.exists(legacy_path), f'Exclude area XML file not found: {legacy_path}'
        with open(legacy_path, 'r') as f:
            return f.read().strip()

    return None


def clip_box_to_rect(
    left: float,
    top: float,
    right: float,
    bottom: float,
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    rect_left, rect_top, rect_right, rect_bottom = rect
    clipped_left = max(left, rect_left)
    clipped_top = max(top, rect_top)
    clipped_right = min(right, rect_right)
    clipped_bottom = min(bottom, rect_bottom)

    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        return None

    return clipped_left, clipped_top, clipped_right, clipped_bottom


def apply_include_exclude_masks(
    left: float,
    top: float,
    right: float,
    bottom: float,
    score: float,
    include_rect: tuple[float, float, float, float] | None,
    exclude_polygon_xml: str | None,
) -> list[float] | None:
    if include_rect is not None:
        clipped = clip_box_to_rect(left, top, right, bottom, include_rect)
        if clipped is None:
            return None
        left, top, right, bottom = clipped

    if exclude_polygon_xml and intersects_polygon(left, top, right, bottom, exclude_polygon_xml):
        return None

    return [float(left), float(top), float(right), float(bottom), float(score)]


def get_frame_count(video_path: str) -> int:
    """Return frame count for the given video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Could not open video {video_path}'
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def resolve_source_video_path(dataset: str, videoset: str, video_name: str) -> str:
    otif_dataset = resolve_otif_dataset_name(dataset)
    source_video_dir = os.path.join(OTIF_DATASET, otif_dataset, videoset, 'video')
    assert os.path.exists(source_video_dir), f'Source video directory not found: {source_video_dir}'

    video_stem, _ = os.path.splitext(video_name)
    match = re.match(r'^(te|tr|va)(\d{2})$', video_stem)
    assert match is not None, f'Invalid split video filename: {video_name}'

    video_idx = int(match.group(2))
    source_video_path = os.path.join(source_video_dir, f'{video_idx}.mp4')
    assert os.path.exists(source_video_path), f'Source video file not found: {source_video_path}'
    return source_video_path


def copy_detection_caldot(dataset: str, video_file: str, gpu_id: int, command_queue: queue.Queue):
    videoset, video_name = parse_video_parts(video_file)
    video_number = parse_segment_index(video_name)

    include_rect = load_include_rect(dataset)
    exclude_polygon_xml = load_exclude_polygon_xml(dataset)

    otif_dataset = resolve_otif_dataset_name(dataset)
    gt_json_path = os.path.join(
        OTIF_DATASET,
        otif_dataset,
        videoset,
        'yolov3-704x480',
        f'{video_number}.json',
    )
    assert os.path.exists(gt_json_path), f'Groundtruth JSON file not found: {gt_json_path}'

    output_path = build_output_path(dataset, video_name)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read preprocessed video frame count
    dataset_video_path = os.path.join(DATASETS_DIR, dataset, video_file)
    frame_count = get_frame_count(dataset_video_path)

    with open(gt_json_path, 'r') as f:
        annotations = json.load(f)

    # assert len(annotations) == frame_count, (
    #     f'Number of annotations ({len(annotations)}) does not match frame count ({frame_count})'
    # )

    command_queue.put((
        'cuda:' + str(gpu_id),
        {
            'description': f'{dataset} {video_name}',
            'completed': 0,
            'total': frame_count,
        },
    ))

    with open(output_path, 'w') as f:
        for frame_idx, frame_annos in enumerate(annotations):
            detections: list[list[float]] = []
            for obj in frame_annos:
                cls = obj['class']
                if cls != 'car':
                    continue

                # Extract GT detection coordinates
                left = float(obj['left'])
                top = float(obj['top'])
                right = float(obj['right'])
                bottom = float(obj['bottom'])

                processed = apply_include_exclude_masks(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    score=float(obj['score']),
                    include_rect=include_rect,
                    exclude_polygon_xml=exclude_polygon_xml,
                )
                if processed is None:
                    continue

                detections.append(processed)

            frame_entry = {
                'frame_idx': frame_idx,
                'detections': detections,
            }
            f.write(json.dumps(frame_entry) + '\n')
            command_queue.put(('cuda:' + str(gpu_id), {'completed': frame_idx + 1}))


def run_detection_ams(dataset: str, video_file: str, gpu_id: int, command_queue: queue.Queue):
    videoset, video_name = parse_video_parts(video_file)

    include_rect = load_include_rect(dataset)
    exclude_polygon_xml = load_exclude_polygon_xml(dataset)

    source_video_path = resolve_source_video_path(dataset, videoset, video_name)
    output_path = build_output_path(dataset, video_name)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    detector = polyis.models.detector.get_detector('ams', gpu_id, batch_size=1)

    cap = cv2.VideoCapture(source_video_path)
    assert cap.isOpened(), f'Could not open video {source_video_path}'
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    command_queue.put((
        'cuda:' + str(gpu_id),
        {
            'description': f'{dataset} {video_name}',
            'completed': 0,
            'total': frame_count,
        },
    ))

    with open(output_path, 'w') as f:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_detections = polyis.models.detector.detect(frame, detector)
            if raw_detections is None:
                raw_detections = np.empty((0, 5), dtype=np.float32)

            detections: list[list[float]] = []
            for det in raw_detections:
                processed = apply_include_exclude_masks(
                    left=float(det[0]),
                    top=float(det[1]),
                    right=float(det[2]),
                    bottom=float(det[3]),
                    score=float(det[4]),
                    include_rect=include_rect,
                    exclude_polygon_xml=exclude_polygon_xml,
                )
                if processed is None:
                    continue

                detections.append(processed)

            frame_entry = {
                'frame_idx': frame_idx,
                'detections': detections,
            }
            f.write(json.dumps(frame_entry) + '\n')

            frame_idx += 1
            command_queue.put(('cuda:' + str(gpu_id), {'completed': frame_idx}))

    cap.release()
    polyis.models.detector.delete(detector)


def get_single_source_video(dataset: str) -> str:
    source_dataset_dir = os.path.join(SOURCE_DIR, dataset)
    assert os.path.exists(source_dataset_dir), f'Source dataset directory not found: {source_dataset_dir}'

    video_files = [
        file_name for file_name in os.listdir(source_dataset_dir)
        if file_name.endswith('.mp4') and os.path.isfile(os.path.join(source_dataset_dir, file_name))
    ]
    assert len(video_files) == 1, (
        f'Expected exactly one source video in {source_dataset_dir}, found {len(video_files)}'
    )

    return os.path.join(source_dataset_dir, video_files[0])


def get_corner_crops(width: int, height: int) -> list[tuple[int, int, int, int]]:
    crop_width = max(1, int(width * 2 / 3))
    crop_height = max(1, int(height * 2 / 3))

    right_x = max(0, width - crop_width)
    bottom_y = max(0, height - crop_height)

    return [
        (0, 0, crop_width, crop_height),
        (right_x, 0, crop_width, crop_height),
        (0, bottom_y, crop_width, crop_height),
        (right_x, bottom_y, crop_width, crop_height),
    ]


def offset_and_clamp_coordinate(value: float, offset: int, limit: int) -> float:
    shifted_value = value + float(offset)
    clamped_value = max(0.0, min(shifted_value, limit - 1.0))
    return clamped_value


def scale_and_clamp_coordinate(value: float, scale: float, limit: int) -> float:
    scaled_value = value * scale
    clamped_value = max(0.0, min(scaled_value, limit - 1.0))
    return clamped_value


def detect_frame_with_corner_crops(detector, frame: np.ndarray) -> list[list[float]]:
    frame_height, frame_width = frame.shape[:2]
    corner_crops = get_corner_crops(frame_width, frame_height)
    merged_boxes: list[list[float]] = []
    merged_scores: list[float] = []

    for crop_x, crop_y, crop_w, crop_h in corner_crops:
        crop_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
        crop_detections = polyis.models.detector.detect(crop_frame, detector)
        if crop_detections is None:
            continue

        for det in crop_detections:
            det_left = offset_and_clamp_coordinate(float(det[0]), crop_x, frame_width)
            det_top = offset_and_clamp_coordinate(float(det[1]), crop_y, frame_height)
            det_right = offset_and_clamp_coordinate(float(det[2]), crop_x, frame_width)
            det_bottom = offset_and_clamp_coordinate(float(det[3]), crop_y, frame_height)
            det_score = float(det[4])

            if det_right <= det_left or det_bottom <= det_top:
                continue

            merged_boxes.append([det_left, det_top, det_right, det_bottom])
            merged_scores.append(det_score)

    if merged_boxes:
        nms_boxes, nms_scores = nms(merged_boxes, merged_scores, NMS_THRESHOLD)
    else:
        nms_boxes, nms_scores = [], []

    detections: list[list[float]] = []
    for box, score in zip(nms_boxes, nms_scores):
        detections.append([
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(score),
        ])
    return detections


def scale_detections(
    detections: list[list[float]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[list[float]]:
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled_detections: list[list[float]] = []

    for det in detections:
        scaled_left = scale_and_clamp_coordinate(det[0], scale_x, target_width)
        scaled_top = scale_and_clamp_coordinate(det[1], scale_y, target_height)
        scaled_right = scale_and_clamp_coordinate(det[2], scale_x, target_width)
        scaled_bottom = scale_and_clamp_coordinate(det[3], scale_y, target_height)
        score = det[4]

        if scaled_right <= scaled_left or scaled_bottom <= scaled_top:
            continue

        scaled_detections.append([
            float(scaled_left),
            float(scaled_top),
            float(scaled_right),
            float(scaled_bottom),
            float(score),
        ])

    return scaled_detections


def run_detection_jnc(dataset: str, video_file: str, gpu_id: int, command_queue: queue.Queue):
    _, video_name = parse_video_parts(video_file)
    segment_idx = parse_segment_index(video_name)

    source_video_path = get_single_source_video(dataset)
    source_video_name = os.path.basename(source_video_path)

    cap = cv2.VideoCapture(source_video_path)
    assert cap.isOpened(), f'Could not open source video {source_video_path}'

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame, end_frame = get_segment_frame_range(total_frames, segment_idx, NUM_SEGMENTS)
    total_segment_frames = max(0, end_frame - start_frame)

    annotations_path = os.path.join(SOURCE_DIR, 'b3d', 'annotations.xml')
    top, bottom, left, right, bitmap = build_b3d_mask_and_crop(
        file_name=source_video_name,
        annotations_path=annotations_path,
        frame_width=source_width,
        frame_height=source_height,
    )
    bitmask = bitmap[0].astype(bool)
    masked_height = bottom - top
    masked_width = right - left
    is_vertical = masked_height > masked_width

    output_path = build_output_path(dataset, video_name)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    detector = polyis.models.detector.get_detector(dataset, gpu_id, batch_size=1)

    command_queue.put((
        'cuda:' + str(gpu_id),
        {
            'description': f'{dataset} {video_name}',
            'completed': 0,
            'total': total_segment_frames,
        },
    ))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with open(output_path, 'w') as f:
        processed_count = 0
        frame_idx = 0

        while cap.isOpened() and processed_count < total_segment_frames:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[top:bottom, left:right, :]
            masked = cropped * bitmask

            if is_vertical:
                masked = np.rot90(masked, k=-1)

            frame_height, frame_width = masked.shape[:2]
            detections = detect_frame_with_corner_crops(detector, masked)
            scaled_detections = scale_detections(
                detections=detections,
                source_width=frame_width,
                source_height=frame_height,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            )

            frame_entry = {
                'frame_idx': frame_idx,
                'detections': scaled_detections,
            }
            f.write(json.dumps(frame_entry) + '\n')

            processed_count += 1
            frame_idx += 1
            command_queue.put(('cuda:' + str(gpu_id), {'completed': processed_count}))

    cap.release()
    polyis.models.detector.delete(detector)


def main():
    args = parse_args()

    splits = []
    if args.test:
        splits.append('test')
    if args.train:
        splits.append('train')
    if args.valid:
        splits.append('valid')
    if not splits:
        splits = ['test']

    # Resolve configured datasets to unique dataset roots.
    datasets_to_process = dedupe_datasets_by_root(EXEC_DATASETS)
    print(f'Resolved datasets for groundtruth preprocessing: {datasets_to_process}')

    funcs = []
    for dataset in datasets_to_process:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        assert os.path.exists(dataset_dir), f'Dataset directory {dataset_dir} does not exist'

        videos: list[str] = []
        for videoset in splits:
            videoset_dir = os.path.join(dataset_dir, videoset)
            assert os.path.exists(videoset_dir), f'Videoset directory {videoset_dir} does not exist'
            videos.extend([
                videoset + '/' + file_name
                for file_name in os.listdir(videoset_dir)
                if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ])
        assert len(videos) > 0, f'No video files found in {dataset_dir}'

        for video in videos:
            print(f'Processing {dataset}/{video}')
            if dataset.startswith('caldot'):
                funcs.append(partial(copy_detection_caldot, dataset, video))
            elif dataset.startswith('ams'):
                funcs.append(partial(run_detection_ams, dataset, video))
            elif dataset.startswith('jnc'):
                funcs.append(partial(run_detection_jnc, dataset, video))
            else:
                raise ValueError(f'Unknown dataset: {dataset}')

    num_gpus = torch.cuda.device_count()
    print(f'Available GPUs: {num_gpus}')

    max_processes = min(len(funcs), num_gpus)
    print(f'Using {max_processes} processes (limited by {num_gpus} GPUs)')

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs), refresh_per_second=10).run_all(funcs)


if __name__ == '__main__':
    main()
