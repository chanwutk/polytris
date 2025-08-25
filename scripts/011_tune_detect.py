#!/usr/local/bin/python

import argparse
import json
import os
import time

import cv2
import tqdm

import polyis.models.retinanet_b3d

CACHE_DIR = '/polyis-cache'
DATA_DIR = '/polyis-data/video-datasets-low'


def format_time(**kwargs):
    return [{ 'op': op, 'time': time } for op, time in kwargs.items()]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--detector', required=False,
                        default='retina',
                        help='Detector name')
    return parser.parse_args()


def detect_retina(cache_dir: str, dataset_dir: str):
    detector = polyis.models.retinanet_b3d.get_detector(device='cuda:0')

    for video in sorted(os.listdir(dataset_dir)):
        video_path = os.path.join(cache_dir, video)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing video {video_path}")

        with (open(os.path.join(video_path, 'segments', 'detection', 'segments.jsonl'), 'r') as f,
              open(os.path.join(video_path, 'segments', 'detection', 'detections.jsonl'), 'w') as fd):
            lines = [*f.readlines()]

            # Construct the path to the video file in the dataset directory
            dataset_video_path = os.path.join(dataset_dir, video)
            cap = cv2.VideoCapture(dataset_video_path)

            for line in tqdm.tqdm(lines):
                snippet = json.loads(line)
                idx = snippet['idx']
                start = snippet['start']
                end = snippet['end']

                frame_idx = start
                # set cap to the start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                while cap.isOpened() and frame_idx < end:
                    start_time = time.time_ns() / 1e6
                    ret, frame = cap.read()
                    end_time = time.time_ns() / 1e6
                    read_time = end_time - start_time

                    assert ret, "Failed to read frame"

                    # Detect objects in the frame
                    start_time = time.time_ns() / 1e6
                    outputs = polyis.models.retinanet_b3d.detect(frame, detector)
                    end_time = time.time_ns() / 1e6
                    detect_time = end_time - start_time

                    fd.write(json.dumps([frame_idx, outputs[:, :4].tolist(), idx, format_time(read=read_time, detect=detect_time)]) + '\n')

                    frame_idx += 1
            cap.release()


def main(args):
    cache_dir = os.path.join(CACHE_DIR, args.dataset)
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    detector = args.detector

    if detector == 'retina':
        detect_retina(cache_dir, dataset_dir)
    else:
        raise ValueError(f"Unknown detector: {detector}")


if __name__ == '__main__':
    main(parse_args())