#!/usr/local/bin/python

import argparse
import json
import os
import time

import cv2
import tqdm

import polyis.models.retinanet_b3d

CACHE_DIR = '/polyis-cache'


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


def detect_retina(dataset_dir: str):
    detector = polyis.models.retinanet_b3d.get_detector(device='cuda:0')

    for video in os.listdir(dataset_dir):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing video {video_path}")

        for snippet in tqdm.tqdm(os.listdir(video_path)):
            snippet_path = os.path.join(video_path, snippet)
            if not os.path.isfile(snippet_path) or not snippet.startswith('d_') or not snippet.endswith('.mp4'):
                continue


            meta = snippet.split('.')[0].split('_')
            start = int(meta[2])

            with open(snippet_path[:-len('.mp4')] + '.jsonl', 'w') as f:
                # Read the video
                cap = cv2.VideoCapture(snippet_path)
                idx = start
                while cap.isOpened():
                    start_time = time.time_ns()
                    ret, frame = cap.read()
                    end_time = time.time_ns()
                    read_time = end_time - start_time

                    if not ret:
                        break

                    # Detect objects in the frame
                    start_time = time.time_ns()
                    outputs = polyis.models.retinanet_b3d.detect(frame, detector)
                    end_time = time.time_ns()
                    detect_time = end_time - start_time

                    f.write(json.dumps([idx, outputs[:, :4].tolist(), format_time(read=read_time, detect=detect_time)]) + '\n')

                    idx += 1

                cap.release()


def main(args):
    dataset_dir = os.path.join(CACHE_DIR, args.dataset)
    detector = args.detector

    if detector == 'retina':
        detect_retina(dataset_dir)
    else:
        raise ValueError(f"Unknown detector: {detector}")


if __name__ == '__main__':
    main(parse_args())