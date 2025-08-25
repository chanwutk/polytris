#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time

import cv2
import torch
import tqdm

import polyis.images

CACHE_DIR = '/polyis-cache'
DATA_DIR = '/polyis-data/video-datasets-low'
TILE_SIZES = [32, 64, 128]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    return parser.parse_args()


def overlapi(interval1: tuple[int, int], interval2: tuple[int, int]):
    return (
        (interval1[0] <= interval2[0] <= interval1[1]) or
        (interval1[0] <= interval2[1] <= interval1[1]) or
        (interval2[0] <= interval1[0] <= interval2[1]) or
        (interval2[0] <= interval1[1] <= interval2[1])
    )

def overlap(b1, b2):
    return overlapi((b1[0], b1[2]), (b2[0], b2[2])) and overlapi((b1[1], b1[3]), (b2[1], b2[3]))


def main(args):
    cache_dir = os.path.join(CACHE_DIR, args.dataset)
    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    for video in sorted(os.listdir(cache_dir)):
        video_dir = os.path.join(cache_dir, video)
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_dir):
            continue

        print(f"Processing video {video_dir}")

        if os.path.exists(os.path.join(video_dir, 'training')):
            shutil.rmtree(os.path.join(video_dir, 'training'))

        for tile_size in TILE_SIZES:
            runtime_path = os.path.join(video_dir, 'training', 'runtime', f'tilesize_{tile_size}')
            if os.path.exists(runtime_path):
                shutil.rmtree(runtime_path)
            os.makedirs(runtime_path, exist_ok=True)

            training_data_path = os.path.join(video_dir, 'training', 'data', f'tilesize_{tile_size}')
            if os.path.exists(training_data_path):
                # remove the existing training data
                shutil.rmtree(training_data_path)
            os.makedirs(training_data_path, exist_ok=True)
            if not os.path.exists(os.path.join(training_data_path, 'pos')):
                os.makedirs(os.path.join(training_data_path, 'pos'), exist_ok=True)
            if not os.path.exists(os.path.join(training_data_path, 'neg')):
                os.makedirs(os.path.join(training_data_path, 'neg'), exist_ok=True)
        
        frs = {
            tile_size: open(os.path.join(video_dir, 'training', 'runtime', f'tilesize_{tile_size}', 'create_training_data.jsonl'), 'w')
            for tile_size in TILE_SIZES
        }

        # Read segments to get frame ranges
        segments_path = os.path.join(video_dir, 'segments', 'detection', 'segments.jsonl')
        detections_path = os.path.join(video_dir, 'segments', 'detection', 'detections.jsonl')
        
        if not os.path.exists(segments_path) or not os.path.exists(detections_path):
            print(f"Skipping {video_dir} - missing segments or detections files")
            continue

        # Construct the path to the video file in the dataset directory
        dataset_video_path = os.path.join(dataset_dir, video)
        cap = cv2.VideoCapture(dataset_video_path)

        with open(segments_path, 'r') as segments_f, open(detections_path, 'r') as detections_f:
            segments_lines = [*segments_f.readlines()]
            # detections_lines = [*detections_f.readlines()]

            for segment in tqdm.tqdm(segments_lines):
                segment_json = json.loads(segment)
                segment_idx = segment_json['idx']
                start = segment_json['start']
                end = segment_json['end']

                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for frame_idx in range(start, end):
                    ret, frame = cap.read()
                    assert ret, f"Failed to read frame {frame_idx}"

                    frame_idx_, dets, segment_idx_, _ = json.loads(detections_f.readline())
                    assert frame_idx_ == frame_idx, f"Frame index mismatch: {frame_idx_} != {frame_idx}"
                    assert segment_idx_ == segment_idx, f"Segment index mismatch: {segment_idx_} != {segment_idx}"

                    for tile_size in TILE_SIZES:
                        split_start_time = time.time()
                        training_data_path = os.path.join(video_dir, 'training', 'data', f'tilesize_{tile_size}')

                        padded_frame = torch.from_numpy(frame).to('cuda:0')
                        assert polyis.images.isHWC(padded_frame), padded_frame.shape

                        patched = polyis.images.splitHWC(padded_frame, tile_size, tile_size)
                        patched = patched.cpu()
                        assert polyis.images.isGHWC(patched), patched.shape
                        split_time = time.time() - split_start_time
                        frs[tile_size].write(json.dumps({
                            'op': 'split',
                            'time': split_time,
                            'frame': frame_idx,
                            'tile_size': tile_size,
                            'frame_shape': frame.shape,
                            'patched_shape': patched.shape,
                        }) + '\n')

                        save_start_time = time.time()
                        for y in range(patched.shape[0]):
                            for x in range(patched.shape[1]):
                                # check if the patch contains any detections
                                fromx, fromy = x * tile_size, y * tile_size
                                tox, toy = fromx + tile_size - 1, fromy + tile_size - 1

                                filename = f'{frame_idx}.{y}.{x}.jpg'
                                patch = patched[y, x].contiguous().numpy()

                                if any(overlap(det, (fromx, fromy, tox, toy)) for det in dets):
                                    cv2.imwrite(os.path.join(training_data_path, 'pos', filename), patch)
                                else:
                                    # For visualizing the negative patches
                                    # frame[fromy:toy, fromx:tox] //= 2
                                    if patched[y, x].any():  # do not save if the patch is completely black
                                        cv2.imwrite(os.path.join(training_data_path, 'neg', filename), patch)
                        save_time = time.time() - save_start_time
                        frs[tile_size].write(json.dumps({
                            'op': 'save',
                            'time': save_time,
                            'frame': frame_idx,
                            'tile_size': tile_size,
                            'frame_shape': frame.shape,
                            'patched_shape': patched.shape,
                        }) + '\n')

        cap.release()
        for fr in frs.values():
            fr.close()


if __name__ == '__main__':
    main(parse_args())