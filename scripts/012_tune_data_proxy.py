import argparse
import json
import os
import shutil

import cv2
import torch

import polyis.images

CACHE_DIR = '/polyis-cache'
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
    dataset_dir = os.path.join(CACHE_DIR, args.dataset)

    for video in os.listdir(dataset_dir):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing video {video_path}")

        for tile_size in TILE_SIZES:
            proxy_data_path = os.path.join(video_path, 'training', f'proxy_{tile_size}')
            if os.path.exists(proxy_data_path):
                # remove the existing proxy data
                shutil.rmtree(proxy_data_path)
            os.makedirs(proxy_data_path, exist_ok=True)
            if not os.path.exists(os.path.join(proxy_data_path, 'pos')):
                os.makedirs(os.path.join(proxy_data_path, 'pos'), exist_ok=True)
            if not os.path.exists(os.path.join(proxy_data_path, 'neg')):
                os.makedirs(os.path.join(proxy_data_path, 'neg'), exist_ok=True)
        

        for snippet in os.listdir(video_path):
            snippet_path = os.path.join(video_path, snippet)
            if not os.path.isfile(snippet_path) or not snippet.startswith('d_') or not snippet.endswith('.mp4'):
                continue

            # Process the snippet
            print(f"Processing {snippet_path}")

            meta = snippet.split('.')[0].split('_')
            start = int(meta[2])
            end = int(meta[3])

            with open(snippet_path[:-len('.mp4')] + '.jsonl', 'r') as f:
                # Read the video
                cap = cv2.VideoCapture(snippet_path)
                idx = start
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    _idx, dets = json.loads(f.readline())
                    assert idx == _idx, (idx, _idx)

                    for tile_size in [32, 64, 128]:
                        proxy_data_path = os.path.join(video_path, 'training', f'proxy_{tile_size}')

                        # todo: create dataset for each patch size
                        padded_frame = torch.from_numpy(frame).to('cuda:0')
                        assert polyis.images.isHWC(padded_frame), padded_frame.shape

                        patched = polyis.images.splitHWC(padded_frame, tile_size, tile_size)
                        patched = patched.cpu()
                        assert polyis.images.isGHWC(patched), patched.shape

                        for y in range(patched.shape[0]):
                            for x in range(patched.shape[1]):
                                # check if the patch contains any detections
                                fromx, fromy = x * tile_size, y * tile_size
                                tox, toy = fromx + tile_size, fromy + tile_size

                                filename = f'{idx}.{y}.{x}.jpg'
                                patch = patched[y, x].contiguous().numpy()

                                if any(overlap(det, (fromx, fromy, tox, toy)) for det in dets):
                                    cv2.imwrite(os.path.join(proxy_data_path, 'pos', filename), patch)
                                else:
                                    # For visualizing the negative patches
                                    # frame[fromy:toy, fromx:tox] //= 2
                                    if not patched[y, x].any():  # do not save if the patch is completely black
                                        continue
                                    cv2.imwrite(os.path.join(proxy_data_path, 'neg', filename), patch)
                    idx += 1


if __name__ == '__main__':
    main(parse_args())