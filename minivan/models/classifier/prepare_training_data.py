import json
import os
from xml.etree import ElementTree
import argparse
import shutil

import cv2
import torch

import minivan.images
from minivan.utils import get_mask


# PATCH_SIZE = 128
# VIDEOFILE = 'jnc00.mp4'
# MASKFILE = 'masks.xml'
# DEVICE = 'cuda:0'

def parseargs():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--videos', type=str, default='/data/chanwutk/data/otif-dataset/dataset/caldot1/test/video/', help='Input video file / directory')
    parser.add_argument('--det', type=str, default='/data/chanwutk/projects/minivan/det', help='Input detection file')
    parser.add_argument('--patch-size', type=int, default=32, help='Patch size')
    parser.add_argument('--output', type=str, default='train-proxy-data-1', help='Output directory')
    return parser.parse_args()


def main():
    args = parseargs()
    videofiles = args.videos
    patch_size = args.patch_size

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'neg'), exist_ok=True)

    if os.path.isdir(videofiles):
        videofiles = [os.path.join(videofiles, f) for f in os.listdir(videofiles) if f.endswith('.mp4')]
    else:
        videofiles = [videofiles]
    
    for videofile in sorted(videofiles):
        cap = cv2.VideoCapture(videofile)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with open(os.path.join(args.det, f'{os.path.basename(videofile)}.jsonl'), 'r') as fpp:
            detections = fpp.readlines()
        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1):
            print(i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frame_masked = torch.from_numpy(frame).to('cuda:0')
            frame_masked = frame_masked.detach()

            assert minivan.images.isHWC(frame_masked), frame_masked.shape
            padded_frame = minivan.images.padHWC(frame_masked, patch_size, patch_size)

            patched = minivan.images.splitHWC(padded_frame, patch_size, patch_size)
            patched = patched.cpu()
            assert minivan.images.isGHWC(patched), patched.shape

            _idx, dets = json.loads(detections[i])
            assert _idx == i, (_idx, i)

            for y in range(patched.shape[0]):
                for x in range(patched.shape[1]):
                    # check if the patch contains any detections
                    fromx, fromy = x * patch_size, y * patch_size
                    tox, toy = fromx + patch_size, fromy + patch_size

                    if any(overlap(det, (fromx, fromy, tox, toy)) for det in dets):
                        cv2.imwrite(os.path.join(args.output, f'pos', f'{os.path.basename(videofile)[:-len('.mp4')]}.{i}.{y}.{x}.jpg'), patched[y, x].contiguous().numpy())
                    else:
                        frame_masked[fromy:toy, fromx:tox] //= 2
                        # do not save if the patch is completely black
                        if not patched[y, x].any():
                            continue
                        cv2.imwrite(os.path.join(args.output, 'neg', f'{os.path.basename(videofile)[:-len('.mp4')]}.{i}.{y}.{x}.jpg'), patched[y, x].contiguous().numpy())
                    
            frame_masked = frame_masked.cpu().numpy()
            for det in dets:
                cv2.rectangle(frame_masked, pt1=(int(det[0]), int(det[1])), pt2=(int(det[2]), int(det[3])), color=(255, 0, 0), thickness=2)
            cv2.imwrite(f'/data/chanwutk/projects/minivan/test_frames/frame_{os.path.basename(videofile)}_{i}.jpg', frame_masked)
        

def overlapi(interval1: tuple[int, int], interval2: tuple[int, int]):
    # return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]
    return (
        (interval1[0] <= interval2[0] <= interval1[1]) or
        (interval1[0] <= interval2[1] <= interval1[1]) or
        (interval2[0] <= interval1[0] <= interval2[1]) or
        (interval2[0] <= interval1[1] <= interval2[1])
    )

def overlap(b1, b2):
    return overlapi((b1[0], b1[2]), (b2[0], b2[2])) and overlapi((b1[1], b1[3]), (b2[1], b2[3]))


if __name__ == '__main__':
    main()