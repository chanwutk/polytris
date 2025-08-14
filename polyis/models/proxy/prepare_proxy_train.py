import json
import os
from xml.etree import ElementTree
import shutil

import cv2
import torch

import polyis.images
from polyis.utils import get_mask


PATCH_SIZE = 128
VIDEOFILE = 'jnc00.mp4'
MASKFILE = 'masks.xml'
DEVICE = 'cuda:0'

SELECTIVITY = 0.01

cap = cv2.VideoCapture(os.path.join('videos', VIDEOFILE))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


tree = ElementTree.parse(MASKFILE)
mask = tree.getroot()
mask = mask.find(f'.//image[@name="{VIDEOFILE[:-len('.mp4')]}.jpg"]')
assert isinstance(mask, ElementTree.Element)
bmmask, btl, bbr = get_mask(mask, width, height)
bmmask = bmmask[btl[0]:bbr[0], btl[1]:bbr[1], :]
bmmask = torch.from_numpy(bmmask).to(DEVICE)

fid = 0
sampling_total = int(video_length * SELECTIVITY)
sampling_sep = int(1 / SELECTIVITY)

with open(os.path.join('track-results', f'{VIDEOFILE}.d.jsonl'), 'r') as fpp:
    detections = fpp.readlines()


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


if os.path.exists('train-proxy-data'):
    shutil.rmtree('train-proxy-data')
os.makedirs('train-proxy-data', exist_ok=True)
os.makedirs(os.path.join('train-proxy-data', 'pos'), exist_ok=True)
os.makedirs(os.path.join('train-proxy-data', 'neg'), exist_ok=True)


for i in range(0, video_length, sampling_sep):
    print(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break

    frame_masked = frame[btl[0]:bbr[0], btl[1]:bbr[1], :]
    frame_masked = torch.from_numpy(frame_masked).to(DEVICE) * bmmask
    frame_masked = frame_masked.detach()

    assert polyis.images.isHWC(frame_masked), frame_masked.shape
    padded_frame = polyis.images.padHWC(frame_masked, PATCH_SIZE, PATCH_SIZE)

    patched = polyis.images.splitHWC(padded_frame, PATCH_SIZE, PATCH_SIZE)
    patched = patched.cpu()
    assert polyis.images.isGHWC(patched), patched.shape

    _idx, dets = json.loads(detections[i])
    assert _idx == i, (_idx, i)

    for y in range(patched.shape[0]):
        for x in range(patched.shape[1]):
            # check if the patch contains any detections
            fromx, fromy = x * PATCH_SIZE, y * PATCH_SIZE
            tox, toy = fromx + PATCH_SIZE, fromy + PATCH_SIZE

            if any(overlap(det, (fromx, fromy, tox, toy)) for det in dets):
                # tags = [0, 0, 0, 0]
                # if any(overlap(det, (fromx, fromy, tox, toy - (PATCH_SIZE // 8))) for det in dets):
                #     # north
                #     tags[0] = 1
                # if any(overlap(det, (fromx + (PATCH_SIZE // 8), fromy, tox, toy)) for det in dets):
                #     # east
                #     tags[1] = 1
                # if any(overlap(det, (fromx, fromy + (PATCH_SIZE // 8), tox, toy)) for det in dets):
                #     # south
                #     tags[2] = 1
                # if any(overlap(det, (fromx, fromy, tox - (PATCH_SIZE // 8), toy)) for det in dets):
                #     # west
                #     tags[3] = 1
                # if not os.path.exists(os.path.join('train-proxy-data', f'pos-{"".join(str(t) for t in tags)}')):
                #     os.makedirs(os.path.join('train-proxy-data', f'pos-{"".join(str(t) for t in tags)}'), exist_ok=True)
                # cv2.imwrite(os.path.join('train-proxy-data', f'pos-{"".join(str(t) for t in tags)}', f'{fid}.{y}.{x}.jpg'), patched[y, x].contiguous().numpy())
                cv2.imwrite(os.path.join('train-proxy-data', f'pos', f'{i}.{y}.{x}.jpg'), patched[y, x].contiguous().numpy())
            else:
                frame_masked[fromy:toy, fromx:tox] //= 2
                # do not save if the patch is completely black
                if not patched[y, x].any():
                    continue
                cv2.imwrite(os.path.join('train-proxy-data', 'neg', f'{i}.{y}.{x}.jpg'), patched[y, x].contiguous().numpy())
            
    frame_masked = frame_masked.cpu().numpy()
    for det in dets:
        cv2.rectangle(frame_masked, pt1=(int(det[0]), int(det[1])), pt2=(int(det[2]), int(det[3])), color=(255, 0, 0), thickness=2)
    cv2.imwrite(f'./test_frames/frame_{i}.jpg', frame_masked)