import sys, pathlib
MODULES_PATH = pathlib.Path().absolute() / 'modules'
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'detectron2'))


import json
import os
import queue
import shutil
import time
from xml.etree import ElementTree

import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import numpy.typing as npt
import torch

from b3d.utils import parse_outputs
from b3d.external.nms import nms
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import polyis.images




def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


CONFIG = './modules/b3d/configs/config_refined.json'
LIMIT = 512
CHUNK_SIZE = 128



def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    # width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))

    # bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


def overlapi(interval1: tuple[int, int], interval2: tuple[int, int]):
    return (
        (interval1[0] <= interval2[0] <= interval1[1]) or
        (interval1[0] <= interval2[1] <= interval1[1]) or
        (interval2[0] <= interval1[0] <= interval2[1]) or
        (interval2[0] <= interval1[1] <= interval2[1])
    )

def overlap(b1, b2):
    return overlapi((b1[0], b1[2]), (b2[0], b2[2])) and overlapi((b1[1], b1[3]), (b2[1], b2[3]))


def render(canvas: npt.NDArray, positions: list[tuple[int, int, int, npt.NDArray, tuple[int, int]]], frame: npt.NDArray):
    for y, x, groupid, mask, offset in positions:
        yfrom, yto = y * CHUNK_SIZE, (y + mask.shape[0]) * CHUNK_SIZE
        xfrom, xto = x * CHUNK_SIZE, (x + mask.shape[1]) * CHUNK_SIZE

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    # patch = patches[i + offset[0], j + offset[1]]
                    patch = frame[(i + offset[0]) * CHUNK_SIZE:(i + offset[0] + 1) * CHUNK_SIZE, (j + offset[1]) * CHUNK_SIZE:(j + offset[1] + 1) * CHUNK_SIZE]
                    # frame2[(i + offset[0]) * CHUNK_SIZE:(i + offset[0] + 1) * CHUNK_SIZE, (j + offset[1]) * CHUNK_SIZE:(j + offset[1] + 1) * CHUNK_SIZE] = ((
                    #     frame2[(i + offset[0]) * CHUNK_SIZE:(i + offset[0] + 1) * CHUNK_SIZE, (j + offset[1]) * CHUNK_SIZE:(j + offset[1] + 1) * CHUNK_SIZE].astype(np.uint32) +
                    #     (np.ones((CHUNK_SIZE, CHUNK_SIZE, 3), dtype=np.uint8) * colors[groupid % len(colors)]).astype(np.uint32)
                    # ) // 2).astype(np.uint8)
                    # cv2.imwrite(f'./packed_images/{idx:03d}.{i}.{j}.{groupid}.jpg', patch)
                    # canvas2[yfrom + (CHUNK_SIZE * i): yfrom + (CHUNK_SIZE * i) + CHUNK_SIZE, xfrom + (CHUNK_SIZE * j): xfrom + (CHUNK_SIZE * j) + CHUNK_SIZE] += (np.ones((CHUNK_SIZE, CHUNK_SIZE, 3), dtype=np.uint8) * colors[groupid % len(colors)]).astype(np.uint8)
                    canvas[yfrom + (CHUNK_SIZE * i): yfrom + (CHUNK_SIZE * i) + CHUNK_SIZE, xfrom + (CHUNK_SIZE * j): xfrom + (CHUNK_SIZE * j) + CHUNK_SIZE] += patch  # .detach().cpu().numpy()
    return canvas


def mark_detections2(detections: list[list[float]], width: int, height: int):
    bitmap = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE), dtype=np.int32)

    for bbox in detections:  # bboxes:
        xfrom, xto = int(bbox[0] // CHUNK_SIZE), int(bbox[2] // CHUNK_SIZE)
        yfrom, yto = int(bbox[1] // CHUNK_SIZE), int(bbox[3] // CHUNK_SIZE)

        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def fill_bitmap(bitmap: npt.NDArray, i: int, j: int):
    value = bitmap[i, j]
    q = queue.Queue()
    q.put((i, j))
    filled: list[tuple[int, int]] = []
    while not q.empty():
        i, j = q.get()
        bitmap[i, j] = value
        filled.append((i, j))
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                q.put((_i, _j))
    return filled


def group_tiles(bitmap: npt.NDArray):
    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int32) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * bitmap

    # Padding with size=1 on all sides
    groups = np.zeros((h + 2, w + 2), dtype=np.int32)
    groups[1:h+1, 1:w+1] = _groups

    visited: set[int] = set()
    bins: list[tuple[int, npt.NDArray, tuple[int, int]]] = []
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            if groups[i, j] == 0 or groups[i, j] in visited:
                continue

            filled = fill_bitmap(groups, i, j)
            filled = np.array(filled, dtype=int).T

            mask = np.zeros((h + 1, w + 1), dtype=np.bool)
            mask[*filled] = True

            offset = np.min(filled, axis=1)
            assert offset.shape == (2,)

            end = np.max(filled, axis=1) + 1
            assert end.shape == (2,)

            mask = mask[offset[0]:end[0], offset[1]:end[1]]
            bins.append((groups[i, j], mask, tuple(offset - 1)))

            visited.add(groups[i, j])

    # bins: a list of tuple (group_id, mask, offset)
    #       mask: cropped mask of the group
    #       offset: offset of mask
    return bins


class BitmapFullException(Exception):
    pass


def pack_append(bins: list[tuple[int, npt.NDArray, tuple[int, int]]], h: int, w: int, bitmap: npt.NDArray | None = None):
    newBitmap = False
    if bitmap is None:
        bitmap = np.zeros((h, w), dtype=np.bool)
        newBitmap = True
    else:
        bitmap = bitmap.copy()

    if len(bins) == 0:
        return bitmap, []

    positions: list[tuple[int, int, int, npt.NDArray, tuple[int, int]]] = []
    for groupid, mask, offset in bins:
        for j in range(w - mask.shape[1] + 1):
            for i in range(h - mask.shape[0] + 1):
                if not np.any(bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] & mask):
                    bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] |= mask
                    positions.append((i, j, groupid, mask, offset))
                    break
            else:
                continue
            break
        else:
            raise BitmapFullException('No space left')
    return bitmap, positions


def main():
    cap = cv2.VideoCapture('./jnc00.mp4')
    success, frame = cap.read()
    print(frame.shape)
    cv2.imwrite('frame.jpg', frame)

    with open(CONFIG) as fp:
        config = json.load(fp)
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('./modules/detectron2/configs', config['config']))
    cfg.MODEL.WEIGHTS = config['weights']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
    cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
    cfg.MODEL.DEVICE = 'cuda:1'
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture('jnc00.mp4')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tree = ElementTree.parse('masks.xml')
    mask = tree.getroot()
    mask_jnc00 = mask.find('.//image[@name="jnc00.jpg"]')
    assert mask_jnc00 is not None

    bitmask, mtl, mbr = get_bitmap(width, height, mask_jnc00)
    bitmask = torch.from_numpy(bitmask).to('cuda:1').to(torch.bool)

    if os.path.exists('./packed_images'):
        shutil.rmtree('./packed_images')

    os.mkdir('./packed_images')



    chunk_list = []
    pack_values = []
    LIMIT = 512
    pack_times = []
    num_detections = []
    num_bins = []
    bm = None
    canvas: "None | npt.NDArray" = None
    inference_time = []

    index_map = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE, 2), dtype=np.int32)
    det_info = {}

    frame_cache = {}
    with open('./track-results-0/jnc00.mp4.d.jsonl') as fp:
        # lines = fp.readlines()
        for idx in range(LIMIT):
            line = fp.readline()
            findex, fdetections = json.loads(line)
            # findex, fdetections = json.loads(lines[idx])
            # assert findex == _idx, (findex, _idx)
            # print(_idx)

            success, frame = cap.read()
            if not success:
                break

            # mask_frame(frame, mask)
            frame = torch.from_numpy(frame).to('cuda:1') * bitmask
            assert polyis.images.isHWC(frame), frame.shape

            height, width = frame.shape[:2]

            # pad so that the width and height are multiples of CHUNK_SIZE
            frame = polyis.images.padHWC(frame, CHUNK_SIZE, CHUNK_SIZE)
            patches = polyis.images.splitHWC(frame, CHUNK_SIZE, CHUNK_SIZE)


            frame = frame.detach().cpu().numpy()#.transpose((1, 2, 0))
            frame = frame.astype(np.uint8)
            # chunks = split_image(frame)

            num_detections.append(len(fdetections))
            fdetections = np.array(fdetections)
            fdetections[:, 0] += mtl[1]
            fdetections[:, 1] += mtl[0]
            fdetections[:, 2] += mtl[1]
            fdetections[:, 3] += mtl[0]

            start = time.time()

            # Place holder: TODO: need to replace with the proxy model.
            # Based on the content of the image, mark tiles in the image that contain detections.
            # Tiles do not overlap and placed in a grid.
            # Tile size is CHUNK_SIZE x CHUNK_SIZE
            # bitmap -> represent the grid of tiles, 0: tile without detection, 1: tile with detection
            bitmap = mark_detections2(fdetections, width, height)
            
            # Group tiles with detections together
            # tetrominoes -> list of tuple (group_id, mask, offset) representing the group of tiles (we will call a group of tile a tetromino)
            # - group_id: unique id of the group
            # - mask: masking of the tetromino. 1: tile is part of the tetromino, 0: tile is not part of the tetromino
            #         The mask is cropped to the minimum size that contains all the tiles of the tetromino
            # - offset: offset of the mask from the top left corner of the bitmap.
            polyominoes = group_tiles(bitmap)

            num_bins.append(len(polyominoes))
            frame_cache[idx] = frame
            
            try:
                # Sort tetrominoes by size (descending).
                polyominoes = sorted(polyominoes, key=lambda x: x[1].sum(), reverse=True)

                # Try packing tetrominoes from our proxy model.
                # Raise BitmapFullException if it fails.
                # bm -> bitmap representing occupied areas of the packed image. 0: available tile in the packed image (canvas), 1: occupied tile
                # positions -> list of (y, x, groupid, mask, offset) representing the position of the tetrominoes in the packed image
                # - y, x: top left corner of the tetrominoes in bm
                # - groupid: unique id of the group
                # - mask: masking of the tetromino. 1: tile is part of the tetromino, 0: tile is not part of the tetromino
                #         The mask is cropped to the minimum size that contains all the tiles of the tetromino
                # - offset: offset of the mask from the top left corner of the bitmap (that represents the original image -- not the packed image).
                bm, positions = pack_append(polyominoes, bitmap.shape[0], bitmap.shape[1], bm)
                if canvas is None:
                    # Initialize Canvas for packed image. Initially empty (black).
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Additively render the packed image from all the tetrominoes (positions) from the current frame.
                canvas = render(canvas, positions, frame)

                # Update index_map with the packed tetrominoes, marking the group id and the frame index that each tile belongs to.
                # Update det_info with the packed tetrominoes, storing the offset (_offset) of the tetrominoes in the original image and offset (x, y) of the tetrominoes in the packed image.
                for gid, (y, x, _groupid, mask, _offset) in enumerate(positions):
                    assert not np.any(index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] & mask)
                    index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * (gid + 1)
                    index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * idx
                    det_info[(int(idx), int(gid + 1))] = ((y, x), _offset)
            except BitmapFullException as e:
                # If the packed image is full, run detection on the packed image and start a new packed image.
                cv2.imwrite(f'./packed_images/{idx:03d}.jpg', canvas)

                _start = time.time()
                ## Start Detection --------------------------------
                _output = predictor(canvas)
                _bboxes, _scores, _ = parse_outputs(_output, (0, 0))
                nms_threshold = config['nms_threshold']
                nms_bboxes, nms_scores = nms(_bboxes, _scores, nms_threshold)
                detections = np.zeros((len(nms_bboxes), 5))
                detections[:, 0:4] = nms_bboxes
                detections[:, 4] = nms_scores
                ## End Detection ---------------------------------
                _end = time.time()

                inference_time.append((_end - _start) * 1000)
                print(idx, len(detections))

                for det in detections:
                    canvas = cv2.rectangle(canvas, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
                cv2.imwrite(f'./packed_images/{idx:03d}.d.jpg', canvas)

                # Iterate through all detections from the packed image.
                for det in detections:
                    # Get the group id and frame index of the tile that the detection belongs to.
                    _gid, _idx = index_map[
                        int((det[1] + det[3]) // (2 * CHUNK_SIZE)),
                        int((det[0] + det[2]) // (2 * CHUNK_SIZE)),
                    ]
                    print(det.astype(int))
                    if int(_gid) == 0:
                        # Blank tile
                        continue

                    # Get the offset of the tetromino in the original image and the packed image.
                    (y, x), _offset = det_info[(int(_idx), int(_gid))]
                    det = det.copy()
                    # Recover the bounding box of the detection in the original image.
                    det[[0, 2]] += (_offset[1] - x) * CHUNK_SIZE
                    det[[1, 3]] += (_offset[0] - y) * CHUNK_SIZE

                    f = frame_cache[int(_idx)]
                    # draw bounding box 
                    f = cv2.rectangle(f, (int(det[0]), int(det[1])), (int(det[2]), int(det[3]),), (0, 255, 0), 2)
                    frame_cache[_idx] = f
                
                for _idx, f in frame_cache.items():
                    cv2.imwrite(f'./unpacked_detections/{_idx:03d}.jpg', f)

                # Reset the packed image and start a new packed image.
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                det_info = {}
                frame_cache = {idx: frame_cache[idx]}
                index_map = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE, 2), dtype=np.int32)
                # Done reset ------------------------------------------------------------------------

                # Redo packing + rednering canvas + updating index_map and det_info
                bm, positions = pack_append(sorted(polyominoes, key=lambda x: x[1].sum(), reverse=True), bitmap.shape[0], bitmap.shape[1])
                canvas = render(canvas, positions, frame)
                for gid, (y, x, _groupid, mask, offset) in enumerate(positions):
                    index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * gid
                    index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * idx
                    det_info[(int(idx), int(gid))] = ((y, x), offset)

            end = time.time()
            pack_times.append((end - start) * 1000)


    pack_values.append(len(chunk_list))

    print(pack_values)
    print(sum(pack_times) / len(pack_times))


    t = [*range(len(pack_times))]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_ylabel('# of Bins', color=color)  # we already handled the x-label with ax1
    ax1.set_xlabel('frame')
    ax1.plot(t, num_bins, color=color)
    ax1.tick_params(axis='y', labelcolor=color)



    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Packing Time (ms)', color=color)
    ax2.plot(t, pack_times, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # plt.plot(range(len(pack_times)), pack_times)
    # plt.plot(range(len(pack_times)), num_detections)
    # plt.xlabel("X-axis")  # add X-axis label
    # plt.ylabel("Y-axis")  # add Y-axis label
    # plt.title("Any suitable title")  # add title
    # plt.show()
    plt.savefig('pack_times.png')


    print('inference time', sum(inference_time) / len(inference_time), 'ms')