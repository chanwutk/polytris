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
import minivan.images


CHUNK_SIZE = 128
CONFIG = './modules/b3d/configs/config_refined.json'
LIMIT = 512


TetrominoPosition = tuple[int, int, npt.NDArray, tuple[int, int]]


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    """
    Get the bitmap of the domain (relevant areas) from the mask.
    The bitmap is a binary image where 1 represents the domain and 0 represents the background.
    The bitmap is cropped to the minimum size that contains the domain.

    Args:
    - width: width of the original image
    - height: height of the original image
    - mask: ElementTree.Element polygon representing the mask
    """
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


def mark_detections(detections: npt.NDArray, width: int, height: int):
    bitmap = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE), dtype=np.int32)
    for bbox in detections:  # bboxes:
        xfrom, xto = int(bbox[0] // CHUNK_SIZE), int(bbox[2] // CHUNK_SIZE)
        yfrom, yto = int(bbox[1] // CHUNK_SIZE), int(bbox[3] // CHUNK_SIZE)

        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def fill_bitmap(bitmap: npt.NDArray, i: int, j: int):
    """
    Fill the connected component of the bitmap starting from (i, j) with the same value (value at bitmap[i, j]).
    Return the list of filled pixels.
    """
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
    """
    Group connected tiles together to form terominoes.
    Return a list of tuple (mask, offset) representing the tetrominoes.
    """
    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int32) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * bitmap

    # Padding with size=1 on all sides
    groups = np.zeros((h + 2, w + 2), dtype=np.int32)
    groups[1:h+1, 1:w+1] = _groups

    visited: set[int] = set()
    bins: list[tuple[npt.NDArray, tuple[int, int]]] = []
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
            bins.append((mask, tuple(offset - 1)))

            visited.add(groups[i, j])

    # bins: a list of tuple (mask, offset)
    #       mask: cropped mask of the group
    #       offset: offset of mask from the origin of the bitmap
    return bins

class BitmapFullException(Exception):
    pass


def pack_append(bins: list[tuple[npt.NDArray, tuple[int, int]]], h: int, w: int, bitmap: npt.NDArray | None = None):
    if bitmap is None:
        bitmap = np.zeros((h, w), dtype=np.bool)
    else:
        bitmap = bitmap.copy()

    if len(bins) == 0:
        return bitmap, []

    positions: list[TetrominoPosition] = []
    for mask, offset in bins:
        for j in range(w - mask.shape[1] + 1):
            for i in range(h - mask.shape[0] + 1):
                if not np.any(bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] & mask):
                    bitmap[i:i+mask.shape[0], j:j+mask.shape[1]] |= mask
                    positions.append((i, j, mask, offset))
                    break
            else:
                continue
            break
        else:
            raise BitmapFullException('No space left')
    return bitmap, positions


def render(canvas: npt.NDArray, positions: list[TetrominoPosition], frame: npt.NDArray):
    for y, x, mask, offset in positions:
        yfrom = y * CHUNK_SIZE
        xfrom = x * CHUNK_SIZE

        oy, ox = offset
        _canvas = canvas[yfrom:, xfrom:]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    patch = frame[(oy + i) * CHUNK_SIZE:, (ox + j) * CHUNK_SIZE:][:CHUNK_SIZE, :CHUNK_SIZE]
                    _canvas[(i * CHUNK_SIZE):, (j * CHUNK_SIZE):][:CHUNK_SIZE, :CHUNK_SIZE] += patch
    return canvas


def remap_detection(index_map: npt.NDArray, det_info: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], det: npt.NDArray):
    # Get the group id and frame index of the tile that the detection belongs to.
    gid, fid = index_map[
        int((det[1] + det[3]) // (2 * CHUNK_SIZE)),
        int((det[0] + det[2]) // (2 * CHUNK_SIZE)),
    ]
    if int(gid) == 0:
        # Detection is on a blank tile
        return None, None

    # Get the offset of the tetromino in the original image and the packed image.
    (y, x), offset = det_info[(int(fid), int(gid))]
    det = det.copy()

    # Recover the bounding box of the detection in the original image.
    det[[0, 2]] += (offset[1] - x) * CHUNK_SIZE
    det[[1, 3]] += (offset[0] - y) * CHUNK_SIZE
    return det, fid


def main():
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
    if os.path.exists('./unpacked_detections'):
        shutil.rmtree('./unpacked_detections')
    os.mkdir('./unpacked_detections')


    LIMIT = 512
    num_detections = []
    num_bins = []
    bm = None
    canvas: "None | npt.NDArray" = None
    inference_time = []

    index_map = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE, 2), dtype=np.int32)
    det_info = {}

    frame_cache = {}
    with open('./track-results-0/jnc00.mp4.d.jsonl') as fp:
        for idx in range(LIMIT):
            line = fp.readline()
            findex, fdetections = json.loads(line)
            assert findex == idx, (findex, idx)

            success, frame = cap.read()
            if not success:
                break

            frame = torch.from_numpy(frame).to('cuda:1') * bitmask
            frame = frame[mtl[0]:mbr[0], mtl[1]:mbr[1]]
            assert minivan.images.isHWC(frame), frame.shape

            height, width = frame.shape[:2]

            # pad so that the width and height are multiples of CHUNK_SIZE
            frame = minivan.images.padHWC(frame, CHUNK_SIZE, CHUNK_SIZE)
            patches = minivan.images.splitHWC(frame, CHUNK_SIZE, CHUNK_SIZE)


            frame = frame.detach().cpu().numpy()#.transpose((1, 2, 0))
            frame = frame.astype(np.uint8)
            # chunks = split_image(frame)

            num_detections.append(len(fdetections))
            fdetections = np.array(fdetections)
            # fdetections[:, 0] += mtl[1]
            # fdetections[:, 1] += mtl[0]
            # fdetections[:, 2] += mtl[1]
            # fdetections[:, 3] += mtl[0]

            # Place holder: TODO: replace with the proxy model
            # Based on the content of the image, mark tiles in the image that contain detections.
            # Tiles do not overlap and placed in a grid.
            # Tile size is CHUNK_SIZE x CHUNK_SIZE
            # bitmap -> represent the grid of tiles, 0: tile without detection, 1: tile with detection
            bitmap = mark_detections(fdetections, width, height)
            
            # Group tiles with detections together
            # tetrominoes -> list of tuple (group_id, mask, offset) representing the group of tiles (we will call a group of tile a tetromino)
            # - group_id: unique id of the group
            # - mask: masking of the tetromino. 1: tile is part of the tetromino, 0: tile is not part of the tetromino
            #         The mask is cropped to the minimum size that contains all the tiles of the tetromino
            # - offset: offset of the mask from the top left corner of the bitmap.
            tetrominoes = group_tiles(bitmap)

            num_bins.append(len(tetrominoes))
            frame_cache[idx] = frame
            
            tetrominoes = sorted(tetrominoes, key=lambda x: x[0].sum(), reverse=True)
            try:
                # Try packing tetrominoes from our proxy model.
                # Raise BitmapFullException if it fails.
                # bm -> bitmap representing occupied areas of the packed image. 0: available tile in the packed image (canvas), 1: occupied tile
                # positions -> list of (y, x, groupid, mask, offset) representing the position of the tetrominoes in the packed image
                # - y, x: top left corner of the tetrominoes in bm
                # - groupid: unique id of the group
                # - mask: masking of the tetromino. 1: tile is part of the tetromino, 0: tile is not part of the tetromino
                #         The mask is cropped to the minimum size that contains all the tiles of the tetromino
                # - offset: offset of the mask from the top left corner of the bitmap (that represents the original image -- not the packed image).
                bm, positions = pack_append(tetrominoes, bitmap.shape[0], bitmap.shape[1], bm)
            except BitmapFullException as e:
                # If the packed image is full, run detection on the packed image and start a new packed image.
                assert canvas is not None
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
                # print(idx, len(detections))

                # Draw bounding boxes on the packed image.
                for det in detections:
                    canvas = cv2.rectangle(canvas, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
                cv2.imwrite(f'./packed_images/{idx:03d}.d.jpg', canvas)

                # Iterate through all detections from the packed image.
                for det in detections:
                    det, fid = remap_detection(index_map, det_info, det)

                    if det is None:
                        # det is on a blank tile. Skip.
                        # TODO: rivisit because the object detector incorrectly detect objects on blank tiles.
                        continue
                    assert fid is not None

                    f = frame_cache[int(fid)]
                    # draw bounding box 
                    f = cv2.rectangle(f, (int(det[0]), int(det[1])), (int(det[2]), int(det[3]),), (0, 255, 0), 2)
                    frame_cache[int(fid)] = f

                for _idx, f in frame_cache.items():
                    cv2.imwrite(f'./unpacked_detections/{_idx:03d}.jpg', f)

                # Reset the packed image and start a new packed image.
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                det_info = {}
                frame_cache = {idx: frame_cache[idx]}
                index_map = np.zeros((height // CHUNK_SIZE, width // CHUNK_SIZE, 2), dtype=np.int32)

                bm, positions = pack_append(tetrominoes, bitmap.shape[0], bitmap.shape[1])

            if canvas is None:
                # Initialize Canvas for packed image. Initially empty (black).
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Additively render the packed image from all the tetrominoes (positions) from the current frame.
            canvas = render(canvas, positions, frame)

            # Update index_map with the packed tetrominoes, marking the group id and the frame index that each tile belongs to.
            # Update det_info with the packed tetrominoes, storing the offset (_offset) of the tetrominoes in the original image and offset (x, y) of the tetrominoes in the packed image.
            for gid, (y, x, mask, _offset) in enumerate(positions):
                assert not np.any(index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] & mask)
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * (gid + 1)
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * idx
                det_info[(int(idx), int(gid + 1))] = ((y, x), _offset)

    print('inference time', sum(inference_time) / len(inference_time), 'ms')


if __name__ == '__main__':
    main()