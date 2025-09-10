import json
from queue import Queue
from typing import NamedTuple, TypeGuard, TypeVar, Any

import numpy as np
import numpy.typing as npt
import torch

import polyis.images
from polyis.dtypes import InPipe, NPImage, Array, D2, IdPolyominoOffset, OutPipe, FArray


# CHUNK_SIZE = 32


def mark_detections2(detections: "list[list[float]] | Array[*D2, np.floating]", width: int, height: int, chunk_size: int):
    bitmap: "Array[*D2, np.int32]" = np.zeros((height // chunk_size, width // chunk_size), dtype=np.int32)

    for bbox in detections:  # bboxes:
        xfrom, xto = int(bbox[0] // chunk_size), int(bbox[2] // chunk_size)
        yfrom, yto = int(bbox[1] // chunk_size), int(bbox[3] // chunk_size)

        bitmap[yfrom:yto+1, xfrom:xto+1] = 1
    
    return bitmap


def find_connected_tiles(bitmap: npt.NDArray, i: int, j: int):
    """
    Find all connected tiles in the bitmap starting from the tile at (i, j).

    Parameters:
    - bitmap: 2D numpy array representing the grid of tiles,
              where 1 indicates a tile with detection and 0 indicates no detection.
    - i: row index of the starting tile
    - j: column index of the starting tile

    Returns: a list of tuples representing the coordinates of all connected tiles.
    """
    value = bitmap[i, j]
    q = Queue()
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


def group_tiles(bitmap: Array[*D2, np.int32]):
    """
    Group groups of connected tiles into polyominoes.

    Parameters:
    - bitmap: 2D numpy array representing the grid of tiles,
              where 1 indicates a tile with detection and 0 indicates no detection.
    Returns: a list of polyominoes, where each polyomino is represented by list[(group_id, mask, offset)]
    - group_id: unique id of the group
    - mask:     masking of the polyomino as a 2D numpy array.
                    1: tile is part of the polyomino,
                    0: tile is not part of the polyomino
    - offset:   offset of the mask from the top left corner of the bitmap.
                the mask is cropped to the minimum size that contains all the tiles of the polyomino.
    """
    # TODO: use scipy.ndimage.label or skimage.measure.label for more efficient grouping of connected components.
    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int32) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * bitmap

    # Padding with size=1 on all sides
    groups = np.zeros((h + 2, w + 2), dtype=np.int32)
    groups[1:h+1, 1:w+1] = _groups

    visited: set[int] = set()
    bins: list[IdPolyominoOffset] = []
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            if groups[i, j] == 0 or groups[i, j] in visited:
                continue

            connected_tiles = find_connected_tiles(groups, i, j)
            connected_tiles = np.array(connected_tiles, dtype=int).T
            mask = np.zeros((h + 1, w + 1), dtype=np.bool)
            mask[*connected_tiles] = True

            offset = np.min(connected_tiles, axis=1)
            assert offset.shape == (2,)

            end = np.max(connected_tiles, axis=1) + 1
            assert end.shape == (2,)

            mask = mask[offset[0]:end[0], offset[1]:end[1]]
            assert is2D(mask), mask.shape
            bins.append((groups[i, j], mask, tuple(offset - 1)))

            visited.add(groups[i, j])
    return bins


class BitmapFullException(Exception):
    pass


P = TypeVar("P", bound=np.generic)
def is2D(x: Array[*tuple[int, ...], P]) -> TypeGuard[Array[*D2, P]]:
    return len(x.shape) == 2


class PolyominoInfo(NamedTuple):
    frame_idx: int
    frame: NPImage
    bitmap: Array[*D2, np.int32]
    polyominoes: list[IdPolyominoOffset]


def chunk(
    imgQueue: "InPipe[NPImage]",
    polyominoQueue: "OutPipe[PolyominoInfo]",
    groundtruth: str,
    chunk_size: int = 128,
    inv_sampling_rate: int = 1,
):
    fp = open(groundtruth, 'r')
    # fp = open('./track-results-0/jnc00.mp4.d.jsonl', 'r')
    flog = open('./chunker.py.log', 'w')
    idx = 0
    while True:
        frame = imgQueue.get()
        if frame is None:
            break
        line = fp.readline()
        flog.flush()
        findex, fdetections = json.loads(line)
        flog.write(f"{findex} {fdetections}\n")
        assert findex == idx, (findex, idx)

        if idx % inv_sampling_rate != 0:
            idx += 1
            continue

        frame = torch.from_numpy(frame).to('cuda:1')
        assert polyis.images.isHWC(frame), frame.shape

        height, width = frame.shape[:2]

        # pad so that the width and height are multiples of CHUNK_SIZE
        frame = polyis.images.padHWC(frame, chunk_size, chunk_size)
        patches = polyis.images.splitHWC(frame, chunk_size, chunk_size)


        frame = frame.detach().cpu().numpy()#.transpose((1, 2, 0))
        frame = frame.astype(np.uint8)
        # chunks = split_image(frame)

        # num_detections.append(len(fdetections))
        if len(fdetections) == 0:
            # No detections in the frame
            polyominoQueue.put(PolyominoInfo(idx, frame, np.zeros((height // chunk_size, width // chunk_size), dtype=np.int32), []))
            idx += 1
            continue
        fdetections = np.array(fdetections, dtype=np.float32)[:, -4:]
        assert is2D(fdetections), fdetections.shape
        # fdetections[:, 0] += mtl[1]
        # fdetections[:, 1] += mtl[0]
        # fdetections[:, 2] += mtl[1]
        # fdetections[:, 3] += mtl[0]

        # Place holder: TODO: need to replace with the proxy model.
        # Based on the content of the image, mark tiles in the image that contain detections.
        # Tiles do not overlap and placed in a grid.
        # Tile size is CHUNK_SIZE x CHUNK_SIZE
        # bitmap -> represent the grid of tiles, 0: tile without detection, 1: tile with detection
        bitmap = mark_detections2(fdetections, width, height, chunk_size)
        
        # Group tiles with detections together
        # tetrominoes -> list of tuple (group_id, mask, offset) representing the group of tiles (we will call a group of tile a tetromino)
        # - group_id: unique id of the group
        # - mask: masking of the tetromino. 1: tile is part of the tetromino, 0: tile is not part of the tetromino
        #         The mask is cropped to the minimum size that contains all the tiles of the tetromino
        # - offset: offset of the mask from the top left corner of the bitmap.
        polyominoes = group_tiles(bitmap)

        polyominoQueue.put(PolyominoInfo(idx, frame, bitmap, polyominoes))
        idx += 1

    polyominoQueue.put(None)
    flog.write('done\n')
    fp.close()
    flog.close()
