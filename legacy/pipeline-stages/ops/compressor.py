import os
from typing import Literal, NamedTuple
from queue import Queue

import cv2
import numpy as np
import numpy.typing as npt


from polyis.dtypes import S2, InPipe, NPImage, Array, D2, IdPolyominoOffset, OutPipe

from .chunker import PolyominoInfo


# CHUNK_SIZE = 128


class BitmapFullException(Exception):
    pass


def pack_append(
    bins: list[tuple[int, npt.NDArray, tuple[int, int]]],
    h: int,
    w: int,
    bitmap: "Array[*D2, np.bool] | None" = None
):
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


def render(canvas: NPImage, positions: list[tuple[int, int, int, npt.NDArray, tuple[int, int]]], frame: NPImage, chunk_size: int):
    for y, x, groupid, mask, offset in positions:
        yfrom, yto = y * chunk_size, (y + mask.shape[0]) * chunk_size
        xfrom, xto = x * chunk_size, (x + mask.shape[1]) * chunk_size

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    # patch = patches[i + offset[0], j + offset[1]]
                    patch = frame[
                        (i + offset[0]) * chunk_size:(i + offset[0] + 1) * chunk_size,
                        (j + offset[1]) * chunk_size:(j + offset[1] + 1) * chunk_size,
                    ]
                    # frame2[(i + offset[0]) * chunk_size:(i + offset[0] + 1) * chunk_size, (j + offset[1]) * chunk_size:(j + offset[1] + 1) * chunk_size] = ((
                    #     frame2[(i + offset[0]) * chunk_size:(i + offset[0] + 1) * chunk_size, (j + offset[1]) * chunk_size:(j + offset[1] + 1) * chunk_size].astype(np.uint32) +
                    #     (np.ones((chunk_size, chunk_size, 3), dtype=np.uint8) * colors[groupid % len(colors)]).astype(np.uint32)
                    # ) // 2).astype(np.uint8)
                    # cv2.imwrite(f'./packed_images/{idx:03d}.{i}.{j}.{groupid}.jpg', patch)
                    # canvas2[yfrom + (chunk_size * i): yfrom + (chunk_size * i) + chunk_size,
                    #         xfrom + (chunk_size * j): xfrom + (chunk_size * j) + chunk_size,
                    # ] += (np.ones((chunk_size, chunk_size, 3), dtype=np.uint8) * colors[groupid % len(colors)]).astype(np.uint8)
                    canvas[
                        yfrom + (chunk_size * i): yfrom + (chunk_size * i) + chunk_size,
                        xfrom + (chunk_size * j): xfrom + (chunk_size * j) + chunk_size,
                    ] += patch  # .detach().cpu().numpy()
    return canvas


FrameId_GroupId = tuple[int, int]
Offset = tuple[int, int]


class PolyominoMapping(NamedTuple):
    index_map: Array[int, int, S2, np.int32]
    det_info: dict[FrameId_GroupId, tuple[Offset, Offset]]
    frame_idx: int
    frame_cache: dict[int, NPImage]
    canvas: NPImage
    frame_range: slice


def compress(
    polyominoQueue: "InPipe[PolyominoInfo]",
    imgQueue: "OutPipe[NPImage]",
    mapQueue: "OutPipe[PolyominoMapping]",
    chunk_size: int = 128
):
    canvas: "NPImage | None" = None
    bm: Array[*D2, np.bool] | None = None
    index_map: Array[int, int, S2, np.int32] | None = None
    det_info: dict[FrameId_GroupId, tuple[Offset, Offset]] = dict()
    frame_cache = dict()
    start_idx = 0

    # if os.path.exists('./packed_images'):
    #     os.system('rm -rf ./packed_images')
    # os.makedirs('./packed_images', exist_ok=True)

    flog = open('./compressor.py.log', 'w')

    last_idx: int | None = None
    while True:
        polyominoes = polyominoQueue.get()
        if polyominoes is None:
            break

        idx, frame, bitmap, polyominoes = polyominoes
        last_idx = idx
        height, width = frame.shape[:2]
        if index_map is None:
            index_map = np.zeros((height // chunk_size, width // chunk_size, 2), dtype=np.int32)
        
        flog.write(f"{idx} {bitmap.shape} {bitmap.sum()}\n")
        flog.flush()
        
        frame_cache[idx] = frame

        try:
            # Sort polyominoes by size (descending).
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
            canvas = render(canvas, positions, frame, chunk_size)

            # Update index_map with the packed tetrominoes, marking the group id and the frame index that each tile belongs to.
            # Update det_info with the packed tetrominoes, storing the offset (_offset) of the tetrominoes in the original image
            #        and offset (x, y) of the tetrominoes in the packed image.
            for gid, (y, x, _groupid, mask, _offset) in enumerate(positions):
                assert not np.any(index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] & mask)
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * (gid + 1)
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * idx
                det_info[(int(idx), int(gid + 1))] = ((y, x), _offset)
        except BitmapFullException:
            # If the packed image is full, run detection on the packed image and start a new packed image.
            assert canvas is not None
            # cv2.imwrite(f'./packed_images/{idx:03d}.jpg', canvas)

            imgQueue.put(canvas)
            assert index_map is not None
            mapQueue.put(PolyominoMapping(index_map, det_info, idx, frame_cache, canvas, slice(start_idx, idx)))

            # Reset the packed image and start a new packed image -------------------------------
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            det_info = {}
            frame_cache = {idx: frame_cache[idx]}
            index_map = np.zeros((height // chunk_size, width // chunk_size, 2), dtype=np.int32)
            start_idx = idx
            # Done reset ------------------------------------------------------------------------

            # Redo packing + rednering canvas + updating index_map and det_info
            bm, positions = pack_append(sorted(polyominoes, key=lambda x: x[1].sum(), reverse=True), bitmap.shape[0], bitmap.shape[1])
            canvas = render(canvas, positions, frame, chunk_size)
            for gid, (y, x, _groupid, mask, offset) in enumerate(positions):
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 0] += mask.astype(np.int32) * gid
                index_map[y:y+mask.shape[0], x:x+mask.shape[1], 1] += mask.astype(np.int32) * idx
                det_info[(int(idx), int(gid))] = ((y, x), offset)
    
    assert canvas is not None
    assert last_idx is not None
    # cv2.imwrite(f'./packed_images/{last_idx+1:03d}.jpg', canvas)

    imgQueue.put(canvas)
    assert index_map is not None
    mapQueue.put(PolyominoMapping(index_map, det_info, last_idx+1, frame_cache, canvas, slice(start_idx, last_idx+1)))

    imgQueue.put(None)
    mapQueue.put(None)

    flog.write('done\n')
    flog.close()