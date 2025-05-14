import os
from typing import TypeVar, TypeAlias
from queue import Queue

import numpy as np
import numpy.typing as npt
import cv2

from minivan.dtypes import S5, DetArray, InPipe, NPImage, Array, D2, IdPolyominoOffset, OutPipe, is_det_array, is_np_image
from .compressor import PolyominoMapping

# CHUNK_SIZE = 128


def uncompress(
    bboxQueue: "InPipe[DetArray]",
    mapQueue: "InPipe[PolyominoMapping]",
    outbboxQueue: "OutPipe[tuple[int, DetArray]]",
    chunk_size: int = 128,
):
    flog = open('uncompressor.py.log', 'w')
    # if os.path.exists('./unpacked_detections'):
    #     os.system('rm -rf ./unpacked_detections')
    # os.makedirs('./unpacked_detections', exist_ok=True)

    while True:
        compress_info = mapQueue.get()
        detections = bboxQueue.get()

        assert (compress_info is None) == (detections is None), (compress_info, detections)
        if detections is None or compress_info is None:
            break

        index_map, det_info, idx, frame_cache, canvas, idx_range = compress_info

        flog.write(f"Uncompressing {idx_range}...\n")
        flog.flush()


        # for det in detections:
        #     canvas = cv2.rectangle(canvas, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
        # cv2.imwrite(f'./packed_images/{idx:03d}.d.jpg', canvas)

        multiframes_detections: dict[int, list[npt.NDArray]] = {}

        # Iterate through all detections from the packed image.
        for det in detections:
            # Get the group id and frame index of the tile that the detection belongs to.
            _gid, _idx = index_map[
                int((det[1] + det[3]) // (2 * chunk_size)),
                int((det[0] + det[2]) // (2 * chunk_size)),
            ]
            _gid = int(_gid)
            _idx = int(_idx)
            if _gid == 0:
                # Blank tile
                continue

            # Get the offset of the tetromino in the original image and the packed image.
            (y, x), _offset = det_info[(_idx, _gid)]
            det = det.copy()
            # Recover the bounding box of the detection in the original image.
            det[[0, 2]] += (_offset[1] - x) * chunk_size
            det[[1, 3]] += (_offset[0] - y) * chunk_size

            if _idx not in multiframes_detections:
                multiframes_detections[_idx] = []
            
            multiframes_detections[_idx].append(det)

            f = frame_cache[_idx]
            # draw bounding box 
            f = cv2.rectangle(f, (int(det[0]), int(det[1])), (int(det[2]), int(det[3]),), (0, 255, 0), 2)
            assert is_np_image(f), f.shape
            frame_cache[_idx] = f
        
        for i in range(idx_range.start, idx_range.stop):
            if i not in multiframes_detections:
                # No detections in this frame
                outbboxQueue.put((i, np.empty((0, 5))))
            else:
                dets = np.array(multiframes_detections[i])
                assert is_det_array(dets), dets.shape
                outbboxQueue.put((i, dets))
        
        # for _idx, f in frame_cache.items():
        #     cv2.imwrite(f'./unpacked_detections/{_idx:03d}.jpg', f)
    
    outbboxQueue.put(None)

    flog.write('done\n')
    flog.close()

