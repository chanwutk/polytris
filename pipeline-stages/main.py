import queue
import  threading
import os
from typing import TypeVar, TypeAlias, Literal
import time
import json

import cv2
import numpy as np

import detector
import tracker
import chunker
import compressor
import uncompressor
import filter_patches


NPDType = TypeVar("NPDType", bound=np.generic)
D = tuple
_1D = D[int]
_2D = D[int, int]
_3D = D[int, int, int]
_4D = D[int, int, int, int]
_ShapeType = TypeVar("_ShapeType", bound=tuple[int, ...])
Array: TypeAlias = np.ndarray[_ShapeType, np.dtype[NPDType]]
NPImage = Array[D[int, int, Literal[3]], np.uint8]
IdPolyominoOffset = tuple[int, Array[_2D, np.bool], tuple[int, int]]


def save_track(trackQueue: queue.Queue, lu):  #, benchmarkQueue: queue.Queue):
    l, u = lu
    idx = 0
    with open(f'tracked_objects.{l}.{u}.jsonl', 'w') as f:
        while True:
            tracked_objects = trackQueue.get()

            if tracked_objects is None:
                break

            f.write(f"{json.dumps([idx, tracked_objects.astype(int).tolist()])}\n")
            f.flush()
            idx += 1


PACK_IMG_DIR = './packed_images'
UNPK_DET_DIR = './unpacked_detections'


def main():
    for l in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for width in [0.1, 0.2, 0.3]:
            u = l + width

            if os.path.exists( PACK_IMG_DIR):
                os.system('rm -rf ' + PACK_IMG_DIR)
            os.makedirs( PACK_IMG_DIR, exist_ok=True)

            if os.path.exists( UNPK_DET_DIR):
                os.system('rm -rf ' + UNPK_DET_DIR)
            os.makedirs( UNPK_DET_DIR, exist_ok=True)

            start = time.time()

            imgQueue = queue.Queue(maxsize=10)
            polQueue = queue.Queue(maxsize=10)
            pol2Queue = queue.Queue(maxsize=10)
            im2Queue = queue.Queue(maxsize=10)
            mapQueue = queue.Queue(maxsize=10)
            boxQueue = queue.Queue(maxsize=10)
            outboxQueue = queue.Queue(maxsize=10)
            trackQueue = queue.Queue(maxsize=10)
            benchmarkQueue = queue.Queue()

            chunkerThread = threading.Thread(
                target=chunker.chunk,
                args=(imgQueue, polQueue),
                daemon=True,
            )
            chunkerThread.start()
            print("Chunker thread started")

            filterPatchThread = threading.Thread(
                target=filter_patches.filter_patches,
                args=(benchmarkQueue, polQueue, pol2Queue, (l, u)),
                daemon=True,
            )
            filterPatchThread.start()

            compressorThread = threading.Thread(
                target=compressor.compress,
                args=(pol2Queue, im2Queue, mapQueue),
                daemon=True,
            )
            compressorThread.start()
            print("Compressor thread started")

            detectorThread = threading.Thread(
                target=detector.detect,
                args=(im2Queue, boxQueue, 'cuda:0'),
                daemon=True,
            )
            detectorThread.start()
            print("Detector thread started")

            uncompressorThread = threading.Thread(
                target=uncompressor.uncompress,
                args=(boxQueue, mapQueue, outboxQueue),
                daemon=True,
            )
            uncompressorThread.start()
            print("Uncompressor thread started")

            trackerThread = threading.Thread(
                target=tracker.track,
                args=(outboxQueue, trackQueue, benchmarkQueue, 35),
                daemon=True,
            )
            trackerThread.start()
            print("Tracker thread started")

            saveTrackThread = threading.Thread(
                target=save_track,
                args=(trackQueue, (l, u)),
                daemon=True,
            )
            saveTrackThread.start()

            cap = cv2.VideoCapture("./video-masked/jnc00.mp4")
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or idx >= 2000:
                    break
                print(idx)
                imgQueue.put(frame)
                idx += 1
            
            
            imgQueue.put(None)

            chunkerThread.join()
            print("Chunker thread finished")

            compressorThread.join()
            print("Compressor thread finished")

            detectorThread.join()
            print("Detector thread finished")

            uncompressorThread.join()
            print("Uncompressor thread finished")

            trackerThread.join()
            print("Tracker thread finished")

            filterPatchThread.join()
            print("Filter patches thread finished")

            saveTrackThread.join()
            print("All threads finished")

            end = time.time()
            print(f"Total time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()