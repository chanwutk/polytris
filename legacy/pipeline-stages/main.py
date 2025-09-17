import queue
import  threading
import os
from typing import TypeVar, TypeAlias, Literal
import time
import json

import cv2
import numpy as np

import ops.detector
import ops.tracker
import ops.chunker
import ops.compressor
import ops.uncompressor
import ops.filter_patches
import ops.interpolator


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


VIDEO_MASK = './video-masked'
LIMIT = 512



def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


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


def save_video(trackQueue: queue.Queue, infile: str, filename: str):
    cap = cv2.VideoCapture(infile)
    writer: cv2.VideoWriter | None = None
    fidx = 0
    while cap.isOpened():
        print('------------', fidx)
        track = trackQueue.get()
        if track is None:
            break
        print(track)

        try:
            ret, frame = cap.read()
            if not ret:
                break
        except:
            break
        print(frame.shape)

        if frame is None:
            break

        # fidx, frame = frame
        tidx, track = track
        assert tidx == fidx, (tidx, fidx)

        width, height = frame.shape[1], frame.shape[0]

        if writer is None:
            writer = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*'mp4v'), 30, (width, height))

        for oid, *box in track:
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[oid % len(colors)], 2)
            frame = cv2.putText(frame, str(oid), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[oid % len(colors)], 2)
    
        writer.write(frame)
        fidx += 1
    
    if writer is not None:
        writer.release()


PACK_IMG_DIR = './packed_images'
UNPK_DET_DIR = './unpacked_detections'


def noOp():
    if os.path.exists(PACK_IMG_DIR + 'no_op'):
        os.system('rm -rf ' + PACK_IMG_DIR + 'no_op')
    os.makedirs(PACK_IMG_DIR + 'no_op', exist_ok=True)

    if os.path.exists(UNPK_DET_DIR + 'no_op'):
        os.system('rm -rf ' + UNPK_DET_DIR + 'no_op')
    os.makedirs(UNPK_DET_DIR + 'no_op', exist_ok=True)

    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
    itrackQueue = queue.Queue()


    detectorThread = threading.Thread(
        target=ops.detector.detectIdx,
        args=(imgQueue, boxQueue, 'cuda:0'),
        daemon=True,
    )
    detectorThread.start()
    print("Detector thread started")

    trackerThread = threading.Thread(
        target=ops.tracker.track,
        args=(boxQueue, trackQueue, None, 35),
        daemon=True,
    )
    trackerThread.start()
    print("Tracker thread started")

    interpolateThread = threading.Thread(
        target=ops.interpolator.interpolate,
        args=(trackQueue, itrackQueue, './tracked_no_op.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    saveVideoThread = threading.Thread(
        target=save_video,
        args=(itrackQueue, os.path.join(VIDEO_MASK, 'jnc00.mp4'), 'tracked_no_op.mp4'),
        daemon=True,
    )
    saveVideoThread.start()

    cap = cv2.VideoCapture( os.path.join(VIDEO_MASK, 'jnc00.mp4'))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= LIMIT:
            break
        print(idx)
        imgQueue.put((idx, frame))
        idx += 1
    
    
    imgQueue.put(None)

    detectorThread.join()
    print("Detector thread finished")

    trackerThread.join()
    print("Tracker thread finished")

    interpolateThread.join()
    print("Interpolate thread finished")

    saveVideoThread.join()

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")


def op():
    if os.path.exists(PACK_IMG_DIR + 'op'):
        os.system('rm -rf ' +PACK_IMG_DIR + 'op')
    os.makedirs(PACK_IMG_DIR + 'op', exist_ok=True)

    if os.path.exists(UNPK_DET_DIR + 'op'):
        os.system('rm -rf ' +UNPK_DET_DIR + 'op')
    os.makedirs(UNPK_DET_DIR + 'op', exist_ok=True)

    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    polQueue = queue.Queue(maxsize=10)
    im2Queue = queue.Queue(maxsize=10)
    mapQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    outboxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
    benchmarkQueue = queue.Queue()
    itrackQueue = queue.Queue()

    chunkerThread = threading.Thread(
        target=ops.chunker.chunk,
        args=(imgQueue, polQueue),
        daemon=True,
    )
    chunkerThread.start()
    print("Chunker thread started")

    compressorThread = threading.Thread(
        target=ops.compressor.compress,
        args=(polQueue, im2Queue, mapQueue),
        daemon=True,
    )
    compressorThread.start()
    print("Compressor thread started")

    detectorThread = threading.Thread(
        target=ops.detector.detect,
        args=(im2Queue, boxQueue, 'cuda:0'),
        daemon=True,
    )
    detectorThread.start()
    print("Detector thread started")

    uncompressorThread = threading.Thread(
        target=ops.uncompressor.uncompress,
        args=(boxQueue, mapQueue, outboxQueue),
        daemon=True,
    )
    uncompressorThread.start()
    print("Uncompressor thread started")

    trackerThread = threading.Thread(
        target=ops.tracker.track,
        args=(outboxQueue, trackQueue, benchmarkQueue, 35),
        daemon=True,
    )
    trackerThread.start()
    print("Tracker thread started")

    interpolateThread = threading.Thread(
        target=ops.interpolator.interpolate,
        args=(trackQueue, itrackQueue, './tracked_op.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    saveVideoThread = threading.Thread(
        target=save_video,
        args=(itrackQueue, os.path.join(VIDEO_MASK, 'jnc00.mp4'), 'tracked_op.mp4'),
        daemon=True,
    )
    saveVideoThread.start()

    cap = cv2.VideoCapture( os.path.join(VIDEO_MASK, 'jnc00.mp4'))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= LIMIT:
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

    interpolateThread.join()

    saveVideoThread.join()

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")


def opFilter():
    if os.path.exists(PACK_IMG_DIR + 'op'):
        os.system('rm -rf ' +PACK_IMG_DIR + 'op')
    os.makedirs(PACK_IMG_DIR + 'op', exist_ok=True)

    if os.path.exists(UNPK_DET_DIR + 'op'):
        os.system('rm -rf ' +UNPK_DET_DIR + 'op')
    os.makedirs(UNPK_DET_DIR + 'op', exist_ok=True)

    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    polQueue = queue.Queue(maxsize=10)
    outPolQueue = queue.Queue(maxsize=10)
    im2Queue = queue.Queue(maxsize=10)
    mapQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    outboxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
    benchmarkQueue = queue.Queue()
    itrackQueue = queue.Queue()

    chunkerThread = threading.Thread(
        target=ops.chunker.chunk,
        args=(imgQueue, polQueue),
        daemon=True,
    )
    chunkerThread.start()
    print("Chunker thread started")

    filterThread = threading.Thread(
        target=ops.filter_patches.filter_patches,
        args=(benchmarkQueue, polQueue, outPolQueue),
        daemon=True,
    )
    filterThread.start()
    print("Filter thread started")

    compressorThread = threading.Thread(
        target=ops.compressor.compress,
        args=(outPolQueue, im2Queue, mapQueue),
        daemon=True,
    )
    compressorThread.start()
    print("Compressor thread started")

    detectorThread = threading.Thread(
        target=ops.detector.detect,
        args=(im2Queue, boxQueue, 'cuda:0'),
        daemon=True,
    )
    detectorThread.start()
    print("Detector thread started")

    uncompressorThread = threading.Thread(
        target=ops.uncompressor.uncompress,
        args=(boxQueue, mapQueue, outboxQueue),
        daemon=True,
    )
    uncompressorThread.start()
    print("Uncompressor thread started")

    trackerThread = threading.Thread(
        target=ops.tracker.track,
        args=(outboxQueue, trackQueue, benchmarkQueue, 35),
        daemon=True,
    )
    trackerThread.start()
    print("Tracker thread started")

    interpolateThread = threading.Thread(
        target=ops.interpolator.interpolate,
        args=(trackQueue, itrackQueue, './tracked_opFilter.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    saveVideoThread = threading.Thread(
        target=save_video,
        args=(itrackQueue, os.path.join(VIDEO_MASK, 'jnc00.mp4'), 'tracked_opFilter.mp4'),
        daemon=True,
    )
    saveVideoThread.start()

    cap = cv2.VideoCapture( os.path.join(VIDEO_MASK, 'jnc00.mp4'))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= LIMIT:
            break
        print(idx)
        imgQueue.put(frame)
        idx += 1
    
    
    imgQueue.put(None)

    chunkerThread.join()
    print("Chunker thread finished")

    filterThread.join()
    print("Filter thread finished")

    compressorThread.join()
    print("Compressor thread finished")

    detectorThread.join()
    print("Detector thread finished")

    uncompressorThread.join()
    print("Uncompressor thread finished")

    trackerThread.join()
    print("Tracker thread finished")

    interpolateThread.join()
    print("Interpolate thread finished")

    saveVideoThread.join()
    print("Save video thread finished")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == '__main__':
    opFilter()
    # op()
    # noOp()