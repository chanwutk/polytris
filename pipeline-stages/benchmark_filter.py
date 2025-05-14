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
import ops.filter_overlap
import ops.filter_frames


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
LIMIT = 1024



def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def save_track(trackQueue: queue.Queue, filename: str):  #, benchmarkQueue: queue.Queue):
    idx = 0
    with open(filename, 'w') as f:
        while True:
            tracked_objects = trackQueue.get()

            if tracked_objects is None:
                break

            f.write(f"{json.dumps([idx, tracked_objects.astype(int).tolist()])}\n")
            f.flush()
            idx += 1


edgeRegions = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 15), (0, 16),
    (1, 15), (1, 16),
    (2, 15), (2, 16),
    (7, 7), (7, 8),
    (8, 7), (8, 8),
    (6, 15), (6, 16),
    (7, 15), (7, 16),
    (8, 15), (8, 16),
    (9, 15),
]


def noOp():
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
        args=(trackQueue, None, './tracking_results/tracked_no_op.jsonl'),
        daemon=True,
    )
    interpolateThread.start()
    print("Interpolate thread started")

    # filteredThread = threading.Thread(
    #     target=ops.filter_overlap.filter_overlap,
    #     args=(itrackQueue, None, edgeRegions, './tracking_results/tracked_no_op_ofiltered.jsonl'),
    #     daemon=True,
    # )
    # filteredThread.start()
    # print("Filter thread started")

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

    # filteredThread.join()
    # print("Filter thread finished")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    with open('./tracking_results/runtime_no_op.json', 'w') as f:
        f.write(json.dumps({"time": end - start, "fps": LIMIT / (end - start), "frames": LIMIT}))


def op():
    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    polQueue = queue.Queue(maxsize=10)
    im2Queue = queue.Queue(maxsize=10)
    mapQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    outboxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
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
        args=(outboxQueue, trackQueue, None, 35),
        daemon=True,
    )
    trackerThread.start()
    print("Tracker thread started")

    interpolateThread = threading.Thread(
        target=ops.interpolator.interpolate,
        args=(trackQueue, None, './tracking_results/tracked_op.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    # filteredThread = threading.Thread(
    #     target=ops.filter_overlap.filter_overlap,
    #     args=(itrackQueue, None, edgeRegions, './tracking_results/tracked_op_ofiltered.jsonl'),
    #     daemon=True,
    # )
    # filteredThread.start()
    # print("Filter thread started")

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
    print("Interpolate thread finished")

    # filteredThread.join()
    # print("Filter thread finished")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    with open('./tracking_results/runtime_op.json', 'w') as f:
        f.write(json.dumps({"time": end - start, "fps": LIMIT / (end - start), "frames": LIMIT}))


def opFilter(tl: float, th: float, isr: int = 1):
    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    polQueue = queue.Queue(maxsize=10)
    outPolQueue = queue.Queue(maxsize=10)
    im2Queue = queue.Queue(maxsize=10)
    mapQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    outboxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
    itrackQueue = queue.Queue()
    benchmarkQueue = queue.Queue()

    chunkerThread = threading.Thread(
        target=ops.chunker.chunk,
        args=(imgQueue, polQueue, isr),
        daemon=True,
    )
    chunkerThread.start()
    print("Chunker thread started")

    filterThread = threading.Thread(
        target=ops.filter_patches.filter_patches,
        args=(benchmarkQueue, polQueue, outPolQueue, (tl, th)),
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
        args=(trackQueue, None, f'./tracking_results/tracked_opFilter_{tl}_{th}_{isr}_.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    # filteredThread = threading.Thread(
    #     target=ops.filter_overlap.filter_overlap,
    #     args=(itrackQueue, None, edgeRegions, f'./tracking_results/tracked_opFilter_{tl}_{th}_ofiltered.jsonl'),
    #     daemon=True,
    # )
    # filteredThread.start()
    # print("Filter thread started")

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

    # filteredThread.join()
    # print("Filter thread finished")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    with open(f'./tracking_results/runtime_opFilter_{tl}_{th}_{isr}_.json', 'w') as f:
        f.write(json.dumps({"time": end - start, "fps": LIMIT / (end - start), "frames": LIMIT}))


def opFrameFilter(isr: int):
    start = time.time()

    imgQueue = queue.Queue(maxsize=10)
    polQueue = queue.Queue(maxsize=10)
    outPolQueue = queue.Queue(maxsize=10)
    im2Queue = queue.Queue(maxsize=10)
    mapQueue = queue.Queue(maxsize=10)
    boxQueue = queue.Queue(maxsize=10)
    outboxQueue = queue.Queue(maxsize=10)
    trackQueue = queue.Queue(maxsize=10)
    itrackQueue = queue.Queue()
    benchmarkQueue = queue.Queue()

    chunkerThread = threading.Thread(
        target=ops.chunker.chunk,
        args=(imgQueue, polQueue, isr),
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
        args=(trackQueue, None, f'./tracking_results/tracked_opFrameFilter_{isr}_.jsonl'),
        daemon=True,
    )
    interpolateThread.start()

    # filteredThread = threading.Thread(
    #     target=ops.filter_overlap.filter_overlap,
    #     args=(itrackQueue, None, edgeRegions, f'./tracking_results/tracked_opFrameFilter_{isr}_ofiltered.jsonl'),
    #     daemon=True,
    # )
    # filteredThread.start()
    # print("Filter thread started")

    cap = cv2.VideoCapture( os.path.join(VIDEO_MASK, 'jnc00.mp4'))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= LIMIT:
            break
        # print(idx)
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
    print("Interpolate thread finished")

    # filteredThread.join()
    # print("Filter thread finished")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    with open(f'./tracking_results/runtime_opFrameFilter_{isr}_.json', 'w') as f:
        f.write(json.dumps({"time": end - start, "fps": LIMIT / (end - start), "frames": LIMIT}))



if __name__ == '__main__':
    if not os.path.exists('./tracking_results'):
        os.makedirs('./tracking_results')

    noOp()
    op()
    thresholds: list[tuple[float, float]] = [
        # gap = 0.2
        (0.7, 0.9),
        # (0.6, 0.8),
        # (0.5, 0.7),
        # (0.4, 0.6),
        (0.3, 0.5),

        # gap = 0.3
        (0.6, 0.9),
        # (0.5, 0.8),
        # (0.4, 0.7),
        (0.3, 0.6),

        # gap = 0.4
        (0.5, 0.9),
        # (0.4, 0.8),
        (0.3, 0.7),

        # gap = 0.5
        (0.4, 0.9),
        (0.3, 0.8),

        # gap = 0.6
        (0.3, 0.9),
    ]
    inv_sampling_rates: list[int] = [1, 2, 4, 8, 16]
    for tl, th in thresholds:
        for isr in inv_sampling_rates:
            print(f"Running opFilter with thresholds: {tl}, {th} and inv_sampling_rate: {isr}")
            opFilter(tl, th, isr)
            print(f"Finished opFilter with thresholds: {tl}, {th} and inv_sampling_rate: {isr}")
    
    inv_sampling_rates: list[int] = [1, 2, 4, 8, 16, 32, 64]
    for isr in inv_sampling_rates:
        print(f"Running opFrameFilter with inv_sampling_rate: {isr}")
        opFrameFilter(isr)
        print(f"Finished opFrameFilter with inv_sampling_rate: {isr}")