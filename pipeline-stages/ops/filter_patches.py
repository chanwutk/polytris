import json
from queue import Queue
import time

import cv2
import numpy as np

from minivan.dtypes import NPImage, Array, D2, IdPolyominoOffset


CHUNK_SIZE = 128
PRINT_SIZE = 128


# def draw(bm, max_value: float | int = 64):
#     frame = np.zeros((bm.shape[0] * PRINT_SIZE, bm.shape[1] * PRINT_SIZE, 3), dtype=np.uint8)

#     for y in range(bm.shape[0]):
#         for x in range(bm.shape[1]):
#             if bm[y, x] < max_value:
#                 val = bm[y, x]
#                 cv2.rectangle(
#                     frame,
#                     (x * PRINT_SIZE, y * PRINT_SIZE),
#                     ((x + 1) * PRINT_SIZE, (y + 1) * PRINT_SIZE),
#                     (int((max_value - val) * 255 / max_value), int((max_value - val) * 255 / max_value), 255),
#                     -1,
#                 )
#                 # write bm[y, x] to the center of the rectangle
#                 text = f"{int(bm[y, x])}"
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 scale = 0.5
#                 thickness = 2
#                 textsize = cv2.getTextSize(text, font, scale, thickness)[0]
#                 cv2.putText(
#                     frame,
#                     f"{int(bm[y, x])}",
#                     (x * PRINT_SIZE + PRINT_SIZE // 2 - (textsize[0] // 2), y * PRINT_SIZE + PRINT_SIZE // 2 + (textsize[1] // 2)),
#                     font,
#                     scale,
#                     (255, 255, 255),
#                     thickness,
#                 )
#     return frame


def filter_patches(
    benchQueue: Queue,
    polyominoQueue: "Queue[tuple[int, NPImage, Array[*D2, np.int32], list[IdPolyominoOffset]] | None]",
    outPolyominoQueue: "Queue[tuple[int, NPImage, Array[*D2, np.int32], list[IdPolyominoOffset]] | None]",
    iou_thresholds: tuple[float, float] = (0.7, 0.9),
):
    iou_threshold_l, iou_threshold_u = iou_thresholds
    assert iou_threshold_u > iou_threshold_l, "Upper IOU threshold must be greater than lower IOU threshold."
    last_frame_map = None
    skip_map = None
    low = None

    flog = open("filter_patches.py.log", "w")

    max_skip = 0
    prune_count = 0

    lfm_writer = None
    skp_writer = None
    low_writer = None
    draw_low = None

    patches_count = 0
    patches_dropped = 0

    evaluate_time = 0.
    filter_time = 0.

    while True:
        polyomino = polyominoQueue.get()
        if polyomino is None:
            break
        idx, frame, bitmap, polyominoes = polyomino
        flog.write(f"Processing frame {idx}...\n")
        flog.flush()

        if last_frame_map is None:
            last_frame_map = np.zeros_like(bitmap, dtype=np.int32)
        # if lfm_writer is None:
        #     lfm_writer = cv2.VideoWriter(f"./tracking_results/filtervis_last_frame_map_{iou_threshold_l}_{iou_threshold_u}_.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 30, (bitmap.shape[1] * PRINT_SIZE, bitmap.shape[0] * PRINT_SIZE))

        if skip_map is None:
            skip_map = np.ones_like(bitmap, dtype=np.int32)
        # if skp_writer is None:
        #     skp_writer = cv2.VideoWriter(f"./tracking_results/filtervis_tmp_skip_map_{iou_threshold_l}_{iou_threshold_u}_.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 30, (bitmap.shape[1] * PRINT_SIZE, bitmap.shape[0] * PRINT_SIZE))

        if low is None:
            low = np.zeros_like(bitmap, dtype=np.float32)
        # if low_writer is None:
        #     low_writer = cv2.VideoWriter(f"./tracking_results/filtervis_tmp_low_{iou_threshold_l}_{iou_threshold_u}_.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 30, (bitmap.shape[1] * PRINT_SIZE, bitmap.shape[0] * PRINT_SIZE))
        
        patches_count += len(polyominoes)
        if idx < 50:
            outPolyominoQueue.put(polyomino)
        else:
            if idx % 32 == 0:
                # update iou_map
                benchmarks = []
                while benchQueue.qsize() > 0:
                    benchmark = benchQueue.get()
                    if benchmark is None:
                        break
                    benchmarks.append(benchmark)

                start = time.time()
                mask = np.zeros_like(bitmap, dtype=np.int8)
                low = np.ones_like(bitmap, dtype=np.float32)
                for benchmark in benchmarks:
                    for bbox, score in benchmark:
                        xfrom, xto = int(bbox[0] // CHUNK_SIZE), int(bbox[2] // CHUNK_SIZE)
                        yfrom, yto = int(bbox[1] // CHUNK_SIZE), int(bbox[3] // CHUNK_SIZE)

                        mask[yfrom:yto+1, xfrom:xto+1] = 1
                        low[yfrom:yto+1, xfrom:xto+1] = np.fmin(
                            low[yfrom:yto+1, xfrom:xto+1],
                            np.ones_like(low[yfrom:yto+1, xfrom:xto+1]) * score,
                        )
                
                decrease = (low < iou_threshold_l) & mask
                increase = (low > iou_threshold_u) & mask
                skip_map = np.where(decrease, skip_map // 2, np.where(increase, skip_map * 2, skip_map))
                skip_map = np.clip(skip_map, 1, 32).astype(np.int32)  # Ensure skip_map stays within bounds
                max_skip = int(np.max(skip_map))
                # draw_low = draw((low * 100).astype(np.int32), max_value=100)
                # for benchmark in benchmarks:
                #     for bbox, score in benchmark:
                #         xfrom, xto = int(bbox[0] // CHUNK_SIZE), int(bbox[2] // CHUNK_SIZE)
                #         yfrom, yto = int(bbox[1] // CHUNK_SIZE), int(bbox[3] // CHUNK_SIZE)

                #         cv2.rectangle(
                #             draw_low,
                #             (int(bbox[0] * PRINT_SIZE / CHUNK_SIZE), int(bbox[1] * PRINT_SIZE / CHUNK_SIZE)),
                #             (int(bbox[2] * PRINT_SIZE / CHUNK_SIZE), int(bbox[3] * PRINT_SIZE / CHUNK_SIZE)),
                #             (0, 255, 0),
                #             2,
                #         )
                end = time.time()
                evaluate_time += end - start
            
            start = time.time()
            included_polyominoes = []
            for p in polyominoes:
                pid, poly, offset = p
                _y, _x = offset
                to_sample = False
                for y in range(poly.shape[0]):
                    if to_sample:
                        break
                    for x in range(poly.shape[1]):
                        if poly[y, x]:
                            if skip_map[y + _y, x + _x] <= idx - last_frame_map[y + _y, x + _x]:
                                to_sample = True
                                included_polyominoes.append(p)
                                break
                if not to_sample:
                    prune_count += 1
                
                if to_sample:
                    for y in range(poly.shape[0]):
                        for x in range(poly.shape[1]):
                            if poly[y, x]:
                                last_frame_map[y + _y, x + _x] = idx
            end = time.time()
            filter_time += end - start
            outPolyominoQueue.put((idx, frame, bitmap, included_polyominoes))
            patches_dropped += len(polyominoes) - len(included_polyominoes)
            # outPolyominoQueue.put(polyomino)

        # skp_writer.write(draw(skip_map, max_value=33))
        # lfm_writer.write(draw(idx - last_frame_map))
        # if draw_low is None:
        #     draw_low = draw(np.zeros_like(bitmap, dtype=np.uint8))
        # low_writer.write(draw_low)
    outPolyominoQueue.put(None)

    # assert skp_writer is not None
    # skp_writer.release()

    # assert lfm_writer is not None
    # lfm_writer.release()

    # assert low_writer is not None
    # low_writer.release()

    flog.write("Filter patches finished.\n")
    flog.close()

    # with open(f"filter_benchmark.{iou_threshold_l}.{iou_threshold_u}.log", 'w') as f:
    #     f.write(f"Max skip value: {max_skip}\n")
    #     f.write(f"Pruned polyominoes count: {prune_count}\n")
    #     f.flush()
    
    with open(f"./tracking_results/filter_patches_benchmark_{iou_threshold_l}_{iou_threshold_u}_.json", 'w') as f:
        f.write(json.dumps({"patches_count": patches_count, "patches_dropped": patches_dropped, "evaluation_time": evaluate_time, "filter_time": filter_time}, indent=2))