from queue import Queue

import numpy as np

from minivan.dtypes import NPImage, Array, D2, IdPolyominoOffset


CHUNK_SIZE = 128


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

    flog = open("filter_patches.py.log", "w")

    max_skip = 0
    prune_count = 0

    while True:
        poolyomino = polyominoQueue.get()
        if poolyomino is None:
            break
        idx, frame, bitmap, polyominoes = poolyomino
        flog.write(f"Processing frame {idx}...\n")
        flog.flush()

        if last_frame_map is None:
            last_frame_map = np.zeros_like(bitmap, dtype=np.int32)
        if skip_map is None:
            skip_map = np.ones_like(bitmap, dtype=np.int32)
        
        if idx < 50 or idx > 1950:
            outPolyominoQueue.put(poolyomino)
        else:
            if idx % 32 == 0:
                # update iou_map
                benchmarks = []
                while benchQueue.qsize() > 0:
                    benchmark = benchQueue.get()
                    if benchmark is None:
                        break
                    benchmarks.append(benchmark)

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
                
                low += (1 - mask).astype(np.float32) * (iou_threshold_u + iou_threshold_l) / 2.

                skip_map = np.where(low < iou_threshold_l, skip_map / 2, np.where(low > iou_threshold_u, skip_map * 2, skip_map))
                skip_map = np.clip(skip_map, 1, 32).astype(np.int32)  # Ensure skip_map stays within bounds
                max_skip = int(np.max(skip_map))
            
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
                            if skip_map[y + _y, x + _x] >= idx - last_frame_map[y + _y, x + _x]:
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
            
            outPolyominoQueue.put((idx, frame, bitmap, included_polyominoes))
    outPolyominoQueue.put(None)

    flog.write("Filter patches finished.\n")
    flog.close()

    with open(f"filter_benchmark.{iou_threshold_l}.{iou_threshold_u}.log", 'w') as f:
        f.write(f"Max skip value: {max_skip}\n")
        f.write(f"Pruned polyominoes count: {prune_count}\n")
        f.flush()