from queue import Queue

import numpy as np

from polyis.dtypes import NPImage, Array, D2, IdPolyominoOffset


def filter_frames(
    polyominoQueue: "Queue[tuple[int, NPImage, Array[*D2, np.int32], list[IdPolyominoOffset]] | None]",
    outPolyominoQueue: "Queue[tuple[int, NPImage, Array[*D2, np.int32], list[IdPolyominoOffset]] | None]",
    inv_sampling_rate: int,
):
    flog = open("filter_frames.py.log", "w")

    while True:
        polyomino = polyominoQueue.get()
        if polyomino is None:
            break
        idx, img, *_ = polyomino
        flog.write(f"Processing frame {idx}...\n")
        flog.flush()
        
        if idx % inv_sampling_rate == 0:
            outPolyominoQueue.put(polyomino)
        # else:
        #     outPolyominoQueue.put((idx, img, np.empty((0, 4), dtype=np.int32), []))
        # outPolyominoQueue.put(polyomino)
    outPolyominoQueue.put(None)

    flog.write("Filter frames finished.\n")
    flog.close()
