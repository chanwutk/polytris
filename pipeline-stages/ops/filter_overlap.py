import json

from minivan.dtypes import InPipe, OutPipe


CHUNK_SIZE = 128


def bbox_overlap(bbox1, bbox2) -> bool:
    # Calculate the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        return True
    return False


def filter_overlap(
    trackQueue: InPipe[tuple[int, list[list[float | int]]]],
    outQueue: OutPipe[tuple[int, list[list[float | int]]]] | None,
    regions: list[tuple[int, int]],
    filename: str
):
    boxes = [
        (
          CHUNK_SIZE * x,
          CHUNK_SIZE * y,
          CHUNK_SIZE * (x + 1),
          CHUNK_SIZE * (y + 1),
        )
        for y, x in regions
    ]
    with open(filename, 'w') as f:
        while True:
            tracked_objects = trackQueue.get()

            if tracked_objects is None:
                break

            idx, objs = tracked_objects
            objs = [
                o for o in objs
                if len(o) == 5
                and all(not bbox_overlap(o[:4], box) for box in boxes)
            ]

            f.write(f"{json.dumps([idx, objs])}\n")
            f.flush()
            if outQueue is not None:
                outQueue.put((idx, objs))
    if outQueue is not None:
        outQueue.put(None)
    