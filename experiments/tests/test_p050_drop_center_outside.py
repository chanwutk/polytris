"""p050's opt-in filter for detections centered on empty canvas space."""
import numpy as np

from scripts.p050_exec_uncompress import unpack_detections

TILE = 10
# A 2x3-tile canvas with one polyomino (group 1) in the left column, placed from source tile (5, 7) of frame 3.
INDEX_MAP = np.array([[1, 0, 0],
                      [1, 0, 0]])
OFFSETS = [((0, 0), (5, 7), 3)]
# Center (6, 10) on the polyomino; center (15, 10) on empty space with its top-left corner on the polyomino.
INSIDE = [2.0, 5.0, 10.0, 15.0, 0.9]
CENTER_OUTSIDE = [5.0, 5.0, 25.0, 15.0, 0.8]
# Every corner and the center on empty space.
FAR = [21.0, 1.0, 29.0, 9.0, 0.7]


def test_default_keeps_boxes_centered_on_empty_space():
    frames, dropped, center_out = unpack_detections([INSIDE, CENTER_OUTSIDE, FAR], INDEX_MAP, OFFSETS, TILE)
    # Both boxes touching the polyomino are mapped by its offset (+70, +50); the far box is dropped as before.
    assert frames == {3: [[72.0, 55.0, 80.0, 65.0, 0.9], [75.0, 55.0, 95.0, 65.0, 0.8]]}
    assert dropped == [FAR]
    assert CENTER_OUTSIDE in center_out


def test_flag_drops_boxes_centered_on_empty_space_only():
    frames, dropped, center_out = unpack_detections([INSIDE, CENTER_OUTSIDE, FAR], INDEX_MAP, OFFSETS, TILE,
                                                    drop_center_outside=True)
    assert frames == {3: [[72.0, 55.0, 80.0, 65.0, 0.9]]}
    assert CENTER_OUTSIDE in center_out
