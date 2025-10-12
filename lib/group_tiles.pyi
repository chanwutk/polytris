"""Type stubs for group_tiles module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2


Polyomino = tuple[Array[*D2, np.uint8], tuple[int, int]]


def group_tiles(bitmap_input: Array[*D2, np.uint8]) -> int:
    """
    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array memoryview (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection

    Returns:
        list: A pointer to a list of polyominoes, where each polyomino is:
            - mask: a list of [x, y, x, y, ...], where (x, y) are the coordinates of the masked tile
            - offset: a tuple of (i, j), where i is the horizontal offset
                      and j is the vertical offset of the mask of its original position
    """
    ...
