"""Type stubs for group_tiles module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2


Polyomino = tuple[Array[*D2, np.uint8], tuple[int, int]]


def group_tiles(bitmap_input: Array[*D2, np.uint8]) -> int:
    """
    Fast Cython implementation of group_tiles.

    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection

    Returns:
        int: a pointer to a list of polyominoes, where each polyomino is:
            - mask: masking of the polyomino as a list of (i, j) tuples,
                    where (i, j) are the coordinates of the mask tile
            - offset: offset of the mask from the top left corner of the bitmap
    """
    ...
