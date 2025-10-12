"""Type stubs for pack_append module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2, PolyominoPositions, Polyomino


def pack_append(
    polyominoes: int,
    h: int,
    w: int,
    occupied_tiles: Array[*D2, np.uint8]
) -> list[PolyominoPositions] | None:
    """
    Fast Cython implementation of pack_append.
    
    Args:
        polyominoes: a pointer to a list of (mask, offset) tuples
                    where mask is a list of (i, j) tuples,
                    where (i, j) are the coordinates of the mask tile,
                    and offset is a tuple of (i, j)
        h: Height of the bitmap
        w: Width of the bitmap
        occupied_tiles: Existing bitmap to append to (modified in-place)
        
    Returns:
        list: positions or None if packing fails
    """
    ...
