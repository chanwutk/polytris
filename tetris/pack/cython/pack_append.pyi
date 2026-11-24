"""Type stubs for pack_append module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D1, D2, PolyominoPositions


def pack_append(
    polyominoes: np.uint64,
    h: int,
    w: int,
    occupied_tiles: Array[*D2, np.uint8]
) -> list[PolyominoPositions] | None:
    """
    Pack all polyominoes into the occupied tiles.
    If not possible, return None.
    Note:
        - occupied_tiles will be reverted to the original state if packing fails.
    
    Args:
        polyominoes: Memory address as numpy.uint64
            - mask: a list of [x, y, x, y, ...], where (x, y) are the coordinates of the masked tile
            - offset_i: horizontal offset of the mask from the top left corner of its original position
            - offset_j: vertical offset of the mask from the top left corner of its original position
        
        h: Height of the bitmap
        w: Width of the bitmap
        occupied_tiles: Existing bitmap to append polyominoes to (modified in-place)
        
    Returns:
        list: positions or None if packing fails, where each position contains:
                - i: y-coordinate of the position
                - j: x-coordinate of the position
                - mask_array: a list of [x, y, x, y, ...],
                              where (x, y) are the coordinates of the masked tile
                - offset: a tuple of (i, j), where i is the horizontal offset
                          and j is the vertical offset of the mask of its original position
    """
    ...
