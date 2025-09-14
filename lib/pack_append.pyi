"""Type stubs for pack_append module."""

import numpy as np
import numpy.typing as npt


Polyomino = tuple[npt.NDArray[np.uint8], tuple[int, int]]
PolyominoPositions = tuple[int, int, npt.NDArray[np.uint8], tuple[int, int]]


def pack_append(
    polyominoes: list[Polyomino],
    h: int,
    w: int,
    occupied_tiles: npt.NDArray[np.uint8]
) -> list[PolyominoPositions] | None:
    """
    Fast Cython implementation of pack_append.
    
    Args:
        polyominoes: list of (mask, offset) tuples
        h: Height of the bitmap
        w: Width of the bitmap
        occupied_tiles: Existing bitmap to append to (modified in-place)
        
    Returns:
        list: positions or None if packing fails
    """
    ...
