"""Type stubs for cbinpack.adapters Cython module."""

from numpy.typing import NDArray
import numpy as np
from polyis.pack.pack_ffd import PyPolyominoPosition

def c_group_tiles(
    bitmap_input: NDArray[np.uint8],
    tilepadding_mode: int
) -> list[tuple[NDArray[np.uint8], tuple[int, int]]]:
    """
    Group connected tiles into polyominoes using C implementation.

    Parameters:
        bitmap_input: 2D numpy array of uint8 representing the tile grid
                     where 1 indicates a tile with detection and 0 indicates no detection
        tilepadding_mode: The mode of tile padding to apply
                         - 0: No padding
                         - 1: Connected padding
                         - 2: Disconnected padding

    Returns:
        List of tuples (mask, (offset_i, offset_j)) where:
        - mask is a 2D numpy array representing the polyomino shape
        - offset_i, offset_j are the top-left coordinates of the polyomino
    """
    ...

def convert_collages_to_bitmap(
    collages: list[list[PyPolyominoPosition]]
) -> list[list[PyPolyominoPosition]]:
    """Convert PyPolyominoPosition shapes from coordinate lists to bitmap masks.

    Mutates and returns the same nested structure (list[list[PyPolyominoPosition]]),
    replacing each position's ``shape`` (Nx2 int16 coordinates) with a 2D uint8 bitmap
    tightly bounded to the shape. Other fields (oy, ox, py, px, frame) are preserved.
    This function modifies the input in place and also returns it for convenience.
    """
    ...