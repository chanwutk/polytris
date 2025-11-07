"""Type stubs for group_tiles module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2


Polyomino = tuple[Array[*D2, np.uint8], tuple[int, int]]


def group_tiles(bitmap_input: Array[*D2, np.uint8], mode: int = 0) -> np.uint64:
    """
    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array memoryview (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection
        mode: The mode of tile padding to apply
            - 0: No padding
            - 1: Disconnected padding
            - 2: Connected padding
    Returns:
        numpy.uint64: Memory address (as uint64) of a list of polyominoes, where each polyomino is:
            - mask: a list of [x, y, x, y, ...], where (x, y) are the coordinates of the masked tile
            - offset_i: the horizontal offset of the mask from its original position
            - offset_j: the vertical offset of the mask from its original position
    """
    ...


def free_polyimino_stack(polyomino_stack_ptr: np.uint64) -> int:
    """
    Cleans up a polyomino stack.

    This function frees the memory allocated for the polyomino stack.
    
    Args:
        polyomino_stack_ptr: Memory address as numpy.uint64, int, or compatible type

    Returns:
        int: The number of polyominoes in the stack
    """
    ...