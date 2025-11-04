"""Type stubs for group_tiles module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2


Polyomino = tuple[Array[*D2, np.uint8], tuple[int, int]]


def group_tiles(bitmap_input: Array[*D2, np.uint8], tilepadding_mode: int) -> list[Polyomino]:
    """
    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array memoryview (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection
        tilepadding_mode: The mode of tile padding to apply
            - 0: No padding
            - 1: Connected padding
            - 2: Disconnected padding
    Returns:
        list[Polyomino]: A list of polyominoes, where each polyomino is:
            - mask: a list of [x, y, x, y, ...], where (x, y) are the coordinates of the masked tile
            - offset: a tuple of (i, j), where i is the horizontal offset
                      and j is the vertical offset of the mask of its original position
    """
    ...


def free_polyimino_stack(polyomino_stack_ptr: int) -> int:
    """
    Cleans up a polyomino stack.

    This function frees the memory allocated for the polyomino stack.
    
    Args:
        polyomino_stack_ptr: A pointer to a polyomino stack

    Returns:
        int: The number of polyominoes in the stack
    """
    ...