"""Type stubs for pack_all module."""

import typing

import numpy as np


if typing.TYPE_CHECKING:
    from polyis.dtypes import Array, D2, PolyominoPositions


def pack_all(
    polyominoes_stacks: list[int],
    h: int,
    w: int
) -> list[list[PolyominoPositions]]:
    """
    Pack all polyominoes from multiple stacks into boards.
    Tries to pack all polyominoes into as fewest boards as possible.
    
    Args:
        polyominoes_stacks: List of pointers to polyomino stacks.
                           Each pointer points to a PolyominoStack containing multiple polyominoes.
        h: Height of each board
        w: Width of each board
        
    Returns:
        list: A list of boards, where each board contains a list of positions.
              Each position contains:
                - i: y-coordinate of the position
                - j: x-coordinate of the position
                - mask_array: a numpy array of [x, y, x, y, ...],
                              where (x, y) are the coordinates of the masked tile
                - offset: a tuple of (i, j), where i is the horizontal offset
                          and j is the vertical offset of the mask of its original position
    """
    ...
