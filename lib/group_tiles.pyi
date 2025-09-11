"""Type stubs for group_tiles module."""

import numpy as np
import numpy.typing as npt


Polyomino = tuple[int, npt.NDArray[np.uint8], tuple[int, int]]


def group_tiles(bitmap_input: npt.NDArray[np.uint8]) -> list[Polyomino]:
    """
    Fast Cython implementation of group_tiles.

    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection

    Returns:
        list: List of polyominoes, where each polyomino is:
            - group_id: unique id of the group
            - mask: masking of the polyomino as a 2D numpy array
            - offset: offset of the mask from the top left corner of the bitmap
    """
    ...
