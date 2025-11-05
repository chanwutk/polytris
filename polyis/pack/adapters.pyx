# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free
import cython

from polyis.pack.group_tiles import group_tiles  # type: ignore[import-untyped]


# Declare C structures from utilities_.h
cdef extern from "c/utilities_.h":
    ctypedef struct UShortArray:
        unsigned short *data  # type: ignore
        int size
        int capacity

    ctypedef struct Polyomino:
        UShortArray mask
        int offset_i
        int offset_j

    ctypedef struct PolyominoArray:
        Polyomino *data
        int size
        int capacity

    # Declare utility functions
    void PolyominoArray_cleanup(PolyominoArray *array)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def c_group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode) -> list:
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
    cdef int polyomino_stack

    # Call group_tiles function from group_tiles.pyx
    # Cast the returned pointer (as int) back to PolyominoArray*
    # Note: group_tiles is defined in group_tiles.pyx, compiled into the same extension
    polyomino_stack = group_tiles(bitmap_input, tilepadding_mode)  # type: ignore[name-defined]

    if polyomino_stack == 0:
        return []

    # Convert result to Python format
    result = format_polyominoes(<PolyominoArray*><unsigned long long>polyomino_stack)

    # Free the array (format_polyominoes already handles cleanup)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef format_polyominoes(PolyominoArray *polyomino_array):
    """
    Convert a C PolyominoArray to Python list format.

    Parameters:
        polyomino_array: Pointer to PolyominoArray from C code

    Returns:
        List of tuples (mask, (offset_i, offset_j))
    """
    cdef list bins = []
    cdef Polyomino polyomino
    cdef UShortArray connected_tiles
    cdef unsigned short max_i, max_j, tile_i, tile_j, num_pairs
    cdef unsigned short *data_
    cdef int i, k
    cdef int mask_h, mask_w
    cdef cnp.uint8_t[:, :] mask_view

    # Process each polyomino in the array
    for i in range(polyomino_array.size):
        polyomino = polyomino_array.data[i]  # type: ignore
        connected_tiles = polyomino.mask
        num_pairs = <unsigned short>(connected_tiles.size // 2)
        data_ = connected_tiles.data  # type: ignore

        # Initialize with first coordinate pair
        max_i = data_[0]  # type: ignore
        max_j = data_[1]  # type: ignore

        # Find max coordinates through all coordinate pairs
        for k in range(1, num_pairs):
            tile_i = data_[k << 1]        # type: ignore
            tile_j = data_[(k << 1) + 1]  # type: ignore

            if tile_i > max_i:
                max_i = tile_i

            if tile_j > max_j:
                max_j = tile_j

        # Create mask with dimensions based on max coordinates
        mask_h = max_i + 1
        mask_w = max_j + 1
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_view = mask  # type: ignore

        # Fill mask - iterate through UShortArray data directly
        for k in range(num_pairs):
            tile_i = data_[k << 1]        # type: ignore
            tile_j = data_[(k << 1) + 1]  # type: ignore
            mask_view[tile_i, tile_j] = 1  # type: ignore

        # Append as tuple: (mask, (offset_i, offset_j))
        bins.append((mask, (polyomino.offset_i, polyomino.offset_j)))

    # Clean up the polyomino array
    PolyominoArray_cleanup(polyomino_array)
    free(<void*>polyomino_array)

    return bins
