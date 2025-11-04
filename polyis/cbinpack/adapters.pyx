# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import cython

# Declare C structures from utilities.h
cdef extern from "utilities.h":
    ctypedef struct IntStack:
        unsigned short *data
        int top
        int capacity

    ctypedef struct Polyomino:
        IntStack mask
        int offset_i
        int offset_j

    ctypedef struct PolyominoStack:
        Polyomino *data
        int top
        int capacity

    # Declare utility functions
    void PolyominoStack_cleanup(PolyominoStack *stack)

# Declare C functions from group_tiles.h
cdef extern from "group_tiles_.h":
    # Main function to group tiles into polyominoes
    # bitmap_input: 2D array (flattened) of uint8_t representing the grid of tiles
    #               where 1 indicates a tile with detection and 0 indicates no detection
    # width: width of the bitmap
    # height: height of the bitmap
    # tilepadding_mode: The mode of tile padding to apply
    #                   - 0: No padding
    #                   - 1: Connected padding
    #                   - 2: Disconnected padding
    # Returns: Pointer to PolyominoStack containing all found polyominoes
    PolyominoStack* group_tiles_(
        unsigned char *bitmap_input,
        int width,
        int height,
        int tilepadding_mode
    )


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
    cdef int height = bitmap_input.shape[0]
    cdef int width = bitmap_input.shape[1]
    cdef PolyominoStack *polyomino_stack

    # Create contiguous copy for C function
    # Need to copy because C function modifies the bitmap
    cdef cnp.uint8_t[:, :] bitmap_copy = np.ascontiguousarray(bitmap_input.copy(), dtype=np.uint8)

    # Call C function
    polyomino_stack = group_tiles_(&bitmap_copy[0, 0], width, height, tilepadding_mode)

    if polyomino_stack == NULL:
        return []

    # Convert result to Python format
    result = format_polyominoes(polyomino_stack)

    # Free the stack (format_polyominoes already handles cleanup)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef format_polyominoes(PolyominoStack *polyomino_stack):
    """
    Convert a C PolyominoStack to Python list format.

    Parameters:
        polyomino_stack: Pointer to PolyominoStack from C code

    Returns:
        List of tuples (mask, (offset_i, offset_j))
    """
    cdef list bins = []
    cdef Polyomino polyomino
    cdef IntStack connected_tiles
    cdef unsigned short max_i, max_j, tile_i, tile_j, num_pairs
    cdef unsigned short *data
    cdef int i, k
    cdef int mask_h, mask_w
    cdef cnp.uint8_t[:, :] mask_view

    # Process each polyomino in the stack
    for i in range(polyomino_stack.top):
        polyomino = polyomino_stack.data[i]
        connected_tiles = polyomino.mask
        num_pairs = <unsigned short>(connected_tiles.top // 2)
        data = connected_tiles.data

        # Initialize with first coordinate pair
        max_i = data[0]
        max_j = data[1]

        # Find max coordinates through all coordinate pairs
        for k in range(1, num_pairs):
            tile_i = data[k << 1]        # k * 2
            tile_j = data[(k << 1) + 1]  # k * 2 + 1

            if tile_i > max_i:
                max_i = tile_i

            if tile_j > max_j:
                max_j = tile_j

        # Create mask with dimensions based on max coordinates
        mask_h = max_i + 1
        mask_w = max_j + 1
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_view = mask

        # Fill mask - iterate through IntStack data directly
        for k in range(num_pairs):
            tile_i = data[k << 1]        # k * 2
            tile_j = data[(k << 1) + 1]  # k * 2 + 1
            mask_view[tile_i, tile_j] = 1

        # Append as tuple: (mask, (offset_i, offset_j))
        bins.append((mask, (polyomino.offset_i, polyomino.offset_j)))

    # Clean up the polyomino stack
    PolyominoStack_cleanup(polyomino_stack)
    free(<void*>polyomino_stack)

    return bins
