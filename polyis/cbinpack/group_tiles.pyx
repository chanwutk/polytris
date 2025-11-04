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


# Declare C functions from group_tiles_.h
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

    # Free a polyomino stack allocated by group_tiles
    # Returns the number of polyominoes that were freed
    int free_polyomino_stack(PolyominoStack *polyomino_stack)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode) -> int:
    """
    Group connected tiles into polyominoes using C implementation.

    Parameters:
        bitmap_input: 2D numpy array of uint8 representing the tile grid
                     where 1 indicates a tile with detection and 0 indicates no detection
                     must be contiguous
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
    return <unsigned long long>group_tiles_(&bitmap[0, 0], width, height, tilepadding_mode)