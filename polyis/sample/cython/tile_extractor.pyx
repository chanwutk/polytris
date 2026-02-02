# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Efficient tile extraction from polyomino C structures.

This module provides direct access to tile coordinates without mask conversion.
"""

cimport numpy as cnp
import numpy as np
from libc.stdint cimport int16_t, uint64_t
import cython


# Declare C structures from utilities.h
cdef extern from "../../pack/c/utilities.h":
    ctypedef struct Coordinate:
        int16_t y
        int16_t x

    ctypedef struct CoordinateArray:
        Coordinate *data
        int size
        int capacity

    ctypedef struct Polyomino:
        CoordinateArray mask
        int offset_y
        int offset_x

    ctypedef struct PolyominoArray:
        Polyomino *data
        int size
        int capacity


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef list extract_tiles_from_polyominoes(uint64_t polyomino_array_addr):
    """
    Extract tile coordinates directly from polyomino C structures.

    This function avoids creating mask arrays by directly extracting
    the tile coordinates from the C structures.

    Parameters:
        polyomino_array_addr: Memory address (uint64) pointing to PolyominoArray

    Returns:
        List of polyominoes, where each polyomino is a list of (row, col) tuples
        representing absolute tile positions (including offsets).
    """
    # Convert uint64 address to pointer
    cdef PolyominoArray *polyomino_array = <PolyominoArray*>polyomino_array_addr

    cdef list polyominoes = []
    cdef list tiles
    cdef Polyomino polyomino
    cdef CoordinateArray connected_tiles
    cdef Coordinate *data_
    cdef int i, k
    cdef int tile_row, tile_col
    cdef int offset_y, offset_x

    # Process each polyomino in the array
    for i in range(polyomino_array.size):
        polyomino = polyomino_array.data[i]
        connected_tiles = polyomino.mask
        data_ = connected_tiles.data
        offset_y = polyomino.offset_y
        offset_x = polyomino.offset_x

        # Extract all tile coordinates for this polyomino
        tiles = []
        for k in range(connected_tiles.size):
            # Get relative coordinates and add offsets to get absolute position
            tile_row = data_[k].y + offset_y
            tile_col = data_[k].x + offset_x
            tiles.append((tile_row, tile_col))

        polyominoes.append(tiles)

    # Note: We do NOT free the polyomino array here
    # The caller is responsible for calling free_polyomino_array
    return polyominoes