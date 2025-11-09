# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free
from libc.stdint cimport int16_t
import cython

from polyis.pack.group_tiles import group_tiles  # type: ignore[import-untyped]


# Declare C structures from utilities_.h
cdef extern from "c/utilities.h":
    ctypedef struct Coordinate:
        int16_t y
        int16_t x

    ctypedef struct CoordinateArray:
        Coordinate *data  # type: ignore
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

    # Declare utility functions
    void PolyominoArray_cleanup(PolyominoArray *array)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def c_group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode) -> list[tuple[np.ndarray, tuple[int, int]]]:
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
    # Call group_tiles function from group_tiles.pyx
    # group_tiles returns numpy.uint64, pass directly to format_polyominoes
    # Note: group_tiles is defined in group_tiles.pyx, compiled into the same extension
    return format_polyominoes(<cnp.uint64_t>group_tiles(bitmap_input, tilepadding_mode))


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef list[tuple[np.ndarray, tuple[int, int]]] format_polyominoes(cnp.uint64_t polyomino_array_addr):
    """
    Convert a C PolyominoArray to Python list format.

    Parameters:
        polyomino_array_addr: Memory address as uint64 pointing to PolyominoArray from C code

    Returns:
        List of tuples (mask, (offset_i, offset_j))
    """
    # Convert uint64 address to pointer
    cdef PolyominoArray *polyomino_array = <PolyominoArray*>polyomino_array_addr
    
    cdef list[tuple[np.ndarray, tuple[int, int]]] bins = []
    cdef Polyomino polyomino
    cdef CoordinateArray connected_tiles
    cdef int max_i, max_j, tile_i, tile_j
    cdef Coordinate *data_
    cdef int i, k
    cdef int mask_h, mask_w
    cdef cnp.uint8_t[:, :] mask_view

    # Process each polyomino in the array
    for i in range(polyomino_array.size):
        polyomino = polyomino_array.data[i]  # type: ignore
        connected_tiles = polyomino.mask
        data_ = connected_tiles.data  # type: ignore

        # Initialize with first coordinate
        max_i = data_[0].y  # type: ignore
        max_j = data_[0].x  # type: ignore

        # Find max coordinates through all coordinates
        for k in range(1, connected_tiles.size):
            tile_i = data_[k].y  # type: ignore
            tile_j = data_[k].x  # type: ignore

            if tile_i > max_i:
                max_i = tile_i

            if tile_j > max_j:
                max_j = tile_j

        # Create mask with dimensions based on max coordinates
        mask_h = max_i + 1
        mask_w = max_j + 1
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_view = mask  # type: ignore

        # Fill mask - iterate through CoordinateArray data directly
        for k in range(connected_tiles.size):
            tile_i = data_[k].y  # type: ignore
            tile_j = data_[k].x  # type: ignore
            mask_view[tile_i, tile_j] = 1  # type: ignore

        # Append as tuple: (mask, (offset_y, offset_x))
        bins.append((mask, (polyomino.offset_y, polyomino.offset_x)))

    # Clean up the polyomino array
    PolyominoArray_cleanup(polyomino_array)
    free(<void*>polyomino_array)

    return bins


def convert_collages_to_bitmap(collages):
    """
    Convert PyPolyominoPosition ``shape`` from coordinate lists to bitmap masks.

    Mutates and returns the same nested structure (list[list[PyPolyominoPosition]]),
    replacing each position's ``shape`` (Nx2 int16 coordinates) with a 2D uint8 bitmap
    tightly bounded to the shape. All other fields (oy, ox, py, px, frame) remain unchanged.
    """
    out = []
    for collage in collages:  # iterate collages
        new_collage = []
        for pos in collage:  # iterate positions within a collage
            coords_np = pos.shape  # numpy array of coordinates
            # Compute bounding box of coordinates
            max_y = int(np.max(coords_np[:, 0]))
            max_x = int(np.max(coords_np[:, 1]))
            mask_h = max_y + 1
            mask_w = max_x + 1
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            # Fill mask using translated coordinates
            for y, x in coords_np:
                mask[int(y), int(x)] = 1
            # Replace shape with bitmap mask
            pos.shape = mask
            new_collage.append(pos)
        out.append(new_collage)
    return out
