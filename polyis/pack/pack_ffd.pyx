# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import cython


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
        Polyomino *data  # type: ignore
        int size
        int capacity


# Declare C structures from pack_ffd_.h
cdef extern from "c/pack_ffd_.h":
    ctypedef struct Coordinate:
        int y
        int x

    ctypedef struct CoordinateArray:
        Coordinate *data  # type: ignore
        int size
        int capacity

    ctypedef struct PolyominoPosition:
        int oy
        int ox
        int py
        int px
        int rotation
        int frame
        CoordinateArray shape

    ctypedef struct PolyominoPositionArray:
        PolyominoPosition *data  # type: ignore
        int size
        int capacity

    ctypedef struct CollageArray:
        PolyominoPositionArray *data
        int size
        int capacity

    # Declare the main packing function
    CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w)

    # Cleanup functions
    void CollageArray_cleanup(CollageArray *list)


# Python class for PolyominoPosition (matches Python implementation)
# This is a regular Python class, not a cdef class
cdef class PyPolyominoPosition:
    """Represents the position and orientation of a polyomino in a collage."""
    cdef public int oy, ox, py, px, rotation, frame
    cdef public object shape

    def __init__(self, int oy, int ox, int py, int px, int rotation, int frame, object shape):
        self.oy = oy
        self.ox = ox
        self.py = py
        self.px = px
        self.rotation = rotation
        self.frame = frame
        self.shape = shape

    def __repr__(self):
        shape_info = f"shape={self.shape}"
        return f"PolyominoPosition(oy={self.oy}, ox={self.ox}, py={self.py}, px={self.px}, rotation={self.rotation}, frame={self.frame}, {shape_info})"


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_all(list polyominoes_stacks, int h, int w):
    """Packs all polyominoes from multiple stacks into collages using the C FFD algorithm.

    This function takes multiple stacks of polyominoes (memory addresses) and attempts to
    pack them into rectangular collages of the specified dimensions. It uses a first-fit
    decreasing algorithm, trying to place the largest polyominoes first in collages with
    the most empty space.

    Args:
        polyominoes_stacks: List of integers, each representing a memory address pointing
                            to a PolyominoArray from C/Cython code.
                            Each stack of polyominoes corresponds to a video frame.
        h: Height of each collage in pixels
        w: Width of each collage in pixels

    Returns:
        List of lists, where each inner list represents a collage containing
        PolyominoPosition objects representing all polyominoes packed into that collage.
    """
    cdef int num_arrays = len(polyominoes_stacks)
    cdef PolyominoArray **arrays_ptr
    cdef CollageArray *result
    cdef list collages
    cdef int i, j, k

    if num_arrays == 0:
        return []

    # Allocate array of pointers to PolyominoArray
    arrays_ptr = <PolyominoArray**>malloc(<size_t>num_arrays * sizeof(PolyominoArray*))
    if arrays_ptr == NULL:
        raise MemoryError("Failed to allocate memory for polyominoes arrays")

    # Convert Python list of integers (memory addresses) to array of pointers
    for i in range(num_arrays):
        arrays_ptr[i] = <PolyominoArray*><unsigned long long>polyominoes_stacks[i]  # type: ignore

    # Call the C packing function
    result = pack_all_(arrays_ptr, num_arrays, h, w)

    if result == NULL:
        raise MemoryError("pack_all_ returned NULL")

    # Convert result to Python format
    collages = convert_collage_array_to_python(result)

    # Clean up the result
    CollageArray_cleanup(result)
    free(<void*>result)

    return collages


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef list convert_collage_array_to_python(CollageArray *collage_array):
    """Convert a C CollageArray to Python list format.

    Parameters:
        collage_array: Pointer to CollageArray from C code

    Returns:
        List of lists of PolyominoPosition objects
    """
    cdef list result = []
    cdef list collage_positions
    cdef PolyominoPositionArray *position_array
    cdef PolyominoPosition *pos
    cdef CoordinateArray *coords
    cdef cnp.uint8_t[:, :] mask_view
    cdef int i, j, k
    cdef int min_y, max_y, min_x, max_x
    cdef int mask_h, mask_w

    # Process each collage
    for i in range(collage_array.size):
        position_array = &collage_array.data[i]
        collage_positions = []

        # Process each polyomino position in this collage
        for j in range(position_array.size):
            pos = &position_array.data[j]
            coords = &pos.shape

            if coords.size == 0:
                continue

            # Find bounding box of the shape
            min_y = coords.data[0].y
            max_y = coords.data[0].y
            min_x = coords.data[0].x
            max_x = coords.data[0].x

            for k in range(1, coords.size):
                if coords.data[k].y < min_y:
                    min_y = coords.data[k].y
                if coords.data[k].y > max_y:
                    max_y = coords.data[k].y
                if coords.data[k].x < min_x:
                    min_x = coords.data[k].x
                if coords.data[k].x > max_x:
                    max_x = coords.data[k].x

            # Create mask array
            mask_h = max_y - min_y + 1
            mask_w = max_x - min_x + 1
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            mask_view = mask

            # Fill mask with shape coordinates
            for k in range(coords.size):
                mask_view[coords.data[k].y - min_y, coords.data[k].x - min_x] = 1

            # Create PolyominoPosition object
            poly_pos = PyPolyominoPosition(
                oy=pos.oy,
                ox=pos.ox,
                py=pos.py,
                px=pos.px,
                rotation=pos.rotation,
                frame=pos.frame,
                shape=mask
            )

            collage_positions.append(poly_pos)

        result.append(collage_positions)

    return result
