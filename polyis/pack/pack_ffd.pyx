# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int16_t
import cython


# Declare C structures from utilities.h
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
        Polyomino *data  # type: ignore
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

# Declare C structures from pack_ffd.h
cdef extern from "c/pack_ffd.h":
    # Declare the main packing function
    CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w)

    # Cleanup functions
    void CollageArray_cleanup(CollageArray *list)


# Python class for PolyominoPosition (matches Python implementation)
# This is a regular Python class, not a cdef class
cdef class PyPolyominoPosition:
    """Represents the position and orientation of a polyomino in a collage."""
    cdef public int oy, ox, py, px, frame
    cdef public object shape

    def __init__(self, int oy, int ox, int py, int px, int frame, object shape):
        self.oy = oy
        self.ox = ox
        self.py = py
        self.px = px
        self.frame = frame
        self.shape = shape

    def __repr__(self):
        shape_info = f"shape={self.shape}"
        return f"PolyominoPosition(oy={self.oy}, ox={self.ox}, py={self.py}, px={self.px}, frame={self.frame}, {shape_info})"


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_all(cnp.uint64_t[:] polyominoes_stacks, int h, int w) -> list[list[PyPolyominoPosition]]:
    """Packs all polyominoes from multiple stacks into collages using the C FFD algorithm.

    This function takes multiple stacks of polyominoes (memory addresses) and attempts to
    pack them into rectangular collages of the specified dimensions. It uses a first-fit
    decreasing algorithm, trying to place the largest polyominoes first in collages with
    the most empty space.

    Args:
        polyominoes_stacks: cnp.uint64_t[:], a list of memory addresses pointing
                            to a PolyominoArray from C/Cython code. Each stack of polyominoes
                            corresponds to a video frame.
        h: Height of each collage in pixels
        w: Width of each collage in pixels

    Returns:
        List of lists, where each inner list represents a collage containing
        PolyominoPosition objects representing all polyominoes packed into that collage
    """
    cdef int num_arrays = polyominoes_stacks.shape[0]
    cdef PolyominoArray **arrays_ptr
    cdef CollageArray *result
    cdef list[list[PyPolyominoPosition]] collages
    cdef int i

    if num_arrays == 0:
        raise ValueError("polyominoes_stacks cannot be empty")

    # Allocate array of pointers to PolyominoArray
    arrays_ptr = <PolyominoArray**>malloc(<size_t>num_arrays * sizeof(PolyominoArray*))
    if arrays_ptr == NULL:  # type: ignore
        raise MemoryError("Failed to allocate memory for polyominoes arrays")

    # Convert Python list of memory addresses (numpy.uint64 or int) to array of pointers
    for i in range(num_arrays):
        arrays_ptr[i] = <PolyominoArray*>polyominoes_stacks[i]  # type: ignore

    # Call the C packing function
    result = pack_all_(arrays_ptr, num_arrays, h, w)
    free(<void*>arrays_ptr)

    if result == NULL:  # type: ignore
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
cdef list[list[PyPolyominoPosition]] convert_collage_array_to_python(CollageArray *collage_array):
    """Convert a C CollageArray to Python list format.

    Parameters:
        collage_array: Pointer to CollageArray from C code

    Returns:
        List of lists of PolyominoPosition objects
    """
    cdef list[list[PyPolyominoPosition]] result = []
    cdef list collage_positions
    cdef PolyominoPositionArray *position_array
    cdef PolyominoPosition *pos
    cdef CoordinateArray *coords
    cdef cnp.int16_t[:, :] coords_view
    cdef int i, j, k
    cdef int min_y, max_y, min_x, max_x
    cdef int mask_h, mask_w

    # Process each collage
    for i in range(collage_array.size):
        position_array = &collage_array.data[i]
        collage_positions: list[PyPolyominoPosition] = []

        # Process each polyomino position in this collage
        for j in range(position_array.size):
            pos = &position_array.data[j]
            coords = &pos.shape

            if coords.size == 0:
                continue

            coords_np = np.zeros((coords.size, 2), dtype=np.int16)
            coords_view = coords_np

            for k in range(coords.size):
                coords_view[k, 0] = coords.data[k].y
                coords_view[k, 1] = coords.data[k].x

            # Create PolyominoPosition object
            poly_pos = PyPolyominoPosition(
                oy=pos.oy,
                ox=pos.ox,
                py=pos.py,
                px=pos.px,
                frame=pos.frame,
                shape=coords_np
            )

            collage_positions.append(poly_pos)

        result.append(collage_positions)

    return result
