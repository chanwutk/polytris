# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import cython

# Import data structures from the shared module
from polyis.pack.cython.utilities cimport (
    Coordinate, CoordinateStack, Polyomino, PolyominoStack,
    CoordinateStack_init, CoordinateStack_push,
    PolyominoStack_init, PolyominoStack_push,
    PolyominoStack_cleanup
)

from polyis.pack.cython.group_tiles import group_tiles as group_tiles_cython
from polyis.pack.cython.pack_append import pack_append as pack_append_cython


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode) -> list:
    # group_tiles_cython returns numpy.uint64, convert to pointer
    return format_polyominoes(group_tiles_cython(bitmap_input, tilepadding_mode))


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def format_polyominoes(cnp.uint64_t polyomino_stack_ptr):
    """
    Format polyominoes from a memory address.
    
    Parameters:
        polyomino_stack_ptr: Memory address as numpy.uint64
    """
    cdef PolyominoStack *polyomino_stack = <PolyominoStack*>polyomino_stack_ptr
    cdef list bins = []
    cdef Polyomino polyomino
    cdef CoordinateStack connected_tiles
    cdef int max_i, max_j, tile_i, tile_j
    cdef Coordinate *data

    for i in range(polyomino_stack.top):
        polyomino = polyomino_stack.mo_data[i]  # type: ignore
        connected_tiles = polyomino.mask
        data = connected_tiles.data

        # Initialize with first coordinate
        max_i = data[0].y  # type: ignore
        max_j = data[0].x  # type: ignore

        # Find max coordinates through all coordinates
        for k in range(1, connected_tiles.top):
            tile_i = data[k].y  # type: ignore
            tile_j = data[k].x  # type: ignore

            if tile_i > max_i:
                max_i = tile_i

            if tile_j > max_j:
                max_j = tile_j

        # Create mask
        mask_h = max_i + 1
        mask_w = max_j + 1
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_view = mask

        # Fill mask - iterate through CoordinateStack data directly
        for k in range(connected_tiles.top):
            tile_i = data[k].y  # type: ignore
            tile_j = data[k].x  # type: ignore
            mask_view[tile_i, tile_j] = 1
        bins.append((mask, (polyomino.offset_i, polyomino.offset_j)))
    PolyominoStack_cleanup(polyomino_stack)
    free(<void*>polyomino_stack)
    return bins


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_append(list polyominoes, int h, int w, cnp.uint8_t[:, :] occupied_tiles):
    # get_polyominoes returns numpy.uint64, convert to pointer for C function
    cdef cnp.uint64_t polyominoes_ptr = get_polyominoes(polyominoes)
    cdef list positions = pack_append_cython(polyominoes_ptr, h, w, occupied_tiles)
    PolyominoStack_cleanup(<PolyominoStack*>polyominoes_ptr)
    free(<void*>polyominoes_ptr)
    return format_positions(positions)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def get_polyominoes(list polyominoes) -> np.uint64:
    # convert polyominoes to PolyominoStack
    cdef PolyominoStack *polyominoes_stack = <PolyominoStack*>malloc(sizeof(PolyominoStack))
    PolyominoStack_init(polyominoes_stack, len(polyominoes))

    cdef Polyomino polyomino
    cdef CoordinateStack mask
    cdef Coordinate coord
    cdef cnp.uint8_t[:, :] mask_array
    cdef tuple offset
    cdef int i, j, mask_h, mask_w

    for i in range(len(polyominoes)):
        mask_array, offset = polyominoes[i]
        mask_h = mask_array.shape[0]
        mask_w = mask_array.shape[1]

        # Initialize CoordinateStack for mask coordinates
        CoordinateStack_init(&mask, 16)

        # Convert mask array to coordinates
        for i in range(mask_h):
            for j in range(mask_w):
                if mask_array[i, j]:  # type: ignore
                    coord.y = <short>i
                    coord.x = <short>j
                    CoordinateStack_push(&mask, coord)

        # Create Polyomino structure
        polyomino.mask = mask
        polyomino.offset_i = offset[0]
        polyomino.offset_j = offset[1]

        PolyominoStack_push(polyominoes_stack, polyomino)

    # Convert pointer to numpy.uint64 for safe type handling
    return np.uint64(<cnp.uint64_t>polyominoes_stack)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def format_positions(list positions) -> "list[tuple[int, int, np.ndarray, tuple[int, int]]] | None":
    if positions is None:
        return None
    
    positions_ret = []
    for position in positions:
        mask_array_ = position[2]
        max_i = 0
        max_j = 0
        for k in range(mask_array_.shape[0] // 2):
            i = mask_array_[(k * 2)]
            j = mask_array_[(k * 2) + 1]
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j
        mask_ = np.zeros((max_i + 1, max_j + 1), dtype=np.uint8)
        for k in range(mask_array_.shape[0] // 2):
            i = mask_array_[(k * 2)]
            j = mask_array_[(k * 2) + 1]
            mask_[i, j] = 1
        positions_ret.append((position[0], position[1], mask_, position[3]))

    return positions_ret