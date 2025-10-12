# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc
import cython

# Import data structures from the shared module
from utilities cimport (
    IntStack, Polyomino, PolyominoStack,
    IntStack_init, IntStack_push,
    PolyominoStack_init, PolyominoStack_push,
    PolyominoStack_cleanup
)

from group_tiles import group_tiles as group_tiles_cython
from pack_append import pack_append as pack_append_cython


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
def group_tiles(cnp.uint8_t[:, :] bitmap_input) -> list:
    cdef unsigned long long polyomino_stack_ptr = group_tiles_cython(bitmap_input)
    cdef PolyominoStack *polyomino_stack = <PolyominoStack*>polyomino_stack_ptr
    cdef list bins = []
    cdef Polyomino polyomino
    cdef IntStack connected_tiles
    cdef unsigned short max_i, max_j, tile_i, tile_j, num_pairs
    cdef unsigned short * data

    for i in range(polyomino_stack.top):
        polyomino = polyomino_stack.mo_data[i]
        connected_tiles = polyomino.mask
        num_pairs = <unsigned short>(connected_tiles.top // 2)
        data = connected_tiles.data

        # Initialize with first coordinate pair
        max_i = data[0]
        max_j = data[1]
        
        # Find min/max through all coordinate pairs
        for k in range(1, num_pairs):
            tile_i = data[k << 1]        # i coordinate
            tile_j = data[(k << 1) + 1]  # j coordinate
            
            if tile_i > max_i:
                max_i = tile_i
                
            if tile_j > max_j:
                max_j = tile_j

        # Create mask
        mask_h = max_i + 1
        mask_w = max_j + 1
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_view = mask

        # Fill mask - iterate through IntStack data directly
        for k in range(num_pairs):
            tile_i = data[k << 1]        # i coordinate
            tile_j = data[(k << 1) + 1]  # j coordinate
            mask_view[tile_i, tile_j] = 1
        bins.append((mask, (polyomino.offset_i, polyomino.offset_j)))
    PolyominoStack_cleanup(polyomino_stack)
    # free(polyomino_stack)
    return bins


def pack_append(
    list polyominoes,
    int h,
    int w,
    cnp.uint8_t[:, :] occupied_tiles
):
    # convert polyominoes to PolyominoStack
    cdef PolyominoStack polyominoes_stack
    PolyominoStack_init(&polyominoes_stack, len(polyominoes))
    
    cdef Polyomino polyomino
    cdef IntStack mask
    cdef cnp.uint8_t[:, :] mask_array
    cdef tuple offset
    cdef int i, j, k, mask_h, mask_w
    
    for input_polyomino in polyominoes:
        mask_array, offset = input_polyomino
        mask_h = mask_array.shape[0]
        mask_w = mask_array.shape[1]
        
        # Initialize IntStack for mask coordinates
        IntStack_init(&mask, 16)
        
        # Convert mask array to coordinate pairs
        for i in range(mask_h):
            for j in range(mask_w):
                if mask_array[i, j]:
                    IntStack_push(&mask, <unsigned short>i)
                    IntStack_push(&mask, <unsigned short>j)
        
        # Create Polyomino structure
        polyomino.mask = mask
        polyomino.offset_i = offset[0]
        polyomino.offset_j = offset[1]
        
        PolyominoStack_push(&polyominoes_stack, polyomino)

    positions = pack_append_cython(<unsigned long long>&polyominoes_stack, h, w, occupied_tiles)
    if positions is None:
        return None

    PolyominoStack_cleanup(&polyominoes_stack)

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


def get_polyominoes(list polyominoes):
    # convert polyominoes to PolyominoStack
    cdef PolyominoStack *polyominoes_stack = <PolyominoStack*>malloc(sizeof(PolyominoStack))
    PolyominoStack_init(polyominoes_stack, len(polyominoes))
    
    cdef Polyomino polyomino
    cdef IntStack mask
    cdef cnp.uint8_t[:, :] mask_array
    cdef tuple offset
    cdef int i, j, mask_h, mask_w
    
    for input_polyomino in polyominoes:
        mask_array, offset = input_polyomino
        mask_h = mask_array.shape[0]
        mask_w = mask_array.shape[1]
        
        # Initialize IntStack for mask coordinates
        IntStack_init(&mask, 16)
        
        # Convert mask array to coordinate pairs
        for i in range(mask_h):
            for j in range(mask_w):
                if mask_array[i, j]:
                    IntStack_push(&mask, <unsigned short>i)
                    IntStack_push(&mask, <unsigned short>j)
        
        # Create Polyomino structure
        polyomino.mask = mask
        polyomino.offset_i = offset[0]
        polyomino.offset_j = offset[1]
        
        PolyominoStack_push(polyominoes_stack, polyomino)

    return <unsigned long long>polyominoes_stack


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