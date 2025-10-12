# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc

import numpy as np

# Import data structures from the shared module
from utilities cimport (
    IntStack, Polyomino, PolyominoStack,
    IntStack_init, IntStack_push, IntStack_cleanup,
    PolyominoStack_init, PolyominoStack_push,
    PolyominoStack_cleanup
)

ctypedef cnp.uint8_t DTYPE_t


def pack_append(
    unsigned long long polyominoes,
    int h,
    int w,
    cnp.uint8_t[:, :] occupied_tiles
):
    """
    Fast Cython implementation of pack_append.
    
    Args:
        polyominoes: List of (groupid, mask, offset) tuples
        h: Height of the bitmap
        w: Width of the bitmap
        occupied_tiles: Existing bitmap to append to (modified in-place)
        
    Returns:
        list: positions or None if packing fails
    """
    cdef int idx, i, j, k, mask_h, mask_w
    cdef tuple offset
    cdef bint valid, placed
    cdef list positions = []
    cdef PolyominoStack *polyominoes_stack = <PolyominoStack*>polyominoes
    cdef Polyomino polyomino
    cdef IntStack mask
    cdef IntStack appending_tiles
    cdef unsigned short tile_i, tile_j
    cdef cnp.uint8_t[:] mask_array_view
    
    # Create appending tiles tracker
    IntStack_init(&appending_tiles, 16)
    
    for idx in range(polyominoes_stack.top):
        polyomino = polyominoes_stack.mo_data[idx]
        mask = polyomino.mask
        offset = (polyomino.offset_i, polyomino.offset_j)
        
        # Calculate mask dimensions from the mask data
        mask_h = 0
        mask_w = 0
        for k in range(mask.top // 2):
            tile_i = mask.data[k << 1]
            tile_j = mask.data[(k << 1) + 1]
            if tile_i >= mask_h:
                mask_h = tile_i + 1
            if tile_j >= mask_w:
                mask_w = tile_j + 1
        
        placed = <bint>False
        
        # Try to place at each position
        for j in range(w - mask_w + 1):
            for i in range(h - mask_h + 1):
                valid = <bint>True
                
                # Check for collisions
                for k in range(mask.top // 2):
                    tile_i = mask.data[k << 1]
                    tile_j = mask.data[(k << 1) + 1]
                    if occupied_tiles[i + tile_i, j + tile_j]:
                        valid = <bint>False
                        break
                
                if valid:
                    mask_array = np.empty((mask.top), dtype=np.uint8)
                    mask_array_view = mask_array

                    # Place the polyomino
                    for k in range(mask.top // 2):
                        tile_i = mask.data[k << 1]
                        tile_j = mask.data[(k << 1) + 1]
                        occupied_tiles[i + tile_i, j + tile_j] = 1
                        # appending_tiles_view[i + tile_i, j + tile_j] = 1
                        IntStack_push(&appending_tiles, <unsigned short>(i + tile_i))
                        IntStack_push(&appending_tiles, <unsigned short>(j + tile_j))

                        mask_array_view[k << 1] = tile_i
                        mask_array_view[(k << 1) + 1] = tile_j

                    positions.append((i, j, mask_array, offset))
                    placed = <bint>True
                    break
            
            if placed:
                break
        
        if not placed:
            # Revert changes by clearing appending_tiles
            for i in range(appending_tiles.top // 2):
                tile_i = appending_tiles.data[i << 1]
                tile_j = appending_tiles.data[(i << 1) + 1]
                occupied_tiles[tile_i, tile_j] = 0
            IntStack_cleanup(&appending_tiles)
            return None
    
    IntStack_cleanup(&appending_tiles)
    return positions
