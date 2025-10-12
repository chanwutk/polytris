# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp


ctypedef cnp.uint8_t DTYPE_t


# @cython.boundscheck(False)
# @cython.wraparound(False)
def pack_append(
    list polyominoes,
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
    cdef int i, j, k, row, col, mask_h, mask_w, groupid
    cdef tuple offset
    cdef cnp.uint8_t[:] mask_view
    cdef cnp.uint8_t[:, :] appending_tiles_view
    cdef bint valid, placed
    cdef list positions = []
    
    # Create appending tiles tracker
    appending_tiles_view = np.zeros((h, w), dtype=np.uint8)
    
    for polyomino_data in polyominoes:
        mask = polyomino_data[0]
        offset = polyomino_data[1]
        mask_view = mask
        # mask_h = mask.shape[0]
        # mask_w = mask.shape[1]

        mask_h = 0
        mask_w = 0
        for k in range(mask.shape[0] // 2):
            mask_h = max(mask_h, mask_view[k << 1] + 1)
            mask_w = max(mask_w, mask_view[(k << 1) + 1] + 1)
        
        placed = False
        
        # Try to place at each position
        for j in range(w - mask_w + 1):
            for i in range(h - mask_h + 1):
                valid = True
                
                # Check for collisions
                for k in range(mask.shape[0] // 2):
                    if occupied_tiles[i + mask_view[k << 1], j + mask_view[(k << 1) + 1]]:
                        valid = False
                        break
                
                if valid:
                    for k in range(mask.shape[0] // 2):
                        occupied_tiles[i + mask_view[k << 1], j + mask_view[(k << 1) + 1]] = 1
                        appending_tiles_view[i + mask_view[k << 1], j + mask_view[(k << 1) + 1]] = 1
                    
                    positions.append((i, j, mask, offset))
                    placed = True
                    break
            
            if placed:
                break
        
        if not placed:
            # Revert changes by XORing with appending_tiles
            for i in range(h):
                for j in range(w):
                    if appending_tiles_view[i, j]:
                        occupied_tiles[i, j] = 0
            return None
    
    return positions
