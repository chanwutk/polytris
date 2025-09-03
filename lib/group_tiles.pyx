# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
cimport cython


ctypedef cnp.uint16_t GROUP_t
ctypedef cnp.uint8_t MASK_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list find_connected_tiles(GROUP_t[:, :] bitmap, int start_i, int start_j):
    """
    Fast Cython implementation of find_connected_tiles using a manual stack.

    This function modifies the bitmap in-place to mark visited tiles.
    The algorithm uses flood fill: it starts with a unique value at (start_i, start_j)
    and spreads that value to all connected tiles, collecting their positions.

    Args:
        bitmap: 2D memoryview of the groups bitmap (modified in-place)
        start_i: Starting row index
        start_j: Starting column index

    Returns:
        list: List of (i, j) tuples for connected tiles
    """
    cdef int h = bitmap.shape[0]
    cdef int w = bitmap.shape[1]
    cdef GROUP_t value = bitmap[start_i, start_j]
    cdef list filled = []
    cdef list stack = [(start_i, start_j)]
    cdef int i, j, _i, _j, di
    cdef tuple pos

    # Directions: up, left, down, right
    cdef int[4] directions_i = [-1, 0, 1, 0]
    cdef int[4] directions_j = [0, -1, 0, 1]

    while stack:
        pos = stack.pop()
        i, j = pos[0], pos[1]

        # Mark current position as visited and add to result
        bitmap[i, j] = value
        filled.append((i, j))

        # Check all 4 directions for unvisited connected tiles
        for di in range(4):
            _i = i + directions_i[di]
            _j = j + directions_j[di]

            if 0 <= _i < h and 0 <= _j < w:
                # Add neighbors that are non-zero and different from current value
                # (meaning they haven't been visited yet)
                if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                    stack.append((_i, _j))

    return filled


@cython.boundscheck(False)
@cython.wraparound(False)
def group_tiles(cnp.uint8_t[:, :] bitmap_input):
    """
    Fast Cython implementation of group_tiles.

    Groups connected tiles into polyominoes.

    Args:
        bitmap_input: 2D numpy array memoryview (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection

    Returns:
        list: List of polyominoes, where each polyomino is:
            - group_id: unique id of the group
            - mask: masking of the polyomino as a 2D numpy array
            - offset: offset of the mask from the top left corner of the bitmap
    """
    # Convert to numpy array for operations that need it
    bitmap_np = np.asarray(bitmap_input, dtype=np.uint8)

    cdef int h = bitmap_np.shape[0]
    cdef int w = bitmap_np.shape[1]
    cdef int i, j, group_id, min_i, min_j, max_i, max_j, mask_i, mask_j
    cdef list connected_tiles, bins = []
    # cdef set visited = set()

    # Create groups array with unique IDs
    groups = np.arange(1, h * w + 1, dtype=np.uint16).reshape((h, w))
    # Mask groups by bitmap - only keep group IDs where bitmap has 1s
    groups = np.where(bitmap_np, groups, 0)
    cdef GROUP_t[:, :] groups_view = groups

    # Process each cell
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            group_id = groups_view[i, j]
            # if group_id == 0 or group_id in visited:
            if group_id == 0 or bitmap_np[(group_id - 1) // w, (group_id - 1) % w] == 0:
                continue

            # Find connected tiles
            connected_tiles = find_connected_tiles(groups_view, i, j)
            if not connected_tiles:
                continue

            # Convert to numpy array and find bounds
            connected_array = np.array(connected_tiles, dtype=np.uint8)
            if connected_array.size == 0:
                continue

            # Find bounding box
            min_i = np.min(connected_array[:, 0])
            max_i = np.max(connected_array[:, 0])
            min_j = np.min(connected_array[:, 1])
            max_j = np.max(connected_array[:, 1])

            # Create mask
            mask_h = max_i - min_i + 1
            mask_w = max_j - min_j + 1
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

            # Fill mask
            for tile_i, tile_j in connected_tiles:
                mask[tile_i - min_i, tile_j - min_j] = 1

            bins.append((group_id, mask, (min_i, min_j)))
            # visited.add(group_id)
            bitmap_np[i, j] = 0

    return bins
