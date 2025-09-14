# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc
from cython.view cimport array as cvarray
cimport cython


ctypedef cnp.uint16_t GROUP_t
ctypedef cnp.uint8_t MASK_t

cdef struct IntVector:
    unsigned short *data
    int top
    int capacity

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef int IntVector_init(IntVector *int_vector, int initial_capacity) noexcept nogil:
    """Initialize an integer vector with initial capacity"""
    if not int_vector:
        return -1
    
    int_vector.data = <unsigned short*>malloc(initial_capacity * sizeof(unsigned short))
    if not int_vector.data:
        return -1
    
    int_vector.top = 0
    int_vector.capacity = initial_capacity
    return 0

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef int IntVector_push(IntVector *int_vector, unsigned short value) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef unsigned short *new_data
    
    if not int_vector:
        return -1
    
    # Check if we need to expand
    if int_vector.top >= int_vector.capacity:
        new_capacity = int_vector.capacity * 2
        new_data = <unsigned short*>realloc(int_vector.data, new_capacity * sizeof(unsigned short))
        if not new_data:
            return -1  # Memory allocation failed
        
        int_vector.data = new_data
        int_vector.capacity = new_capacity
    
    # Push the value
    int_vector.data[int_vector.top] = value
    int_vector.top += 1
    return 0

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef unsigned short IntVector_pop(IntVector *int_vector) noexcept:
#     """Pop a value from the vector"""
#     if not int_vector or int_vector.top <= 0:
#         return -1  # Stack is empty or invalid
    
#     int_vector.top -= 1
#     return int_vector.data[int_vector.top]

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void IntVector_cleanup(IntVector *int_vector) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if int_vector:
        if int_vector.data:
            free(<void*>int_vector.data)
            int_vector.data = NULL
        int_vector.top = 0
        int_vector.capacity = 0


# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef IntVector _find_connected_tiles(
    GROUP_t[:, :] bitmap,
    unsigned short start_i,
    unsigned short start_j
) noexcept nogil:
    """
    Fast Cython implementation of find_connected_tiles using a C-based stack.

    This function modifies the bitmap in-place to mark visited tiles.
    The algorithm uses flood fill: it starts with a unique value at (start_i, start_j)
    and spreads that value to all connected tiles, collecting their positions.

    Args:
        bitmap: 2D memoryview of the groups bitmap (modified in-place)
        start_i: Starting row index
        start_j: Starting column index

    Returns:
        IntVector: IntVector containing coordinate pairs for connected tiles
    """
    cdef unsigned short h = bitmap.shape[0]
    cdef unsigned short w = bitmap.shape[1]
    cdef GROUP_t value = bitmap[start_i, start_j]
    
    # Create IntVector for filled coordinates on stack memory
    cdef IntVector filled
    if IntVector_init(&filled, 16) == -1:  # Initial capacity of 32 (16 coordinate pairs)
        # Return empty IntVector on initialization failure
        IntVector_cleanup(&filled)
        return filled
    
    # Create C stack for coordinates on stack memory
    cdef IntVector stack
    if IntVector_init(&stack, 16):  # Initial capacity of 16
        # Initialization failed, cleanup filled and return empty
        IntVector_cleanup(&stack)
        IntVector_cleanup(&filled)
        return filled
    
    cdef unsigned short i, j, _i, _j, di

    # Directions: up, left, down, right
    cdef unsigned short[4] directions_i = [-1, 0, 1, 0]
    cdef unsigned short[4] directions_j = [0, -1, 0, 1]

    # Push initial coordinates
    IntVector_push(&stack, start_i)
    IntVector_push(&stack, start_j)

    while stack.top > 0:
        j = stack.data[stack.top - 1]
        i = stack.data[stack.top - 2]
        stack.top -= 2
        # j = IntVector_pop(&stack)
        # i = IntVector_pop(&stack)

        # Mark current position as visited and add to result
        bitmap[i, j] = value
        if IntVector_push(&filled, i) or IntVector_push(&filled, j):
            # Memory allocation failed
            break

        # Check all 4 directions for unvisited connected tiles
        for di in range(4):
            _i = i + directions_i[di]
            _j = j + directions_j[di]

            if 0 <= _i < h and 0 <= _j < w:
                # Add neighbors that are non-zero and different from current value
                # (meaning they haven't been visited yet)
                if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                    # If either push failed, we have a memory issue
                    if IntVector_push(&stack, _i) or IntVector_push(&stack, _j):
                        # Memory allocation failed
                        IntVector_cleanup(&stack)
                        IntVector_cleanup(&filled)
                        return filled

    # Free the stack's data before returning
    IntVector_cleanup(&stack)
    return filled


# @cython.boundscheck(False)
# @cython.wraparound(False)
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
    cdef unsigned short h = bitmap_input.shape[0]
    cdef unsigned short w = bitmap_input.shape[1]
    cdef unsigned short i, j, k, group_id, min_i, min_j, max_i, max_j, mask_i, mask_j, tile_i, tile_j, num_pairs
    cdef IntVector connected_tiles
    cdef list bins = []
    # cdef set visited = set()

    # Create groups array with unique IDs
    groups = np.arange(1, h * w + 1, dtype=np.uint16).reshape((h, w))
    # Mask groups by bitmap - only keep group IDs where bitmap has 1s
    groups = np.where(bitmap_input, groups, 0)
    cdef GROUP_t[:, :] groups_view = groups
    cdef MASK_t[:, :] mask_view

    # Process each cell
    for i in range(h):
        for j in range(w):
            group_id = groups_view[i, j]
            # if group_id == 0 or group_id in visited:
            if group_id == 0 or bitmap_input[(group_id - 1) // w, (group_id - 1) % w] == 0:
                continue

            # Find connected tiles - returns IntVector
            connected_tiles = _find_connected_tiles(groups_view, i, j)
            if connected_tiles.top == 0:
                # Clean up empty IntVector
                IntVector_cleanup(&connected_tiles)
                continue
            
            # Find bounding box directly from IntVector data
            num_pairs = connected_tiles.top // 2
            
            # Initialize with first coordinate pair
            min_i = connected_tiles.data[0]
            max_i = connected_tiles.data[0]
            min_j = connected_tiles.data[1]
            max_j = connected_tiles.data[1]
            
            # Find min/max through all coordinate pairs
            for k in range(1, num_pairs):
                tile_i = connected_tiles.data[k << 1]        # i coordinate
                tile_j = connected_tiles.data[(k << 1) + 1]  # j coordinate
                
                if tile_i < min_i:
                    min_i = tile_i
                elif tile_i > max_i:
                    max_i = tile_i
                    
                if tile_j < min_j:
                    min_j = tile_j
                elif tile_j > max_j:
                    max_j = tile_j

            # Create mask
            mask_h = max_i - min_i + 1
            mask_w = max_j - min_j + 1
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            mask_view = mask

            # Fill mask - iterate through IntVector data directly
            for k in range(num_pairs):
                tile_i = connected_tiles.data[k << 1]        # i coordinate
                tile_j = connected_tiles.data[(k << 1) + 1]  # j coordinate
                mask_view[tile_i - min_i, tile_j - min_j] = 1
            # Clean up IntVector memory
            IntVector_cleanup(&connected_tiles)
            bitmap_input[i, j] = 0

            bins.append((mask, (min_i, min_j)))

    return bins
