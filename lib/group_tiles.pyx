# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free, realloc
import cython


ctypedef cnp.uint16_t GROUP_t
ctypedef cnp.uint8_t MASK_t


cdef struct IntStack:
    unsigned short *data
    int top
    int capacity


# Directions: up, left, down, right
cdef unsigned short[4] DIRECTIONS_I = [-1, 0, 1, 0]
cdef unsigned short[4] DIRECTIONS_J = [0, -1, 0, 1]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef int IntStack_init(IntStack *stack, int initial_capacity) noexcept nogil:
    """Initialize an integer vector with initial capacity"""
    # if not stack:
    #     return -1
    
    stack.data = <unsigned short*>malloc(<size_t>initial_capacity * sizeof(unsigned short))
    # if not stack.data:
    #     return -1
    
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef int IntStack_push(IntStack *stack, unsigned short value) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef unsigned short *new_data
    
    # if not stack:
    #     return -1
    
    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <unsigned short*>realloc(<void*>stack.data,
                                            <size_t>new_capacity * sizeof(unsigned short))
        # if not new_data:
        #     return -1  # Memory allocation failed
        
        stack.data = new_data
        stack.capacity = new_capacity
    
    # Push the value
    stack.data[stack.top] = value
    stack.top += 1
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef void IntStack_cleanup(IntStack *stack) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if stack:
        if stack.data:
            free(<void*>stack.data)
            stack.data = NULL
        stack.top = 0
        stack.capacity = 0


cdef struct MaskAndOffset:
    IntStack mask
    int offset_i
    int offset_j


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef void MaskAndOffset_cleanup(MaskAndOffset *mask_and_offset) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if mask_and_offset:
        if mask_and_offset.mask.data:
            IntStack_cleanup(&mask_and_offset.mask)


cdef struct MaskAndOffsetStack:
    MaskAndOffset *mo_data
    int top
    int capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef int MaskAndOffsetStack_init(
    MaskAndOffsetStack *stack,
    int initial_capacity
) noexcept nogil:
    """Initialize an mask and offset pointer vector with initial capacity"""
    # if not stack:
    #     return -1
    
    stack.mo_data = <MaskAndOffset*>malloc(<size_t>initial_capacity * sizeof(MaskAndOffset))
    # if not stack.mo_data:
    #     return -1
    
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef int MaskAndOffsetStack_push(
    MaskAndOffsetStack *stack,
    MaskAndOffset value
) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef MaskAndOffset *new_data
    
    # if not stack:
    #     return -1
    
    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <MaskAndOffset*>realloc(<void*>stack.mo_data,
                                            <size_t>new_capacity * sizeof(MaskAndOffset))
        # if not new_data:
        #     return -1  # Memory allocation failed
        
        stack.mo_data = new_data
        stack.capacity = new_capacity
    
    # Push the value
    stack.mo_data[stack.top] = value
    stack.top += 1
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef void MaskAndOffsetStack_cleanup(MaskAndOffsetStack *stack) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if stack:
        if stack.mo_data:
            for i in range(stack.top):
                MaskAndOffset_cleanup(&stack.mo_data[i])
            free(<void*>stack.mo_data)
            stack.mo_data = NULL
        stack.top = 0
        stack.capacity = 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef IntStack _find_connected_tiles(
    # unsigned int[:, :] bitmap,
    unsigned int* bitmap,
    unsigned short h,
    unsigned short w,
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
        IntStack: IntStack containing coordinate pairs for connected tiles
    """
    # cdef unsigned short h = <unsigned short>bitmap.shape[0]
    # cdef unsigned short w = <unsigned short>bitmap.shape[1]
    # cdef unsigned int value = bitmap[start_i, start_j]
    cdef unsigned int value = bitmap[start_i * w + start_j]
    
    # Create IntStack for filled coordinates on stack memory
    cdef IntStack filled
    if IntStack_init(&filled, 16) == -1:  # Initial capacity of 32 (16 coordinate pairs)
        # Return empty IntStack on initialization failure
        IntStack_cleanup(&filled)
        return filled
    
    # Create C stack for coordinates on stack memory
    cdef IntStack stack
    if IntStack_init(&stack, 16):  # Initial capacity of 16
        # Initialization failed, cleanup filled and return empty
        IntStack_cleanup(&stack)
        IntStack_cleanup(&filled)
        return filled
    
    cdef unsigned short i, j, _i, _j, di

    # Push initial coordinates
    IntStack_push(&stack, start_i)
    IntStack_push(&stack, start_j)

    while stack.top > 0:
        j = stack.data[stack.top - 1]
        i = stack.data[stack.top - 2]
        stack.top -= 2
        # j = IntStack_pop(&stack)
        # i = IntStack_pop(&stack)

        # Mark current position as visited and add to result
        # bitmap[i, j] = value
        bitmap[i * w + j] = value
        IntStack_push(&filled, i)
        IntStack_push(&filled, j)
        # if IntStack_push(&filled, i) or IntStack_push(&filled, j):
        #     # Memory allocation failed
        #     break

        # Check all 4 directions for unvisited connected tiles
        for di in range(4):
            _i = i + DIRECTIONS_I[di]
            _j = j + DIRECTIONS_J[di]

            if 0 <= _i < h and 0 <= _j < w:
                # Add neighbors that are non-zero and different from current value
                # (meaning they haven't been visited yet)
                # if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                if bitmap[_i * w + _j] != 0 and bitmap[_i * w + _j] != value:
                    # If either push failed, we have a memory issue
                    IntStack_push(&stack, _i)
                    IntStack_push(&stack, _j)
                    # if IntStack_push(&stack, _i) or IntStack_push(&stack, _j):
                    #     # Memory allocation failed
                    #     IntStack_cleanup(&stack)
                    #     IntStack_cleanup(&filled)
                    #     return filled

    # Free the stack's data before returning
    IntStack_cleanup(&stack)
    return filled


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef unsigned long long group_tiles(cnp.uint8_t[:, :] bitmap_input):
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
    cdef unsigned short h = <unsigned short>bitmap_input.shape[0]
    cdef unsigned short w = <unsigned short>bitmap_input.shape[1]
    cdef unsigned short group_id, min_i, min_j, max_i, max_j, tile_i, tile_j, num_pairs
    cdef int i, j, k
    cdef IntStack connected_tiles
    cdef list bins = []
    cdef MaskAndOffset mask_and_offset
    cdef MaskAndOffsetStack *mask_and_offset_stack
    mask_and_offset_stack = <MaskAndOffsetStack*>malloc(sizeof(MaskAndOffsetStack))
    MaskAndOffsetStack_init(mask_and_offset_stack, 16)

    # Create groups array with unique IDs
    cdef unsigned int* groups = <unsigned int*>calloc(h * w, sizeof(unsigned int))
    # Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for i in range(h):
        for j in range(w):
            if bitmap_input[i, j]:
                groups[i * w + j] = i * w + j + 1
    # cdef MASK_t[:, :] mask_view

    # Process each cell
    for i in range(h):
        for j in range(w):
            # group_id = groups_view[i, j]
            group_id = groups[i * w + j]
            # if group_id == 0 or group_id in visited:
            if group_id == 0 or bitmap_input[(group_id - 1) // w, (group_id - 1) % w] == 0:
                continue

            # Find connected tiles - returns IntStack
            # connected_tiles = _find_connected_tiles(groups_view, i, j)
            connected_tiles = _find_connected_tiles(groups, h, w, i, j)
            if connected_tiles.top == 0:
                # Clean up empty IntStack
                IntStack_cleanup(&connected_tiles)
                continue
            
            # Find bounding box directly from IntStack data
            num_pairs = <unsigned short>(connected_tiles.top // 2)
            
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

            for k in range(num_pairs):
                connected_tiles.data[k << 1] -= min_i        # i coordinate
                connected_tiles.data[(k << 1) + 1] -= min_j  # j coordinate
            
            mask_and_offset.mask = connected_tiles
            mask_and_offset.offset_i = min_i
            mask_and_offset.offset_j = min_j
            MaskAndOffsetStack_push(mask_and_offset_stack, mask_and_offset)

            # # Create mask
            # mask_h = max_i - min_i + 1
            # mask_w = max_j - min_j + 1
            # mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            # mask_view = mask

            # # Fill mask - iterate through IntStack data directly
            # for k in range(num_pairs):
            #     tile_i = connected_tiles.data[k << 1]        # i coordinate
            #     tile_j = connected_tiles.data[(k << 1) + 1]  # j coordinate
            #     mask_view[tile_i - min_i, tile_j - min_j] = 1
            # # Clean up IntStack memory
            # IntStack_cleanup(&connected_tiles)
            bitmap_input[i, j] = 0

            # bins.append((mask, (min_i, min_j)))

    free(groups)
    return <unsigned long long>mask_and_offset_stack


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
def group_tiles_adapter(cnp.uint8_t[:, :] bitmap_input) -> list:
    cdef MaskAndOffsetStack *mask_and_offset_stack = <MaskAndOffsetStack*>group_tiles(bitmap_input)
    cdef list bins = []
    cdef MaskAndOffset mask_and_offset
    cdef IntStack connected_tiles
    cdef unsigned short max_i, max_j, tile_i, tile_j, num_pairs
    cdef unsigned short * data

    for i in range(mask_and_offset_stack.top):
        mask_and_offset = mask_and_offset_stack.mo_data[i]
        connected_tiles = mask_and_offset.mask
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

        # Clean up IntStack memory
        IntStack_cleanup(&connected_tiles)
        bins.append((mask, (mask_and_offset.offset_i, mask_and_offset.offset_j)))
    MaskAndOffsetStack_cleanup(mask_and_offset_stack)
    free(mask_and_offset_stack)
    return bins
