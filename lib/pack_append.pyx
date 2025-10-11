# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc

ctypedef cnp.uint8_t DTYPE_t


cdef struct IntStack:
    unsigned short *data
    int top
    int capacity

cdef struct MaskAndOffset:
    IntStack mask
    int offset_i
    int offset_j

cdef struct MaskAndOffsetStack:
    MaskAndOffset *mo_data
    int top
    int capacity


cdef int IntStack_init(IntStack *stack, int initial_capacity) noexcept nogil:
    """Initialize an integer vector with initial capacity"""
    stack.data = <unsigned short*>malloc(<size_t>initial_capacity * sizeof(unsigned short))
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


cdef int IntStack_push(IntStack *stack, unsigned short value) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef unsigned short *new_data
    
    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <unsigned short*>realloc(<void*>stack.data,
                                            <size_t>new_capacity * sizeof(unsigned short))
        stack.data = new_data
        stack.capacity = new_capacity
    
    # Push the value
    stack.data[stack.top] = value
    stack.top += 1
    return 0


cdef void IntStack_cleanup(IntStack *stack) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if stack:
        if stack.data:
            free(<void*>stack.data)
            stack.data = NULL
        stack.top = 0
        stack.capacity = 0


cdef void MaskAndOffset_cleanup(MaskAndOffset *mask_and_offset) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if mask_and_offset:
        if mask_and_offset.mask.data:
            IntStack_cleanup(&mask_and_offset.mask)


cdef int MaskAndOffsetStack_init(
    MaskAndOffsetStack *stack,
    int initial_capacity
) noexcept nogil:
    """Initialize an mask and offset pointer vector with initial capacity"""
    stack.mo_data = <MaskAndOffset*>malloc(<size_t>initial_capacity * sizeof(MaskAndOffset))
    stack.top = 0
    stack.capacity = initial_capacity
    return 0


cdef int MaskAndOffsetStack_push(
    MaskAndOffsetStack *stack,
    MaskAndOffset value
) noexcept nogil:
    """Push a value onto the vector, expanding if necessary"""
    cdef int new_capacity
    cdef MaskAndOffset *new_data
    
    # Check if we need to expand
    if stack.top >= stack.capacity:
        new_capacity = stack.capacity * 2
        new_data = <MaskAndOffset*>realloc(<void*>stack.mo_data,
                                            <size_t>new_capacity * sizeof(MaskAndOffset))
        stack.mo_data = new_data
        stack.capacity = new_capacity
    
    # Push the value
    stack.mo_data[stack.top] = value
    stack.top += 1
    return 0


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
    cdef MaskAndOffsetStack *polyominoes_stack = <MaskAndOffsetStack*>polyominoes
    cdef MaskAndOffset mask_and_offset
    cdef IntStack mask
    cdef IntStack appending_tiles
    cdef unsigned short tile_i, tile_j
    
    # Create appending tiles tracker
    IntStack_init(&appending_tiles, 16)
    
    for idx in range(polyominoes_stack.top):
        mask_and_offset = polyominoes_stack.mo_data[idx]
        mask = mask_and_offset.mask
        offset = (mask_and_offset.offset_i, mask_and_offset.offset_j)
        
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
                    # Place the polyomino
                    for k in range(mask.top // 2):
                        tile_i = mask.data[k << 1]
                        tile_j = mask.data[(k << 1) + 1]
                        occupied_tiles[i + tile_i, j + tile_j] = 1
                        # appending_tiles_view[i + tile_i, j + tile_j] = 1
                        IntStack_push(&appending_tiles, <unsigned short>(i + tile_i))
                        IntStack_push(&appending_tiles, <unsigned short>(j + tile_j))
                    
                    positions.append((i, j, offset))
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


def pack_append_adapter(
    list polyominoes,
    int h,
    int w,
    cnp.uint8_t[:, :] occupied_tiles
):
    # convert polyominoes to MaskAndOffsetStack
    cdef MaskAndOffsetStack polyominoes_stack
    MaskAndOffsetStack_init(&polyominoes_stack, len(polyominoes))
    
    cdef MaskAndOffset mask_and_offset
    cdef IntStack mask
    cdef cnp.uint8_t[:, :] mask_array
    cdef tuple offset
    cdef int i, j, k, mask_h, mask_w
    
    for polyomino in polyominoes:
        mask_array, offset = polyomino
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
        
        # Create MaskAndOffset structure
        mask_and_offset.mask = mask
        mask_and_offset.offset_i = offset[0]
        mask_and_offset.offset_j = offset[1]
        
        MaskAndOffsetStack_push(&polyominoes_stack, mask_and_offset)

    return pack_append(<unsigned long long>&polyominoes_stack, h, w, occupied_tiles)