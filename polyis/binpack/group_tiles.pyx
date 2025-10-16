# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free, qsort
import cython

from utilities cimport IntStack, Polyomino, PolyominoStack, IntStack_init, \
                       IntStack_push, IntStack_cleanup, PolyominoStack_init, \
                       PolyominoStack_push, PolyominoStack_cleanup

ctypedef cnp.uint16_t GROUP_t
ctypedef cnp.uint8_t MASK_t


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef int compare_polyomino_by_mask_length(const void *a, const void *b) noexcept nogil:
    """
    Comparison function for qsort to sort polyominoes by mask length (descending order).
    Returns negative if a should come before b, positive if b should come before a.
    """
    # Compare by mask length (top field of IntStack) in descending order
    # Larger masks first (negative return means a comes before b)
    return (<Polyomino*>b).mask.top - (<Polyomino*>a).mask.top


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
cdef IntStack _find_connected_tiles(
    unsigned int* bitmap,
    unsigned short h,
    unsigned short w,
    unsigned short start_i,
    unsigned short start_j
) noexcept nogil:
    """
    Groups connected tiles into a polyomino.

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
    cdef unsigned short i, j, _i, _j
    cdef int di
    cdef IntStack filled, stack
    cdef char[4] DIRECTIONS_I = [-1, 0, 1, 0]  # type: ignore
    cdef char[4] DIRECTIONS_J = [0, -1, 0, 1]  # type: ignore
    cdef unsigned int value = bitmap[start_i * w + start_j]  # type: ignore
    
    if IntStack_init(&filled, 16):
        # Return empty IntStack on initialization failure
        IntStack_cleanup(&filled)
        return filled
    
    if IntStack_init(&stack, 16):
        # Initialization failed, cleanup filled and return empty
        IntStack_cleanup(&stack)
        IntStack_cleanup(&filled)
        return filled

    # Push initial coordinates
    IntStack_push(&stack, start_i)
    IntStack_push(&stack, start_j)

    while stack.top > 0:
        j = stack.data[stack.top - 1]  # type: ignore
        i = stack.data[stack.top - 2]  # type: ignore
        stack.top -= 2

        # Mark current position as visited and add to result
        bitmap[i * w + j] = value  # type: ignore
        IntStack_push(&filled, i)
        IntStack_push(&filled, j)
        # if IntStack_push(&filled, i) or IntStack_push(&filled, j):
        #     # Memory allocation failed
        #     break

        # Check all 4 directions for unvisited connected tiles
        for di in range(4):
            _i = i + DIRECTIONS_I[di]  # type: ignore
            _j = j + DIRECTIONS_J[di]  # type: ignore

            if 0 <= _i < h and 0 <= _j < w:
                # Add neighbors that are non-zero and different from current value
                # (meaning they haven't been visited yet)
                if bitmap[_i * w + _j] != 0 and bitmap[_i * w + _j] != value:  # type: ignore
                    IntStack_push(&stack, _i)
                    IntStack_push(&stack, _j)
                    # If either push failed, we have a memory issue
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
def group_tiles(cnp.uint8_t[:, :] bitmap_input):
    cdef unsigned short h = <unsigned short>bitmap_input.shape[0]
    cdef unsigned short w = <unsigned short>bitmap_input.shape[1]
    cdef unsigned short group_id, min_i, min_j, tile_i, tile_j, num_pairs
    cdef int i, j, k
    cdef IntStack connected_tiles
    cdef Polyomino polyomino
    cdef PolyominoStack *polyomino_stack
    polyomino_stack = <PolyominoStack*>malloc(sizeof(PolyominoStack))
    PolyominoStack_init(polyomino_stack, 16)

    # Create groups array with unique IDs
    cdef unsigned int* groups = <unsigned int*>calloc(h * w, sizeof(unsigned int))
    # Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for i in range(h):
        for j in range(w):
            if bitmap_input[i, j]:  # type: ignore
                groups[i * w + j] = i * w + j + 1  # type: ignore

    # Process each cell
    for i in range(h):
        for j in range(w):
            # group_id = groups_view[i, j]
            group_id = groups[i * w + j]  # type: ignore
            # if group_id == 0 or group_id in visited:
            if group_id == 0 or bitmap_input[(group_id - 1) // w, (group_id - 1) % w] == 0:  # type: ignore
                continue

            # Find connected tiles - returns IntStack
            connected_tiles = _find_connected_tiles(groups, h, w, <unsigned short>i, <unsigned short>j)
            if connected_tiles.top == 0:
                # Clean up empty IntStack
                IntStack_cleanup(&connected_tiles)
                continue
            
            # Find bounding box directly from IntStack data
            num_pairs = <unsigned short>(connected_tiles.top // 2)
            
            # Initialize with first coordinate pair
            min_i = connected_tiles.data[0]  # type: ignore
            min_j = connected_tiles.data[1]  # type: ignore
            
            # Find min/max through all coordinate pairs
            for k in range(1, num_pairs):
                tile_i = connected_tiles.data[k << 1]        # type: ignore
                tile_j = connected_tiles.data[(k << 1) + 1]  # type: ignore
                
                if tile_i < min_i:
                    min_i = tile_i
                    
                if tile_j < min_j:
                    min_j = tile_j

            for k in range(num_pairs):
                connected_tiles.data[k << 1] -= min_i        # type: ignore
                connected_tiles.data[(k << 1) + 1] -= min_j  # type: ignore
            
            polyomino.mask = connected_tiles
            polyomino.offset_i = min_i
            polyomino.offset_j = min_j
            PolyominoStack_push(polyomino_stack, polyomino)

            bitmap_input[i, j] = 0  # type: ignore

    # Sort polyominoes by mask length (descending order) before returning
    qsort(polyomino_stack.mo_data,
          polyomino_stack.top,
          sizeof(Polyomino),
          &compare_polyomino_by_mask_length)

    free(groups)
    return <unsigned long long>polyomino_stack


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
def free_polyimino_stack(unsigned long long polyomino_stack_ptr) -> int:
    cdef int num_polyominoes = (<PolyominoStack*>polyomino_stack_ptr).top
    PolyominoStack_cleanup(<PolyominoStack*>polyomino_stack_ptr)
    free(<void*>polyomino_stack_ptr)
    return num_polyominoes
