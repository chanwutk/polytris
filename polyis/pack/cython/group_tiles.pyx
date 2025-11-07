# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free, qsort
import cython

from polyis.pack.cython.utilities cimport Coordinate, CoordinateStack, Polyomino, PolyominoStack, \
                       CoordinateStack_init, CoordinateStack_push, CoordinateStack_cleanup, \
                       PolyominoStack_init, PolyominoStack_push, PolyominoStack_cleanup

ctypedef cnp.uint16_t GROUP_t
ctypedef cnp.uint8_t MASK_t


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
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
@cython.nonecheck(False)  # type: ignore
cdef CoordinateStack _find_connected_tiles(
    unsigned int* bitmap,
    short h,
    short w,
    short start_i,
    short start_j,
    cnp.uint8_t[:, :] bitmap_input,
    int mode
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
        bitmap_input: 2D numpy array memoryview (uint8) representing the grid of tiles,
                     where 1 indicates a tile with detection and 0 indicates no detection
        mode: The mode of tile padding to apply
            - 0: No padding
            - 1: Connected padding
            - 2: Disconnected padding
    Returns:
        CoordinateStack: CoordinateStack containing coordinates for connected tiles
    """
    cdef short i, j, _i, _j
    cdef int di
    cdef CoordinateStack filled, stack
    cdef Coordinate coord
    cdef char[4] DIRECTIONS_I = [-1, 0, 1, 0]  # type: ignore
    cdef char[4] DIRECTIONS_J = [0, -1, 0, 1]  # type: ignore
    cdef unsigned int value = bitmap[start_i * w + start_j]  # type: ignore
    cdef unsigned char curr_occupancy
    cdef unsigned int next_group

    if CoordinateStack_init(&filled, 16):
        # Return empty CoordinateStack on initialization failure
        CoordinateStack_cleanup(&filled)
        return filled

    if CoordinateStack_init(&stack, 16):
        # Initialization failed, cleanup filled and return empty
        CoordinateStack_cleanup(&stack)
        CoordinateStack_cleanup(&filled)
        return filled

    # Push initial coordinates
    coord.y = start_i
    coord.x = start_j
    CoordinateStack_push(&stack, coord)

    while stack.top > 0:
        # Pop coordinate from stack
        stack.top -= 1
        coord = stack.data[stack.top]  # type: ignore
        i = coord.y
        j = coord.x

        if bitmap[i * w + j] == value and (j != start_j or i != start_i):  # type: ignore
            continue  # Already visited

        # Mark current position as visited and add to result
        bitmap[i * w + j] = value  # type: ignore
        coord.y = i
        coord.x = j
        CoordinateStack_push(&filled, coord)

        curr_occupancy = bitmap_input[i, j]
        # Check all 4 directions for unvisited connected tiles
        for di in range(4):
            _i = i + DIRECTIONS_I[di]  # type: ignore
            _j = j + DIRECTIONS_J[di]  # type: ignore
            next_group = bitmap[_i * w + _j]

            if 0 <= _i < h and 0 <= _j < w:
                # Add neighbors that are non-zero and different from current value
                # (meaning they haven't been visited yet)
                if next_group != 0 and next_group != value:  # type: ignore
                    if mode == 0 or mode == 2 or (mode == 1 and (curr_occupancy == 1 or bitmap_input[_i, _j] == 1)):
                        coord.y = _i
                        coord.x = _j
                        CoordinateStack_push(&stack, coord)

    # Free the stack's data before returning
    CoordinateStack_cleanup(&stack)
    return filled


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void _add_padding(cnp.uint8_t[:, :] bitmap,
                       unsigned short h,
                       unsigned short w) noexcept nogil:
    cdef char[4] DIRECTIONS_I = [-1, 0, 1, 0]  # type: ignore
    cdef char[4] DIRECTIONS_J = [0, -1, 0, 1]  # type: ignore
    cdef int i, j, di
    cdef int _i, _j

    for i in range(h):
        for j in range(w):
            if bitmap[i, j] != 1:  # type: ignore
                continue

            for di in range(4):
                _i = i + DIRECTIONS_I[di]  # type: ignore
                _j = j + DIRECTIONS_J[di]  # type: ignore

                if 0 <= _i < h and 0 <= _j < w and bitmap[_i, _j] == 0:  # type: ignore
                    bitmap[_i, _j] = 2  # type: ignore


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def group_tiles(cnp.uint8_t[:, :] bitmap_input, int tilepadding_mode):
    cdef unsigned short h = <unsigned short>bitmap_input.shape[0]
    cdef unsigned short w = <unsigned short>bitmap_input.shape[1]
    cdef short group_id, min_i, min_j
    cdef int i, j, k, tile_i, tile_j
    cdef CoordinateStack connected_tiles
    cdef Polyomino polyomino
    cdef PolyominoStack *polyomino_stack
    polyomino_stack = <PolyominoStack*>malloc(<size_t>sizeof(PolyominoStack))
    PolyominoStack_init(polyomino_stack, 16)

    if tilepadding_mode != 0:
        _add_padding(bitmap_input, h, w)

    # Create groups array with unique IDs
    cdef unsigned int* groups = <unsigned int*>calloc(<size_t>(h * w), <size_t>sizeof(unsigned int))
    # Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for i in range(h):
        for j in range(w):
            if bitmap_input[i, j]:  # type: ignore
                groups[i * w + j] = i * w + j + 1  # type: ignore

    # Process each cell
    for i in range(h):
        for j in range(w):
            group_id = groups[i * w + j]  # type: ignore
            if group_id == 0 or bitmap_input[(group_id - 1) // w, (group_id - 1) % w] == 0:  # type: ignore
                continue

            # Find connected tiles - returns CoordinateStack
            connected_tiles = _find_connected_tiles(groups, h, w, <short>i, <short>j,
                                                    bitmap_input, tilepadding_mode)
            if connected_tiles.top == 0:
                # Clean up empty CoordinateStack
                CoordinateStack_cleanup(&connected_tiles)
                continue

            # Find bounding box directly from CoordinateStack data
            # Initialize with first coordinate
            min_i = connected_tiles.data[0].y  # type: ignore
            min_j = connected_tiles.data[0].x  # type: ignore

            # Find min coordinates through all coordinates
            for k in range(1, connected_tiles.top):
                tile_i = connected_tiles.data[k].y  # type: ignore
                tile_j = connected_tiles.data[k].x  # type: ignore

                if tile_i < min_i:
                    min_i = tile_i

                if tile_j < min_j:
                    min_j = tile_j

            # Normalize coordinates by subtracting min_i and min_j
            for k in range(connected_tiles.top):
                connected_tiles.data[k].y -= min_i  # type: ignore
                connected_tiles.data[k].x -= min_j  # type: ignore

            polyomino.mask = connected_tiles
            polyomino.offset_i = min_i
            polyomino.offset_j = min_j
            PolyominoStack_push(polyomino_stack, polyomino)

            bitmap_input[i, j] = 0  # type: ignore

    # Sort polyominoes by mask length (descending order) before returning
    qsort(<void*>polyomino_stack.mo_data,
          <size_t>polyomino_stack.top,
          <size_t>sizeof(Polyomino),
          &compare_polyomino_by_mask_length)

    free(<void*>groups)
    return <unsigned long long>polyomino_stack


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def free_polyimino_stack(unsigned long long polyomino_stack_ptr) -> int:
    cdef int num_polyominoes = (<PolyominoStack*>polyomino_stack_ptr).top
    PolyominoStack_cleanup(<PolyominoStack*>polyomino_stack_ptr)
    free(<void*>polyomino_stack_ptr)
    return num_polyominoes
