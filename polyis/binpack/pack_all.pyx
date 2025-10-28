# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport numpy as cnp
import cython
from libc.stdlib cimport malloc, free

import numpy as np

from polyis.binpack.utilities cimport IntStack, Polyomino, PolyominoStack, \
                       IntStack_init, IntStack_push, IntStack_cleanup, \
                       Polyomino_cleanup


cdef struct Position:
    # Original offset of the polyomino
    unsigned char ox
    unsigned char oy

    # Packed offset of the polyomino
    unsigned char px
    unsigned char py

    # Mask of the polyomino
    Polyomino polyomino


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void Position_cleanup(Position *position) noexcept nogil:
    """Free the stack's data array (stack itself is on stack memory)"""
    if position:
        Polyomino_cleanup(&(position.polyomino))


cdef struct PositionStack:
    Position * position_data
    int position_top
    int position_capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int PositionStack_init(PositionStack *stack, int initial_capacity) noexcept nogil:
    """Initialize an mask and offset pointer vector with initial capacity"""
    # if not stack:
    #     return -1
    
    stack.position_data = <Position*>malloc(<size_t>initial_capacity * sizeof(Position))
    # if not stack.position_data:
    #     return -1
    
    stack.position_top = 0
    stack.position_capacity = initial_capacity
    return 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
def pack_all(list polyominoes_stacks, int h, int w) -> list:
    # Try to pack all polyominoes from all stacks into the smallest board possible
    # Collect all polyominoes with their stack references
    cdef unsigned long long polyominoes_stack_ptr
    cdef PolyominoStack *polyominoes_stack
    cdef int stack_idx, poly_idx
    cdef Polyomino * all_polyominoes
    cdef Polyomino polyomino
    cdef int count_polyominos = 0

    # Try to pack all polyominoes greedily
    cdef list all_positions = []
    cdef cnp.uint8_t[:, :] occupied_tiles = np.zeros((h, w), dtype=np.uint8)
    cdef list current_positions = []

    for polyominoes_stack_ptr in polyominoes_stacks:
        polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
        count_polyominos += polyominoes_stack.top
    
    # If no polyominoes, return empty result
    if count_polyominos == 0:
        return []

    all_polyominoes = <Polyomino*>malloc(count_polyominos * sizeof(Polyomino))
    
    # Iterate through all polyominoes stacks and collect individual polyominoes
    cdef int i = 0
    for polyominoes_stack_ptr in polyominoes_stacks:
        polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
        for poly_idx in range(polyominoes_stack.top):
            all_polyominoes[i] = polyominoes_stack.mo_data[poly_idx]  # type: ignore
            i += 1
    
    # Try to pack each polyomino into the current board
    for poly_idx in range(count_polyominos):
        polyomino = all_polyominoes[poly_idx]  # type: ignore

        positions = pack_single_polyomino(
            polyominoes_stack_ptr,
            poly_idx,
            h,
            w,
            occupied_tiles
        )
        
        if positions is not None:
            # Successfully packed
            current_positions.extend(positions)
        else:
            # Current board is full, add it to results and start a new board
            if len(current_positions) > 0:
                all_positions.append(current_positions)
            current_positions = []
            occupied_tiles = np.zeros((h, w), dtype=np.uint8)
            
            # Try to pack into the new board
            positions = pack_single_polyomino(
                polyominoes_stack_ptr,
                poly_idx,
                h,
                w,
                occupied_tiles
            )
            if positions is not None:
                current_positions.extend(positions)
    
    # Add the last board if it has any positions
    if len(current_positions) > 0:
        all_positions.append(current_positions)
    
    free(<void*>all_polyominoes)
    return all_positions


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void pack_single_polyomino(
    unsigned long long polyominoes_stack_ptr,
    int poly_idx,
    int h,
    int w,
    cnp.uint8_t[:, :] occupied_tiles
) noexcept nogil:
    # Extract the single polyomino we want to pack
    cdef PolyominoStack *polyominoes_stack = <PolyominoStack*>polyominoes_stack_ptr
    cdef Polyomino polyomino = polyominoes_stack.mo_data[poly_idx]  # type: ignore
    cdef IntStack mask = polyomino.mask
    cdef tuple offset = (polyomino.offset_i, polyomino.offset_j)
    
    cdef int i, j, k, mask_h, mask_w
    cdef bint valid, placed
    cdef list positions = []
    cdef IntStack appending_tiles
    cdef unsigned short tile_i, tile_j
    cdef cnp.uint8_t[:] mask_array_view
    
    # Calculate mask dimensions from the mask data
    mask_h = 0
    mask_w = 0
    for k in range(mask.top // 2):
        tile_i = mask.data[k << 1]  # type: ignore
        tile_j = mask.data[(k << 1) + 1]  # type: ignore
        if tile_i >= mask_h:
            mask_h = tile_i + 1
        if tile_j >= mask_w:
            mask_w = tile_j + 1
    
    placed = <bint>False
    
    # Create appending tiles tracker
    IntStack_init(&appending_tiles, 16)
    
    # Try to place at each position
    for j in range(w - mask_w + 1):
        for i in range(h - mask_h + 1):
            valid = <bint>True
            
            # Check for collisions
            for k in range(mask.top // 2):
                tile_i = mask.data[k << 1]  # type: ignore
                tile_j = mask.data[(k << 1) + 1]  # type: ignore
                if occupied_tiles[i + tile_i, j + tile_j]:  # type: ignore
                    valid = <bint>False
                    break
            
            if valid:
                # Allocate a NumPy array for the mask
                mask_array = np.empty(mask.top, dtype=np.uint8)
                mask_array_view = mask_array
                
                # Place the polyomino
                for k in range(mask.top // 2):
                    tile_i = mask.data[k << 1]  # type: ignore
                    tile_j = mask.data[(k << 1) + 1]  # type: ignore
                    occupied_tiles[i + tile_i, j + tile_j] = 1  # type: ignore
                    IntStack_push(&appending_tiles, <unsigned short>(i + tile_i))
                    IntStack_push(&appending_tiles, <unsigned short>(j + tile_j))
                    
                    mask_array_view[k << 1] = tile_i  # type: ignore
                    mask_array_view[(k << 1) + 1] = tile_j  # type: ignore
                
                positions.append((i, j, mask_array, offset))
                placed = <bint>True
                IntStack_cleanup(&appending_tiles)
                return positions
        
        if placed:
            break
    
    # Failed to pack
    IntStack_cleanup(&appending_tiles)
    return None