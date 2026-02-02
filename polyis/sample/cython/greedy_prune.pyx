# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Greedy temporal covering algorithm for polyomino pruning.

This module implements the greedy algorithm described in p022_exec_prune_PLAN.md
for selecting the minimal number of polyominoes to maintain temporal coverage.
"""

cimport numpy as cnp
import numpy as np
import heapq
from typing import List, Tuple, Dict, Set
from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t
import cython

# Import group_tiles functionality for polyomino extraction
from polyis.pack.group_tiles import group_tiles, free_polyomino_array
from polyis.sample.cython.tile_extractor import extract_tiles_from_polyominoes

# Structure to track polyomino information
cdef class PolyominoInfo:
    """Information about a polyomino in a specific frame."""
    cdef public int32_t frame_idx
    cdef public int32_t poly_id
    cdef public list tiles  # List of (row, col) tuples

    def __init__(self, int32_t frame_idx, int32_t poly_id, list tiles):
        self.frame_idx = frame_idx
        self.poly_id = poly_id
        self.tiles = tiles


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list greedy_prune_polyominoes(
    list polyomino_arrays,
    uint8_t[:, :, :] relevance_bitmaps,
    int32_t[:, :] max_gaps,
    double threshold=0.5
):
    """
    Greedy temporal covering algorithm for polyomino pruning.

    This implements the algorithm from p022_exec_prune_PLAN.md which selects
    the minimum number of polyominoes to maintain temporal coverage constraints.

    Parameters:
        polyomino_arrays: List of polyomino array pointers (uint64) from group_tiles
        relevance_bitmaps: 3D array [frames, height, width] of relevance scores (0-255)
        max_gaps: 2D array [height, width] of maximum sampling gaps per tile position
        threshold: Threshold for converting relevance to binary (default 0.5 = 127/255)

    Returns:
        List of selected frame indices
    """
    cdef int32_t num_frames = relevance_bitmaps.shape[0]
    cdef int32_t height = relevance_bitmaps.shape[1]
    cdef int32_t width = relevance_bitmaps.shape[2]
    cdef uint8_t threshold_uint8 = <uint8_t>(threshold * 255)

    # Convert relevance scores to binary bitmaps
    cdef cnp.ndarray[uint8_t, ndim=3] bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)
    cdef int32_t f, i, j
    for f in range(num_frames):
        for i in range(height):
            for j in range(width):
                if relevance_bitmaps[f, i, j] >= threshold_uint8:
                    bitmaps[f, i, j] = 1

    # Build mapping of which frames contain positive tiles at each position
    # positive_frames[i][j] = sorted list of frame indices where tile (i,j) is positive
    positive_frames = {}
    for i in range(height):
        for j in range(width):
            frames_with_positive = []
            for f in range(num_frames):
                if bitmaps[f, i, j] == 1:
                    frames_with_positive.append(f)
            if frames_with_positive:
                positive_frames[(i, j)] = frames_with_positive

    # Build polyomino information for each frame
    frame_polyominoes = []
    for f in range(num_frames):
        # Extract polyominoes for this frame using group_tiles
        frame_bitmap = bitmaps[f, :, :]

        # Get polyomino pointer and extract tiles directly
        poly_ptr = group_tiles(frame_bitmap, mode=0)  # No padding
        polyominoes = extract_tiles_from_polyominoes(poly_ptr)
        free_polyomino_array(poly_ptr)

        # Store polyomino info with tile positions
        frame_polys = []
        for poly_idx, tiles_list in enumerate(polyominoes):
            poly_info = PolyominoInfo(f, poly_idx, tiles_list)
            frame_polys.append(poly_info)
        frame_polyominoes.append(frame_polys)

    # Track which frames/polyominoes are selected
    selected_frames = set()
    selected_polyominoes = []  # List of (frame_idx, poly_id) tuples

    # Initialize: select all polyominoes at frame 0 that contain positive tiles
    if len(frame_polyominoes) > 0 and len(frame_polyominoes[0]) > 0:
        for poly in frame_polyominoes[0]:
            selected_frames.add(0)
            selected_polyominoes.append((0, poly.poly_id))

    # Track last selected frame for each tile position
    last_selected = {}
    for i in range(height):
        for j in range(width):
            if (i, j) in positive_frames and 0 in positive_frames[(i, j)]:
                last_selected[(i, j)] = 0

    # Priority queue: (deadline, row, col)
    pq = []

    # Initialize priority queue with deadlines from frame 0
    for (i, j) in last_selected:
        deadline = last_selected[(i, j)] + max_gaps[i, j]
        if deadline < num_frames:
            heapq.heappush(pq, (deadline, i, j))

    # Main greedy algorithm loop
    while pq:
        deadline, row, col = heapq.heappop(pq)

        # Skip if this tile was already updated
        if (row, col) in last_selected and last_selected[(row, col)] >= deadline - max_gaps[row, col]:
            continue

        # Find latest frame <= deadline where (row, col) is positive
        pos_frames = positive_frames.get((row, col), [])
        if not pos_frames:
            continue

        target_frame = -1
        for f in reversed(pos_frames):
            if f <= deadline:
                target_frame = f
                break

        # Handle Impossible Covering Constraint (Constraint 4)
        if target_frame == -1 or target_frame <= last_selected.get((row, col), -1):
            # Find next positive frame after deadline
            for f in pos_frames:
                if f > deadline:
                    target_frame = f
                    break

            if target_frame == -1:
                continue  # No more positive frames for this tile

        # Select the polyomino containing (row, col) at target_frame
        for poly in frame_polyominoes[target_frame]:
            if (row, col) in poly.tiles:
                # Add this polyomino if not already selected
                if target_frame not in selected_frames or (target_frame, poly.poly_id) not in selected_polyominoes:
                    selected_frames.add(target_frame)
                    selected_polyominoes.append((target_frame, poly.poly_id))

                # Update last_selected for all tiles in this polyomino
                for (tile_row, tile_col) in poly.tiles:
                    last_selected[(tile_row, tile_col)] = target_frame

                    # Add new deadline to priority queue
                    new_deadline = target_frame + max_gaps[tile_row, tile_col]
                    if new_deadline < num_frames:
                        heapq.heappush(pq, (new_deadline, tile_row, tile_col))
                break

    # Also select all polyominoes at the last frame that contain positive tiles
    if len(frame_polyominoes) > 0 and num_frames > 0:
        last_frame = num_frames - 1
        if last_frame not in selected_frames:
            for poly in frame_polyominoes[last_frame]:
                # Check if polyomino contains any positive tile
                has_positive = False
                for (tile_row, tile_col) in poly.tiles:
                    if bitmaps[last_frame, tile_row, tile_col] == 1:
                        has_positive = True
                        break

                if has_positive:
                    selected_frames.add(last_frame)
                    selected_polyominoes.append((last_frame, poly.poly_id))

    # Return sorted list of selected frame indices
    return sorted(list(selected_frames))