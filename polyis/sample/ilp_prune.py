"""
Integer Linear Programming (ILP) implementation for polyomino pruning.

This module implements the ILP algorithm described in p022_exec_prune_PLAN.md
for selecting the minimal number of tiles while maintaining temporal coverage.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import pulp
from polyis.pack.group_tiles import group_tiles, free_polyomino_array
from polyis.sample.cython.tile_extractor import extract_tiles_from_polyominoes


def ilp_prune_polyominoes(
    polyomino_arrays: List[int],
    relevance_bitmaps: np.ndarray,
    max_gaps: np.ndarray,
    threshold: float = 0.5,
    solver: Optional[str] = None,
    time_limit: Optional[int] = None
) -> List[int]:
    """
    Integer Linear Programming algorithm for polyomino pruning.

    This implements the ILP formulation from p022_exec_prune_PLAN.md which
    minimizes the total number of tiles selected while maintaining temporal
    coverage constraints.

    Parameters:
        polyomino_arrays: List of polyomino array pointers (uint64) from group_tiles
        relevance_bitmaps: 3D array [frames, height, width] of relevance scores (0-255)
        max_gaps: 2D array [height, width] of maximum sampling gaps per tile position
        threshold: Threshold for converting relevance to binary (default 0.5 = 127/255)
        solver: PuLP solver to use (None for default, 'GLPK', 'CBC', etc.)
        time_limit: Time limit in seconds for solver (None for no limit)

    Returns:
        List of selected frame indices
    """
    num_frames, height, width = relevance_bitmaps.shape
    threshold_uint8 = int(threshold * 255)

    # Convert relevance scores to binary bitmaps
    bitmaps = (relevance_bitmaps >= threshold_uint8).astype(np.uint8)

    # Build polyomino information for each frame
    frame_polyominoes = []
    polyomino_vars = []  # Will store (frame, poly_id, tiles, var)

    for f in range(num_frames):
        # Extract polyominoes for this frame using group_tiles
        frame_bitmap = bitmaps[f, :, :]

        # Get polyomino pointer and extract tiles directly
        poly_ptr = group_tiles(frame_bitmap, mode=0)  # No padding
        polyominoes = extract_tiles_from_polyominoes(poly_ptr)
        free_polyomino_array(poly_ptr)

        # Store polyomino info
        frame_polys = []
        for poly_idx, tiles_list in enumerate(polyominoes):
            frame_polys.append({
                'frame': f,
                'poly_id': poly_idx,
                'tiles': tiles_list
            })
        frame_polyominoes.append(frame_polys)

    # Create ILP problem
    prob = pulp.LpProblem("MinimizeTiles", pulp.LpMinimize)

    # Create decision variables for each polyomino
    x_vars = {}
    for f, frame_polys in enumerate(frame_polyominoes):
        for poly in frame_polys:
            var_name = f"x_{f}_{poly['poly_id']}"
            x = pulp.LpVariable(var_name, cat='Binary')
            x_vars[(f, poly['poly_id'])] = x
            polyomino_vars.append({
                'frame': f,
                'poly_id': poly['poly_id'],
                'tiles': poly['tiles'],
                'var': x
            })

    # Objective: Minimize total number of tiles selected
    prob += pulp.lpSum([
        len(p['tiles']) * p['var']
        for p in polyomino_vars
    ]), "TotalTiles"

    # Build mapping of which frames contain positive tiles at each position
    positive_frames = {}
    for i in range(height):
        for j in range(width):
            frames_with_positive = []
            for f in range(num_frames):
                if bitmaps[f, i, j] == 1:
                    frames_with_positive.append(f)
            if frames_with_positive:
                positive_frames[(i, j)] = frames_with_positive

    # Build mapping of which polyominoes contain each tile position
    tile_to_polys = {}
    for p in polyomino_vars:
        for (tile_row, tile_col) in p['tiles']:
            key = (p['frame'], tile_row, tile_col)
            if key not in tile_to_polys:
                tile_to_polys[key] = []
            tile_to_polys[key].append(p)

    # Add temporal coverage constraints
    for (row, col), pos_frames in positive_frames.items():
        if not pos_frames:
            continue

        # For each positive instance of this tile, ensure coverage within max_gap
        for idx, curr_frame in enumerate(pos_frames[:-1]):
            # Get the polyomino variable for this tile at curr_frame
            curr_polys = tile_to_polys.get((curr_frame, row, col), [])
            if not curr_polys:
                continue

            # For simplicity, assume one polyomino per tile (connected components)
            curr_var = curr_polys[0]['var']

            # Find next positive frame for this tile
            next_frame = pos_frames[idx + 1]
            max_gap = max_gaps[row, col]
            deadline = curr_frame + max_gap

            if next_frame > deadline:
                # Impossible Covering Constraint (Constraint 4)
                # Both current and next must be selected
                prob += curr_var == 1, f"Impossible_{row}_{col}_{curr_frame}_curr"

                # Find polyomino at next_frame containing (row, col)
                next_polys = tile_to_polys.get((next_frame, row, col), [])
                if next_polys:
                    prob += next_polys[0]['var'] == 1, f"Impossible_{row}_{col}_{curr_frame}_next"
            else:
                # Normal temporal constraint: at least one frame in window must be selected
                # if current frame is selected
                window_frames = [f for f in pos_frames[idx + 1:] if f <= deadline]
                if window_frames:
                    window_vars = []
                    for wf in window_frames:
                        w_polys = tile_to_polys.get((wf, row, col), [])
                        if w_polys:
                            window_vars.append(w_polys[0]['var'])

                    if window_vars:
                        # If current is selected, at least one in window must be selected
                        prob += pulp.lpSum(window_vars) >= curr_var, f"Temporal_{row}_{col}_{curr_frame}"

    # Boundary constraints: force first and last frame polyominoes with positive tiles
    # First frame
    if frame_polyominoes and positive_frames:
        for poly in frame_polyominoes[0]:
            # Check if polyomino contains any positive tile
            has_positive = False
            for (tile_row, tile_col) in poly['tiles']:
                if bitmaps[0, tile_row, tile_col] == 1:
                    has_positive = True
                    break

            if has_positive and (0, poly['poly_id']) in x_vars:
                prob += x_vars[(0, poly['poly_id'])] == 1, f"Boundary_first_{poly['poly_id']}"

    # Last frame
    if num_frames > 0 and frame_polyominoes:
        last_frame = num_frames - 1
        for poly in frame_polyominoes[last_frame]:
            # Check if polyomino contains any positive tile
            has_positive = False
            for (tile_row, tile_col) in poly['tiles']:
                if bitmaps[last_frame, tile_row, tile_col] == 1:
                    has_positive = True
                    break

            if has_positive and (last_frame, poly['poly_id']) in x_vars:
                prob += x_vars[(last_frame, poly['poly_id'])] == 1, f"Boundary_last_{poly['poly_id']}"

    # Solve the ILP problem
    if solver:
        if solver == 'GLPK':
            solver_obj = pulp.GLPK(msg=0, options=['--tmlim', str(time_limit)] if time_limit else [])
        elif solver == 'CBC':
            solver_obj = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        else:
            solver_obj = None
    else:
        solver_obj = None

    if solver_obj:
        prob.solve(solver_obj)
    else:
        prob.solve()

    # Extract solution: selected frame indices
    selected_frames = set()
    for (frame, poly_id), var in x_vars.items():
        if var.varValue == 1:
            selected_frames.add(frame)

    # Return sorted list of selected frame indices
    return sorted(list(selected_frames))