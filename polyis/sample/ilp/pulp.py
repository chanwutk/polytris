import pulp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


ILP_SOLVER_TIME_LIMIT_SECONDS = 10


def build_ilp_solver():
    # Fail fast with an actionable error when the Gurobi backend is missing from PuLP.
    if not hasattr(pulp, 'GUROBI'):
        raise RuntimeError(
            "PuLP Gurobi solver is unavailable. Install `gurobipy` and configure a valid Gurobi license."
        )

    # Use the in-process Gurobi backend and keep the existing solve time limit.
    return pulp.GUROBI(msg=False, timeLimit=ILP_SOLVER_TIME_LIMIT_SECONDS)


def solve_ilp(
    tile_to_polyomino_id: "np.ndarray",
    polyomino_lengths: list[list[int]],
    max_sampling_distance: "np.ndarray",
    grid_height: int,
    grid_width: int
) -> set[tuple[int, int]]:
    """
    Solve integer linear program to select minimum set of polyominoes.

    Args:
        tile_to_polyomino_id: 3D array [frame, row, col] -> polyomino_id (-1 if no polyomino)
        polyomino_lengths: polyomino_lengths[frame][polyomino_id] = number of cells
        max_sampling_distance: 2D array [grid_height, grid_width] of max sampling distance per tile
        grid_height: Height of the grid
        grid_width: Width of the grid

    Returns:
        Set of (frame_idx, polyomino_id) tuples representing selected polyominoes
    """
    B = len(polyomino_lengths)  # Number of frames

    # Create optimization problem
    prob = pulp.LpProblem("MinCells", pulp.LpMinimize)

    # Create variables: x[b, k] = 1 if polyomino k in frame b is selected
    x = {}
    for b in range(B):
        for k in range(len(polyomino_lengths[b])):
            x[(b, k)] = pulp.LpVariable(f"x_{b}_{k}", cat='Binary')

    # Objective: minimize total number of cells in selected polyominoes
    prob += pulp.lpSum([x[(b, k)] * polyomino_lengths[b][k] for (b, k) in x])

    # Temporal constraints: for each cell (n, m)
    for n in range(grid_height):
        for m in range(grid_width):
            # Find all frames where this cell is covered by a polyomino
            pos = [b for b in range(B) if tile_to_polyomino_id[b][n, m] >= 0]

            if len(pos) <= 1:
                continue  # No temporal constraint needed

            # For each consecutive pair of frames where this cell appears
            for i in range(len(pos) - 1):
                b_curr = pos[i]
                b_next_avail = pos[i + 1]
                t_limit = b_curr + max_sampling_distance[n, m]  # Per-tile sampling distance

                # Get the polyomino IDs at current and next frames
                k_curr = tile_to_polyomino_id[b_curr][n, m]
                k_next = tile_to_polyomino_id[b_next_avail][n, m]

                curr_var = x[(b_curr, k_curr)]

                if b_next_avail > t_limit:
                    # Gap too large: both polyominoes must be selected (mandatory bridge)
                    prob += curr_var == 1
                    prob += x[(b_next_avail, k_next)] == 1
                else:
                    # Window constraint: if current is selected, at least one in window must be selected
                    # Use pointer-based iteration to avoid O(n²) list slicing
                    j = i + 1
                    while j < len(pos) and pos[j] <= t_limit:
                        j += 1
                    window_frames = pos[i+1:j]  # Single slice, no filtering needed
                    if window_frames:
                        window_vars = [x[(b, tile_to_polyomino_id[b][n, m])] 
                                      for b in window_frames]
                        prob += pulp.lpSum(window_vars) >= curr_var

            # Always choose first and last time tile is covered
            b_first = pos[0]
            b_last = pos[-1]
            k_first = tile_to_polyomino_id[b_first][n, m]
            k_last = tile_to_polyomino_id[b_last][n, m]
            prob += x[(b_first, k_first)] == 1
            prob += x[(b_last, k_last)] == 1

    # Build one solver for this problem, then close it to release WLS resources promptly.
    solver = build_ilp_solver()
    # Solve the ILP with the configured Gurobi backend and a bounded solve time.
    prob.solve(solver)
    # Dispose of the solver model after the solve so the Gurobi environment can clean up.
    solver.close()

    # Extract selected polyominoes
    selected = set()
    for (b, k), var in x.items():
        if pulp.value(var) == 1:
            selected.add((b, k))

    return selected