import time
import gurobipy
from gurobipy import GRB
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ILPResult(NamedTuple):
    """Return value of solve_ilp, bundling the solution with build and solve timings."""
    selected: set[tuple[int, int]]
    build_ms: float  # Time spent building the model (variables + objective + constraints)
    solve_ms: float  # Time spent in model.optimize()


def solve_ilp(
    tile_to_polyomino_id: "np.ndarray",
    polyomino_lengths: list[list[int]],
    max_sampling_distance: "np.ndarray",
    grid_height: int,
    grid_width: int,
    time_limit_seconds: float = 0.5,
) -> ILPResult:
    """
    Solve integer linear program to select minimum set of polyominoes.

    Args:
        tile_to_polyomino_id: 3D array [frame, row, col] -> polyomino_id (-1 if no polyomino)
        polyomino_lengths: polyomino_lengths[frame][polyomino_id] = number of cells
        max_sampling_distance: 2D array [grid_height, grid_width] of max sampling distance per tile
        grid_height: Height of the grid
        grid_width: Width of the grid
        time_limit_seconds: Maximum solver wall-clock time in seconds

    Returns:
        ILPResult with selected polyominoes and build/solve timings in milliseconds
    """
    B = len(polyomino_lengths)  # Number of frames

    # --- Build phase: create env, model, variables, objective, constraints ---
    build_start = time.monotonic()

    # Create a Gurobi environment with empty=True so parameters are set before license handshake,
    # suppressing the banner even in multiprocessing workers using a WLS license.
    env = gurobipy.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    # Create optimization model within the managed environment.
    model = gurobipy.Model("MinCells", env=env)

    # Set time limit for the optimizer to the caller-provided value.
    model.Params.TimeLimit = time_limit_seconds

    # Create variables: x[b, k] = 1 if polyomino k in frame b is selected
    x = {}
    for b in range(B):
        for k in range(len(polyomino_lengths[b])):
            x[(b, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{b}_{k}")

    # Flush pending variable additions before building objective and constraints.
    model.update()

    # Objective: minimize total number of cells in selected polyominoes
    model.setObjective(
        gurobipy.quicksum(x[(b, k)] * polyomino_lengths[b][k] for (b, k) in x),
        GRB.MINIMIZE
    )

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
                    model.addConstr(curr_var == 1)
                    model.addConstr(x[(b_next_avail, k_next)] == 1)
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
                        model.addConstr(gurobipy.quicksum(window_vars) >= curr_var)

            # Always choose first and last time tile is covered
            b_first = pos[0]
            b_last = pos[-1]
            k_first = tile_to_polyomino_id[b_first][n, m]
            k_last = tile_to_polyomino_id[b_last][n, m]
            model.addConstr(x[(b_first, k_first)] == 1)
            model.addConstr(x[(b_last, k_last)] == 1)

    build_ms = (time.monotonic() - build_start) * 1000

    # --- Solve phase: call the optimizer ---
    solve_start = time.monotonic()
    model.optimize()

    # Extract selected polyominoes; guard against no feasible solution found within time limit.
    selected = set()
    if model.SolCount > 0:
        for (b, k), var in x.items():
            # Use > 0.5 threshold for binary variables to handle floating-point precision.
            if var.X > 0.5:
                selected.add((b, k))

    # Dispose model then environment to release WLS license resources promptly.
    model.dispose()
    env.dispose()
    solve_ms = (time.monotonic() - solve_start) * 1000

    return ILPResult(selected=selected, build_ms=build_ms, solve_ms=solve_ms)