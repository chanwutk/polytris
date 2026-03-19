import numpy as np
from polyis.sample.ilp.gurobi import ILPResult

def solve_ilp(
    tile_to_polyomino_id_arr: np.ndarray,
    polyomino_lengths: list[list[int]],
    max_sampling_distance_arr: np.ndarray,
    grid_height: int,
    grid_width: int,
    time_limit_seconds: float = 0.1,
) -> ILPResult:
    """
    Solve the polyomino-pruning ILP using the Gurobi C API.

    Identical formulation to ``polyis.sample.ilp.gurobi.solve_ilp`` but
    bypasses the gurobipy Python layer and calls libgurobi directly, removing
    Python-object overhead from model construction and solution extraction.

    Variable indexing
    -----------------
    Variables are numbered 0 … num_vars-1.  Frame *b*, polyomino *k* maps to::

        var_index(b, k) = frame_offsets[b] + k

    where ``frame_offsets[b] = sum(num_polyominoes_per_frame[0..b-1])``.

    Constraint encoding
    -------------------
    For each tile ``(n, m)`` covered in two or more frames:

    * **First / last frame forced** — ``x[v_first] = 1``, ``x[v_last] = 1``.
    * **Consecutive pair** ``(b_curr, b_next)``:

      - Gap too large (``b_next > b_curr + max_dist[n, m]``):
        both endpoints forced (mandatory bridge).
      - Otherwise (window constraint):
        ``Σ x[window_vars] ≥ x[b_curr]``.

    Args:
        tile_to_polyomino_id_arr: int16 array of shape
            ``[num_frames, grid_height, grid_width]``.  Entry ``[b, n, m]``
            is the polyomino ID covering tile ``(n, m)`` in frame *b*, or
            ``-1`` if the tile is uncovered.
        polyomino_lengths: ``polyomino_lengths[b][k]`` is the number of cells
            in polyomino *k* of frame *b*.
        max_sampling_distance_arr: float64 array of shape
            ``[grid_height, grid_width]``.  Maximum allowed frame gap before a
            mandatory bridge constraint is inserted for tile ``(n, m)``.
        grid_height: Number of tile rows.
        grid_width: Number of tile columns.
        time_limit_seconds: Maximum solver wall-clock time in seconds.

    Returns:
        ILPResult with the set of selected ``(frame, polyomino_id)`` pairs and
        separate build / solve timings in milliseconds.

    Raises:
        MemoryError: If any internal C allocation fails.
        RuntimeError: If ``GRBemptyenv``, ``GRBstartenv``, or ``GRBnewmodel``
            returns a non-zero error code (e.g. invalid license).
    """
    ...
