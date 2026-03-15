# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython wrapper around the Gurobi C API for the polyomino-pruning ILP solver.

The solver formulation is identical to polyis/sample/ilp/gurobi.py; this
module bypasses the gurobipy Python layer and calls libgurobi directly,
removing Python-object overhead from model construction and solution extraction.

Variable indexing
-----------------
Variables are numbered 0 … num_vars-1.  Frame b, polyomino k maps to

    var_index(b, k) = frame_offsets[b] + k

where frame_offsets[b] = sum(num_polyominoes_per_frame[0..b-1]).

Constraint encoding
-------------------
For each tile (n, m) with len(pos) >= 2:

  * First and last coverage frames are forced:
        x[v_first] = 1,  x[v_last] = 1          (GRB_EQUAL, rhs=1)

  * Consecutive pair (b_curr, b_next):
    - If b_next > b_curr + max_dist[n,m]  (gap too large):
          x[v_curr] = 1,  x[v_next] = 1         (mandatory bridge)
    - Otherwise (window constraint):
          Σ x[window_vars]  ≥  x[v_curr]
          ↔ Σ x[window_vars] − x[v_curr] ≥ 0    (GRB_GREATER_EQUAL, rhs=0)
"""

from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport int16_t

cimport numpy as cnp
import numpy as np

# ── Gurobi C API declarations ────────────────────────────────────────────────
cdef extern from "gurobi_c.h":
    ctypedef struct GRBenv:
        pass
    ctypedef struct GRBmodel:
        pass

    # Variable type and constraint sense constants
    char GRB_BINARY           # 'B'
    char GRB_GREATER_EQUAL    # '>'
    char GRB_EQUAL            # '='

    # Environment
    int  GRBemptyenv(GRBenv **envP)
    int  GRBstartenv(GRBenv *env)
    void GRBfreeenv(GRBenv *env)

    # Model
    int GRBnewmodel(GRBenv *env, GRBmodel **modelP, const char *Pname,
                    int numvars, double *obj, double *lb, double *ub,
                    char *vtype, char **varnames)
    int GRBfreemodel(GRBmodel *model)
    int GRBupdatemodel(GRBmodel *model)
    int GRBoptimize(GRBmodel *model)

    # Parameters
    int GRBsetintparam(GRBenv *env, const char *paramname, int value)
    int GRBsetdblparam(GRBenv *env, const char *paramname, double value)

    # Constraints
    int GRBaddconstr(GRBmodel *model, int numnz, int *cind, double *cval,
                     char sense, double rhs, const char *constrname)

    # Attributes
    int GRBgetintattr(GRBmodel *model, const char *attrname, int *valueP)
    int GRBgetdblattrelement(GRBmodel *model, const char *attrname,
                             int element, double *valueP)

    # Model environment (for post-creation parameter setting)
    GRBenv *GRBgetenv(GRBmodel *model)

# ── Timing ───────────────────────────────────────────────────────────────────
# Use Python's time.monotonic() — call overhead (<1 µs) is negligible relative
# to Gurobi model-build and solve times (typically in the tens of milliseconds).
from time import monotonic as _monotonic

cdef inline double _now_ms():
    """Return current time in milliseconds."""
    return _monotonic() * 1000.0

# ── Public solver ────────────────────────────────────────────────────────────
from polyis.sample.ilp.gurobi import ILPResult, ILP_SOLVER_TIME_LIMIT_SECONDS


def solve_ilp(
    cnp.ndarray tile_to_polyomino_id_arr not None,
    list polyomino_lengths,
    cnp.ndarray max_sampling_distance_arr not None,
    int grid_height,
    int grid_width,
) -> ILPResult:
    """
    Solve the polyomino-pruning ILP using the Gurobi C API.

    Args:
        tile_to_polyomino_id_arr: int16 array [num_frames, grid_height, grid_width]
        polyomino_lengths: list[list[int]], cell counts per (frame, polyomino)
        max_sampling_distance_arr: float64 array [grid_height, grid_width]
        grid_height: grid height
        grid_width: grid width

    Returns:
        ILPResult(selected, build_ms, solve_ms)
    """
    cdef double build_start = _now_ms()

    # Ensure correct dtypes and memory layout for typed memoryviews.
    cdef cnp.int16_t[:, :, :]  tile_ids = np.ascontiguousarray(
        tile_to_polyomino_id_arr, dtype=np.int16)
    cdef cnp.float64_t[:, :]   max_dist  = np.ascontiguousarray(
        max_sampling_distance_arr, dtype=np.float64)

    cdef int B = len(polyomino_lengths)

    # ── Variable-index bookkeeping ───────────────────────────────────────────
    # frame_offsets[b] = first variable index for frame b
    # frame_offsets[B] = total number of variables
    cdef int *frame_offsets = <int *>malloc((B + 1) * sizeof(int))
    if not frame_offsets:
        raise MemoryError("frame_offsets")

    frame_offsets[0] = 0
    cdef int b, k
    for b in range(B):
        frame_offsets[b + 1] = frame_offsets[b] + len(polyomino_lengths[b])
    cdef int num_vars = frame_offsets[B]

    # ── Objective / bound / type arrays for GRBnewmodel ─────────────────────
    cdef double *obj_arr  = <double *>malloc(num_vars * sizeof(double))
    cdef double *lb_arr   = <double *>calloc(num_vars,  sizeof(double))  # 0.0
    cdef double *ub_arr   = <double *>malloc(num_vars * sizeof(double))
    cdef char   *type_arr = <char   *>malloc(num_vars * sizeof(char))

    if not obj_arr or not lb_arr or not ub_arr or not type_arr:
        free(frame_offsets); free(obj_arr); free(lb_arr); free(ub_arr); free(type_arr)
        raise MemoryError("variable arrays")

    # Fill objective coefficients (cell counts) and variable metadata.
    cdef int var_idx
    cdef list lengths_b
    for b in range(B):
        lengths_b = polyomino_lengths[b]
        for k in range(len(lengths_b)):
            var_idx          = frame_offsets[b] + k
            obj_arr[var_idx] = <double>lengths_b[k]
            ub_arr[var_idx]  = 1.0
            type_arr[var_idx] = GRB_BINARY

    # ── Temporary buffers for constraint building ────────────────────────────
    # pos   : frames where the current tile is covered  (max B entries)
    # cind  : variable indices in one constraint         (max B+1 entries)
    # cval  : coefficients in one constraint             (max B+1 entries)
    cdef int    *pos  = <int    *>malloc(B * sizeof(int))
    cdef int    *cind = <int    *>malloc((B + 1) * sizeof(int))
    cdef double *cval = <double *>malloc((B + 1) * sizeof(double))

    if not pos or not cind or not cval:
        free(frame_offsets); free(obj_arr); free(lb_arr); free(ub_arr); free(type_arr)
        free(pos); free(cind); free(cval)
        raise MemoryError("constraint buffers")

    # ── Build phase ──────────────────────────────────────────────────────────
    # Create environment with output suppressed (before license handshake).
    cdef GRBenv   *env   = NULL
    cdef GRBmodel *model = NULL

    if GRBemptyenv(&env) != 0:
        free(frame_offsets); free(obj_arr); free(lb_arr); free(ub_arr); free(type_arr)
        free(pos); free(cind); free(cval)
        raise RuntimeError("GRBemptyenv failed")

    GRBsetintparam(env, "OutputFlag", 0)

    if GRBstartenv(env) != 0:
        GRBfreeenv(env)
        free(frame_offsets); free(obj_arr); free(lb_arr); free(ub_arr); free(type_arr)
        free(pos); free(cind); free(cval)
        raise RuntimeError("GRBstartenv failed — check Gurobi license")

    # Create model with all variables pre-populated.
    if GRBnewmodel(env, &model, "MinCells",
                   num_vars, obj_arr, lb_arr, ub_arr, type_arr, NULL) != 0:
        GRBfreeenv(env)
        free(frame_offsets); free(obj_arr); free(lb_arr); free(ub_arr); free(type_arr)
        free(pos); free(cind); free(cval)
        raise RuntimeError("GRBnewmodel failed")

    # Release variable arrays — Gurobi has copied the data internally.
    free(obj_arr);  obj_arr  = NULL
    free(lb_arr);   lb_arr   = NULL
    free(ub_arr);   ub_arr   = NULL
    free(type_arr); type_arr = NULL

    # Set per-model parameters via the model's own environment.
    GRBsetdblparam(GRBgetenv(model), "TimeLimit",
                   <double>ILP_SOLVER_TIME_LIMIT_SECONDS)

    # Flush pending variable additions so indices are stable.
    GRBupdatemodel(model)

    # ── Add temporal constraints ─────────────────────────────────────────────
    cdef int    n, m, i, j
    cdef int    pos_count, window_count, c_idx
    cdef int    b_curr, b_next_avail, b_first, b_last
    cdef int    k_curr, k_next, k_first, k_last
    cdef double t_limit

    for n in range(grid_height):
        for m in range(grid_width):
            # Collect frames where this tile is covered by some polyomino.
            pos_count = 0
            for b in range(B):
                if tile_ids[b, n, m] >= 0:
                    pos[pos_count] = b
                    pos_count += 1

            if pos_count <= 1:
                continue  # No temporal constraint for this tile

            # Force first coverage frame.
            b_first = pos[0]
            k_first = tile_ids[b_first, n, m]
            cind[0] = frame_offsets[b_first] + k_first
            cval[0] = 1.0
            GRBaddconstr(model, 1, cind, cval, GRB_EQUAL, 1.0, NULL)

            # Force last coverage frame (skip if same as first).
            b_last = pos[pos_count - 1]
            k_last = tile_ids[b_last, n, m]
            if b_last != b_first:
                cind[0] = frame_offsets[b_last] + k_last
                cval[0] = 1.0
                GRBaddconstr(model, 1, cind, cval, GRB_EQUAL, 1.0, NULL)

            # Per-pair temporal constraints.
            for i in range(pos_count - 1):
                b_curr       = pos[i]
                b_next_avail = pos[i + 1]
                t_limit      = b_curr + max_dist[n, m]
                k_curr       = tile_ids[b_curr,       n, m]
                k_next       = tile_ids[b_next_avail, n, m]

                if b_next_avail > t_limit:
                    # Gap exceeds sampling distance: both endpoints forced.
                    cind[0] = frame_offsets[b_curr] + k_curr
                    cval[0] = 1.0
                    GRBaddconstr(model, 1, cind, cval, GRB_EQUAL, 1.0, NULL)

                    cind[0] = frame_offsets[b_next_avail] + k_next
                    cval[0] = 1.0
                    GRBaddconstr(model, 1, cind, cval, GRB_EQUAL, 1.0, NULL)

                else:
                    # Window constraint: Σ(window_vars) − x[curr] ≥ 0
                    # Find end of window (pointer scan, O(1) amortised).
                    j = i + 1
                    while j < pos_count and pos[j] <= t_limit:
                        j += 1
                    window_count = j - (i + 1)
                    if window_count == 0:
                        continue

                    # Window variable coefficients (+1 each).
                    for c_idx in range(window_count):
                        b       = pos[i + 1 + c_idx]
                        cind[c_idx] = frame_offsets[b] + tile_ids[b, n, m]
                        cval[c_idx] = 1.0

                    # Current variable coefficient (−1).
                    cind[window_count] = frame_offsets[b_curr] + k_curr
                    cval[window_count] = -1.0

                    GRBaddconstr(model, window_count + 1, cind, cval,
                                 GRB_GREATER_EQUAL, 0.0, NULL)

    cdef double build_ms = _now_ms() - build_start

    # ── Solve phase ──────────────────────────────────────────────────────────
    cdef double solve_start = _now_ms()
    GRBoptimize(model)

    # ── Extract solution ─────────────────────────────────────────────────────
    cdef int    sol_count = 0
    cdef double var_val
    selected = set()

    GRBgetintattr(model, "SolCount", &sol_count)
    if sol_count > 0:
        for b in range(B):
            for k in range(len(polyomino_lengths[b])):
                var_idx = frame_offsets[b] + k
                GRBgetdblattrelement(model, "X", var_idx, &var_val)
                if var_val > 0.5:
                    selected.add((b, k))

    # ── Cleanup ──────────────────────────────────────────────────────────────
    GRBfreemodel(model)
    GRBfreeenv(env)
    free(frame_offsets)
    free(pos)
    free(cind)
    free(cval)

    cdef double solve_ms = _now_ms() - solve_start

    return ILPResult(selected=selected, build_ms=build_ms, solve_ms=solve_ms)
