# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython matching functions for ByteTrack.

Only the C-level APIs are the public surface for BYTETracker: compute_iou_cost,
fuse_score, lapjv_solve, linear_assignment. Used by bytetrack.pyx via cimport.
"""

cimport cython
from libc.math cimport fmax, fmin
from libc.stdlib cimport malloc, calloc, free

# External C function for linear assignment problem
cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t
    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


# ============================================================
# C-level cdef functions (no Python objects, nogil)
# ============================================================

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void compute_iou_cost(
    double *atlbrs, int N,
    double *btlbrs, int M,
    double *cost_out
) noexcept nogil:
    """
    Compute IOU-based cost matrix (1 - IOU) from flat tlbr arrays.

    atlbrs: flat array of N bounding boxes, each [x1, y1, x2, y2] (N*4 total)
    btlbrs: flat array of M bounding boxes, each [x1, y1, x2, y2] (M*4 total)
    cost_out: flat output array of size N*M, filled with (1 - IOU)
    """
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2
    cdef int ai, bj

    for i in range(N):
        # Precompute area of bbox i (with +1 for PASCAL VOC formula)
        ai = i * 4
        area1 = (atlbrs[ai + 2] - atlbrs[ai + 0] + 1.0) * (atlbrs[ai + 3] - atlbrs[ai + 1] + 1.0)

        for j in range(M):
            bj = j * 4

            # Compute intersection coordinates
            xx1 = fmax(atlbrs[ai + 0], btlbrs[bj + 0])
            yy1 = fmax(atlbrs[ai + 1], btlbrs[bj + 1])
            xx2 = fmin(atlbrs[ai + 2], btlbrs[bj + 2])
            yy2 = fmin(atlbrs[ai + 3], btlbrs[bj + 3])

            # Intersection area with +1 for PASCAL VOC formula
            w = fmax(0.0, xx2 - xx1 + 1.0)
            h = fmax(0.0, yy2 - yy1 + 1.0)
            wh = w * h

            # Area of bbox j with +1
            area2 = (btlbrs[bj + 2] - btlbrs[bj + 0] + 1.0) * (btlbrs[bj + 3] - btlbrs[bj + 1] + 1.0)

            # Cost = 1 - IOU
            cost_out[i * M + j] = 1.0 - wh / (area1 + area2 - wh + 1e-9)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void fuse_score(
    double *cost_matrix,
    double *det_scores,
    int N, int M
) noexcept nogil:
    """
    Fuse detection scores into cost matrix in-place.

    cost_matrix: flat N*M cost matrix (modified in-place)
    det_scores: M detection scores
    N: number of rows (tracks)
    M: number of columns (detections)

    Formula: cost = 1 - (1 - cost) * score  =  1 - iou_sim * score
    """
    cdef int i, j
    cdef double iou_sim

    for i in range(N):
        for j in range(M):
            # Convert cost to similarity, fuse with score, convert back
            iou_sim = 1.0 - cost_matrix[i * M + j]
            cost_matrix[i * M + j] = 1.0 - iou_sim * det_scores[j]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void lapjv_solve(
    double *cost_matrix, int n_rows, int n_cols,
    int *raw_match_a, int *raw_match_b, int *n_raw_matches
) noexcept nogil:
    """
    Solve the linear assignment problem using LAPJV on raw pointer arrays.

    cost_matrix: flat n_rows * n_cols cost matrix
    raw_match_a: output row indices of matches (pre-allocated, size >= min(n_rows, n_cols))
    raw_match_b: output col indices of matches (pre-allocated, size >= min(n_rows, n_cols))
    n_raw_matches: output number of raw matches (before threshold filtering)
    """
    cdef int n = n_rows if n_rows > n_cols else n_cols
    cdef int i, x_val

    # Allocate extended square cost matrix (padded with zeros)
    cdef double *cost_extended = <double *> calloc(n * n, sizeof(double))
    for i in range(n_rows):
        for x_val in range(n_cols):
            cost_extended[i * n + x_val] = cost_matrix[i * n_cols + x_val]

    # Build pointer array for lapjv_internal
    cdef double **cost_ptr = <double **> malloc(n * sizeof(double *))
    for i in range(n):
        cost_ptr[i] = &cost_extended[i * n]

    # Allocate assignment arrays
    cdef int_t *x_c = <int_t *> malloc(n * sizeof(int_t))
    cdef int_t *y_c = <int_t *> malloc(n * sizeof(int_t))

    # Solve assignment
    lapjv_internal(n, cost_ptr, x_c, y_c)

    # Free temporary arrays
    free(cost_ptr)
    free(cost_extended)

    # Extract valid matches (row i matched to column x_c[i])
    cdef int count = 0
    for i in range(n_rows):
        x_val = x_c[i]
        if x_val >= 0 and x_val < n_cols:
            raw_match_a[count] = i
            raw_match_b[count] = x_val
            count += 1

    free(x_c)
    free(y_c)
    n_raw_matches[0] = count


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void linear_assignment(
    double *cost_matrix, int N, int M,
    double thresh,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_a, int *n_unmatched_a,
    int *unmatched_b, int *n_unmatched_b
) noexcept nogil:
    """
    Solve linear assignment problem with threshold filtering.

    cost_matrix: flat N*M cost matrix
    N: number of rows
    M: number of columns
    thresh: cost threshold for valid matches
    match_a: output row indices of valid matches (pre-allocated, size >= min(N,M))
    match_b: output col indices of valid matches (pre-allocated, size >= min(N,M))
    n_matches: output number of valid matches
    unmatched_a: output unmatched row indices (pre-allocated, size >= N)
    n_unmatched_a: output number of unmatched rows
    unmatched_b: output unmatched col indices (pre-allocated, size >= M)
    n_unmatched_b: output number of unmatched cols
    """
    cdef int i, j, ra, rb
    cdef int n_raw = 0
    cdef int matched_count = 0
    cdef int ua_count = 0
    cdef int ub_count = 0

    # Handle empty cost matrix
    if N == 0 or M == 0:
        n_matches[0] = 0
        # All rows are unmatched
        for i in range(N):
            unmatched_a[i] = i
        n_unmatched_a[0] = N
        # All cols are unmatched
        for i in range(M):
            unmatched_b[i] = i
        n_unmatched_b[0] = M
        return

    # Allocate temporary arrays for raw LAPJV result
    cdef int max_matches = N if N < M else M
    cdef int *raw_a = <int *> malloc(max_matches * sizeof(int))
    cdef int *raw_b = <int *> malloc(max_matches * sizeof(int))

    # Solve assignment
    lapjv_solve(cost_matrix, N, M, raw_a, raw_b, &n_raw)

    # Track which rows/cols are matched (using bitmask arrays)
    cdef int *matched_row = <int *> calloc(N, sizeof(int))
    cdef int *matched_col = <int *> calloc(M, sizeof(int))

    # Filter by threshold
    for i in range(n_raw):
        ra = raw_a[i]
        rb = raw_b[i]
        if cost_matrix[ra * M + rb] <= thresh:
            match_a[matched_count] = ra
            match_b[matched_count] = rb
            matched_count += 1
            matched_row[ra] = 1
            matched_col[rb] = 1

    n_matches[0] = matched_count

    # Collect unmatched rows
    for i in range(N):
        if matched_row[i] == 0:
            unmatched_a[ua_count] = i
            ua_count += 1
    n_unmatched_a[0] = ua_count

    # Collect unmatched cols
    for i in range(M):
        if matched_col[i] == 0:
            unmatched_b[ub_count] = i
            ub_count += 1
    n_unmatched_b[0] = ub_count

    # Free temporary arrays
    free(raw_a)
    free(raw_b)
    free(matched_row)
    free(matched_col)
