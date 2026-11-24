# cython: language_level=3

# C-level matching function declarations for cimport by bytetrack.pyx

# Compute IOU-based cost matrix (1 - IOU) from flat tlbr arrays
cdef void compute_iou_cost(
    double *atlbrs, int N,
    double *btlbrs, int M,
    double *cost_out
) noexcept nogil

# Fuse detection scores into cost matrix in-place
cdef void fuse_score(
    double *cost_matrix,
    double *det_scores,
    int N, int M
) noexcept nogil

# Solve linear assignment problem with threshold, returns via pointer params
cdef void linear_assignment(
    double *cost_matrix, int N, int M,
    double thresh,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_a, int *n_unmatched_a,
    int *unmatched_b, int *n_unmatched_b
) noexcept nogil
