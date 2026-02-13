# cython: language_level=3

# C-level association function declarations for cimport by ocsort.pyx

# Association function type enum
cdef enum AssoFuncType:
    ASSO_IOU = 0
    ASSO_GIOU = 1
    ASSO_DIOU = 2
    ASSO_CIOU = 3
    ASSO_CT_DIST = 4

# Compute IOU-based distance from flat bbox arrays (no PASCAL VOC +1)
cdef void iou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil

# Dispatch to the correct distance metric based on enum
cdef void asso_dispatch(int func_type, double *bb1, int N, double *bb2, int M, double *out) noexcept nogil

# Solve linear assignment on a NEGATED cost matrix, returns raw match pairs
cdef void linear_assignment(
    double *cost_matrix, int N, int M,
    int *match_a, int *match_b, int *n_matches
) noexcept nogil

# Full first-round association with velocity direction consistency (VDC)
cdef void associate(
    double *dets, int n_dets,
    double *trks, int n_trks,
    double iou_threshold,
    double *velocities,
    double *previous_obs,
    double vdc_weight,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_dets, int *n_unmatched_dets,
    int *unmatched_trks, int *n_unmatched_trks
) noexcept nogil
