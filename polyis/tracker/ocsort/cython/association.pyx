# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of association functions for OC-SORT.
C-level cdef functions (iou_batch_c, associate_c, linear_assignment_c, etc.)
are used by ocsort.pyx via cimport. Python def wrappers are kept for
backward compatibility.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, fmax, fmin, acos, atan, fabs, M_PI
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport int32_t

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
cdef void iou_batch_c(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute IOU between two sets of flat bbox arrays [x1,y1,x2,y2]."""
    cdef int i, j, ai, bj
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2
    for i in range(N):
        ai = i * 4
        area1 = (bb1[ai+2] - bb1[ai+0]) * (bb1[ai+3] - bb1[ai+1])
        for j in range(M):
            bj = j * 4
            xx1 = fmax(bb1[ai+0], bb2[bj+0])
            yy1 = fmax(bb1[ai+1], bb2[bj+1])
            xx2 = fmin(bb1[ai+2], bb2[bj+2])
            yy2 = fmin(bb1[ai+3], bb2[bj+3])
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            area2 = (bb2[bj+2] - bb2[bj+0]) * (bb2[bj+3] - bb2[bj+1])
            out[i * M + j] = wh / (area1 + area2 - wh + 1e-9)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void giou_batch_c(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute GIOU between two sets of flat bbox arrays."""
    cdef int i, j, ai, bj
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_val, iou_val
    cdef double xxc1, yyc1, xxc2, yyc2, area_enclose
    for i in range(N):
        ai = i * 4
        area1 = (bb1[ai+2] - bb1[ai+0]) * (bb1[ai+3] - bb1[ai+1])
        for j in range(M):
            bj = j * 4
            xx1 = fmax(bb1[ai+0], bb2[bj+0]); yy1 = fmax(bb1[ai+1], bb2[bj+1])
            xx2 = fmin(bb1[ai+2], bb2[bj+2]); yy2 = fmin(bb1[ai+3], bb2[bj+3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bb2[bj+2] - bb2[bj+0]) * (bb2[bj+3] - bb2[bj+1])
            union_val = area1 + area2 - wh
            iou_val = wh / (union_val + 1e-9)
            xxc1 = fmin(bb1[ai+0], bb2[bj+0]); yyc1 = fmin(bb1[ai+1], bb2[bj+1])
            xxc2 = fmax(bb1[ai+2], bb2[bj+2]); yyc2 = fmax(bb1[ai+3], bb2[bj+3])
            area_enclose = (xxc2 - xxc1) * (yyc2 - yyc1)
            out[i * M + j] = (iou_val - (area_enclose - union_val) / (area_enclose + 1e-9) + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void diou_batch_c(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute DIOU between two sets of flat bbox arrays."""
    cdef int i, j, ai, bj
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_val, iou_val
    cdef double cx1, cy1, cx2, cy2, inner_diag
    cdef double xxc1, yyc1, xxc2, yyc2, outer_diag
    for i in range(N):
        ai = i * 4
        area1 = (bb1[ai+2] - bb1[ai+0]) * (bb1[ai+3] - bb1[ai+1])
        for j in range(M):
            bj = j * 4
            xx1 = fmax(bb1[ai+0], bb2[bj+0]); yy1 = fmax(bb1[ai+1], bb2[bj+1])
            xx2 = fmin(bb1[ai+2], bb2[bj+2]); yy2 = fmin(bb1[ai+3], bb2[bj+3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bb2[bj+2] - bb2[bj+0]) * (bb2[bj+3] - bb2[bj+1])
            union_val = area1 + area2 - wh
            iou_val = wh / (union_val + 1e-9)
            cx1 = (bb1[ai+0] + bb1[ai+2]) / 2.0; cy1 = (bb1[ai+1] + bb1[ai+3]) / 2.0
            cx2 = (bb2[bj+0] + bb2[bj+2]) / 2.0; cy2 = (bb2[bj+1] + bb2[bj+3]) / 2.0
            inner_diag = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2)
            xxc1 = fmin(bb1[ai+0], bb2[bj+0]); yyc1 = fmin(bb1[ai+1], bb2[bj+1])
            xxc2 = fmax(bb1[ai+2], bb2[bj+2]); yyc2 = fmax(bb1[ai+3], bb2[bj+3])
            outer_diag = (xxc2 - xxc1) * (xxc2 - xxc1) + (yyc2 - yyc1) * (yyc2 - yyc1)
            out[i * M + j] = (iou_val - inner_diag / (outer_diag + 1e-9) + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ciou_batch_c(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute CIOU between two sets of flat bbox arrays."""
    cdef int i, j, ai, bj
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_val, iou_val
    cdef double cx1, cy1, cx2, cy2, inner_diag
    cdef double xxc1, yyc1, xxc2, yyc2, outer_diag
    cdef double w1, h1, w2, h2, arctan_val, v, S, alpha_val
    for i in range(N):
        ai = i * 4
        area1 = (bb1[ai+2] - bb1[ai+0]) * (bb1[ai+3] - bb1[ai+1])
        for j in range(M):
            bj = j * 4
            xx1 = fmax(bb1[ai+0], bb2[bj+0]); yy1 = fmax(bb1[ai+1], bb2[bj+1])
            xx2 = fmin(bb1[ai+2], bb2[bj+2]); yy2 = fmin(bb1[ai+3], bb2[bj+3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bb2[bj+2] - bb2[bj+0]) * (bb2[bj+3] - bb2[bj+1])
            union_val = area1 + area2 - wh
            iou_val = wh / (union_val + 1e-9)
            cx1 = (bb1[ai+0] + bb1[ai+2]) / 2.0; cy1 = (bb1[ai+1] + bb1[ai+3]) / 2.0
            cx2 = (bb2[bj+0] + bb2[bj+2]) / 2.0; cy2 = (bb2[bj+1] + bb2[bj+3]) / 2.0
            inner_diag = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2)
            xxc1 = fmin(bb1[ai+0], bb2[bj+0]); yyc1 = fmin(bb1[ai+1], bb2[bj+1])
            xxc2 = fmax(bb1[ai+2], bb2[bj+2]); yyc2 = fmax(bb1[ai+3], bb2[bj+3])
            outer_diag = (xxc2 - xxc1) * (xxc2 - xxc1) + (yyc2 - yyc1) * (yyc2 - yyc1)
            w1 = bb1[ai+2] - bb1[ai+0]; h1 = bb1[ai+3] - bb1[ai+1] + 1.0
            w2 = bb2[bj+2] - bb2[bj+0]; h2 = bb2[bj+3] - bb2[bj+1] + 1.0
            arctan_val = atan(w2 / h2) - atan(w1 / h1)
            v = (4.0 / (M_PI * M_PI)) * (arctan_val * arctan_val)
            S = 1.0 - iou_val
            alpha_val = v / (S + v + 1e-9)
            out[i * M + j] = (iou_val - inner_diag / (outer_diag + 1e-9) - alpha_val * v + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ct_dist_c(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute center distance between two sets of flat bbox arrays."""
    cdef int i, j, ai, bj
    cdef double cx1, cy1, cx2, cy2, d, max_dist
    # First pass: compute distances and find max
    max_dist = 0.0
    for i in range(N):
        ai = i * 4
        cx1 = (bb1[ai+0] + bb1[ai+2]) / 2.0; cy1 = (bb1[ai+1] + bb1[ai+3]) / 2.0
        for j in range(M):
            bj = j * 4
            cx2 = (bb2[bj+0] + bb2[bj+2]) / 2.0; cy2 = (bb2[bj+1] + bb2[bj+3]) / 2.0
            d = sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2))
            out[i * M + j] = d
            if d > max_dist:
                max_dist = d
    # Second pass: normalize
    if max_dist > 1e-9:
        for i in range(N):
            for j in range(M):
                out[i * M + j] = max_dist - out[i * M + j]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void asso_dispatch_c(int func_type, double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Dispatch to the correct distance metric based on enum."""
    if func_type == 0:    # ASSO_IOU
        iou_batch_c(bb1, N, bb2, M, out)
    elif func_type == 1:  # ASSO_GIOU
        giou_batch_c(bb1, N, bb2, M, out)
    elif func_type == 2:  # ASSO_DIOU
        diou_batch_c(bb1, N, bb2, M, out)
    elif func_type == 3:  # ASSO_CIOU
        ciou_batch_c(bb1, N, bb2, M, out)
    elif func_type == 4:  # ASSO_CT_DIST
        ct_dist_c(bb1, N, bb2, M, out)
    else:
        iou_batch_c(bb1, N, bb2, M, out)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void lapjv_solve_c(
    double *cost_matrix, int n_rows, int n_cols,
    int *raw_match_a, int *raw_match_b, int *n_raw_matches
) noexcept nogil:
    """Solve linear assignment using LAPJV on raw pointer arrays."""
    cdef int n = n_rows if n_rows > n_cols else n_cols
    cdef int i, x_val

    # Allocate extended square cost matrix
    cdef double *cost_ext = <double *>calloc(n * n, sizeof(double))
    for i in range(n_rows):
        for x_val in range(n_cols):
            cost_ext[i * n + x_val] = cost_matrix[i * n_cols + x_val]

    # Build pointer array for lapjv_internal
    cdef double **cost_ptr = <double **>malloc(n * sizeof(double *))
    for i in range(n):
        cost_ptr[i] = &cost_ext[i * n]

    # Solve
    cdef int_t *x_c = <int_t *>malloc(n * sizeof(int_t))
    cdef int_t *y_c = <int_t *>malloc(n * sizeof(int_t))
    lapjv_internal(n, cost_ptr, x_c, y_c)
    free(cost_ptr)
    free(cost_ext)

    # Extract valid match pairs
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
cdef void linear_assignment_c(
    double *cost_matrix, int N, int M,
    int *match_a, int *match_b, int *n_matches
) noexcept nogil:
    """Solve linear assignment on a cost matrix. Returns raw match pairs."""
    if N == 0 or M == 0:
        n_matches[0] = 0
        return
    cdef int max_m = N if N < M else M
    cdef int *ra = <int *>malloc((max_m + 1) * sizeof(int))
    cdef int *rb = <int *>malloc((max_m + 1) * sizeof(int))
    cdef int n_raw = 0
    lapjv_solve_c(cost_matrix, N, M, ra, rb, &n_raw)
    cdef int i
    for i in range(n_raw):
        match_a[i] = ra[i]
        match_b[i] = rb[i]
    n_matches[0] = n_raw
    free(ra)
    free(rb)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void associate_c(
    double *dets, int n_dets,
    double *trks, int n_trks,
    double iou_threshold,
    double *velocities,
    double *previous_obs,
    double vdc_weight,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_dets, int *n_unmatched_dets,
    int *unmatched_trks, int *n_unmatched_trks
) noexcept nogil:
    """
    Full first-round association with velocity direction consistency (VDC).
    dets: n_dets * 5 flat (x1,y1,x2,y2,score)
    trks: n_trks * 5 flat (x1,y1,x2,y2,0)
    velocities: n_trks * 2 flat (dy, dx)
    previous_obs: n_trks * 5 flat (x1,y1,x2,y2,valid_flag)
    """
    cdef int i, j, d, t
    cdef int matched_count = 0
    cdef int ud_count = 0
    cdef int ut_count = 0

    # Handle empty trackers case
    if n_trks == 0:
        n_matches[0] = 0
        for i in range(n_dets):
            unmatched_dets[i] = i
        n_unmatched_dets[0] = n_dets
        n_unmatched_trks[0] = 0
        return

    # ---- Compute speed direction batch: dets vs previous_obs ----
    # Y[t * n_dets + d] = dy, X[t * n_dets + d] = dx   (track x det)
    cdef double *Y = <double *>malloc(n_trks * n_dets * sizeof(double))
    cdef double *X = <double *>malloc(n_trks * n_dets * sizeof(double))
    cdef double dcx, dcy, tcx, tcy, ddx, ddy, norm_val
    cdef int ti, di

    for t in range(n_trks):
        ti = t * 5
        tcx = (previous_obs[ti + 0] + previous_obs[ti + 2]) / 2.0
        tcy = (previous_obs[ti + 1] + previous_obs[ti + 3]) / 2.0
        for d in range(n_dets):
            di = d * 5
            dcx = (dets[di + 0] + dets[di + 2]) / 2.0
            dcy = (dets[di + 1] + dets[di + 3]) / 2.0
            ddx = dcx - tcx
            ddy = dcy - tcy
            norm_val = sqrt(ddx * ddx + ddy * ddy) + 1e-6
            # Y = dy direction, X = dx direction (track x det layout)
            Y[t * n_dets + d] = ddy / norm_val
            X[t * n_dets + d] = ddx / norm_val

    # ---- Compute angle diff cost ----
    cdef double *angle_diff_cost = <double *>calloc(n_dets * n_trks, sizeof(double))
    cdef double inertia_y, inertia_x, diff_cos, diff_angle, valid, score_val

    for t in range(n_trks):
        inertia_y = velocities[t * 2 + 0]
        inertia_x = velocities[t * 2 + 1]
        # Check if previous obs is valid (5th element >= 0)
        valid = 1.0
        if previous_obs[t * 5 + 4] < 0:
            valid = 0.0
        for d in range(n_dets):
            # Dot product of inertia and direction
            diff_cos = inertia_x * X[t * n_dets + d] + inertia_y * Y[t * n_dets + d]
            # Clip to [-1, 1]
            if diff_cos > 1.0:
                diff_cos = 1.0
            elif diff_cos < -1.0:
                diff_cos = -1.0
            # angle_diff = (pi/2 - |arccos(cos)|) / pi
            diff_angle = (M_PI / 2.0 - fabs(acos(diff_cos))) / M_PI
            # Apply valid mask, vdc_weight, and detection score
            score_val = dets[d * 5 + 4]
            # Layout: angle_diff_cost[d * n_trks + t] (det x trk)
            angle_diff_cost[d * n_trks + t] = valid * diff_angle * vdc_weight * score_val

    free(Y)
    free(X)

    # ---- Compute IOU matrix ----
    # Extract bbox-only arrays (4 values per entry)
    cdef double *det_bb = <double *>malloc(n_dets * 4 * sizeof(double))
    cdef double *trk_bb = <double *>malloc(n_trks * 4 * sizeof(double))
    for i in range(n_dets):
        for j in range(4):
            det_bb[i * 4 + j] = dets[i * 5 + j]
    for i in range(n_trks):
        for j in range(4):
            trk_bb[i * 4 + j] = trks[i * 5 + j]

    cdef double *iou_matrix = <double *>malloc(n_dets * n_trks * sizeof(double))
    iou_batch_c(det_bb, n_dets, trk_bb, n_trks, iou_matrix)
    free(det_bb)
    free(trk_bb)

    # ---- Check for one-to-one shortcut ----
    cdef int use_shortcut = 0
    cdef int *row_sum = <int *>calloc(n_dets, sizeof(int))
    cdef int *col_sum = <int *>calloc(n_trks, sizeof(int))
    cdef int max_row_sum = 0, max_col_sum = 0

    for i in range(n_dets):
        for j in range(n_trks):
            if iou_matrix[i * n_trks + j] > iou_threshold:
                row_sum[i] += 1
                col_sum[j] += 1

    for i in range(n_dets):
        if row_sum[i] > max_row_sum:
            max_row_sum = row_sum[i]
    for j in range(n_trks):
        if col_sum[j] > max_col_sum:
            max_col_sum = col_sum[j]

    # Build matched_indices
    cdef int *mi_a = NULL
    cdef int *mi_b = NULL
    cdef int n_mi = 0
    cdef double *neg_cost = NULL
    cdef int max_dim = 0

    if n_dets > 0 and n_trks > 0:
        if max_row_sum == 1 and max_col_sum == 1:
            # One-to-one shortcut: directly read pairs from threshold mask
            use_shortcut = 1
            mi_a = <int *>malloc((n_dets + 1) * sizeof(int))
            mi_b = <int *>malloc((n_dets + 1) * sizeof(int))
            n_mi = 0
            for i in range(n_dets):
                for j in range(n_trks):
                    if iou_matrix[i * n_trks + j] > iou_threshold:
                        mi_a[n_mi] = i
                        mi_b[n_mi] = j
                        n_mi += 1
                        break
        else:
            # Build negated cost matrix: -(iou + angle_diff_cost)
            neg_cost = <double *>malloc(n_dets * n_trks * sizeof(double))
            for i in range(n_dets):
                for j in range(n_trks):
                    neg_cost[i * n_trks + j] = -(iou_matrix[i * n_trks + j] + angle_diff_cost[i * n_trks + j])

            max_dim = n_dets if n_dets < n_trks else n_trks
            mi_a = <int *>malloc((max_dim + 1) * sizeof(int))
            mi_b = <int *>malloc((max_dim + 1) * sizeof(int))
            linear_assignment_c(neg_cost, n_dets, n_trks, mi_a, mi_b, &n_mi)
            free(neg_cost)

    free(row_sum)
    free(col_sum)
    free(angle_diff_cost)

    # ---- Filter matches by IOU threshold and build output ----
    cdef int *matched_d = <int *>calloc(n_dets, sizeof(int))
    cdef int *matched_t = <int *>calloc(n_trks, sizeof(int))
    matched_count = 0

    for i in range(n_mi):
        d = mi_a[i]
        t = mi_b[i]
        if iou_matrix[d * n_trks + t] >= iou_threshold:
            match_a[matched_count] = d
            match_b[matched_count] = t
            matched_count += 1
            matched_d[d] = 1
            matched_t[t] = 1

    # Also mark filtered-out matches as unmatched
    for i in range(n_mi):
        d = mi_a[i]
        t = mi_b[i]
        if iou_matrix[d * n_trks + t] < iou_threshold:
            if matched_d[d] == 0:
                matched_d[d] = 0  # keep as unmatched
            if matched_t[t] == 0:
                matched_t[t] = 0  # keep as unmatched

    n_matches[0] = matched_count

    # Collect unmatched detections
    ud_count = 0
    for i in range(n_dets):
        if matched_d[i] == 0:
            unmatched_dets[ud_count] = i
            ud_count += 1
    n_unmatched_dets[0] = ud_count

    # Collect unmatched trackers
    ut_count = 0
    for i in range(n_trks):
        if matched_t[i] == 0:
            unmatched_trks[ut_count] = i
            ut_count += 1
    n_unmatched_trks[0] = ut_count

    if mi_a != NULL:
        free(mi_a)
    if mi_b != NULL:
        free(mi_b)
    free(matched_d)
    free(matched_t)
    free(iou_matrix)


# ============================================================
# Legacy memoryview-based internal functions (kept for def wrappers)
# ============================================================

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int linear_assignment_internal(
    cnp.float64_t[:, :] cost_c,
    cnp.int16_t[:, :] matches
) noexcept nogil:
    """Solve linear assignment problem using lapjv (memoryview version)."""
    cdef int_t n_rows = cost_c.shape[0]
    cdef int_t n_cols = cost_c.shape[1]
    cdef int_t n = 0
    n = max(n_rows, n_cols)
    cdef double *cost_extended = <double *> calloc(n * n, sizeof(double))
    cdef int i, j
    for i in range(n_rows):
        for j in range(n_cols):
            cost_extended[i * n + j] = cost_c[i, j]
    cdef double **cost_ptr
    cost_ptr = <double **> malloc(n * sizeof(double *))
    for i in range(n):
        cost_ptr[i] = &cost_extended[i * n]
    cdef int_t *x_c = <int_t *> malloc(n * sizeof(int_t))
    cdef int_t *y_c = <int_t *> malloc(n * sizeof(int_t))
    cdef int ret = lapjv_internal(n, cost_ptr, x_c, y_c)
    free(cost_ptr)
    free(cost_extended)
    cdef int x, y
    cdef int count = 0
    for i in range(n_rows):
        x = x_c[i]
        y = y_c[x]
        if x >= n_cols:
            x = -1
        if y >= n_rows:
            y = -1
        if x >= 0:
            matches[count, 0] = y
            matches[count, 1] = x
            count += 1
    free(x_c)
    free(y_c)
    return count


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void iou_batch_internal(
    cnp.float64_t[:, :] bboxes1,
    cnp.float64_t[:, :] bboxes2,
    cnp.float64_t[:, :] output
) noexcept nogil:
    """Compute IOU (memoryview version)."""
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            output[i, j] = wh / (area1 + area2 - wh + 1e-9)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void giou_batch_internal(
    cnp.float64_t[:, :] bboxes1, cnp.float64_t[:, :] bboxes2, cnp.float64_t[:, :] output
) noexcept nogil:
    """Compute GIOU (memoryview version)."""
    cdef int N = bboxes1.shape[0], M = bboxes2.shape[0]
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_v, iou_v
    cdef double xxc1, yyc1, xxc2, yyc2, wc, hc, area_enc
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0]); yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2]); yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union_v = area1 + area2 - wh; iou_v = wh / (union_v + 1e-9)
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0]); yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2]); yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            area_enc = (xxc2 - xxc1) * (yyc2 - yyc1)
            output[i, j] = (iou_v - (area_enc - union_v) / (area_enc + 1e-9) + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void diou_batch_internal(
    cnp.float64_t[:, :] bboxes1, cnp.float64_t[:, :] bboxes2, cnp.float64_t[:, :] output
) noexcept nogil:
    """Compute DIOU (memoryview version)."""
    cdef int N = bboxes1.shape[0], M = bboxes2.shape[0]
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_v, iou_v
    cdef double cx1, cy1, cx2, cy2, inner_diag, xxc1, yyc1, xxc2, yyc2, outer_diag
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0]); yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2]); yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union_v = area1 + area2 - wh; iou_v = wh / (union_v + 1e-9)
            cx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0; cy1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            cx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0; cy2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0]); yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2]); yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
            output[i, j] = (iou_v - inner_diag / (outer_diag + 1e-9) + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ciou_batch_internal(
    cnp.float64_t[:, :] bboxes1, cnp.float64_t[:, :] bboxes2, cnp.float64_t[:, :] output
) noexcept nogil:
    """Compute CIOU (memoryview version)."""
    cdef int N = bboxes1.shape[0], M = bboxes2.shape[0]
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union_v, iou_v
    cdef double cx1, cy1, cx2, cy2, inner_diag, xxc1, yyc1, xxc2, yyc2, outer_diag
    cdef double w1, h1, w2, h2, arctan_v, v, S, alpha_v
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0]); yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2]); yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            w = fmax(0.0, xx2 - xx1); h = fmax(0.0, yy2 - yy1); wh = w * h
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union_v = area1 + area2 - wh; iou_v = wh / (union_v + 1e-9)
            cx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0; cy1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            cx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0; cy2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0]); yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2]); yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
            w1 = bboxes1[i, 2] - bboxes1[i, 0]; h1 = bboxes1[i, 3] - bboxes1[i, 1] + 1.0
            w2 = bboxes2[j, 2] - bboxes2[j, 0]; h2 = bboxes2[j, 3] - bboxes2[j, 1] + 1.0
            arctan_v = atan(w2 / h2) - atan(w1 / h1)
            v = (4.0 / (M_PI * M_PI)) * (arctan_v * arctan_v)
            S = 1.0 - iou_v; alpha_v = v / (S + v + 1e-9)
            output[i, j] = (iou_v - inner_diag / (outer_diag + 1e-9) - alpha_v * v + 1.0) / 2.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ct_dist_internal(
    cnp.float64_t[:, :] bboxes1, cnp.float64_t[:, :] bboxes2, cnp.float64_t[:, :] output
) noexcept nogil:
    """Compute center distance (memoryview version)."""
    cdef int N = bboxes1.shape[0], M = bboxes2.shape[0]
    cdef int i, j
    cdef double cx1, cy1, cx2, cy2, d, max_dist
    max_dist = 0.0
    for i in range(N):
        cx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0; cy1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
        for j in range(M):
            cx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0; cy2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            d = sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            output[i, j] = d
            if d > max_dist:
                max_dist = d
    if max_dist > 1e-9:
        for i in range(N):
            for j in range(M):
                output[i, j] = max_dist - output[i, j]


# ============================================================
# Python def wrappers (backward compatibility)
# ============================================================

def iou_batch(bboxes1, bboxes2):
    """From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]"""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    iou_batch_internal(b1, b2, result)
    return result

def giou_batch(bboxes1, bboxes2):
    """Compute GIOU between two sets of bboxes."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    giou_batch_internal(b1, b2, result)
    return result

def diou_batch(bboxes1, bboxes2):
    """Compute DIOU between two sets of bboxes."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    diou_batch_internal(b1, b2, result)
    return result

def ciou_batch(bboxes1, bboxes2):
    """Compute CIOU between two sets of bboxes."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    ciou_batch_internal(b1, b2, result)
    return result

def ct_dist(bboxes1, bboxes2):
    """Measure the center distance between two sets of bounding boxes."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    ct_dist_internal(b1, b2, result)
    return result

def speed_direction_batch(dets, tracks):
    """Compute speed direction between detections and tracks."""
    dets = np.asarray(dets, dtype=np.float64)
    tracks = np.asarray(tracks, dtype=np.float64)
    if tracks.ndim == 2:
        tracks = tracks[:, :, np.newaxis]
    CX1 = (dets[:, 0] + dets[:, 2]) / 2.0
    CY1 = (dets[:, 1] + dets[:, 3]) / 2.0
    CX2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    CY2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

def linear_assignment(cost_matrix):
    """Solve linear assignment problem."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    """Associate detections to trackers using IOU and velocity direction consistency."""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    Y, X = speed_direction_batch(detections, previous_obs)
    velocities = np.asarray(velocities, dtype=np.float64)
    previous_obs = np.asarray(previous_obs, dtype=np.float64)
    inertia_Y = velocities[:, 0]
    inertia_X = velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight):
    """Associate detections to trackers for KITTI dataset with category matching."""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    Y, X = speed_direction_batch(detections, previous_obs)
    velocities = np.asarray(velocities, dtype=np.float64)
    previous_obs = np.asarray(previous_obs, dtype=np.float64)
    inertia_Y = velocities[:, 0]
    inertia_X = velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    iou_matrix = iou_batch(detections, trackers)
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6
    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
