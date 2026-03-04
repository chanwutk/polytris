# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython association functions for OC-SORT.

Only the C-level APIs are the public surface for OCSort: iou_batch,
giou_batch, diou_batch, ciou_batch, ct_dist, asso_dispatch,
lapjv_solve, linear_assignment, associate. Used by ocsort.pyx via cimport.
"""

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
cdef void iou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
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
cdef void giou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
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
cdef void diou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
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
cdef void ciou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
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
cdef void ct_dist(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
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
cdef void asso_dispatch(int func_type, double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Dispatch to the correct distance metric based on enum."""
    if func_type == 0:    # ASSO_IOU
        iou_batch(bb1, N, bb2, M, out)
    elif func_type == 1:  # ASSO_GIOU
        giou_batch(bb1, N, bb2, M, out)
    elif func_type == 2:  # ASSO_DIOU
        diou_batch(bb1, N, bb2, M, out)
    elif func_type == 3:  # ASSO_CIOU
        ciou_batch(bb1, N, bb2, M, out)
    elif func_type == 4:  # ASSO_CT_DIST
        ct_dist(bb1, N, bb2, M, out)
    else:
        iou_batch(bb1, N, bb2, M, out)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void lapjv_solve(
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
cdef void linear_assignment(
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
    lapjv_solve(cost_matrix, N, M, ra, rb, &n_raw)
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
    iou_batch(det_bb, n_dets, trk_bb, n_trks, iou_matrix)
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
            linear_assignment(neg_cost, n_dets, n_trks, mi_a, mi_b, &n_mi)
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
