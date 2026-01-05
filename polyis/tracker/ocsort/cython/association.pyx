# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, fmax, fmin, acos, atan, fabs, M_PI
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport int32_t

cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t
    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int linear_assignment_internal(
    cnp.float64_t[:, :] cost_c,
    cnp.int16_t[:, :] matches
) noexcept nogil:
    """
    Solve linear assignment problem using lapjv.
    """
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

    # Convert to array of pairs
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
    """
    Compute IOU between two sets of bboxes in the form [x1,y1,x2,y2].
    """
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
            
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            
            output[i, j] = wh / (area1 + area2 - wh + 1e-9)


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    
    iou_batch_internal(b1, b2, result)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void giou_batch_internal(
    cnp.float64_t[:, :] bboxes1,
    cnp.float64_t[:, :] bboxes2,
    cnp.float64_t[:, :] output
) noexcept nogil:
    """
    Compute GIOU between two sets of bboxes.
    """
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]
    
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union, iou
    cdef double xxc1, yyc1, xxc2, yyc2, wc, hc, area_enclose, giou
    
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union = area1 + area2 - wh
            iou = wh / (union + 1e-9)
            
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0])
            yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2])
            yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            wc = xxc2 - xxc1
            hc = yyc2 - yyc1
            area_enclose = wc * hc
            
            giou = iou - (area_enclose - union) / (area_enclose + 1e-9)
            output[i, j] = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)


def giou_batch(bboxes1, bboxes2):
    """
    Compute GIOU between two sets of bboxes.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    
    giou_batch_internal(b1, b2, result)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void diou_batch_internal(
    cnp.float64_t[:, :] bboxes1,
    cnp.float64_t[:, :] bboxes2,
    cnp.float64_t[:, :] output
) noexcept nogil:
    """
    Compute DIOU between two sets of bboxes.
    """
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]
    
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union, iou
    cdef double centerx1, centery1, centerx2, centery2, inner_diag
    cdef double xxc1, yyc1, xxc2, yyc2, outer_diag, diou
    
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union = area1 + area2 - wh
            iou = wh / (union + 1e-9)
            
            centerx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0
            centery1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            centerx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0
            centery2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            
            inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
            
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0])
            yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2])
            yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            
            outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
            diou = iou - inner_diag / (outer_diag + 1e-9)
            
            output[i, j] = (diou + 1.0) / 2.0  # resize from (-1,1) to (0,1)


def diou_batch(bboxes1, bboxes2):
    """
    Compute DIOU between two sets of bboxes.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    
    diou_batch_internal(b1, b2, result)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ciou_batch_internal(
    cnp.float64_t[:, :] bboxes1,
    cnp.float64_t[:, :] bboxes2,
    cnp.float64_t[:, :] output
) noexcept nogil:
    """
    Compute CIOU between two sets of bboxes.
    """
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]
    
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2, union, iou
    cdef double centerx1, centery1, centerx2, centery2, inner_diag
    cdef double xxc1, yyc1, xxc2, yyc2, outer_diag
    cdef double w1, h1, w2, h2, arctan, v, S, alpha, ciou
    
    for i in range(N):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])
            
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            union = area1 + area2 - wh
            iou = wh / (union + 1e-9)
            
            centerx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0
            centery1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            centerx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0
            centery2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            
            inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
            
            xxc1 = fmin(bboxes1[i, 0], bboxes2[j, 0])
            yyc1 = fmin(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = fmax(bboxes1[i, 2], bboxes2[j, 2])
            yyc2 = fmax(bboxes1[i, 3], bboxes2[j, 3])
            
            outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
            
            w1 = bboxes1[i, 2] - bboxes1[i, 0]
            h1 = bboxes1[i, 3] - bboxes1[i, 1]
            w2 = bboxes2[j, 2] - bboxes2[j, 0]
            h2 = bboxes2[j, 3] - bboxes2[j, 1]
            
            # prevent dividing over zero. add one pixel shift
            h2 = h2 + 1.0
            h1 = h1 + 1.0
            arctan = atan(w2/h2) - atan(w1/h1)
            v = (4.0 / (M_PI * M_PI)) * (arctan * arctan)
            S = 1.0 - iou
            alpha = v / (S + v + 1e-9)
            ciou = iou - inner_diag / (outer_diag + 1e-9) - alpha * v
            
            output[i, j] = (ciou + 1.0) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2):
    """
    Compute CIOU between two sets of bboxes.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    
    ciou_batch_internal(b1, b2, result)
    return result


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void ct_dist_internal(
    cnp.float64_t[:, :] bboxes1,
    cnp.float64_t[:, :] bboxes2,
    cnp.float64_t[:, :] output
) noexcept nogil:
    """
    Measure the center distance between two sets of bounding boxes.
    """
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]
    
    cdef int i, j
    cdef double centerx1, centery1, centerx2, centery2, ct_dist2, ct_dist, max_dist
    
    # First pass: compute all distances and find max
    max_dist = 0.0
    for i in range(N):
        centerx1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0
        centery1 = (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
        for j in range(M):
            centerx2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0
            centery2 = (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            
            ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
            ct_dist = sqrt(ct_dist2)
            output[i, j] = ct_dist
            if ct_dist > max_dist:
                max_dist = ct_dist
    
    # Second pass: normalize
    if max_dist > 1e-9:
        for i in range(N):
            for j in range(M):
                output[i, j] = max_dist - output[i, j]  # resize to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
    Measure the center distance between two sets of bounding boxes.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)
    
    ct_dist_internal(b1, b2, result)
    return result


def speed_direction_batch(dets, tracks):
    """
    Compute speed direction between detections and tracks.
    Returns (Y, X) where Y and X are normalized direction vectors.
    """
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
    
    return dy, dx  # size: num_track x num_det


def linear_assignment(cost_matrix):
    """
    Solve linear assignment problem.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    """
    Associate detections to trackers using IOU and velocity direction consistency.
    """
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
    
    # filter out matched with low IOU
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
    """
    Associate detections to trackers for KITTI dataset with category matching.
    """
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
    
    # Category mismatch cost
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
    
    # filter out matched with low IOU
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

