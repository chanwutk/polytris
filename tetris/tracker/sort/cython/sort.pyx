# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of SORT tracker.

Sort is a cdef class with C-typed vector fields. All internal operations
use cdef functions and C structs. The class provides a thin Python-callable
update() method for the wrapper and tests.
"""

from __future__ import print_function

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, fmax, fmin, isnan
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap

from polyis.tracker.sort.cython.kalman_filter cimport KalmanFilter, kf_init, kf_predict, kf_update


cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t
    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


# Set random seed for reproducibility
np.random.seed(0)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void lapjv_solve(
    double *cost_matrix, int n_rows, int n_cols,
    int *raw_match_a, int *raw_match_b, int *n_raw_matches
) noexcept nogil:
    """Solve linear assignment using LAPJV on raw pointer arrays (C-level)."""
    cdef int n = n_rows if n_rows > n_cols else n_cols
    cdef int i, x_val
    cdef double *cost_ext = <double *>calloc(n * n, sizeof(double))
    for i in range(n_rows):
        for x_val in range(n_cols):
            cost_ext[i * n + x_val] = cost_matrix[i * n_cols + x_val]
    cdef double **cost_ptr = <double **>malloc(n * sizeof(double *))
    for i in range(n):
        cost_ptr[i] = &cost_ext[i * n]
    cdef int_t *x_c = <int_t *>malloc(n * sizeof(int_t))
    cdef int_t *y_c = <int_t *>malloc(n * sizeof(int_t))
    lapjv_internal(n, cost_ptr, x_c, y_c)
    free(cost_ptr)
    free(cost_ext)
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
cdef void iou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil:
    """Compute IOU between two sets of flat bbox arrays [x1,y1,x2,y2] (C-level)."""
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
            out[i * M + j] = wh / (area1 + area2 - wh)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void convert_bbox_to_z(double *bbox, double *z) noexcept nogil:
    """
    Convert bounding box from [x1,y1,x2,y2] to [x,y,s,r] format.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        z: Pointer to array of size 4 to store result [x, y, s, r]
    """
    cdef double w = bbox[2] - bbox[0]
    cdef double h = bbox[3] - bbox[1]
    z[0] = bbox[0] + w / 2.0
    z[1] = bbox[1] + h / 2.0
    z[2] = w * h  # scale is just area
    z[3] = w / h


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void convert_x_to_bbox(double *x, double *bbox) noexcept nogil:
    """
    Convert bounding box from [x,y,s,r] to [x1,y1,x2,y2] format.

    Args:
        x: State vector [x, y, s, r]
        bbox: Pointer to array of size 4 to store result [x1, y1, x2, y2]
    """
    cdef double w = sqrt(x[2] * x[3])
    cdef double h = x[2] / w
    bbox[0] = x[0] - w/2.
    bbox[1] = x[1] - h/2.
    bbox[2] = x[0] + w/2.
    bbox[3] = x[1] + h/2.


cdef struct KalmanBoxTracker:
    KalmanFilter kf
    int time_since_update
    int id
    int hits
    int hit_streak
    int age

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_init(KalmanBoxTracker *self, double *bbox, int id) noexcept nogil:
    """
    Initialize tracker using initial bounding box.

    Args:
        bbox: Initial bounding box [x1, y1, x2, y2]
    """
    # Initialize Kalman Filter struct
    kf_init(&self.kf)

    # Define constant velocity model
    # F is identity from init. Set specific values.
    self.kf.F[0][4] = 1.0
    self.kf.F[1][5] = 1.0
    self.kf.F[2][6] = 1.0

    # H is zeros from init. Set specific values.
    self.kf.H[0][0] = 1.0
    self.kf.H[1][1] = 1.0
    self.kf.H[2][2] = 1.0
    self.kf.H[3][3] = 1.0

    # Adjust covariance matrices
    # R[2:, 2:] *= 10.0
    self.kf.R[2][2] *= 10.0
    self.kf.R[3][3] *= 10.0

    # Give high uncertainty to the unobservable initial velocities
    # P[4:, 4:] *= 1000.0
    self.kf.P[4][4] *= 1000.0
    self.kf.P[5][5] *= 1000.0
    self.kf.P[6][6] *= 1000.0

    # P *= 10.0
    cdef int i, j
    for i in range(7):
        for j in range(7):
            self.kf.P[i][j] *= 10.0

    # Q[-1, -1] *= 0.01
    self.kf.Q[6][6] *= 0.01
    # Q[4:, 4:] *= 0.01
    self.kf.Q[4][4] *= 0.01
    self.kf.Q[5][5] *= 0.01
    self.kf.Q[6][6] *= 0.01 # Applied twice as in original code

    # Initialize state with bbox
    convert_bbox_to_z(bbox, self.kf.x)

    self.time_since_update = 0
    self.id = id
    # self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_update(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """
    Update state vector with observed bbox.

    Args:
        bbox: Observed bounding box [x1, y1, x2, y2]
    """
    self.time_since_update = 0
    # self.history = []
    self.hits += 1
    self.hit_streak += 1

    cdef double z[4]
    convert_bbox_to_z(bbox, z)

    kf_update(&self.kf, z)

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_predict(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """
    Advance state vector and return predicted bounding box estimate.

    Returns:
        Predicted bounding box [x1, y1, x2, y2]
    """
    if (self.kf.x[6] + self.kf.x[2]) <= 0:
        self.kf.x[6] = 0.0

    kf_predict(&self.kf)

    self.age += 1
    if self.time_since_update > 0:
        self.hit_streak = 0
    self.time_since_update += 1

    convert_x_to_bbox(self.kf.x, bbox)

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_get_state(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """
    Return current bounding box estimate.

    Returns:
        Current bounding box [x1, y1, x2, y2]
    """
    convert_x_to_bbox(self.kf.x, bbox)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void associate(
    double *det_bb, int n_dets,
    double *trk_bb, int n_trks,
    double iou_threshold,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_dets, int *n_unmatched_dets
) noexcept nogil:
    """
    Associate detections to trackers using IOU (C-level).
    det_bb: n_dets * 4 flat [x1,y1,x2,y2]
    trk_bb: n_trks * 4 flat [x1,y1,x2,y2]
    """
    cdef int i, j, d, t
    cdef int matched_count = 0
    cdef int ud_count = 0
    cdef double *iou_matrix = NULL
    cdef int *row_sum = NULL
    cdef int *col_sum = NULL
    cdef int max_row = 0, max_col = 0
    cdef int *mi_a = NULL
    cdef int *mi_b = NULL
    cdef int n_mi = 0
    cdef double *neg_iou = NULL
    cdef int min_dim
    cdef int *matched_d = NULL

    # Handle empty trackers
    if n_trks == 0:
        n_matches[0] = 0
        for i in range(n_dets):
            unmatched_dets[i] = i
        n_unmatched_dets[0] = n_dets
        return

    # Compute IOU matrix
    iou_matrix = <double *>malloc(n_dets * n_trks * sizeof(double))
    iou_batch(det_bb, n_dets, trk_bb, n_trks, iou_matrix)

    # Check for one-to-one shortcut
    row_sum = <int *>calloc(n_dets, sizeof(int))
    col_sum = <int *>calloc(n_trks, sizeof(int))

    for i in range(n_dets):
        for j in range(n_trks):
            if iou_matrix[i * n_trks + j] > iou_threshold:
                row_sum[i] += 1
                col_sum[j] += 1
    for i in range(n_dets):
        if row_sum[i] > max_row:
            max_row = row_sum[i]
    for j in range(n_trks):
        if col_sum[j] > max_col:
            max_col = col_sum[j]

    if n_dets > 0 and n_trks > 0:
        if max_row == 1 and max_col == 1:
            # One-to-one shortcut
            mi_a = <int *>malloc((n_dets + 1) * sizeof(int))
            mi_b = <int *>malloc((n_dets + 1) * sizeof(int))
            for i in range(n_dets):
                for j in range(n_trks):
                    if iou_matrix[i * n_trks + j] > iou_threshold:
                        mi_a[n_mi] = i
                        mi_b[n_mi] = j
                        n_mi += 1
                        break
        else:
            # Use linear assignment on negated IOU
            neg_iou = <double *>malloc(n_dets * n_trks * sizeof(double))
            for i in range(n_dets * n_trks):
                neg_iou[i] = -iou_matrix[i]
            min_dim = n_dets if n_dets < n_trks else n_trks
            mi_a = <int *>malloc((min_dim + 1) * sizeof(int))
            mi_b = <int *>malloc((min_dim + 1) * sizeof(int))
            lapjv_solve(neg_iou, n_dets, n_trks, mi_a, mi_b, &n_mi)
            free(neg_iou)

    free(row_sum)
    free(col_sum)

    # Filter by IOU threshold
    matched_d = <int *>calloc(n_dets, sizeof(int))
    matched_count = 0
    for i in range(n_mi):
        d = mi_a[i]
        t = mi_b[i]
        if iou_matrix[d * n_trks + t] >= iou_threshold:
            match_a[matched_count] = d
            match_b[matched_count] = t
            matched_count += 1
            matched_d[d] = 1
    n_matches[0] = matched_count

    # Collect unmatched detections (includes those filtered by low IOU)
    ud_count = 0
    for i in range(n_dets):
        if matched_d[i] == 0:
            unmatched_dets[ud_count] = i
            ud_count += 1
    n_unmatched_dets[0] = ud_count

    if mi_a != NULL:
        free(mi_a)
    if mi_b != NULL:
        free(mi_b)
    free(matched_d)
    free(iou_matrix)


cdef class Sort:
    """
    SORT tracker implementation in Cython.
    """
    cdef int _max_age
    cdef int _min_hits
    cdef double _iou_threshold
    cdef vector[KalmanBoxTracker*] _trackers
    cdef int _frame_count
    cdef public int track_id_counter

    def __init__(self, int max_age=1, int min_hits=3, double iou_threshold=0.3):
        """
        Set key parameters for SORT.

        Args:
            max_age: Maximum age of a track before deletion
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: IOU threshold for matching
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._frame_count = 0
        self.track_id_counter = 0

    def __dealloc__(self):
        """
        Clean up allocated memory for trackers.
        """
        cdef int i
        for i in range(<int>self._trackers.size()):
            if self._trackers[i] != NULL:
                free(self._trackers[i])
        self._trackers.clear()

    @cython.boundscheck(False)  # type: ignore
    @cython.wraparound(False)  # type: ignore
    @cython.nonecheck(False)  # type: ignore
    cdef int _update_c(self, double *dets_bb, int n_dets, double *output):
        """
        C-level update.
        dets_bb: n_dets * 4 flat [x1,y1,x2,y2]
        output: pre-allocated buffer for results (max_tracks * 5)
        Returns: number of output tracks
        """
        cdef int i, j, k, t, d, idx
        cdef int n_trks
        cdef double pred_bbox[4]
        cdef double state_bbox[4]
        cdef int has_nan
        cdef int *to_del = NULL
        cdef int n_del = 0
        cdef double *trks = NULL
        cdef int min_dim
        cdef int *match_a = NULL
        cdef int *match_b = NULL
        cdef int n_matches = 0
        cdef int *unmatched_d = NULL
        cdef int n_unmatched_d = 0
        cdef KalmanBoxTracker *tracker = NULL
        cdef KalmanBoxTracker *new_trk = NULL
        cdef int n_output = 0
        cdef vector[int] dead_indices

        self._frame_count += 1

        # Get predicted locations from existing trackers
        n_trks = <int>self._trackers.size()
        trks = <double *>calloc((n_trks + 1) * 4, sizeof(double))
        to_del = <int *>malloc((n_trks + 1) * sizeof(int))

        for t in range(n_trks):
            KalmanBoxTracker_predict(self._trackers[t], pred_bbox)
            for k in range(4):
                trks[t * 4 + k] = pred_bbox[k]
            has_nan = 0
            for k in range(4):
                if isnan(pred_bbox[k]):
                    has_nan = 1
                    break
            if has_nan:
                to_del[n_del] = t
                n_del += 1

        # Remove invalid trackers
        for i in range(n_del - 1, -1, -1):
            t = to_del[i]
            free(self._trackers[t])
            self._trackers.erase(self._trackers.begin() + t)
            # Compact trks array
            for j in range(t, <int>self._trackers.size()):
                for k in range(4):
                    trks[j * 4 + k] = trks[(j + 1) * 4 + k]

        n_trks = <int>self._trackers.size()
        free(to_del)

        # Associate detections to trackers
        min_dim = n_dets if n_dets < n_trks else n_trks
        match_a = <int *>malloc((min_dim + 1) * sizeof(int))
        match_b = <int *>malloc((min_dim + 1) * sizeof(int))
        unmatched_d = <int *>malloc((n_dets + 1) * sizeof(int))

        associate(dets_bb, n_dets, trks, n_trks, self._iou_threshold,
                   match_a, match_b, &n_matches,
                   unmatched_d, &n_unmatched_d)

        # Update matched trackers with assigned detections
        for i in range(n_matches):
            d = match_a[i]
            t = match_b[i]
            KalmanBoxTracker_update(self._trackers[t], &dets_bb[d * 4])

        free(match_a)
        free(match_b)

        # Create and initialize new trackers for unmatched detections
        for i in range(n_unmatched_d):
            d = unmatched_d[i]
            new_trk = <KalmanBoxTracker *>malloc(sizeof(KalmanBoxTracker))
            KalmanBoxTracker_init(new_trk, &dets_bb[d * 4], self.track_id_counter)
            self.track_id_counter += 1
            self._trackers.push_back(new_trk)

        free(unmatched_d)
        free(trks)

        # Collect results and remove dead tracklets
        for idx in range(<int>self._trackers.size() - 1, -1, -1):
            tracker = self._trackers[idx]
            KalmanBoxTracker_get_state(tracker, state_bbox)

            if tracker.time_since_update < 1 and (tracker.hit_streak >= self._min_hits or self._frame_count <= self._min_hits):
                # +1 as MOT benchmark requires positive IDs
                output[n_output * 5 + 0] = state_bbox[0]
                output[n_output * 5 + 1] = state_bbox[1]
                output[n_output * 5 + 2] = state_bbox[2]
                output[n_output * 5 + 3] = state_bbox[3]
                output[n_output * 5 + 4] = <double>(tracker.id + 1)
                n_output += 1

            # Remove dead tracklet
            if tracker.time_since_update > self._max_age:
                dead_indices.push_back(idx)

        # Remove dead trackers (already in reverse order from the loop)
        for i in range(<int>dead_indices.size()):
            idx = dead_indices[i]
            free(self._trackers[idx])
            self._trackers.erase(self._trackers.begin() + idx)

        return n_output

    def update(self, cnp.ndarray[cnp.float64_t, ndim=2] dets):
        """
        Update tracker with new detections.

        Args:
            dets: Detection array in format [[x1,y1,x2,y2,score], ...]
                 Can be empty array with shape (0, 5) for frames without detections.

        Returns:
            Array of tracked objects [[x1,y1,x2,y2,track_id], ...]
        """
        cdef int n_dets = dets.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=2] dets_bb = np.ascontiguousarray(dets[:, :4], dtype=np.float64)

        # Allocate output buffer
        cdef int max_output = n_dets + <int>self._trackers.size() + 100
        cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.empty((max_output, 5), dtype=np.float64)

        cdef int n_output = self._update_c(
            <double *>dets_bb.data,
            n_dets,
            <double *>result.data
        )

        if n_output > 0:
            return result[:n_output]
        return np.empty((0, 5), dtype=np.float64)
