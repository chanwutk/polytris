# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from __future__ import print_function

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, fmax, fmin, isnan
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from polyis.tracker.cython.kalman_filter cimport KalmanFilter, kf_init, kf_predict, kf_update


cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t

    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


# Set random seed for reproducibility
np.random.seed(0)

# Module-level counter for KalmanBoxTracker IDs  
_kalman_box_tracker_count = 0

def reset_tracker_count():
    """Reset the tracker counter. Used for testing."""
    global _kalman_box_tracker_count
    _kalman_box_tracker_count = 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int linear_assignment(
    cnp.float64_t[:, :] cost_c,
    cnp.int16_t[:, :] matches
) noexcept nogil:
    """
    Solve linear assignment problem using lapjv.
    
    Args:
        cost_matrix: Cost matrix for assignment
        matches: Array of matched pairs [[row, col], ...]
    
    Returns:
        Number of matches
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
    # if ret != 0:
    #     free(x_c)
    #     free(y_c)
    #     if ret == -1:
    #         raise MemoryError('Out of memory.')
    #     raise RuntimeError('Unknown error (lapjv_internal returned %d).' % ret)

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
cdef void iou_batch(
    cnp.float64_t[:, :] bb_test,
    cnp.float64_t[:, :] bb_gt,
    cnp.float64_t[:, :] o_view
) noexcept nogil:
    """
    Compute IOU between two sets of bboxes in the form [x1,y1,x2,y2].
    Optimized with memory views and C loops.
    """
    cdef int N = bb_test.shape[0]
    cdef int M = bb_gt.shape[0]
    
    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area_test, area_gt
    
    for i in range(N):
        area_test = (bb_test[i, 2] - bb_test[i, 0]) * (bb_test[i, 3] - bb_test[i, 1])
        for j in range(M):
            xx1 = fmax(bb_test[i, 0], bb_gt[j, 0])
            yy1 = fmax(bb_test[i, 1], bb_gt[j, 1])
            xx2 = fmin(bb_test[i, 2], bb_gt[j, 2])
            yy2 = fmin(bb_test[i, 3], bb_gt[j, 3])
            
            w = fmax(0.0, xx2 - xx1)
            h = fmax(0.0, yy2 - yy1)
            wh = w * h
            
            area_gt = (bb_gt[j, 2] - bb_gt[j, 0]) * (bb_gt[j, 3] - bb_gt[j, 1])
            
            o_view[i, j] = wh / (area_test + area_gt - wh)


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
    # cdef public list history

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
def associate_detections_to_trackers(
    cnp.float64_t[:, :] detections,
    cnp.float64_t[:, :] trackers,
    double iou_threshold
):
    """
    Assign detections to tracked objects (both represented as bounding boxes).
    
    Args:
        detections: Detection bounding boxes (N, 4)
        trackers: Tracker bounding boxes (M, 4)
        iou_threshold: IOU threshold for matching
        
    Returns:
        Tuple of (matches, unmatched_detections) where:
        - matches: List of matched pairs [det_idx, trk_idx]
        - unmatched_detections: List of unmatched detection indices
    """
    cdef int N = detections.shape[0]
    cdef int M = trackers.shape[0]
    
    if M == 0:
        return [], [i for i in range(N)]
    
    # Compute IOU matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] iou_matrix = np.zeros((N, M), dtype=np.float64)
    iou_batch(detections, trackers, iou_matrix)
    cdef cnp.float64_t[:, :] iou_matrix_view = iou_matrix
    
    cdef cnp.ndarray[cnp.int16_t, ndim=2] a
    cdef int min_dim = min(N, M)
    cdef size
    cdef cnp.int16_t[:, :] matched_indices
    
    if min_dim > 0:
        a = (iou_matrix > iou_threshold).astype(np.int16)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # Simple assignment
            matched_indices = np.stack(np.where(a), axis=1).astype(np.int16)
        else:
            # Use linear assignment
            matched_indices = np.empty((iou_matrix.shape[0], 2), dtype=np.int16)
            size = linear_assignment(-iou_matrix, matched_indices)
            matched_indices = matched_indices[:size]
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=np.int16)
    
    # Find unmatched detections
    cdef list unmatched_detections = []
    cdef set[int] matched_indices_set = set(matched_indices[:, 0])
    cdef int i
    for i in range(N):
        if i not in matched_indices_set:
            unmatched_detections.append(i)
    
    # Filter out matches with low IOU
    cdef list matches = []
    cdef int det_idx, trk_idx
    
    for i in range(matched_indices.shape[0]):
        det_idx = matched_indices[i, 0]
        trk_idx = matched_indices[i, 1]
        if iou_matrix_view[det_idx, trk_idx] < iou_threshold:
            unmatched_detections.append(det_idx)
        else:
            matches.append(matched_indices[i])
    
    return matches, unmatched_detections


cdef public class PySort [object PySortObject, type PySortType]:
    """
    SORT tracker implementation in Cython.
    """
    cdef int max_age
    cdef int min_hits
    cdef double iou_threshold
    cdef vector[KalmanBoxTracker*] trackers
    cdef int frame_count
    
    def __init__(self, int max_age=1, int min_hits=3, double iou_threshold=0.3):
        """
        Set key parameters for SORT.
        
        Args:
            max_age: Maximum age of a track before deletion
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
    
    def __dealloc__(self):
        """
        Clean up allocated memory for trackers.
        """
        cdef int i
        for i in range(<int>self.trackers.size()):
            free(self.trackers[i])
    
    def update(self, cnp.ndarray[cnp.float64_t, ndim=2] dets):
        """
        Update tracker with new detections.
        
        Args:
            dets: Detection array in format [[x1,y1,x2,y2,score], ...]
                 Can be empty array with shape (0, 5) for frames without detections.
                 
        Returns:
            Array of tracked objects [[x1,y1,x2,y2,track_id], ...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        cdef int num_trackers = self.trackers.size()
        cdef cnp.ndarray[cnp.float64_t, ndim=2] trks = np.zeros((num_trackers, 5), dtype=np.float64)
        cdef vector[int] to_del
        cdef list ret = []
        cdef int t
        cdef double pos[4]
        cdef KalmanBoxTracker *tracker
        
        for t in range(num_trackers):
            tracker = self.trackers[t]
            KalmanBoxTracker_predict(tracker, pos)
            # trks[t, :] = [(pos[0]), (pos[1]), (pos[2]), (pos[3]), 0]
            trks[t, 0] = pos[0]
            trks[t, 1] = pos[1]
            trks[t, 2] = pos[2]
            trks[t, 3] = pos[3]
            trks[t, 4] = 0
            if isnan(pos[0]) or isnan(pos[1]) or isnan(pos[2]) or isnan(pos[3]):
                to_del.push_back(t)
        
        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in range(<int>to_del.size() - 1, -1, -1):
            free(self.trackers[t])
            self.trackers.erase(self.trackers.begin() + t)
        
        # Associate detections to trackers
        cdef list matched
        cdef list unmatched_dets
        matched, unmatched_dets = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        cdef int i
        for i in range(len(matched)):
            tracker = self.trackers[matched[i][1]]
            KalmanBoxTracker_update(tracker, &dets[<int>matched[i][0], 0])
        
        # Create and initialize new trackers for unmatched detections
        for i in range(len(unmatched_dets)):
            tracker = <KalmanBoxTracker*>malloc(sizeof(KalmanBoxTracker))
            global _kalman_box_tracker_count
            KalmanBoxTracker_init(tracker, &dets[<int>unmatched_dets[i], 0], _kalman_box_tracker_count)
            _kalman_box_tracker_count += 1
            self.trackers.push_back(tracker)
        
        # Collect results and remove dead tracklets
        for idx in range(self.trackers.size() - 1, -1, -1):
            tracker = self.trackers[idx]
            KalmanBoxTracker_get_state(tracker, pos)
            if (tracker.time_since_update < 1) and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive IDs
                ret.append(np.array([[pos[0], pos[1], pos[2], pos[3], tracker.id + 1]], dtype=np.float64))
            # Remove dead tracklet
            if tracker.time_since_update > self.max_age:
                free(self.trackers[idx])
                self.trackers.erase(self.trackers.begin() + idx)

        
        if len(ret) > 0:
            return np.concatenate(ret, dtype=np.float64)
        return np.empty((0, 5), dtype=np.float64)
