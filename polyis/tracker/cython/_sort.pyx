# Force compiling with Python 3 
# cython: language_level=3

from __future__ import print_function

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, fmax, fmin
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from polyis.tracker.cython._kalman_filter cimport KalmanFilter, kf_init, kf_predict, kf_update
from polyis.tracker.cython._lapjv import lapjv

# Set random seed for reproducibility
np.random.seed(0)

# Module-level counter for KalmanBoxTracker IDs  
_kalman_box_tracker_count = 0

def reset_tracker_count():
    """Reset the tracker counter. Used for testing."""
    global _kalman_box_tracker_count
    _kalman_box_tracker_count = 0


@cython.boundscheck(False)
@cython.wraparound(False)
def linear_assignment(cnp.ndarray[cnp.float64_t, ndim=2] cost_matrix):
    """
    Solve linear assignment problem using lapjv.
    
    Args:
        cost_matrix: Cost matrix for assignment
        
    Returns:
        Array of matched pairs [[row, col], ...]
    """
    x, y = lapjv(cost_matrix, extend_cost=True, return_cost=False)
    # Convert to array of pairs
    cdef list pairs = []
    cdef int i
    for i in range(len(x)):
        if x[i] >= 0:
            pairs.append([y[x[i]], x[i]])
    return np.array(pairs, dtype=np.int16)


@cython.boundscheck(False)
@cython.wraparound(False)
def iou_batch(cnp.ndarray[cnp.float64_t, ndim=2] bb_test, 
              cnp.ndarray[cnp.float64_t, ndim=2] bb_gt):
    """
    Compute IOU between two sets of bboxes in the form [x1,y1,x2,y2].
    
    Args:
        bb_test: Test bounding boxes (N, 4)
        bb_gt: Ground truth bounding boxes (M, 4)
        
    Returns:
        IOU matrix (N, M)
    """
    # Expand dimensions for broadcasting
    cdef cnp.ndarray[cnp.float64_t, ndim=3] bb_gt_ = np.expand_dims(bb_gt, 0)  # (1, M, 4)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] bb_test_ = np.expand_dims(bb_test, 1)  # (N, 1, 4)
    
    # Compute intersection coordinates
    cdef cnp.ndarray[cnp.float64_t, ndim=2] xx1 = np.maximum(bb_test_[..., 0], bb_gt_[..., 0])
    cdef cnp.ndarray[cnp.float64_t, ndim=2] yy1 = np.maximum(bb_test_[..., 1], bb_gt_[..., 1])
    cdef cnp.ndarray[cnp.float64_t, ndim=2] xx2 = np.minimum(bb_test_[..., 2], bb_gt_[..., 2])
    cdef cnp.ndarray[cnp.float64_t, ndim=2] yy2 = np.minimum(bb_test_[..., 3], bb_gt_[..., 3])
    
    # Compute width and height of intersection
    cdef cnp.ndarray[cnp.float64_t, ndim=2] w = np.maximum(0., xx2 - xx1)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] h = np.maximum(0., yy2 - yy1)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] wh = w * h
    
    # Compute IOU
    cdef cnp.ndarray[cnp.float64_t, ndim=2] o = wh / (
        (bb_test_[..., 2] - bb_test_[..., 0]) * (bb_test_[..., 3] - bb_test_[..., 1]) +
        (bb_gt_[..., 2] - bb_gt_[..., 0]) * (bb_gt_[..., 3] - bb_gt_[..., 1]) - wh
    )
    
    return o


@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
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
    
cdef void KalmanBoxTracker_init(KalmanBoxTracker *self, double *bbox):
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
    global _kalman_box_tracker_count
    self.id = _kalman_box_tracker_count
    _kalman_box_tracker_count += 1
    # self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    
cdef void KalmanBoxTracker_update(KalmanBoxTracker *self, double *bbox):
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

cdef void KalmanBoxTracker_predict(KalmanBoxTracker *self, double *bbox):
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

cdef void KalmanBoxTracker_get_state(KalmanBoxTracker *self, double *bbox):
    """
    Return current bounding box estimate.
    
    Returns:
        Current bounding box [x1, y1, x2, y2]
    """
    # cdef double bbox[4]
    convert_x_to_bbox(self.kf.x, bbox)


@cython.boundscheck(False)
@cython.wraparound(False)
def associate_detections_to_trackers(
    cnp.ndarray[cnp.float64_t, ndim=2] detections,
    cnp.ndarray[cnp.float64_t, ndim=2] trackers,
    double iou_threshold = 0.3
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
    if len(trackers) == 0:
        return [], [np.int16(i) for i in range(len(detections))]
    
    # Compute IOU matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] iou_matrix = iou_batch(detections, trackers)
    
    cdef cnp.ndarray[cnp.int16_t, ndim=2] matched_indices
    cdef cnp.ndarray[cnp.int16_t, ndim=2] a
    cdef int min_dim
    min_dim = min(iou_matrix.shape[0], iou_matrix.shape[1])
    if min_dim > 0:
        a = (iou_matrix > iou_threshold).astype(np.int16)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # Simple assignment
            matched_indices = np.stack(np.where(a), axis=1).astype(np.int16)
        else:
            # Use linear assignment
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=np.int16)
    
    # Find unmatched detections
    cdef list unmatched_detections = []
    cdef int d
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(np.int16(d))
    
    # Filter out matches with low IOU
    cdef list matches = []
    cdef int i
    cdef cnp.ndarray[cnp.int16_t, ndim=1] m
    for i in range(len(matched_indices)):
        m = matched_indices[i]
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(np.int16(m[0]))
        else:
            matches.append(m)
    
    return matches, unmatched_detections


cdef public class PySort [object PySortObject, type PySortType]:
    """
    SORT tracker implementation in Cython.
    """
    cdef public int max_age
    cdef public int min_hits
    cdef public double iou_threshold
    cdef vector[KalmanBoxTracker*] trackers
    cdef public int frame_count
    
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
        # self.trackers = []
        self.frame_count = 0
    
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
        cdef list to_del = []
        cdef list ret = []
        cdef int t
        cdef double pos[4]
        cdef KalmanBoxTracker *tracker
        
        for t in range(num_trackers):
            tracker = self.trackers[t]
            KalmanBoxTracker_predict(tracker, pos)
            trks[t, :] = [(pos[0]), (pos[1]), (pos[2]), (pos[3]), 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.erase(self.trackers.begin() + t)
        
        # Associate detections to trackers
        cdef list matched
        cdef list unmatched_dets
        matched, unmatched_dets = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        cdef int i
        cdef cnp.ndarray[cnp.int16_t, ndim=1] m
        for i in range(len(matched)):
            m = matched[i]
            tracker = self.trackers[m[1]]
            KalmanBoxTracker_update(tracker, &dets[m[0], 0])
        
        # Create and initialize new trackers for unmatched detections
        cdef KalmanBoxTracker *new_trk
        for det_idx in unmatched_dets:
            new_trk = <KalmanBoxTracker*>malloc(sizeof(KalmanBoxTracker))
            KalmanBoxTracker_init(new_trk, &dets[<int>det_idx, 0])
            self.trackers.push_back(new_trk)
        
        # Collect results and remove dead tracklets
        cdef cnp.uint64_t trk_ptr
        cdef KalmanBoxTracker *trk
        cdef double d[4]
        for idx in range(self.trackers.size() - 1, -1, -1):
            trk = self.trackers[idx]
            KalmanBoxTracker_get_state(trk, d)
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive IDs
                ret.append(np.array([[d[0], d[1], d[2], d[3], trk.id + 1]], dtype=np.float64))
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.erase(self.trackers.begin() + idx)

        
        if len(ret) > 0:
            return np.concatenate(ret, dtype=np.float64)
        return np.empty((0, 5), dtype=np.float64)
