# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, fmax, fmin, isnan, fabs
from libc.stdlib cimport malloc, calloc, free
from libcpp.vector cimport vector

from polyis.tracker.ocsort.cython.kalman_filter cimport KalmanFilter, kf_init, kf_predict, kf_update, kf_freeze, kf_unfreeze
from polyis.tracker.ocsort.cython.association import iou_batch, giou_batch, diou_batch, ciou_batch, ct_dist, associate, linear_assignment

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
cdef void convert_bbox_to_z(double *bbox, double *z) noexcept nogil:
    """
    Convert bounding box from [x1,y1,x2,y2] to [x,y,s,r] format.
    """
    cdef double w = bbox[2] - bbox[0]
    cdef double h = bbox[3] - bbox[1]
    z[0] = bbox[0] + w / 2.0
    z[1] = bbox[1] + h / 2.0
    z[2] = w * h  # scale is just area
    z[3] = w / (h + 1e-6)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void convert_x_to_bbox(double *x, double *bbox) noexcept nogil:
    """
    Convert bounding box from [x,y,s,r] to [x1,y1,x2,y2] format.
    """
    cdef double w = sqrt(x[2] * x[3])
    cdef double h = x[2] / w
    bbox[0] = x[0] - w/2.
    bbox[1] = x[1] - h/2.
    bbox[2] = x[0] + w/2.
    bbox[3] = x[1] + h/2.


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void speed_direction_internal(double *bbox1, double *bbox2, double *speed) noexcept nogil:
    """
    Compute speed direction between two bounding boxes.
    """
    cdef double cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cdef double cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cdef double cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cdef double cy2 = (bbox2[1] + bbox2[3]) / 2.0
    cdef double dx = cx2 - cx1
    cdef double dy = cy2 - cy1
    cdef double norm = sqrt(dx*dx + dy*dy) + 1e-6
    speed[0] = dy / norm
    speed[1] = dx / norm


cdef struct KalmanBoxTracker:
    KalmanFilter kf
    int time_since_update
    int id
    int hits
    int hit_streak
    int age
    int delta_t
    # Store last observation as [x1, y1, x2, y2, score] or [-1, -1, -1, -1, -1] for placeholder
    double last_observation[5]
    # Velocity as [dy, dx]
    double velocity[2]
    int has_velocity
    # History observations stored as flat array: each observation is 4 doubles (x, y, s, r)
    # We'll use a Python list for this in the actual implementation
    int max_history
    int history_len


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_init(KalmanBoxTracker *self, double *bbox, int id, int delta_t) noexcept nogil:
    """
    Initialize tracker using initial bounding box.
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
    self.kf.R[2][2] *= 10.0
    self.kf.R[3][3] *= 10.0
    
    # Give high uncertainty to the unobservable initial velocities
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
    self.kf.Q[6][6] *= 0.01  # Applied twice as in original code
    
    # Initialize state with bbox
    convert_bbox_to_z(bbox, self.kf.x)
    
    self.time_since_update = 0
    self.id = id
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.delta_t = delta_t
    
    # Initialize last_observation as placeholder
    self.last_observation[0] = -1.0
    self.last_observation[1] = -1.0
    self.last_observation[2] = -1.0
    self.last_observation[3] = -1.0
    self.last_observation[4] = -1.0
    
    self.has_velocity = 0
    self.velocity[0] = 0.0
    self.velocity[1] = 0.0
    
    self.max_history = 1000  # Maximum history size
    self.history_len = 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_update(KalmanBoxTracker *self, double *bbox, double *history_obs, int *history_len) noexcept nogil:
    """
    Update state vector with observed bbox.
    """
    # Declare all variables at the top
    cdef double previous_box[4]
    cdef double prev_bbox[4]
    cdef double z[4]
    cdef int found = 0
    cdef int i, dt, j
    cdef int obs_idx
    cdef double sum_val
    cdef double w, h
    cdef KalmanFilter *kf_ptr

    if bbox is not NULL:
        # Check if we have a previous observation
        if self.last_observation[0] >= 0:
            # Find previous box for velocity calculation
            found = 0

            # Try to find observation delta_t steps ago
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt >= 0 and self.age - dt < history_len[0]:
                    # Check if observation exists (not all zeros)
                    obs_idx = (self.age - dt) * 4
                    if obs_idx < history_len[0] * 4:
                        sum_val = 0.0
                        for j in range(4):
                            sum_val += fabs(history_obs[obs_idx + j])
                        if sum_val > 1e-9:
                            # Found valid observation
                            previous_box[0] = history_obs[obs_idx + 0]
                            previous_box[1] = history_obs[obs_idx + 1]
                            previous_box[2] = history_obs[obs_idx + 2]
                            previous_box[3] = history_obs[obs_idx + 3]
                            # Convert to bbox format for speed_direction
                            w = sqrt(previous_box[2] * previous_box[3])
                            h = previous_box[2] / w
                            prev_bbox[0] = previous_box[0] - w/2.0
                            prev_bbox[1] = previous_box[1] - h/2.0
                            prev_bbox[2] = previous_box[0] + w/2.0
                            prev_bbox[3] = previous_box[1] + h/2.0

                            speed_direction_internal(prev_bbox, bbox, self.velocity)
                            self.has_velocity = 1
                            found = 1
                            break

            if not found:
                # Use last observation
                prev_bbox[0] = self.last_observation[0]
                prev_bbox[1] = self.last_observation[1]
                prev_bbox[2] = self.last_observation[2]
                prev_bbox[3] = self.last_observation[3]
                speed_direction_internal(prev_bbox, bbox, self.velocity)
                self.has_velocity = 1

        # Update last observation
        self.last_observation[0] = bbox[0]
        self.last_observation[1] = bbox[1]
        self.last_observation[2] = bbox[2]
        self.last_observation[3] = bbox[3]
        if bbox[4] > 0:
            self.last_observation[4] = bbox[4]

        # Note: History is NOT added here to avoid double-addition with Python wrapper
        # The Python wrapper manages history_observations list entirely

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Update Kalman filter
        kf_ptr = &self.kf
        convert_bbox_to_z(bbox, z)
        kf_update(kf_ptr, z)
    else:
        # No observation - update without measurement
        kf_ptr = &self.kf
        convert_bbox_to_z(bbox, z)
        kf_update(kf_ptr, NULL)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_predict(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """
    Advance state vector and return predicted bounding box estimate.
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
    """
    convert_x_to_bbox(self.kf.x, bbox)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void k_previous_obs_internal(double *history_obs, int history_len, int age, int delta_t, double *result) noexcept nogil:
    """
    Get previous observation k steps ago.
    Returns [-1, -1, -1, -1, -1] if not found.
    """
    cdef int i, dt, obs_idx, j
    cdef double sum_val
    
    if history_len == 0:
        result[0] = -1.0
        result[1] = -1.0
        result[2] = -1.0
        result[3] = -1.0
        result[4] = -1.0
        return
    
    for i in range(delta_t):
        dt = delta_t - i
        if age - dt >= 0 and age - dt < history_len:
            obs_idx = (age - dt) * 4
            if obs_idx < history_len * 4:
                sum_val = 0.0
                for j in range(4):
                    sum_val += fabs(history_obs[obs_idx + j])
                if sum_val > 1e-9:
                    # Found valid observation
                    result[0] = history_obs[obs_idx + 0]
                    result[1] = history_obs[obs_idx + 1]
                    result[2] = history_obs[obs_idx + 2]
                    result[3] = history_obs[obs_idx + 3]
                    result[4] = 1.0  # Valid observation
                    return
    
    # Use most recent observation
    if history_len > 0:
        obs_idx = (history_len - 1) * 4
        result[0] = history_obs[obs_idx + 0]
        result[1] = history_obs[obs_idx + 1]
        result[2] = history_obs[obs_idx + 2]
        result[3] = history_obs[obs_idx + 3]
        result[4] = 1.0
    else:
        result[0] = -1.0
        result[1] = -1.0
        result[2] = -1.0
        result[3] = -1.0
        result[4] = -1.0


# Python wrapper class for KalmanBoxTracker
cdef class KalmanBoxTrackerPy:
    cdef KalmanBoxTracker *tracker
    cdef object observations_dict  # Python dict for observations
    cdef int history_len
    cdef double *history_obs_array
    cdef int history_array_size
    
    def __cinit__(self, bbox, delta_t) -> None:
        global _kalman_box_tracker_count
        self.tracker = <KalmanBoxTracker*>malloc(sizeof(KalmanBoxTracker))
        cdef double bbox_arr[4]
        bbox_arr[0] = bbox[0]
        bbox_arr[1] = bbox[1]
        bbox_arr[2] = bbox[2]
        bbox_arr[3] = bbox[3]
        KalmanBoxTracker_init(self.tracker, bbox_arr, _kalman_box_tracker_count, delta_t)
        _kalman_box_tracker_count += 1
        
        self.observations_dict = {}
        self.history_array_size = 1000
        self.history_obs_array = <double*>calloc(self.history_array_size * 4, sizeof(double))
    
    def __dealloc__(self):
        if self.tracker is not NULL:
            free(self.tracker)
        if self.history_obs_array is not NULL:
            free(self.history_obs_array)
    
    def update(self, bbox):
        cdef double bbox_arr[5]
        cdef int history_len = self.history_len

        if bbox is not None:
            bbox_arr[0] = bbox[0]
            bbox_arr[1] = bbox[1]
            bbox_arr[2] = bbox[2]
            bbox_arr[3] = bbox[3]
            if len(bbox) > 4:
                bbox_arr[4] = bbox[4]
            else:
                bbox_arr[4] = 1.0

            # Update observations dict
            self.observations_dict[self.tracker.age] = bbox

            # Update Kalman filter
            KalmanBoxTracker_update(self.tracker, bbox_arr, self.history_obs_array, &history_len)
        else:
            # Update without observation
            KalmanBoxTracker_update(self.tracker, NULL, self.history_obs_array, &history_len)
    
    def predict(self):
        cdef double bbox[4]
        KalmanBoxTracker_predict(self.tracker, bbox)
        return np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
    
    def get_state(self):
        cdef double bbox[4]
        KalmanBoxTracker_get_state(self.tracker, bbox)
        # Return shape (1, 4) to match Python version
        return np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape((1, 4))
    
    @property
    def time_since_update(self):
        return self.tracker.time_since_update
    
    @property
    def id(self):
        return self.tracker.id
    
    @property
    def hits(self):
        return self.tracker.hits
    
    @property
    def hit_streak(self):
        return self.tracker.hit_streak
    
    @property
    def age(self):
        return self.tracker.age
    
    @property
    def last_observation(self):
        if self.tracker.last_observation[0] < 0:
            return np.array([-1, -1, -1, -1, -1])
        return np.array([
            self.tracker.last_observation[0],
            self.tracker.last_observation[1],
            self.tracker.last_observation[2],
            self.tracker.last_observation[3],
            self.tracker.last_observation[4]
        ])
    
    @property
    def velocity(self):
        if self.tracker.has_velocity:
            return np.array([self.tracker.velocity[0], self.tracker.velocity[1]])
        return None
    
    @property
    def observations(self):
        return self.observations_dict


# Association function mapping
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist
}


def k_previous_obs(observations, cur_age, k):
    """
    Get previous observation k steps ago.
    """
    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1])
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


cdef public class OCSort [object OCSortObject, type OCSortType]:
    """
    OC-SORT tracker implementation in Cython.
    """
    cdef int max_age
    cdef int min_hits
    cdef double iou_threshold
    cdef double det_thresh
    cdef int delta_t
    cdef double inertia
    cdef int use_byte
    cdef object trackers
    cdef int frame_count
    cdef object asso_func
    
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Set key parameters for OC-SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.trackers = []
        self.frame_count = 0
        self.asso_func = ASSO_FUNCS.get(asso_func, iou_batch)
        global _kalman_box_tracker_count
        _kalman_box_tracker_count = 0
    
    def update(self, output_results, img_info, img_size):
        """
        Update tracker with new detections.
        """
        if output_results is None:
            return np.empty((0, 5))
        
        self.frame_count += 1
        
        # Post-process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            if hasattr(output_results, 'cpu'):
                output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        velocities = np.array([
            trk.velocity if trk.velocity is not None else np.array((0, 0)) 
            for trk in self.trackers
        ])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([
            k_previous_obs(trk.observations, trk.age, self.delta_t) 
            for trk in self.trackers
        ])
        
        # First round of association
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Second round of association by OCR
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        
        for m in unmatched_trks:
            self.trackers[m].update(None)
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTrackerPy(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

