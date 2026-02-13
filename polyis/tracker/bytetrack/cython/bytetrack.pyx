# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of ByteTrack tracker.

BYTETracker is a cdef class with C-typed vector fields (no Python objects
in hot path). All internal operations use cdef functions, C structs, and
C++ vectors/maps. The class provides a thin Python-callable update()
method for the wrapper and tests.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, isnan
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap

from polyis.tracker.bytetrack.cython.kalman_filter cimport KalmanFilter, kf_init, kf_initiate, kf_predict, kf_update
from polyis.tracker.bytetrack.cython.matching cimport compute_iou_cost, fuse_score, linear_assignment


# Track state enum
cdef enum TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# ============================================================
# Bounding box conversion utilities (all nogil)
# ============================================================

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlwh_to_xyah(double *tlwh, double *xyah) noexcept nogil:
    """Convert bounding box from [x, y, w, h] to [cx, cy, aspect, h]."""
    xyah[0] = tlwh[0] + tlwh[2] / 2.0
    xyah[1] = tlwh[1] + tlwh[3] / 2.0
    xyah[2] = tlwh[2] / tlwh[3]
    xyah[3] = tlwh[3]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlwh_to_tlbr(double *tlwh, double *tlbr) noexcept nogil:
    """Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2]."""
    tlbr[0] = tlwh[0]
    tlbr[1] = tlwh[1]
    tlbr[2] = tlwh[0] + tlwh[2]
    tlbr[3] = tlwh[1] + tlwh[3]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlbr_to_tlwh(double *tlbr, double *tlwh) noexcept nogil:
    """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h]."""
    tlwh[0] = tlbr[0]
    tlwh[1] = tlbr[1]
    tlwh[2] = tlbr[2] - tlbr[0]
    tlwh[3] = tlbr[3] - tlbr[1]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void xyah_to_tlwh(double *xyah, double *tlwh) noexcept nogil:
    """Convert bounding box from [cx, cy, aspect, h] to [x, y, w, h]."""
    cdef double w = xyah[2] * xyah[3]
    tlwh[0] = xyah[0] - w / 2.0
    tlwh[1] = xyah[1] - xyah[3] / 2.0
    tlwh[2] = w
    tlwh[3] = xyah[3]


# ============================================================
# STrack struct and cdef functions (all nogil)
# ============================================================

# STrack structure
cdef struct STrack:
    KalmanFilter kf
    double mean[8]
    double covariance[64]  # 8x8 matrix stored flat
    double _tlwh[4]
    int is_activated
    double score
    int tracklet_len
    int state
    int track_id
    int frame_id
    int start_frame


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_init(STrack *self, double *tlwh, double score, int track_id) noexcept nogil:
    """Initialize STrack with bounding box and score."""
    cdef int i

    # Initialize Kalman filter
    kf_init(&self.kf)

    # Store initial bounding box
    self._tlwh[0] = tlwh[0]
    self._tlwh[1] = tlwh[1]
    self._tlwh[2] = tlwh[2]
    self._tlwh[3] = tlwh[3]

    # Initialize mean and covariance to zero
    for i in range(8):
        self.mean[i] = 0.0
    for i in range(64):
        self.covariance[i] = 0.0

    # Initialize state fields
    self.is_activated = 0
    self.score = score
    self.tracklet_len = 0
    self.state = TrackState.New
    self.track_id = track_id
    self.frame_id = 0
    self.start_frame = 0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_predict(STrack *self) noexcept nogil:
    """Predict next state using Kalman filter."""
    cdef int i, j

    # Copy mean and covariance into Kalman filter state
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Set velocity to 0 if not tracked
    if self.state != TrackState.Tracked:
        self.kf.x[7] = 0.0

    # Run Kalman filter prediction
    kf_predict(&self.kf)

    # Copy results back to STrack
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_activate(STrack *self, int frame_id, int *track_id_counter) noexcept nogil:
    """Activate a new tracklet. Increments track_id_counter and assigns new ID."""
    cdef double xyah[4]

    # Convert tlwh to xyah for Kalman filter
    tlwh_to_xyah(self._tlwh, xyah)

    # Initialize Kalman filter with measurement
    kf_initiate(&self.kf, xyah, self.mean, self.covariance)

    # Assign track ID from counter
    track_id_counter[0] += 1
    self.track_id = track_id_counter[0]

    # Reset tracklet length and set state
    self.tracklet_len = 0
    self.state = TrackState.Tracked

    # Only activate on first frame
    if frame_id == 1:
        self.is_activated = 1

    self.frame_id = frame_id
    self.start_frame = frame_id


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_re_activate(STrack *self, double *new_tlwh, double new_score, int frame_id, int new_id, int *track_id_counter) noexcept nogil:
    """Re-activate a lost track with new detection."""
    cdef double xyah[4]
    cdef int i, j

    # Convert new bbox to xyah
    tlwh_to_xyah(new_tlwh, xyah)

    # Copy mean and covariance to Kalman filter
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Update Kalman filter with new measurement
    kf_update(&self.kf, xyah)

    # Copy results back
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]

    # Update state fields
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = 1
    self.frame_id = frame_id

    # Assign new ID if requested
    if new_id:
        track_id_counter[0] += 1
        self.track_id = track_id_counter[0]

    self.score = new_score


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_update(STrack *self, double *new_tlwh, double new_score, int frame_id) noexcept nogil:
    """Update a matched track with new detection."""
    cdef double xyah[4]
    cdef int i, j

    # Update frame and tracklet length
    self.frame_id = frame_id
    self.tracklet_len += 1

    # Convert new bbox to xyah
    tlwh_to_xyah(new_tlwh, xyah)

    # Copy mean and covariance to Kalman filter
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Update Kalman filter with new measurement
    kf_update(&self.kf, xyah)

    # Copy results back
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]

    # Mark as tracked and activated
    self.state = TrackState.Tracked
    self.is_activated = 1
    self.score = new_score


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_mark_lost(STrack *self) noexcept nogil:
    """Mark track as lost."""
    self.state = TrackState.Lost


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_mark_removed(STrack *self) noexcept nogil:
    """Mark track as removed."""
    self.state = TrackState.Removed


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_get_tlwh(STrack *self, double *tlwh) noexcept nogil:
    """Get current position in tlwh format."""
    cdef double xyah[4]
    cdef int i

    if self.mean[0] == 0.0 and self.mean[1] == 0.0:
        # Mean not initialized, use stored tlwh
        for i in range(4):
            tlwh[i] = self._tlwh[i]
    else:
        # Convert mean (xyah) to tlwh
        for i in range(4):
            xyah[i] = self.mean[i]
        xyah_to_tlwh(xyah, tlwh)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_get_tlbr(STrack *self, double *tlbr) noexcept nogil:
    """Get current position in tlbr format."""
    cdef double tlwh[4]
    STrack_get_tlwh(self, tlwh)
    tlwh_to_tlbr(tlwh, tlbr)


# ============================================================
# Track list helper cdef functions (using vector / map)
# ============================================================

cdef void joint_stracks(vector[STrack*] *a, vector[STrack*] *b, vector[STrack*] *out):
    """Combine two track lists into out, removing duplicates by track_id (map)."""
    cdef cppmap[int, int] exists
    cdef int i
    out[0].clear()
    # Add all tracks from a
    for i in range(<int>a[0].size()):
        exists[a[0][i].track_id] = 1
        out[0].push_back(a[0][i])
    # Add tracks from b that are not already in a
    for i in range(<int>b[0].size()):
        if exists.count(b[0][i].track_id) == 0:
            exists[b[0][i].track_id] = 1
            out[0].push_back(b[0][i])


cdef void sub_stracks(vector[STrack*] *a, vector[STrack*] *b, vector[STrack*] *out):
    """Remove from a any track whose track_id appears in b (map)."""
    cdef cppmap[int, int] b_ids
    cdef int i
    out[0].clear()
    # Collect all track_ids from b
    for i in range(<int>b[0].size()):
        b_ids[b[0][i].track_id] = 1
    # Keep only tracks from a whose id is not in b
    for i in range(<int>a[0].size()):
        if b_ids.count(a[0][i].track_id) == 0:
            out[0].push_back(a[0][i])


cdef void remove_duplicate_stracks(
    vector[STrack*] *stracksa, vector[STrack*] *stracksb,
    vector[STrack*] *out_a, vector[STrack*] *out_b
):
    """Remove duplicate tracks between two lists based on IOU distance."""
    cdef int na = <int>stracksa[0].size()
    cdef int nb = <int>stracksb[0].size()
    cdef int i, j, timep, timeq

    # If either list is empty, copy as-is
    if na == 0 or nb == 0:
        out_a[0] = stracksa[0]
        out_b[0] = stracksb[0]
        return

    # Extract tlbr arrays from both lists
    cdef double *atlbrs = <double*>malloc(na * 4 * sizeof(double))
    cdef double *btlbrs = <double*>malloc(nb * 4 * sizeof(double))
    for i in range(na):
        STrack_get_tlbr(stracksa[0][i], &atlbrs[i * 4])
    for i in range(nb):
        STrack_get_tlbr(stracksb[0][i], &btlbrs[i * 4])

    # Compute IOU cost matrix (1 - IOU)
    cdef double *cost = <double*>malloc(na * nb * sizeof(double))
    compute_iou_cost(atlbrs, na, btlbrs, nb, cost)

    # Find duplicates (cost < 0.15 means IOU > 0.85)
    cdef int *dup_a = <int*>calloc(na, sizeof(int))
    cdef int *dup_b = <int*>calloc(nb, sizeof(int))

    for i in range(na):
        for j in range(nb):
            if cost[i * nb + j] < 0.15:
                # Compare track durations to decide which to remove
                timep = stracksa[0][i].frame_id - stracksa[0][i].start_frame
                timeq = stracksb[0][j].frame_id - stracksb[0][j].start_frame
                if timep > timeq:
                    dup_b[j] = 1
                else:
                    dup_a[i] = 1

    # Build filtered result lists
    out_a[0].clear()
    out_b[0].clear()
    for i in range(na):
        if dup_a[i] == 0:
            out_a[0].push_back(stracksa[0][i])
    for i in range(nb):
        if dup_b[i] == 0:
            out_b[0].push_back(stracksb[0][i])

    # Free temporary arrays
    free(atlbrs)
    free(btlbrs)
    free(cost)
    free(dup_a)
    free(dup_b)


cdef void free_strack_vector(vector[STrack*] *v):
    """Free all STrack pointers in a vector and clear it."""
    cdef int i
    for i in range(<int>v[0].size()):
        if v[0][i] != NULL:
            free(v[0][i])
    v[0].clear()


# ============================================================
# BYTETracker cdef class (C-level typed fields, Python interface)
# ============================================================

cdef class BYTETracker:
    """
    ByteTrack tracker implemented as a cdef class with C-typed vector fields.
    All internal operations use C structs and cdef functions for maximum speed.
    The track_id_counter is stored here and exposed as a public attribute.
    """
    # Track lists (C++ vectors of heap-allocated STrack pointers)
    cdef vector[STrack*] _tracked_stracks
    cdef vector[STrack*] _lost_stracks
    cdef vector[STrack*] _removed_stracks
    # Tracker parameters
    cdef int _frame_id
    cdef double _track_thresh
    cdef double _match_thresh
    cdef double _det_thresh
    cdef int _buffer_size
    cdef int _max_time_lost
    cdef int _mot20
    # Track ID counter (public so wrapper/tests can read/reset)
    cdef public int track_id_counter

    def __cinit__(self):
        self._frame_id = 0
        self.track_id_counter = 0

    def __init__(self, args, frame_rate=30):
        self._track_thresh = args.track_thresh
        self._match_thresh = args.match_thresh
        self._det_thresh = args.track_thresh + 0.1
        self._buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self._max_time_lost = self._buffer_size
        self._mot20 = 1 if args.mot20 else 0

    def __dealloc__(self):
        # Collect all unique STrack pointers across all vectors to avoid double-free
        cdef cppmap[long, int] freed
        cdef int i
        cdef long ptr_val
        # Free tracked
        for i in range(<int>self._tracked_stracks.size()):
            ptr_val = <long>self._tracked_stracks[i]
            if freed.count(ptr_val) == 0 and self._tracked_stracks[i] != NULL:
                freed[ptr_val] = 1
                free(self._tracked_stracks[i])
        self._tracked_stracks.clear()
        # Free lost
        for i in range(<int>self._lost_stracks.size()):
            ptr_val = <long>self._lost_stracks[i]
            if freed.count(ptr_val) == 0 and self._lost_stracks[i] != NULL:
                freed[ptr_val] = 1
                free(self._lost_stracks[i])
        self._lost_stracks.clear()
        # Free removed
        for i in range(<int>self._removed_stracks.size()):
            ptr_val = <long>self._removed_stracks[i]
            if freed.count(ptr_val) == 0 and self._removed_stracks[i] != NULL:
                freed[ptr_val] = 1
                free(self._removed_stracks[i])
        self._removed_stracks.clear()

    @cython.boundscheck(False)  # type: ignore
    @cython.wraparound(False)  # type: ignore
    @cython.nonecheck(False)  # type: ignore
    cdef int _update_c(self, double *bboxes, double *scores, int n_dets, double *output):
        """
        C-level update. Receives pre-parsed, pre-scaled detections.

        Args:
            bboxes: flat n_dets*4 array of [x1,y1,x2,y2] bounding boxes
            scores: flat n_dets array of confidence scores
            n_dets: number of detections
            output: pre-allocated output buffer (at least max_tracks*5 doubles)

        Returns:
            Number of output tracks written to output
        """
        cdef int i, j, k
        cdef int n_high, n_low
        cdef int hi_idx, lo_idx

        # Increment frame counter
        self._frame_id += 1

        # ---- Filter detections by threshold ----
        # Count high-score and low-score detections
        n_high = 0
        n_low = 0
        for i in range(n_dets):
            if scores[i] > self._track_thresh:
                n_high += 1
            elif scores[i] > 0.1:
                n_low += 1

        # Allocate filtered bbox/score arrays
        cdef double *high_bb = <double*>malloc((n_high + 1) * 4 * sizeof(double))
        cdef double *high_sc = <double*>malloc((n_high + 1) * sizeof(double))
        cdef double *low_bb = <double*>malloc((n_low + 1) * 4 * sizeof(double))
        cdef double *low_sc = <double*>malloc((n_low + 1) * sizeof(double))

        # Fill filtered arrays
        hi_idx = 0
        lo_idx = 0
        for i in range(n_dets):
            if scores[i] > self._track_thresh:
                for k in range(4):
                    high_bb[hi_idx * 4 + k] = bboxes[i * 4 + k]
                high_sc[hi_idx] = scores[i]
                hi_idx += 1
            elif scores[i] > 0.1:
                for k in range(4):
                    low_bb[lo_idx * 4 + k] = bboxes[i * 4 + k]
                low_sc[lo_idx] = scores[i]
                lo_idx += 1

        # ---- Create high-score detection STracks ----
        cdef vector[STrack*] detections
        cdef STrack *det_ptr
        cdef double det_tlwh_buf[4]
        for i in range(n_high):
            det_ptr = <STrack*>malloc(sizeof(STrack))
            # Convert tlbr to tlwh for STrack initialization
            tlbr_to_tlwh(&high_bb[i * 4], det_tlwh_buf)
            STrack_init(det_ptr, det_tlwh_buf, high_sc[i], -1)
            detections.push_back(det_ptr)

        # ---- Separate tracked / unconfirmed ----
        cdef vector[STrack*] unconfirmed
        cdef vector[STrack*] tracked
        for i in range(<int>self._tracked_stracks.size()):
            if self._tracked_stracks[i].is_activated == 0:
                unconfirmed.push_back(self._tracked_stracks[i])
            else:
                tracked.push_back(self._tracked_stracks[i])

        # ---- Step 2: First association with high-score detections ----
        cdef vector[STrack*] strack_pool
        joint_stracks(&tracked, &self._lost_stracks, &strack_pool)

        # Predict current location for all tracks in pool
        for i in range(<int>strack_pool.size()):
            STrack_predict(strack_pool[i])

        cdef int n_pool = <int>strack_pool.size()
        cdef int n_det = <int>detections.size()

        # Temporary arrays for association
        cdef double *pool_tlbrs = NULL
        cdef double *det_tlbrs = NULL
        cdef double *cost_mat = NULL
        cdef double *det_sc_tmp = NULL
        cdef int *match_a = NULL
        cdef int *match_b = NULL
        cdef int n_matches = 0
        cdef int *unmatched_trk = NULL
        cdef int n_unmatched_trk = 0
        cdef int *unmatched_det = NULL
        cdef int n_unmatched_det = 0
        cdef int min_dim

        # Frame-local track lists
        cdef vector[STrack*] activated_starcks
        cdef vector[STrack*] refind_stracks_v
        cdef vector[STrack*] lost_stracks_v
        cdef vector[STrack*] removed_stracks_v

        if n_pool > 0 and n_det > 0:
            # Extract tlbrs from pool and detections
            pool_tlbrs = <double*>malloc(n_pool * 4 * sizeof(double))
            det_tlbrs = <double*>malloc(n_det * 4 * sizeof(double))
            for i in range(n_pool):
                STrack_get_tlbr(strack_pool[i], &pool_tlbrs[i * 4])
            for i in range(n_det):
                STrack_get_tlbr(detections[i], &det_tlbrs[i * 4])

            # Compute IOU cost matrix
            cost_mat = <double*>malloc(n_pool * n_det * sizeof(double))
            compute_iou_cost(pool_tlbrs, n_pool, det_tlbrs, n_det, cost_mat)

            # Fuse detection scores if not MOT20
            if self._mot20 == 0:
                det_sc_tmp = <double*>malloc(n_det * sizeof(double))
                for i in range(n_det):
                    det_sc_tmp[i] = detections[i].score
                fuse_score(cost_mat, det_sc_tmp, n_pool, n_det)
                free(det_sc_tmp)

            # Solve linear assignment
            min_dim = n_pool if n_pool < n_det else n_det
            match_a = <int*>malloc((min_dim + 1) * sizeof(int))
            match_b = <int*>malloc((min_dim + 1) * sizeof(int))
            unmatched_trk = <int*>malloc((n_pool + 1) * sizeof(int))
            unmatched_det = <int*>malloc((n_det + 1) * sizeof(int))
            linear_assignment(cost_mat, n_pool, n_det, self._match_thresh,
                              match_a, match_b, &n_matches,
                              unmatched_trk, &n_unmatched_trk,
                              unmatched_det, &n_unmatched_det)

            # Free cost / tlbr arrays
            free(cost_mat); free(pool_tlbrs); free(det_tlbrs)
        else:
            # No matches possible: all tracks and detections are unmatched
            n_matches = 0
            unmatched_trk = <int*>malloc((n_pool + 1) * sizeof(int))
            for i in range(n_pool):
                unmatched_trk[i] = i
            n_unmatched_trk = n_pool
            unmatched_det = <int*>malloc((n_det + 1) * sizeof(int))
            for i in range(n_det):
                unmatched_det[i] = i
            n_unmatched_det = n_det

        # Process first-round matches
        cdef STrack *trk
        cdef STrack *det_trk
        cdef double det_tlwh_tmp[4]

        for i in range(n_matches):
            trk = strack_pool[match_a[i]]
            det_trk = detections[match_b[i]]
            STrack_get_tlwh(det_trk, det_tlwh_tmp)
            if trk.state == TrackState.Tracked:
                # Update matched tracked track
                STrack_update(trk, det_tlwh_tmp, det_trk.score, self._frame_id)
                activated_starcks.push_back(trk)
            else:
                # Re-activate matched lost track
                STrack_re_activate(trk, det_tlwh_tmp, det_trk.score, self._frame_id, 0, &self.track_id_counter)
                refind_stracks_v.push_back(trk)

        free(match_a); match_a = NULL
        free(match_b); match_b = NULL

        # ---- Step 3: Second association with low-score detections ----
        # Create low-score detection STracks
        cdef vector[STrack*] detections_second
        for i in range(n_low):
            det_ptr = <STrack*>malloc(sizeof(STrack))
            tlbr_to_tlwh(&low_bb[i * 4], det_tlwh_buf)
            STrack_init(det_ptr, det_tlwh_buf, low_sc[i], -1)
            detections_second.push_back(det_ptr)

        # Build r_tracked: unmatched tracks from pool that are in Tracked state
        cdef vector[STrack*] r_tracked
        for i in range(n_unmatched_trk):
            if strack_pool[unmatched_trk[i]].state == TrackState.Tracked:
                r_tracked.push_back(strack_pool[unmatched_trk[i]])

        free(unmatched_trk); unmatched_trk = NULL

        cdef int n_rtracked = <int>r_tracked.size()
        cdef int n_det2 = <int>detections_second.size()
        cdef int *match_a2 = NULL
        cdef int *match_b2 = NULL
        cdef int n_matches2 = 0
        cdef int *unmatched_trk2 = NULL
        cdef int n_unmatched_trk2 = 0
        cdef int *unmatched_det2 = NULL
        cdef int n_unmatched_det2 = 0

        if n_rtracked > 0 and n_det2 > 0:
            # Extract tlbrs for second-round association
            pool_tlbrs = <double*>malloc(n_rtracked * 4 * sizeof(double))
            det_tlbrs = <double*>malloc(n_det2 * 4 * sizeof(double))
            for i in range(n_rtracked):
                STrack_get_tlbr(r_tracked[i], &pool_tlbrs[i * 4])
            for i in range(n_det2):
                STrack_get_tlbr(detections_second[i], &det_tlbrs[i * 4])

            # Compute cost and solve assignment
            cost_mat = <double*>malloc(n_rtracked * n_det2 * sizeof(double))
            compute_iou_cost(pool_tlbrs, n_rtracked, det_tlbrs, n_det2, cost_mat)

            min_dim = n_rtracked if n_rtracked < n_det2 else n_det2
            match_a2 = <int*>malloc((min_dim + 1) * sizeof(int))
            match_b2 = <int*>malloc((min_dim + 1) * sizeof(int))
            unmatched_trk2 = <int*>malloc((n_rtracked + 1) * sizeof(int))
            unmatched_det2 = <int*>malloc((n_det2 + 1) * sizeof(int))
            linear_assignment(cost_mat, n_rtracked, n_det2, 0.5,
                              match_a2, match_b2, &n_matches2,
                              unmatched_trk2, &n_unmatched_trk2,
                              unmatched_det2, &n_unmatched_det2)

            free(cost_mat); free(pool_tlbrs); free(det_tlbrs)
        else:
            n_matches2 = 0
            unmatched_trk2 = <int*>malloc((n_rtracked + 1) * sizeof(int))
            for i in range(n_rtracked):
                unmatched_trk2[i] = i
            n_unmatched_trk2 = n_rtracked
            unmatched_det2 = <int*>malloc((n_det2 + 1) * sizeof(int))
            for i in range(n_det2):
                unmatched_det2[i] = i
            n_unmatched_det2 = n_det2

        # Process second-round matches
        for i in range(n_matches2):
            trk = r_tracked[match_a2[i]]
            det_trk = detections_second[match_b2[i]]
            STrack_get_tlwh(det_trk, det_tlwh_tmp)
            if trk.state == TrackState.Tracked:
                STrack_update(trk, det_tlwh_tmp, det_trk.score, self._frame_id)
                activated_starcks.push_back(trk)
            else:
                STrack_re_activate(trk, det_tlwh_tmp, det_trk.score, self._frame_id, 0, &self.track_id_counter)
                refind_stracks_v.push_back(trk)

        free(match_a2); free(match_b2)

        # Mark unmatched r_tracked as lost
        for i in range(n_unmatched_trk2):
            trk = r_tracked[unmatched_trk2[i]]
            if trk.state != TrackState.Lost:
                STrack_mark_lost(trk)
                lost_stracks_v.push_back(trk)

        free(unmatched_trk2); free(unmatched_det2)

        # ---- Deal with unconfirmed tracks ----
        # Build remaining detections from first round's unmatched
        cdef vector[STrack*] remaining_dets
        for i in range(n_unmatched_det):
            remaining_dets.push_back(detections[unmatched_det[i]])

        free(unmatched_det); unmatched_det = NULL

        cdef int n_unconf = <int>unconfirmed.size()
        cdef int n_remaining = <int>remaining_dets.size()
        cdef int *match_a3 = NULL
        cdef int *match_b3 = NULL
        cdef int n_matches3 = 0
        cdef int *u_unconf = NULL
        cdef int n_u_unconf = 0
        cdef int *u_det3 = NULL
        cdef int n_u_det3 = 0

        if n_unconf > 0 and n_remaining > 0:
            # Extract tlbrs for unconfirmed association
            pool_tlbrs = <double*>malloc(n_unconf * 4 * sizeof(double))
            det_tlbrs = <double*>malloc(n_remaining * 4 * sizeof(double))
            for i in range(n_unconf):
                STrack_get_tlbr(unconfirmed[i], &pool_tlbrs[i * 4])
            for i in range(n_remaining):
                STrack_get_tlbr(remaining_dets[i], &det_tlbrs[i * 4])

            # Compute cost, fuse scores, solve assignment
            cost_mat = <double*>malloc(n_unconf * n_remaining * sizeof(double))
            compute_iou_cost(pool_tlbrs, n_unconf, det_tlbrs, n_remaining, cost_mat)

            if self._mot20 == 0:
                det_sc_tmp = <double*>malloc(n_remaining * sizeof(double))
                for i in range(n_remaining):
                    det_sc_tmp[i] = remaining_dets[i].score
                fuse_score(cost_mat, det_sc_tmp, n_unconf, n_remaining)
                free(det_sc_tmp)

            min_dim = n_unconf if n_unconf < n_remaining else n_remaining
            match_a3 = <int*>malloc((min_dim + 1) * sizeof(int))
            match_b3 = <int*>malloc((min_dim + 1) * sizeof(int))
            u_unconf = <int*>malloc((n_unconf + 1) * sizeof(int))
            u_det3 = <int*>malloc((n_remaining + 1) * sizeof(int))
            linear_assignment(cost_mat, n_unconf, n_remaining, 0.7,
                              match_a3, match_b3, &n_matches3,
                              u_unconf, &n_u_unconf,
                              u_det3, &n_u_det3)

            free(cost_mat); free(pool_tlbrs); free(det_tlbrs)
        else:
            n_matches3 = 0
            u_unconf = <int*>malloc((n_unconf + 1) * sizeof(int))
            for i in range(n_unconf):
                u_unconf[i] = i
            n_u_unconf = n_unconf
            u_det3 = <int*>malloc((n_remaining + 1) * sizeof(int))
            for i in range(n_remaining):
                u_det3[i] = i
            n_u_det3 = n_remaining

        # Process unconfirmed matches
        for i in range(n_matches3):
            trk = unconfirmed[match_a3[i]]
            det_trk = remaining_dets[match_b3[i]]
            STrack_get_tlwh(det_trk, det_tlwh_tmp)
            STrack_update(trk, det_tlwh_tmp, det_trk.score, self._frame_id)
            activated_starcks.push_back(trk)

        free(match_a3); free(match_b3)

        # Mark unmatched unconfirmed as removed
        for i in range(n_u_unconf):
            trk = unconfirmed[u_unconf[i]]
            STrack_mark_removed(trk)
            removed_stracks_v.push_back(trk)

        free(u_unconf)

        # ---- Initialize new tracks ----
        for i in range(n_u_det3):
            det_trk = remaining_dets[u_det3[i]]
            if det_trk.score < self._det_thresh:
                continue
            # Activate detection as new track
            STrack_activate(det_trk, self._frame_id, &self.track_id_counter)
            activated_starcks.push_back(det_trk)

        free(u_det3)

        # ---- Remove old lost tracks ----
        for i in range(<int>self._lost_stracks.size()):
            trk = self._lost_stracks[i]
            if self._frame_id - trk.frame_id > self._max_time_lost:
                STrack_mark_removed(trk)
                removed_stracks_v.push_back(trk)

        # ---- Update track lists ----
        # tracked = [t for t in tracked_stracks if t.state == Tracked]
        cdef vector[STrack*] new_tracked
        for i in range(<int>self._tracked_stracks.size()):
            if self._tracked_stracks[i].state == TrackState.Tracked:
                new_tracked.push_back(self._tracked_stracks[i])

        # tracked = joint(tracked, activated)
        cdef vector[STrack*] tmp1
        joint_stracks(&new_tracked, &activated_starcks, &tmp1)

        # tracked = joint(tracked, refind)
        cdef vector[STrack*] tmp2
        joint_stracks(&tmp1, &refind_stracks_v, &tmp2)
        self._tracked_stracks = tmp2

        # lost = sub(lost, tracked)
        cdef vector[STrack*] tmp3
        sub_stracks(&self._lost_stracks, &self._tracked_stracks, &tmp3)

        # lost.extend(lost_local)
        for i in range(<int>lost_stracks_v.size()):
            tmp3.push_back(lost_stracks_v[i])

        # removed.extend(removed_local) — extend BEFORE sub to prevent double-ownership
        for i in range(<int>removed_stracks_v.size()):
            self._removed_stracks.push_back(removed_stracks_v[i])

        # lost = sub(lost, removed) — now includes newly removed tracks
        cdef vector[STrack*] tmp4
        sub_stracks(&tmp3, &self._removed_stracks, &tmp4)
        self._lost_stracks = tmp4

        # Deduplicate between tracked and lost
        cdef vector[STrack*] out_tracked, out_lost
        remove_duplicate_stracks(&self._tracked_stracks, &self._lost_stracks, &out_tracked, &out_lost)
        self._tracked_stracks = out_tracked
        self._lost_stracks = out_lost

        # ---- Build output ----
        cdef int n_output = 0
        cdef double out_tlbr[4]
        for i in range(<int>self._tracked_stracks.size()):
            trk = self._tracked_stracks[i]
            if trk.is_activated != 0:
                STrack_get_tlbr(trk, out_tlbr)
                output[n_output * 5 + 0] = out_tlbr[0]
                output[n_output * 5 + 1] = out_tlbr[1]
                output[n_output * 5 + 2] = out_tlbr[2]
                output[n_output * 5 + 3] = out_tlbr[3]
                output[n_output * 5 + 4] = <double>trk.track_id
                n_output += 1

        # ---- Free temporary detection STracks ----
        # Activated detections have state==Tracked; keep them (now in _tracked_stracks)
        # All others have state==New; free them
        for i in range(<int>detections.size()):
            if detections[i].state == TrackState.New:
                free(detections[i])
        # Low-score detections are never activated; always free
        for i in range(<int>detections_second.size()):
            free(detections_second[i])

        # Free filtered bbox/score arrays
        free(high_bb)
        free(high_sc)
        free(low_bb)
        free(low_sc)

        return n_output

    @property
    def tracked_stracks(self):
        """Return tracked tracks as a list of STrackView objects (backward compat)."""
        return [_make_view(self._tracked_stracks[i]) for i in range(<int>self._tracked_stracks.size())]

    @property
    def lost_stracks(self):
        """Return lost tracks as a list of STrackView objects (backward compat)."""
        return [_make_view(self._lost_stracks[i]) for i in range(<int>self._lost_stracks.size())]

    @property
    def removed_stracks(self):
        """Return removed tracks as a list of STrackView objects (backward compat)."""
        return [_make_view(self._removed_stracks[i]) for i in range(<int>self._removed_stracks.size())]

    @staticmethod
    def _tlbr_to_tlwh(tlbr_in):
        """Convert tlbr to tlwh (backward compat for debug tests)."""
        cdef double tlwh_out[4]
        cdef double tlbr_arr[4]
        tlbr_arr[0] = tlbr_in[0]
        tlbr_arr[1] = tlbr_in[1]
        tlbr_arr[2] = tlbr_in[2]
        tlbr_arr[3] = tlbr_in[3]
        tlbr_to_tlwh(tlbr_arr, tlwh_out)
        return np.array([tlwh_out[0], tlwh_out[1], tlwh_out[2], tlwh_out[3]])

    def update(self, output_results, img_info, img_size):
        """
        Update tracker with new detections (Python-callable).

        Args:
            output_results: Detection results (Nx5 or Nx6 array)
            img_info: Image info (height, width)
            img_size: Image size for rescaling

        Returns:
            Array of tracked objects [[x1,y1,x2,y2,track_id], ...]
        """
        # Parse detections from numpy input
        if output_results.shape[1] == 5:
            scores_np = output_results[:, 4]
            bboxes_np = output_results[:, :4]
        else:
            if hasattr(output_results, 'cpu'):
                output_results = output_results.cpu().numpy()
            scores_np = output_results[:, 4] * output_results[:, 5]
            bboxes_np = output_results[:, :4]

        # Apply scale to bounding boxes
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes_np = bboxes_np / scale

        # Ensure contiguous float64 arrays
        cdef cnp.ndarray[cnp.float64_t, ndim=2] bboxes_c = np.ascontiguousarray(bboxes_np, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] scores_c = np.ascontiguousarray(scores_np, dtype=np.float64)
        cdef int n_dets = bboxes_c.shape[0]

        # Allocate output buffer (generous upper bound)
        cdef int max_output = n_dets + <int>self._tracked_stracks.size() + <int>self._lost_stracks.size() + 100
        cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.empty((max_output, 5), dtype=np.float64)

        # Call C-level update
        cdef int n_output = self._update_c(<double*>bboxes_c.data, <double*>scores_c.data, n_dets, <double*>result.data)

        # Return only the filled portion
        if n_output > 0:
            return result[:n_output]
        return np.empty((0, 5))


# ============================================================
# Non-owning STrack view (for Python access to internal tracks)
# ============================================================

cdef class STrackView:
    """Non-owning view of a C-level STrack. Does NOT free on dealloc."""
    cdef STrack *_ptr

    def predict(self):
        """Run Kalman filter prediction."""
        STrack_predict(self._ptr)

    def update(self, new_track, frame_id):
        """Update matched track with detection data."""
        cdef double tlwh[4]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tlwh_arr = new_track.tlwh
        tlwh[0] = tlwh_arr[0]
        tlwh[1] = tlwh_arr[1]
        tlwh[2] = tlwh_arr[2]
        tlwh[3] = tlwh_arr[3]
        STrack_update(self._ptr, tlwh, new_track.score, frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track."""
        cdef double tlwh[4]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tlwh_arr = new_track.tlwh
        tlwh[0] = tlwh_arr[0]
        tlwh[1] = tlwh_arr[1]
        tlwh[2] = tlwh_arr[2]
        tlwh[3] = tlwh_arr[3]
        global _strack_count
        cdef int counter = _strack_count
        STrack_re_activate(self._ptr, tlwh, new_track.score, frame_id, 1 if new_id else 0, &counter)
        _strack_count = counter

    def mark_lost(self):
        """Mark track as lost."""
        STrack_mark_lost(self._ptr)

    def mark_removed(self):
        """Mark track as removed."""
        STrack_mark_removed(self._ptr)

    @property
    def tlwh(self):
        cdef double tlwh[4]
        STrack_get_tlwh(self._ptr, tlwh)
        return np.array([tlwh[0], tlwh[1], tlwh[2], tlwh[3]])

    @property
    def tlbr(self):
        cdef double tlbr[4]
        STrack_get_tlbr(self._ptr, tlbr)
        return np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3]])

    @property
    def score(self):
        return self._ptr.score

    @property
    def track_id(self):
        return self._ptr.track_id

    @property
    def state(self):
        return self._ptr.state

    @property
    def is_activated(self):
        return self._ptr.is_activated != 0

    @property
    def frame_id(self):
        return self._ptr.frame_id

    @property
    def start_frame(self):
        return self._ptr.start_frame

    @property
    def mean(self):
        return np.array([self._ptr.mean[i] for i in range(8)])

    @property
    def covariance(self):
        cov = np.zeros((8, 8), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                cov[i, j] = self._ptr.covariance[i * 8 + j]
        return cov


cdef STrackView _make_view(STrack *ptr):
    """Create a non-owning STrackView around a C STrack pointer."""
    cdef STrackView v = STrackView.__new__(STrackView)
    v._ptr = ptr
    return v
