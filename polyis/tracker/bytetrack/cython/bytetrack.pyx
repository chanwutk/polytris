# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of ByteTrack tracker.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, isnan
from libc.stdlib cimport malloc, calloc, free
from libcpp.vector cimport vector

from polyis.tracker.bytetrack.cython.kalman_filter cimport KalmanFilter, kf_init, kf_initiate, kf_predict, kf_update
from polyis.tracker.bytetrack.cython.matching import iou_distance, fuse_score, linear_assignment

# Set random seed for reproducibility
np.random.seed(0)

# Module-level counter for STrack IDs
_strack_count = 0

def reset_tracker_count():
    """Reset the tracker counter. Used for testing."""
    global _strack_count
    _strack_count = 0


# Track state enum
cdef enum TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlwh_to_xyah(double *tlwh, double *xyah) noexcept nogil:
    """
    Convert bounding box from [x, y, w, h] to [x, y, a, h] format.
    where a is aspect ratio (w/h).
    """
    xyah[0] = tlwh[0] + tlwh[2] / 2.0
    xyah[1] = tlwh[1] + tlwh[3] / 2.0
    xyah[2] = tlwh[2] / tlwh[3]
    xyah[3] = tlwh[3]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlwh_to_tlbr(double *tlwh, double *tlbr) noexcept nogil:
    """
    Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2] format.
    """
    tlbr[0] = tlwh[0]
    tlbr[1] = tlwh[1]
    tlbr[2] = tlwh[0] + tlwh[2]
    tlbr[3] = tlwh[1] + tlwh[3]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void tlbr_to_tlwh(double *tlbr, double *tlwh) noexcept nogil:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format.
    """
    tlwh[0] = tlbr[0]
    tlwh[1] = tlbr[1]
    tlwh[2] = tlbr[2] - tlbr[0]
    tlwh[3] = tlbr[3] - tlbr[1]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void xyah_to_tlwh(double *xyah, double *tlwh) noexcept nogil:
    """
    Convert bounding box from [x, y, a, h] to [x, y, w, h] format.
    """
    cdef double w = xyah[2] * xyah[3]
    tlwh[0] = xyah[0] - w / 2.0
    tlwh[1] = xyah[1] - xyah[3] / 2.0
    tlwh[2] = w
    tlwh[3] = xyah[3]


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
    """
    Initialize STrack with bounding box and score.
    """
    cdef int i, j

    # Initialize Kalman filter
    kf_init(&self.kf)

    # Store initial bounding box
    self._tlwh[0] = tlwh[0]
    self._tlwh[1] = tlwh[1]
    self._tlwh[2] = tlwh[2]
    self._tlwh[3] = tlwh[3]

    # Initialize state
    for i in range(8):
        self.mean[i] = 0.0
    for i in range(64):
        self.covariance[i] = 0.0

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
    """
    Predict next state using Kalman filter.
    """
    cdef int i, j

    # Copy mean and covariance to Kalman filter
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Set velocity to 0 if not tracked
    if self.state != TrackState.Tracked:
        self.kf.x[7] = 0.0

    # Predict
    kf_predict(&self.kf)

    # Copy back
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int STrack_activate(STrack *self, int frame_id) noexcept nogil:
    """
    Activate a new tracklet.
    Returns the assigned track ID.
    """
    cdef double xyah[4]
    cdef int i, j
    cdef int new_id

    # Convert tlwh to xyah
    tlwh_to_xyah(self._tlwh, xyah)

    # Initialize Kalman filter with measurement
    kf_initiate(&self.kf, xyah, self.mean, self.covariance)

    # Assign track ID
    with gil:
        global _strack_count
        _strack_count += 1
        self.track_id = _strack_count
        new_id = _strack_count

    self.tracklet_len = 0
    self.state = TrackState.Tracked

    if frame_id == 1:
        self.is_activated = 1

    self.frame_id = frame_id
    self.start_frame = frame_id

    return new_id


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int STrack_re_activate(STrack *self, double *new_tlwh, double new_score, int frame_id, int new_id) noexcept nogil:
    """
    Re-activate a lost track with new detection.
    Returns the new track ID if new_id is True, otherwise returns the current track ID.
    """
    cdef double xyah[4]
    cdef int i, j
    cdef int result_id

    # Convert tlwh to xyah
    tlwh_to_xyah(new_tlwh, xyah)

    # Copy mean and covariance to Kalman filter
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Update with new measurement
    kf_update(&self.kf, xyah)

    # Copy back
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]

    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = 1
    self.frame_id = frame_id

    if new_id:
        with gil:
            global _strack_count
            _strack_count += 1
            self.track_id = _strack_count
            result_id = _strack_count
    else:
        result_id = self.track_id

    self.score = new_score
    return result_id


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void STrack_update(STrack *self, double *new_tlwh, double new_score, int frame_id) noexcept nogil:
    """
    Update a matched track with new detection.
    """
    cdef double xyah[4]
    cdef int i, j

    self.frame_id = frame_id
    self.tracklet_len += 1

    # Convert tlwh to xyah
    tlwh_to_xyah(new_tlwh, xyah)

    # Copy mean and covariance to Kalman filter
    for i in range(8):
        self.kf.x[i] = self.mean[i]
    for i in range(8):
        for j in range(8):
            self.kf.P[i][j] = self.covariance[i * 8 + j]

    # Update with new measurement
    kf_update(&self.kf, xyah)

    # Copy back
    for i in range(8):
        self.mean[i] = self.kf.x[i]
    for i in range(8):
        for j in range(8):
            self.covariance[i * 8 + j] = self.kf.P[i][j]

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
    """
    Get current position in tlwh format.
    """
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
    """
    Get current position in tlbr format.
    """
    cdef double tlwh[4]
    STrack_get_tlwh(self, tlwh)
    tlwh_to_tlbr(tlwh, tlbr)


# Python wrapper class for STrack
cdef class STrackPy:
    cdef STrack *track

    def __cinit__(self, tlwh, score):
        self.track = <STrack*>malloc(sizeof(STrack))
        cdef double tlwh_arr[4]
        tlwh_arr[0] = tlwh[0]
        tlwh_arr[1] = tlwh[1]
        tlwh_arr[2] = tlwh[2]
        tlwh_arr[3] = tlwh[3]
        # Don't assign ID yet - it will be assigned on activation
        STrack_init(self.track, tlwh_arr, score, -1)

    def __dealloc__(self):
        if self.track is not NULL:
            free(self.track)

    def predict(self):
        STrack_predict(self.track)

    def activate(self, frame_id):
        cdef int new_id = STrack_activate(self.track, frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        cdef double tlwh[4]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tlwh_arr = new_track.tlwh
        tlwh[0] = tlwh_arr[0]
        tlwh[1] = tlwh_arr[1]
        tlwh[2] = tlwh_arr[2]
        tlwh[3] = tlwh_arr[3]
        cdef int new_track_id = STrack_re_activate(self.track, tlwh, new_track.score, frame_id, 1 if new_id else 0)

    def update(self, new_track, frame_id):
        cdef double tlwh[4]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tlwh_arr = new_track.tlwh
        tlwh[0] = tlwh_arr[0]
        tlwh[1] = tlwh_arr[1]
        tlwh[2] = tlwh_arr[2]
        tlwh[3] = tlwh_arr[3]
        STrack_update(self.track, tlwh, new_track.score, frame_id)

    def mark_lost(self):
        STrack_mark_lost(self.track)

    def mark_removed(self):
        STrack_mark_removed(self.track)

    @property
    def tlwh(self):
        cdef double tlwh[4]
        STrack_get_tlwh(self.track, tlwh)
        return np.array([tlwh[0], tlwh[1], tlwh[2], tlwh[3]])

    @property
    def tlbr(self):
        cdef double tlbr[4]
        STrack_get_tlbr(self.track, tlbr)
        return np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3]])

    @property
    def score(self):
        return self.track.score

    @property
    def track_id(self):
        return self.track.track_id

    @property
    def state(self):
        return self.track.state

    @property
    def is_activated(self):
        return self.track.is_activated != 0

    @property
    def frame_id(self):
        return self.track.frame_id

    @property
    def start_frame(self):
        return self.track.start_frame

    @property
    def mean(self):
        return np.array([self.track.mean[i] for i in range(8)])

    @property
    def covariance(self):
        cov = np.zeros((8, 8), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                cov[i, j] = self.track.covariance[i * 8 + j]
        return cov


def joint_stracks(tlista, tlistb):
    """Combine two track lists, removing duplicates."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Remove tracks from tlista that appear in tlistb."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks based on IOU."""
    if len(stracksa) == 0 or len(stracksb) == 0:
        return stracksa, stracksb

    # Compute pairwise IOU
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)

    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


# ByteTrack main class
cdef public class BYTETracker [object BYTETrackerObject, type BYTETrackerType]:
    """
    ByteTrack tracker implementation in Cython.
    """
    cdef public object tracked_stracks
    cdef public object lost_stracks
    cdef public object removed_stracks
    cdef int frame_id
    cdef object args
    cdef double det_thresh
    cdef int buffer_size
    cdef int max_time_lost

    def __init__(self, args, frame_rate=30):
        """
        Initialize ByteTrack tracker.

        Args:
            args: Arguments object with track_thresh, track_buffer, match_thresh, mot20
            frame_rate: Frame rate for buffer size calculation
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        # Reset global counter
        global _strack_count
        _strack_count = 0

    def update(self, output_results, img_info, img_size):
        """
        Update tracker with new detections.

        Args:
            output_results: Detection results (Nx5 or Nx6 array)
            img_info: Image info (height, width)
            img_size: Image size for rescaling

        Returns:
            Array of tracked objects [[x1,y1,x2,y2,track_id], ...]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Parse detections
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

        # Filter detections by threshold
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # Create detections
        if len(dets) > 0:
            detections = [STrackPy(self._tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # Separate tracked and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # First association: with high score detections
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict current location
        for strack in strack_pool:
            strack.predict()

        dists = iou_distance(strack_pool, detections)

        if not self.args.mot20:
            dists = fuse_score(dists, detections)

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Second association: with low score detections
        if len(dets_second) > 0:
            detections_second = [STrackPy(self._tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark unmatched tracks as lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)

        if not self.args.mot20:
            dists = fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Initialize new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        # Remove old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # Get output tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # Format output
        ret = []
        for track in output_stracks:
            tlbr = track.tlbr
            ret.append(np.concatenate((tlbr, [track.track_id])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    @staticmethod
    def _tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh."""
        cdef double tlwh[4]
        cdef double tlbr_arr[4]
        tlbr_arr[0] = tlbr[0]
        tlbr_arr[1] = tlbr[1]
        tlbr_arr[2] = tlbr[2]
        tlbr_arr[3] = tlbr[3]
        tlbr_to_tlwh(tlbr_arr, tlwh)
        return np.array([tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
