# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of matching functions for ByteTrack.
Includes IOU computation, fuse_score, and linear assignment.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fmax, fmin
from libc.stdlib cimport malloc, calloc, free

# External C function for linear assignment problem
cdef extern from "lapjv.h" nogil:
    ctypedef signed int int_t
    ctypedef unsigned int uint_t
    int lapjv_internal(const uint_t n, double *cost[], int_t *x, int_t *y)


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
    Uses PASCAL VOC formula with +1 to match the original Python implementation.
    """
    cdef int N = bboxes1.shape[0]
    cdef int M = bboxes2.shape[0]

    cdef int i, j
    cdef double xx1, yy1, xx2, yy2, w, h, wh, area1, area2

    for i in range(N):
        # Add +1 to match PASCAL VOC formula used in cython_bbox
        area1 = (bboxes1[i, 2] - bboxes1[i, 0] + 1.0) * (bboxes1[i, 3] - bboxes1[i, 1] + 1.0)
        for j in range(M):
            xx1 = fmax(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = fmax(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = fmin(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = fmin(bboxes1[i, 3], bboxes2[j, 3])

            # Add +1 to intersection dimensions
            w = fmax(0.0, xx2 - xx1 + 1.0)
            h = fmax(0.0, yy2 - yy1 + 1.0)
            wh = w * h

            # Add +1 to area2 calculation
            area2 = (bboxes2[j, 2] - bboxes2[j, 0] + 1.0) * (bboxes2[j, 3] - bboxes2[j, 1] + 1.0)

            output[i, j] = wh / (area1 + area2 - wh + 1e-9)


def iou_batch(bboxes1, bboxes2):
    """
    Compute IOU between two sets of bboxes in the form [x1,y1,x2,y2].

    Args:
        bboxes1: First set of bounding boxes (N, 4)
        bboxes2: Second set of bounding boxes (M, 4)

    Returns:
        IOU matrix (N, M)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b1 = np.asarray(bboxes1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b2 = np.asarray(bboxes2, dtype=np.float64)

    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float64)

    iou_batch_internal(b1, b2, result)
    return result


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU.
    Converts track objects or arrays to bounding boxes and computes IoU distance.

    Args:
        atracks: List of tracks or arrays
        btracks: List of tracks or arrays

    Returns:
        Cost matrix (1 - IOU)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] atlbrs
    cdef cnp.ndarray[cnp.float64_t, ndim=2] btlbrs

    # Handle both track objects and numpy arrays
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float64)

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = np.asarray(atracks, dtype=np.float64)
        btlbrs = np.asarray(btracks, dtype=np.float64)
    else:
        # Extract tlbr from track objects
        atlbrs = np.array([track.tlbr for track in atracks], dtype=np.float64)
        btlbrs = np.array([track.tlbr for track in btracks], dtype=np.float64)

    # Compute IOU
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ious = iou_batch(atlbrs, btlbrs)

    # Return distance (1 - IOU)
    return 1.0 - ious


def fuse_score(cost_matrix, detections):
    """
    Fuse detection scores with IOU similarity.

    Args:
        cost_matrix: Cost matrix from IOU distance
        detections: List of detection objects with score attribute

    Returns:
        Fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # Convert to similarity
    cdef cnp.ndarray[cnp.float64_t, ndim=2] iou_sim = 1.0 - cost_matrix

    # Extract detection scores
    cdef cnp.ndarray[cnp.float64_t, ndim=1] det_scores = np.array([det.score for det in detections], dtype=np.float64)

    # Expand scores to match cost matrix shape
    cdef cnp.ndarray[cnp.float64_t, ndim=2] det_scores_expanded = np.expand_dims(det_scores, axis=0)
    det_scores_expanded = np.repeat(det_scores_expanded, cost_matrix.shape[0], axis=0)

    # Fuse: similarity = iou_sim * score
    cdef cnp.ndarray[cnp.float64_t, ndim=2] fuse_sim = iou_sim * det_scores_expanded

    # Convert back to cost
    return 1.0 - fuse_sim


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int linear_assignment_internal(
    cnp.float64_t[:, :] cost_c,
    cnp.int32_t[:, :] matches
) noexcept nogil:
    """
    Solve linear assignment problem using lapjv.

    Args:
        cost_c: Cost matrix
        matches: Output array for matches [[det, trk], ...]

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

    # Convert to array of pairs
    cdef int x, y
    cdef int count = 0
    for i in range(n_rows):
        x = x_c[i]
        if x >= 0 and x < n_cols:
            matches[count, 0] = i
            matches[count, 1] = x
            count += 1
    free(x_c)
    free(y_c)
    return count


def linear_assignment(cost_matrix, thresh):
    """
    Solve linear assignment problem with threshold.

    Args:
        cost_matrix: Cost matrix for assignment
        thresh: Cost threshold for valid matches

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
        - matches: Array of matched pairs [[det_idx, trk_idx], ...]
        - unmatched_a: Array of unmatched detection indices
        - unmatched_b: Array of unmatched track indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )

    cdef cnp.ndarray[cnp.float64_t, ndim=2] cost = np.asarray(cost_matrix, dtype=np.float64)
    cdef int n_rows = cost.shape[0]
    cdef int n_cols = cost.shape[1]

    # Allocate space for matches
    cdef cnp.ndarray[cnp.int32_t, ndim=2] matches_array = np.empty((n_rows, 2), dtype=np.int32)

    # Solve assignment
    cdef int num_matches = linear_assignment_internal(cost, matches_array)

    # Extract matches and filter by threshold
    cdef list matches = []
    cdef set matched_a = set()
    cdef set matched_b = set()

    cdef int i, det_idx, trk_idx
    for i in range(num_matches):
        det_idx = matches_array[i, 0]
        trk_idx = matches_array[i, 1]
        if cost[det_idx, trk_idx] <= thresh:
            matches.append([det_idx, trk_idx])
            matched_a.add(det_idx)
            matched_b.add(trk_idx)

    # Find unmatched
    cdef list unmatched_a = [i for i in range(n_rows) if i not in matched_a]
    cdef list unmatched_b = [i for i in range(n_cols) if i not in matched_b]

    return np.asarray(matches, dtype=np.int32), np.array(unmatched_a, dtype=np.int32), np.array(unmatched_b, dtype=np.int32)
