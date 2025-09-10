from queue import Queue
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from b3d.external.sort import Sort, KalmanBoxTracker, iou_batch, associate_detections_to_trackers
from polyis.dtypes import S5, DetArray, IntDetArray, NPImage, Array, IdPolyominoOffset, Pipe, InPipe, OutPipe, is_det_array


class TrackPointAccuracyScore(NamedTuple):
    det: "npt.NDArray"
    score: "float"


class SortWithBenchmark(Sort):
    def __init__(self, max_age=1, min_hits=0, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.trackers: "list[KalmanBoxTracker]" = []
        self.benchmark: "list[TrackPointAccuracyScore]" = []

    def update(self, dets: "npt.NDArray" = np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
        # matched is a list of tuples, where each tuple is a pair of indices of matched detections and trackers.
        # matched = [(det_idx, trk_idx), (det_idx, trk_idx), ...]
        iou_matrix = iou_batch(dets, trks)  # -- Added for benchmarking tracking results --
        assert iou_matrix.shape == (dets.shape[0], trks.shape[0]), f"iou_matrix shape mismatch: {iou_matrix.shape, (dets.shape, trks.shape)}"  # -- Added for benchmarking tracking results --
        self.benchmark = [
            TrackPointAccuracyScore(det, iou) for det, iou
            in zip(dets[matched[:, 0]], iou_matrix[*matched.T])
        ]  # -- Added for benchmarking tracking results --

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            self.benchmark.append(TrackPointAccuracyScore(dets[i], 0))  # -- Added for benchmarking tracking results --
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # ret.append(np.concatenate((d,[trk.id+1.5])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d,[trk.id+1.1])).reshape(1,-1)) # +1 as MOT benchmark requires positive  # +.1 to avoid x.9999 when converted to float
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            ret = np.concatenate(ret)
            assert is_det_array(ret), ret.shape
            return ret
            # return np.concatenate(ret)
        return np.empty((0,5))


import time


def track(
    bboxQueue: InPipe[tuple[int, DetArray]],
    trackQueue: OutPipe[tuple[int, DetArray]],
    benchmarkQueue: "OutPipe | None" = None,
    max_age: int = 5
):
    tracker = SortWithBenchmark(max_age=max_age)
    flog = open('tracker.py.log', 'w')
    prev_idx = -1
    while True:
        detections = bboxQueue.get()
        if detections is None:
            break
        idx, detections = detections
        assert prev_idx + 1 == idx, (prev_idx, idx)
        prev_idx = idx
        
        # print(f"Tracking {detections.shape}...")
        flog.write(f"Tracking {detections.shape}...\n")
        flog.flush()
        # print('----------', idx, len(detections))

        # if len(detections) != 0:
        tracked_objects = tracker.update(detections)
        # print('--------t--', idx, len(tracked_objects))
        #     # print("-------------------------", idx, len(tracked_objects))
        # else:
        #     tracked_objects = np.empty((0, 5))
        track_benchmark = tracker.benchmark.copy()

        trackQueue.put((idx, tracked_objects))
        if benchmarkQueue is not None:
            benchmarkQueue.put(track_benchmark)

    trackQueue.put(None)
    if benchmarkQueue is not None:
        benchmarkQueue.put(None)

    flog.write("Tracker finished.\n")
    flog.close()