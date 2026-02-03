"""
ByteTrack tracker wrapper.

This wrapper adapts the ByteTrack implementation to match the interface
used by SORT and OC-SORT trackers in this project.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from polyis.tracker.bytetrack.cython.bytetrack import BYTETracker as _BYTETracker


class ByteTrackArgs:
    """
    Simple args object for ByteTrack configuration.
    This mimics the args object expected by BYTETracker.
    """
    def __init__(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.9,
        track_buffer: int = 30,
        mot20: bool = False
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20


class ByteTrack:
    """
    ByteTrack tracker wrapper that matches the SORT interface.

    This wrapper adapts ByteTrack's interface to work with the existing
    tracking pipeline. It accepts detections in the format
    [[x1, y1, x2, y2, score], ...] and returns tracked detections in the
    format [[x1, y1, x2, y2, id], ...].
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ):
        """
        Initialize ByteTrack tracker.

        Args:
            img_size: Image size as (height, width)
            track_thresh: Detection confidence threshold for tracking
            match_thresh: IOU threshold for matching detections to tracks
            track_buffer: Number of frames to keep lost tracks before removal
            frame_rate: Video frame rate (used to compute buffer size)
            mot20: Whether to use MOT20-specific settings
        """
        args = ByteTrackArgs(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            mot20=False,
        )
        self.tracker = _BYTETracker(args, frame_rate=frame_rate)
        self.frame_count = 0
        self.img_size = img_size

    def update(self, dets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # Handle empty detections
        if dets is None or dets.size == 0:
            return np.empty((0, 5), dtype=np.float64)

        img_info = self.img_size

        # Update ByteTrack tracker
        online_targets = self.tracker.update(dets, img_info, img_info)

        # Convert STrack objects to output format [[x1, y1, x2, y2, id], ...]
        if len(online_targets) == 0:
            return np.empty((0, 5), dtype=np.float64)

        results = []
        for track in online_targets:
            # Get bounding box in tlbr format (x1, y1, x2, y2)
            bbox = track[:4]
            track_id = track[4]
            results.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])

        return np.array(results, dtype=np.float64)
