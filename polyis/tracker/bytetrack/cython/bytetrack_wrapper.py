"""
ByteTrack tracker wrapper.

This wrapper adapts the ByteTrack implementation to match the interface
used by SORT and OC-SORT trackers in this project. The track ID counter
is stored on the wrapper (Python) side.
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

    The track ID counter is managed here on the Python side and passed
    to the Cython BYTETracker via the public track_id_counter attribute.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 15,
        frame_rate: int = 15,
    ):
        # Build args object for the Cython tracker
        args = ByteTrackArgs(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            mot20=False,
        )
        # Create Cython tracker (track_id_counter starts at 0)
        self.tracker = _BYTETracker(args, frame_rate=frame_rate)
        self.frame_count = 0
        self.img_size = img_size

    @property
    def track_id_counter(self) -> int:
        """Read the track ID counter from the Cython tracker."""
        return self.tracker.track_id_counter

    @track_id_counter.setter
    def track_id_counter(self, value: int) -> None:
        """Set the track ID counter on the Cython tracker."""
        self.tracker.track_id_counter = value

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

        # Pass image info for scale computation (scale = 1.0 since img_info == img_size)
        img_info = self.img_size

        # Update Cython tracker (returns Nx5 numpy array)
        online_targets = self.tracker.update(dets, img_info, img_info)

        # Return directly since update() now returns a numpy array
        if len(online_targets) == 0:
            return np.empty((0, 5), dtype=np.float64)

        return np.asarray(online_targets, dtype=np.float64)
