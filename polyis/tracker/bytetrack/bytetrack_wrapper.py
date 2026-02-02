"""
ByteTrack tracker wrapper.

This wrapper adapts the ByteTrack implementation to match the interface
used by SORT and OC-SORT trackers in this project.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from polyis.tracker.bytetrack.byte_tracker import BYTETracker as _BYTETracker  # type: ignore


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
        mot20: bool = False
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
            mot20=mot20
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
        
        # Ensure dets is in correct format (N, 5) with [x1, y1, x2, y2, score]
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        
        # Extract bboxes and scores
        if dets.shape[1] >= 5:
            bboxes = dets[:, :4]  # x1, y1, x2, y2
            scores = dets[:, 4]
        else:
            # If only 4 columns, assume no score (shouldn't happen but handle gracefully)
            bboxes = dets[:, :4]
            scores = np.ones(len(dets))
        
        # Create output_results in format expected by ByteTrack
        # ByteTrack expects [x1, y1, x2, y2, score] or [x1, y1, x2, y2, score, class]
        output_results = np.column_stack([bboxes, scores])
        
        img_info = self.img_size
        
        # Update ByteTrack tracker
        online_targets = self.tracker.update(output_results, img_info, img_info)
        
        # Convert STrack objects to output format [[x1, y1, x2, y2, id], ...]
        if len(online_targets) == 0:
            return np.empty((0, 5), dtype=np.float64)
        
        results = []
        for track in online_targets:
            # Get bounding box in tlbr format (x1, y1, x2, y2)
            bbox = track.tlbr
            track_id = track.track_id
            results.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        
        if len(results) > 0:
            return np.array(results, dtype=np.float64)
        else:
            return np.empty((0, 5), dtype=np.float64)
