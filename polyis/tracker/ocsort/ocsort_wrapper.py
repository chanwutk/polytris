"""
OC-SORT tracker wrapper that matches the SORT interface.

This wrapper adapts OC-SORT's interface to work with the existing
tracking pipeline. It accepts detections in the format
[[x1, y1, x2, y2, score], ...] and returns tracked detections in the
format [[x1, y1, x2, y2, id], ...].
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .ocsort import OCSort as _OCSort


class OCSort:
    """
    OC-SORT tracker wrapper that matches the SORT interface.
    
    This wrapper adapts OC-SORT's interface to work with the existing
    tracking pipeline. It accepts detections in the format
    [[x1, y1, x2, y2, score], ...] and returns tracked detections in the
    format [[x1, y1, x2, y2, id], ...].
    """
    
    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False
    ):
        """
        Initialize OC-SORT tracker.
        
        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum age for tracks
            min_hits: Minimum hits for tracks
            iou_threshold: IOU threshold for matching
            delta_t: Delta time for velocity estimation
            asso_func: Association function name
            inertia: Inertia parameter
            use_byte: Whether to use BYTE association
        """
        self.tracker = _OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte
        )
        self.frame_count = 0
        
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
        
        # OC-SORT requires img_info and img_size for scaling
        # Estimate from bboxes or use defaults
        if len(dets) > 0:
            max_h = int(np.max(dets[:, 3]) + 1)
            max_w = int(np.max(dets[:, 2]) + 1)
            img_info = (max_h, max_w)
            img_size = img_info
        else:
            # Default HD resolution when no detections
            img_info = (1080, 1920)
            img_size = img_info
        
        # Update OC-SORT tracker
        tracked_dets = self.tracker.update(dets, img_info, img_size)
        
        return tracked_dets
