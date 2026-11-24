"""
OC-SORT tracker wrapper that matches the SORT interface.

This wrapper adapts OC-SORT's interface to work with the existing
tracking pipeline. The track ID counter is stored on the wrapper side.
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

    The track ID counter is managed here on the Python side via
    the Cython tracker's public track_id_counter attribute.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False
    ):
        # Create Cython tracker
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
        self.img_size = img_size
        self.frame_count = 0

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

        # Ensure dets is in correct format
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        # Pass image info for scale computation (scale = 1.0)
        img_info = self.img_size

        # Update OC-SORT tracker (returns Nx5 numpy array)
        tracked_dets = self.tracker.update(dets, img_info, self.img_size)

        return tracked_dets
