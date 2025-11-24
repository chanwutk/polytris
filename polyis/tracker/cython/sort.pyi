import numpy as np
import numpy.typing as npt


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        ...

    def update(self, dets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...