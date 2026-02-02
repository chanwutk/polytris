import numpy as np
import numpy.typing as npt


class OCSort:
    def __init__(
        self,
        det_thresh: float,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False
    ):
        ...

    def update(
        self,
        output_results: npt.NDArray[np.floating] | None,
        img_info: tuple[int, int],
        img_size: tuple[int, int]
    ) -> npt.NDArray[np.floating]:
        ...

