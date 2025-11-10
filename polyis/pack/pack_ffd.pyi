# Type stubs for pack_ffd.pyx
import numpy as np
import numpy.typing as npt

class PyPolyominoPosition:
    """Represents the position and orientation of a polyomino in a collage."""

    oy: int
    ox: int
    py: int
    px: int
    rotation: int
    frame: int
    shape: npt.NDArray[np.uint8]

    def __init__(
        self,
        oy: int,
        ox: int,
        py: int,
        px: int,
        rotation: int,
        frame: int,
        shape: npt.NDArray[np.uint8]
    ) -> None: ...

    def __repr__(self) -> str: ...


def pack_all(
    polyominoes_stacks: np.ndarray,
    h: int,
    w: int,
    mode: int,
) -> list[list[PyPolyominoPosition]]:
    """Packs all polyominoes from multiple stacks into collages using the C FFD algorithm.

    This function takes multiple stacks of polyominoes (memory addresses) and attempts to
    pack them into rectangular collages of the specified dimensions. It uses a first-fit
    decreasing algorithm, trying to place the largest polyominoes first in collages with
    the most empty space.

    Args:
        polyominoes_stacks: np.ndarray, a list of memory addresses pointing
                            to a PolyominoArray from C/Cython code. Each stack of polyominoes
                            corresponds to a video frame.
        h: Height of each collage in pixels
        w: Width of each collage in pixels
        mode: Packing mode to use
    Returns:
        A list of lists, where each inner list represents a collage containing
        PolyominoPosition objects representing all polyominoes packed into that collage
    """
    ...
