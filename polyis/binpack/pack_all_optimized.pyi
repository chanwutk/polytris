"""Type stubs for pack_all_optimized module."""

import typing

import numpy as np


class PolyominoPosition:
    """Represents the position and orientation of a polyomino in a collage.
    
    Attributes:
        oy: Original y-coordinate offset of the polyomino from its video frame
        ox: Original x-coordinate offset of the polyomino from its video frame
        py: Packed y-coordinate position in the collage
        px: Packed x-coordinate position in the collage
        rotation: Rotation applied to the polyomino (0-3, representing 0°, 90°, 180°, 270°)
        frame: Frame index this polyomino belongs to
        shape: The actual polyomino shape as a numpy array
    """
    oy: int
    ox: int
    py: int
    px: int
    rotation: int
    frame: int
    shape: np.ndarray
    
    def __init__(
        self,
        oy: int,
        ox: int,
        py: int,
        px: int,
        rotation: int,
        frame: int,
        shape: np.ndarray
    ) -> None:
        ...


class Placement:
    """Represents a successful placement of a polyomino in a collage.
    
    Attributes:
        y: Y-coordinate where the polyomino was placed
        x: X-coordinate where the polyomino was placed
        rotation: Rotation applied to the polyomino (0-3)
    """
    y: int
    x: int
    rotation: int
    
    def __init__(self, y: int, x: int, rotation: int) -> None:
        ...


class CollageMetadata:
    """Holds a collage and cached metadata about its unoccupied regions.
    
    Attributes:
        occupied_tiles: 2D numpy array representing the collage (0=empty, 1=occupied)
        unoccupied_spaces: List of numpy array masks, each representing one unoccupied region
        space_sizes: List of integers, parallel to unoccupied_spaces, storing the size of each space
    """
    occupied_tiles: np.ndarray
    unoccupied_spaces: list[np.ndarray]
    space_sizes: list[int]
    
    def __init__(
        self,
        occupied_tiles: np.ndarray,
        unoccupied_spaces: list[np.ndarray],
        space_sizes: list[int]
    ) -> None:
        ...


def extract_unoccupied_spaces(
    occupied_tiles: np.ndarray
) -> tuple[list[np.ndarray], list[int]]:
    """Extract all unoccupied connected regions as numpy array masks and their sizes.
    
    Args:
        occupied_tiles: 2D numpy array representing the collage state
    
    Returns:
        Tuple of (list of numpy array masks, list of sizes)
        Each mask represents one unoccupied region (same shape as occupied_tiles, 1=unoccupied in this region)
        Each size is the number of tiles in the corresponding region
    """
    ...


def pack_all(
    polyominoes_stacks: list[int],
    h: int,
    w: int
) -> list[list[PolyominoPosition]]:
    """
    Packs all polyominoes from multiple stacks into collages using a bin packing algorithm.
    
    This function takes multiple stacks of polyominoes (memory addresses) and attempts to
    pack them into rectangular collages of the specified dimensions. It uses a greedy
    first-fit decreasing algorithm, trying to place the largest polyominoes first.
    
    Args:
        polyominoes_stacks: List of integers, each representing a stack of polyominoes
                            a stack of polyominoes is a memory address pointing to a
                            stack of polyominoes from Cython.
                            Each stack of polyominoes corresponds to a video frame.
        h: Height of each collage in pixels
        w: Width of each collage in pixels
    
    Returns:
        List of lists, where each inner list represents a collage, which contains
        PolyominoPosition objects representing all polyominoes packed into a single collage.
    """
    ...

