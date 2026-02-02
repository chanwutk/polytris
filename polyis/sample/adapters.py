"""
Efficient adapters for polyomino pruning that avoid unnecessary conversions.

This module provides direct access to polyomino tile coordinates without
converting between mask representations.
"""

import numpy as np
from typing import List, Tuple
from polyis.pack.group_tiles import group_tiles, free_polyomino_array


def get_polyomino_tiles(bitmap: np.ndarray, tilepadding_mode: int = 0) -> List[List[Tuple[int, int]]]:
    """
    Extract polyominoes from a bitmap and return as lists of tile coordinates.

    This function avoids the mask conversion overhead by directly extracting
    tile coordinates from the C structures.

    Parameters:
        bitmap: 2D numpy array of uint8 (1 = relevant, 0 = not relevant)
        tilepadding_mode: Padding mode (0 = no padding)

    Returns:
        List of polyominoes, where each polyomino is a list of (row, col) tuples
    """
    # Get the polyomino array pointer
    poly_ptr = group_tiles(bitmap, mode=tilepadding_mode)

    # Import the Cython function to extract tiles directly
    from polyis.sample.cython.tile_extractor import extract_tiles_from_polyominoes

    try:
        # Extract tiles directly without creating masks
        polyominoes = extract_tiles_from_polyominoes(poly_ptr)
    finally:
        # Always free the C memory
        free_polyomino_array(poly_ptr)

    return polyominoes