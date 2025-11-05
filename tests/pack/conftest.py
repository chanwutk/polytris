"""
Pytest configuration for pack_append tests.
"""

import pytest
import numpy as np
import sys
import os

# No need to modify sys.path - polyis package is already accessible


@pytest.fixture
def sample_polyominoes():
    """Sample polyominoes for testing."""
    # 2x2 square
    square = np.array([[True, True], [True, True]], dtype=np.uint8)
    # 1x2 rectangle
    rect_h = np.array([[True, True]], dtype=np.uint8)
    # 2x1 rectangle
    rect_v = np.array([[True], [True]], dtype=np.uint8)
    
    return {
        'square': (square, (0, 0)),
        'rect_h': (rect_h, (0, 0)),
        'rect_v': (rect_v, (0, 0))
    }


@pytest.fixture
def empty_grid():
    """Empty 4x4 grid for testing."""
    return np.zeros((4, 4), dtype=np.uint8)


@pytest.fixture
def small_grid():
    """Empty 3x3 grid for testing."""
    return np.zeros((3, 3), dtype=np.uint8)
