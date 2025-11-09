"""
Test suite for convert_collages_to_bitmap adapter function.

This test verifies that the adapter correctly converts coordinate-based
shapes from pack_ffd.pyx into bitmap format.

Note: PyPolyominoPosition has attributes: oy, ox, py, px, frame, shape
where shape is a numpy array of (y, x) coordinates.
"""

import numpy as np
import pytest


def test_convert_collages_to_bitmap():
    """Test that coordinate-based shapes are correctly converted to bitmaps."""
    # Import the adapter function and PyPolyominoPosition
    from polyis.pack.adapters import convert_collages_to_bitmap
    from polyis.pack.pack_ffd import PyPolyominoPosition
    
    # Create test data: simple L-shape polyomino
    # Coordinates: [(0,0), (0,1), (1,0)]
    coords1 = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int16)
    pos1 = PyPolyominoPosition(oy=5, ox=10, py=20, px=30, frame=0, shape=coords1)  # type: ignore
    
    # Create test data: straight line polyomino
    # Coordinates: [(2,3), (2,4), (2,5)]
    coords2 = np.array([[2, 3], [2, 4], [2, 5]], dtype=np.int16)
    pos2 = PyPolyominoPosition(oy=8, ox=12, py=25, px=35, frame=1, shape=coords2)  # type: ignore
    
    # Test with one collage containing two polyominoes
    collages = [[pos1, pos2]]
    
    # Convert to bitmap
    convert_collages_to_bitmap(collages)
    
    # Verify structure
    assert len(collages) == 1, "Should have one collage"
    assert len(collages[0]) == 2, "Collage should have two polyominoes"
    
    # Verify first polyomino (L-shape)
    # Original coords: [(0,0), (0,1), (1,0)] -> bitmap 2x2
    pos1_result = collages[0][0]
    assert pos1_result.shape.shape == (2, 2), "L-shape bitmap should be 2x2"
    assert pos1_result.shape.dtype == np.uint8, "Shape should be converted to uint8 bitmap"
    assert pos1_result.py == 20 and pos1_result.px == 30, "Placement offsets should match py, px"
    assert pos1_result.oy == 5 and pos1_result.ox == 10, "Original offsets should match oy, ox"
    assert pos1_result.frame == 0, "Frame should be 0"
    # Check L-shape bitmap pattern
    expected_mask1 = np.array([
        [1, 1],
        [1, 0],
    ], dtype=np.uint8)
    assert np.array_equal(pos1_result.shape, expected_mask1), "L-shape bitmap should match"
    
    # Verify second polyomino (straight line)
    # Original coords: [(2,3), (2,4), (2,5)] -> bitmap 1x3 (same y, consecutive x)
    pos2_result = collages[0][1]
    assert pos2_result.shape.shape == (3, 6), "Line bitmap should be 1x3"
    assert pos2_result.shape.dtype == np.uint8, "Shape should be converted to uint8 bitmap"
    assert pos2_result.py == 25 and pos2_result.px == 35, "Placement offsets should match py, px"
    assert pos2_result.oy == 8 and pos2_result.ox == 12, "Original offsets should match oy, ox"
    assert pos2_result.frame == 1, "Frame should be 1"
    # Check line bitmap pattern
    expected_mask2 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.uint8)
    assert np.array_equal(pos2_result.shape, expected_mask2), "Line bitmap should match"


def test_convert_empty_collage():
    """Test handling of empty collages."""
    from polyis.pack.adapters import convert_collages_to_bitmap
    
    # Test with empty collages
    result = convert_collages_to_bitmap([])
    assert result == [], "Empty input should return empty list"
    
    # Test with collage containing no polyominoes
    result = convert_collages_to_bitmap([[]])
    assert result == [[]], "Empty collage should return empty inner list"
