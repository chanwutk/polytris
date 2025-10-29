"""
Pytest tests for the group_tiles Cython implementation.
"""

import pytest
import numpy as np
import time
import sys
import os
from polyis.binpack.adapters import group_tiles
from group_tiles_original import group_tiles as _group_tiles_original
from queue import Queue


def same_results(result1, result2):
    """Helper function to compare two group_tiles results."""
    assert len(result1) == len(result2), \
        f"Different number of polyominoes: Result1={len(result1)}, Result2={len(result2)}"
    for (mask1, offset1), (mask2, offset2) in zip(result1, result2):
        np.testing.assert_array_equal(mask1, mask2, 
            err_msg=f"Masks differ at offset {offset1}")


class TestGroupTiles:
    """Test class for group_tiles functionality."""
    
    def test_single_connected_component(self):
        """Test grouping a single connected component."""
        bitmap = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        # Test Cython implementation
        result_cython = group_tiles(bitmap.copy(), 0)
        
        # Test original implementation
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        
        # Sort results by offset for consistent comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))
        
        for i, ((cython_mask, cython_offset), (orig_mask, orig_offset)) in enumerate(zip(result_cython_sorted, result_original_sorted)):
            # Masks should have the same shape and content
            assert cython_mask.shape == orig_mask.shape, f"Polyomino {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Different mask content")
            assert cython_offset == orig_offset, f"Polyomino {i}: Different offsets"
    
    def test_multiple_connected_components(self):
        """Test grouping multiple separate connected components."""
        bitmap = np.array([
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0]
        ], dtype=np.uint8)
        
        # Test Cython implementation
        result_cython = group_tiles(bitmap.copy(), 0)
        
        # Test original implementation
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        
        # Sort results by offset for consistent comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))
        
        for i, ((cython_mask, cython_offset), (orig_mask, orig_offset)) in enumerate(zip(result_cython_sorted, result_original_sorted)):
            assert cython_mask.shape == orig_mask.shape, f"Polyomino {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Different mask content")
            assert cython_offset == orig_offset, f"Polyomino {i}: Different offsets"
    
    def test_empty_bitmap(self):
        """Test with empty bitmap."""
        bitmap = np.zeros((3, 3), dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        assert len(result_cython) == len(result_original), f"Different results for empty bitmap: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 0, "Empty bitmap should produce no polyominoes"
    
    def test_full_bitmap(self):
        """Test with completely filled bitmap."""
        bitmap = np.ones((3, 3), dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 1, "Fully connected bitmap should produce one polyomino"
        
        # Compare the single polyomino
        cython_mask, cython_offset = result_cython[0]
        orig_mask, orig_offset = result_original[0]
        
        assert cython_mask.shape == orig_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Masks should be identical")
        assert cython_offset == orig_offset, "Offsets should be identical"
    
    def test_single_tile(self):
        """Test with single isolated tile."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 1, "Should find one polyomino"
        
        # Compare the single polyomino
        cython_mask, cython_offset = result_cython[0]
        orig_mask, orig_offset = result_original[0]
        
        assert cython_mask.shape == orig_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Masks should be identical")
        assert cython_offset == orig_offset, "Offsets should be identical"
    
    def test_complex_shape(self):
        """Test with L-shaped polyomino."""
        bitmap = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 1, "L-shape should be one connected component"
        
        # Compare the L-shaped polyomino
        cython_mask, cython_offset = result_cython[0]
        orig_mask, orig_offset = result_original[0]
        
        assert cython_mask.shape == orig_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Masks should be identical")
        assert cython_offset == orig_offset, "Offsets should be identical"
        assert cython_mask.sum() == 5, "L-shape should have 5 tiles"
    
    def test_connected_padding(self):
        bitmap = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        result1 = [(np.array([
            [1, 0],
            [1, 1],
            [1, 0]
        ]), (0, 0)), (np.array([
            [0, 1],
            [1, 1],
            [0, 1]
        ]), (0, 2))]
        result2 = [(np.array([
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1]
        ]), (0, 0))]

        result_cython = group_tiles(bitmap.copy(), 1)
        result_original = _group_tiles_original(bitmap.copy(), 1)
        same_results(result_cython, result1)
        same_results(result_original, result1)

        result_cython = group_tiles(bitmap.copy(), 2)
        result_original = _group_tiles_original(bitmap.copy(), 2)
        same_results(result_cython, result2)
        same_results(result_original, result2)
    
    def test_diagonal_not_connected(self):
        """Test that diagonal tiles are not considered connected."""
        bitmap = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        # Compare results
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 3, "Diagonal tiles should be separate components"
        
        # Sort results for consistent comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))  # Sort by offset
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))
        
        for i, ((cython_mask, cython_offset), (orig_mask, orig_offset)) in enumerate(zip(result_cython_sorted, result_original_sorted)):
            assert cython_mask.shape == orig_mask.shape, f"Component {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Component {i}: Different mask content")
            assert cython_offset == orig_offset, f"Component {i}: Different offsets"
            assert cython_mask.sum() == 1, f"Component {i}: Should have one tile"


class TestGroupTilesPerformance:
    """Performance tests for group_tiles."""
    
    def test_performance_comparison(self):
        """Compare performance between Cython and original implementations."""
        # Create a bitmap with multiple scattered components
        bitmap = np.zeros((30, 30), dtype=np.uint8)
        
        # Add some scattered 2x2 squares and other shapes
        for i in range(0, 30, 6):
            for j in range(0, 30, 6):
                if i + 1 < 30 and j + 1 < 30:
                    bitmap[i:i+2, j:j+2] = 1
        
        # Add some L-shapes
        for i in range(1, 30, 8):
            for j in range(1, 30, 8):
                if i + 2 < 30 and j + 2 < 30:
                    bitmap[i, j:j+3] = 1
                    bitmap[i+1:i+3, j] = 1
        
        # Test original implementation
        start_time = time.time()
        result_original = _group_tiles_original(bitmap.copy())
        original_time = time.time() - start_time
        
        # Test Cython implementation
        start_time = time.time()
        result_cython = group_tiles(bitmap.copy(), 0)
        cython_time = time.time() - start_time
        
        # Verify results are identical
        assert len(result_cython) == len(result_original), f"Different number of polyominoes: Cython={len(result_cython)}, Original={len(result_original)}"
        
        # Sort results for comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))
        
        for i, ((cython_mask, cython_offset), (orig_mask, orig_offset)) in enumerate(zip(result_cython_sorted, result_original_sorted)):
            assert cython_mask.shape == orig_mask.shape, f"Polyomino {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Different mask content")
            assert cython_offset == orig_offset, f"Polyomino {i}: Different offsets"
        
        # Performance reporting
        speedup = original_time / cython_time if cython_time > 0 else float('inf')
        
        print(f"\nPerformance comparison:")
        print(f"  Original time: {original_time:.6f}s")
        print(f"  Cython time:   {cython_time:.6f}s")
        print(f"  Speedup:       {speedup:.2f}x")
        print(f"  Found {len(result_cython)} polyominoes")
        print(f"  Bitmap size: {bitmap.shape}")
        
        assert cython_time < 1.0, "Cython implementation should complete within reasonable time"


def test_import_success():
    """Test that the Cython module can be imported successfully."""
    try:
        from polyis.binpack.adapters import group_tiles
        assert callable(group_tiles), "group_tiles should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import Cython implementation: {e}")


def test_return_format():
    """Test the return format of group_tiles."""
    bitmap = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    
    result = group_tiles(bitmap, 0)
    
    assert isinstance(result, list), "Should return a list"
    assert len(result) > 0, "Should find at least one polyomino"
    
    for item in result:
        assert isinstance(item, tuple), "Each item should be a tuple"
        assert len(item) == 2, "Each tuple should have 2 elements"
        
        mask, offset = item
        assert isinstance(mask, np.ndarray), "Mask should be numpy array"
        assert mask.dtype == np.uint8, "Mask should be uint8"
        assert isinstance(offset, tuple), "Offset should be tuple"
        assert len(offset) == 2, "Offset should have 2 elements"


def test_comprehensive_comparison():
    """Comprehensive test comparing Cython vs original implementation on various patterns."""
    test_bitmaps = [
        # Simple cases
        np.array([[1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
        np.array([[1], [1]], dtype=np.uint8),
        
        # Complex shapes
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.uint8),  # L-shape
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),  # Plus shape
        
        # Multiple components
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8),
        
        # Large connected component
        np.ones((5, 5), dtype=np.uint8),
        
        # Sparse pattern
        np.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=np.uint8),
    ]
    
    for i, bitmap in enumerate(test_bitmaps):
        # Test both implementations
        result_cython = group_tiles(bitmap.copy(), 0)
        result_original = _group_tiles_original(bitmap.copy())
        
        # Verify identical results
        assert len(result_cython) == len(result_original), f"Test {i}: Different number of polyominoes"
        
        if len(result_cython) > 0:
            # Sort for consistent comparison
            result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
            result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))
            
            for j, ((cython_mask, cython_offset), (orig_mask, orig_offset)) in enumerate(zip(result_cython_sorted, result_original_sorted)):
                assert cython_mask.shape == orig_mask.shape, f"Test {i}, Polyomino {j}: Different mask shapes"
                np.testing.assert_array_equal(cython_mask, orig_mask, f"Test {i}, Polyomino {j}: Different mask content")
                assert cython_offset == orig_offset, f"Test {i}, Polyomino {j}: Different offsets"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
