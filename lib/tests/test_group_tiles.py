"""
Pytest tests for the group_tiles Cython implementation.
"""

import pytest
import numpy as np
import time
import sys
import os
from group_tiles import group_tiles
from queue import Queue

# Add parent directory to path to import original implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def find_connected_tiles(bitmap: np.ndarray, i: int, j: int) -> list[tuple[int, int]]:
    """
    Find all connected tiles in the bitmap starting from the tile at (i, j).
    
    Args:
        bitmap: 2D numpy array representing the grid of tiles,
                where 1 indicates a tile with detection and 0 indicates no detection
        i: row index of the starting tile
        j: column index of the starting tile
        
    Returns:
        list[tuple[int, int]]: List of tuples representing the coordinates of all connected tiles
    """
    value = bitmap[i, j]
    q = Queue()
    q.put((i, j))
    filled: list[tuple[int, int]] = []
    while not q.empty():
        i, j = q.get()
        bitmap[i, j] = value
        filled.append((i, j))
        for _i, _j in [(-1, 0), (0, -1), (+1, 0), (0, +1)]:
            _i += i
            _j += j
            if bitmap[_i, _j] != 0 and bitmap[_i, _j] != value:
                q.put((_i, _j))
    return filled


def _group_tiles_original(bitmap: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Original Python implementation of group_tiles (backup).
    """
    h, w = bitmap.shape
    _groups = np.arange(h * w, dtype=np.int16) + 1
    _groups = _groups.reshape(bitmap.shape)
    _groups = _groups * bitmap
    
    # Padding with size=1 on all sides
    groups = np.zeros((h + 2, w + 2), dtype=np.int16)
    groups[1:h+1, 1:w+1] = _groups
    
    visited: set[int] = set()
    bins: list[tuple[np.ndarray, tuple[int, int]]] = []
    
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            if groups[i, j] == 0 or groups[i, j] in visited:
                continue
            
            connected_tiles = find_connected_tiles(groups, i, j)
            if not connected_tiles:
                continue
                
            connected_tiles = np.array(connected_tiles, dtype=int).T
            mask = np.zeros((h + 1, w + 1), dtype=np.uint8)
            mask[*connected_tiles] = True
            
            offset = np.min(connected_tiles, axis=1)
            end = np.max(connected_tiles, axis=1) + 1
            
            mask = mask[offset[0]:end[0], offset[1]:end[1]]
            bins.append((mask, (int(offset[0] - 1), int(offset[1] - 1))))
            visited.add(groups[i, j])
    
    return bins


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
        result_cython = group_tiles(bitmap.copy())
        
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
        result_cython = group_tiles(bitmap.copy())
        
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
        result_cython = group_tiles(bitmap.copy())
        result_original = _group_tiles_original(bitmap.copy())
        
        assert len(result_cython) == len(result_original), f"Different results for empty bitmap: Cython={len(result_cython)}, Original={len(result_original)}"
        assert len(result_cython) == 0, "Empty bitmap should produce no polyominoes"
    
    def test_full_bitmap(self):
        """Test with completely filled bitmap."""
        bitmap = np.ones((3, 3), dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy())
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
        result_cython = group_tiles(bitmap.copy())
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
        result_cython = group_tiles(bitmap.copy())
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
    
    def test_diagonal_not_connected(self):
        """Test that diagonal tiles are not considered connected."""
        bitmap = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.uint8)
        
        # Test both implementations
        result_cython = group_tiles(bitmap.copy())
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
        result_cython = group_tiles(bitmap.copy())
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
        from group_tiles import group_tiles
        assert callable(group_tiles), "group_tiles should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import Cython implementation: {e}")


def test_return_format():
    """Test the return format of group_tiles."""
    bitmap = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    
    result = group_tiles(bitmap)
    
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
        result_cython = group_tiles(bitmap.copy())
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
