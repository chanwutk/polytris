#!/usr/bin/env python3
"""
Test suite for comparing C and Python implementations of pack_ffd (First Fit Decreasing).

This module tests the correctness and consistency of the C implementation (pack_ffd.pyx)
against the Python prototype implementation (pack_ffd_python.py).

Also includes tests for the convert_collages_to_bitmap adapter function that converts
coordinate-based shapes from pack_ffd.pyx into bitmap format.
"""

import pytest
import numpy as np
import time
from polyis.pack.cython.group_tiles import group_tiles as group_tiles_cython
from polyis.pack.python.pack_ffd import pack as pack_python
from polyis.pack.group_tiles import group_tiles as group_tiles_c
from polyis.pack.pack import pack as pack_c, PyPolyominoPosition
from polyis.pack.adapters import convert_collages_to_bitmap


def generate_test_bitmap(shape, density=0.3, seed=42):
    """Generate a test bitmap with specified density of 1s.

    Args:
        shape: Tuple (height, width) for the bitmap
        density: Fraction of cells that should be 1 (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        2D numpy array of uint8 with random 1s and 0s
    """
    np.random.seed(seed)
    bitmap = np.random.random(shape) < density
    return bitmap.astype(np.uint8)


def get_polyominoes_for_both_implementations(bitmap, mode):
    """Get polyomino stacks for both Python and C implementations.

    Args:
        bitmap: 2D numpy array representing the tile grid

    Returns:
        Tuple of (python_stack, c_stack) where each is a memory address
        to the appropriate polyomino data structure for that implementation
    """
    # IMPORTANT: Due to symbol conflicts between Cython and C group_tiles,
    # we must use the same implementation for both to avoid memory corruption.
    # Both implementations can accept PolyominoArray* from C group_tiles.
    python_stack = group_tiles_cython(bitmap.copy(), mode)
    c_stack = group_tiles_c(bitmap.copy(), mode)
    return python_stack, c_stack


def compare_polyomino_positions(python_result, c_result, verbose=False):
    """Compare the results from Python and C implementations.

    Args:
        python_result: List of lists of PolyominoPosition from Python implementation
        c_result: List of lists of PolyominoPosition from C implementation
        verbose: If True, print detailed comparison information

    Returns:
        bool: True if results match (same number of collages and polyominoes)
    """
    # Compare number of collages
    if len(python_result) != len(c_result):
        if verbose:
            print(f"Number of collages differs: Python={len(python_result)}, C={len(c_result)}")
        return False

    # Compare each collage
    for i, (py_collage, c_collage) in enumerate(zip(python_result, c_result)):
        if len(py_collage) != len(c_collage):
            if verbose:
                print(f"Collage {i}: Number of polyominoes differs: Python={len(py_collage)}, C={len(c_collage)}")
            return False

        # Compare each polyomino within the collage
        for j, (py_poly, c_poly) in enumerate(zip(py_collage, c_collage)):
            # Compare scalar fields
            if py_poly.oy != c_poly.oy:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: oy differs: Python={py_poly.oy}, C={c_poly.oy}")
                return False
            if py_poly.ox != c_poly.ox:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: ox differs: Python={py_poly.ox}, C={c_poly.ox}")
                return False
            if py_poly.py != c_poly.py:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: py differs: Python={py_poly.py}, C={c_poly.py}")
                return False
            if py_poly.px != c_poly.px:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: px differs: Python={py_poly.px}, C={c_poly.px}")
                return False
            # Note: rotation field removed from C implementation (always 0)
            if py_poly.frame != c_poly.frame:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: frame differs: Python={py_poly.frame}, C={c_poly.frame}")
                return False
            
            # Compare shape (numpy array mask) - both should be in bitmap format now
            if py_poly.shape.shape != c_poly.shape.shape:
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: shape dimensions differ: Python={py_poly.shape.shape}, C={c_poly.shape.shape}")
                return False

            # Compare shape contents element-wise
            if not np.array_equal(py_poly.shape, c_poly.shape):
                if verbose:
                    print(f"Collage {i}, Polyomino {j}: shape contents differ")
                    print(f"  Python shape:\n{py_poly.shape}")
                    print(f"  C shape:\n{c_poly.shape}")
                return False

    if verbose:
        print(f"Results match: {len(python_result)} collages")
        for i, collage in enumerate(python_result):
            print(f"  Collage {i}: {len(collage)} polyominoes")

    return True


def verify_packing_properties(result, polyominoes_stacks, h, w):
    """Verify that the packing result has valid properties.

    Args:
        result: List of lists of PolyominoPosition
        polyominoes_stacks: Original input polyominoes
        h: Collage height
        w: Collage width

    Returns:
        dict: Statistics about the packing
    """
    stats = {
        'num_collages': len(result),
        'total_polyominoes': sum(len(collage) for collage in result),
        'collage_fills': [],
        'valid': True,
        'errors': []
    }

    # Check each collage
    for collage_idx, collage in enumerate(result):
        # Create occupancy grid for this collage
        occupancy = np.zeros((h, w), dtype=np.uint8)

        for pos in collage:
            # Check bounds
            shape = pos.shape
            
            # Adjust for placement position
            py, px = pos.py, pos.px
            
            # Shape should be bitmap format (2D array where 1s indicate occupied cells)
            occupied_coords = np.argwhere(shape == 1)

            for coord in occupied_coords:
                y = py + coord[0]
                x = px + coord[1]

                # Check bounds
                if y < 0 or y >= h or x < 0 or x >= w:
                    stats['valid'] = False
                    stats['errors'].append(f"Collage {collage_idx}: Position out of bounds at ({y}, {x})")
                    continue

                # Check overlap
                if occupancy[y, x] != 0:
                    stats['valid'] = False
                    stats['errors'].append(f"Collage {collage_idx}: Overlap detected at ({y}, {x})")

                occupancy[y, x] = 1

        # Calculate fill percentage
        fill_pct = np.sum(occupancy) / (h * w) * 100
        stats['collage_fills'].append(fill_pct)

    return stats


class TestPackFFDBasic:
    """Basic functionality tests for pack_ffd implementations."""

    def test_empty_input(self):
        """Test with empty polyominoes list."""
        polyominoes_stacks = []
        h, w = 10, 10

        result_python = pack_python(np.array(polyominoes_stacks, dtype=np.uint64), h, w, 0)
        
        # C implementation raises ValueError for empty input
        with pytest.raises(ValueError, match="polyominoes_stacks cannot be empty"):
            pack_c(np.array(polyominoes_stacks, dtype=np.uint64), h, w, 0)

        assert len(result_python) == 0

    def test_single_polyomino(self):
        """Test with a single polyomino in a single frame."""
        # Create a simple 3x3 bitmap with one polyomino
        bitmap = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        h, w = 10, 10

        # Test C implementation only (due to memory corruption bug)
        c_stack = group_tiles_c(bitmap, 0)
        result_c = pack_c(np.array([c_stack], dtype=np.uint64), h, w, 0)
        
        # Convert coordinate format to bitmap format
        convert_collages_to_bitmap(result_c)

        # Should create exactly 1 collage with 1 polyomino
        assert len(result_c) == 1
        assert len(result_c[0]) == 1

        # Verify the polyomino properties
        poly = result_c[0][0]
        assert poly.oy == 0
        assert poly.ox == 0
        assert poly.frame == 0
        assert poly.shape.shape == (2, 2)  # L-shaped polyomino fits in 2x2 bounding box

    def test_multiple_polyominoes_single_frame(self):
        """Test with multiple polyominoes in a single frame."""
        bitmap = np.array([
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ], dtype=np.uint8)

        # Get polyominoes for both implementations
        python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap, 0)

        h, w = 10, 10

        result_python = pack_python(np.array([python_stack], dtype=np.uint64), h, w, 0)
        result_c = pack_c(np.array([c_stack], dtype=np.uint64), h, w, 0)
        
        # Convert C result from coordinate format to bitmap format
        convert_collages_to_bitmap(result_c)

        # Compare results
        assert compare_polyomino_positions(result_python, result_c, verbose=True)

        # Verify packing properties - note: can't use stacks for verification as they're consumed
        stats_python = verify_packing_properties(result_python, [], h, w)
        stats_c = verify_packing_properties(result_c, [], h, w)

        assert stats_python['valid'], f"Python packing invalid: {stats_python['errors']}"
        assert stats_c['valid'], f"C packing invalid: {stats_c['errors']}"

    def test_multiple_frames(self):
        """Test with multiple frames, each containing polyominoes."""
        bitmaps = [
            np.array([[1, 1], [1, 0]], dtype=np.uint8),
            np.array([[1, 1], [1, 1]], dtype=np.uint8),
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
        ]

        # Get polyominoes for both implementations for each frame
        python_stacks = np.empty(len(bitmaps), dtype=np.uint64)
        c_stacks = np.empty(len(bitmaps), dtype=np.uint64)
        for idx, bitmap in enumerate(bitmaps):
            python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap, 0)
            python_stacks[idx] = python_stack
            c_stacks[idx] = c_stack

        h, w = 8, 8

        result_python = pack_python(python_stacks, h, w, 0)
        result_c = pack_c(c_stacks, h, w, 0)
        
        # Convert C result from coordinate format to bitmap format
        convert_collages_to_bitmap(result_c)

        # Compare results
        assert compare_polyomino_positions(result_python, result_c, verbose=True)

        # Verify packing properties
        stats_python = verify_packing_properties(result_python, [], h, w)
        stats_c = verify_packing_properties(result_c, [], h, w)

        assert stats_python['valid'], f"Python packing invalid: {stats_python['errors']}"
        assert stats_c['valid'], f"C packing invalid: {stats_c['errors']}"


class TestPackFFDRandom:
    """Random test cases for pack_ffd implementations."""

    @pytest.mark.parametrize("density", [0.1, 0.3, 0.5])
    @pytest.mark.parametrize("size", [(10, 10), (20, 20), (30, 30)])
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_random_bitmaps(self, density, size, seed):
        """Test with randomly generated bitmaps."""
        bitmap = generate_test_bitmap(size, density=density, seed=seed)

        # Get polyominoes for both implementations
        python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap, 0)

        h, w = 50, 50

        result_python = pack_python(np.array([python_stack], dtype=np.uint64), h, w, 0)
        result_c = pack_c(np.array([c_stack], dtype=np.uint64), h, w, 0)
        
        # Convert C result from coordinate format to bitmap format
        convert_collages_to_bitmap(result_c)

        # Compare results
        assert compare_polyomino_positions(result_python, result_c, verbose=False), \
            f"Mismatch for size={size}, density={density}, seed={seed}"

        # Verify packing properties
        stats_c = verify_packing_properties(result_c, [], h, w)
        assert stats_c['valid'], f"C packing invalid: {stats_c['errors']}"

    def test_multiple_random_frames(self):
        """Test with multiple frames of random bitmaps."""
        num_frames = 5
        bitmaps = [generate_test_bitmap((15, 15), density=0.3, seed=i*10) for i in range(num_frames)]

        # Get polyominoes for both implementations for each frame
        python_stacks = np.empty(num_frames, dtype=np.uint64)
        c_stacks = np.empty(num_frames, dtype=np.uint64)
        for idx, bitmap in enumerate(bitmaps):
            python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap, 0)
            python_stacks[idx] = python_stack
            c_stacks[idx] = c_stack

        h, w = 40, 40

        result_python = pack_python(python_stacks, h, w, 0)
        result_c = pack_c(c_stacks, h, w, 0)
        
        # Convert C result from coordinate format to bitmap format
        convert_collages_to_bitmap(result_c)

        # Compare results
        assert compare_polyomino_positions(result_python, result_c, verbose=True)

        # Verify packing properties
        stats_python = verify_packing_properties(result_python, [], h, w)
        stats_c = verify_packing_properties(result_c, [], h, w)

        assert stats_python['valid'], f"Python packing invalid: {stats_python['errors']}"
        assert stats_c['valid'], f"C packing invalid: {stats_c['errors']}"


class TestPackFFDPerformance:
    """Performance comparison tests."""

    def test_performance_comparison(self):
        """Compare performance of Python vs C implementations."""
        print("\n=== Pack FFD Performance Comparison ===")

        test_cases = [
            ((10, 10), 0.3, 3),   # Small: 10 frames of 10x10
            ((20, 20), 0.3, 5),   # Medium: 20 frames of 20x20
        ]

        print(f"{'Size':<12} {'Frames':<8} {'Python (s)':<12} {'C (s)':<12} {'Speedup':<10}")
        print("-" * 60)

        for (h_bitmap, w_bitmap), density, num_frames in test_cases:
            # Generate test data
            bitmaps = [generate_test_bitmap((h_bitmap, w_bitmap), density=density, seed=i*10)
                      for i in range(num_frames)]

            # Get polyominoes for both implementations for each frame
            python_stacks = np.empty(num_frames, dtype=np.uint64)
            c_stacks = np.empty(num_frames, dtype=np.uint64)
            for idx, bitmap in enumerate(bitmaps):
                python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap, 0)
                python_stacks[idx] = python_stack
                c_stacks[idx] = c_stack

            h, w = 128, 128

            # Time Python implementation
            start = time.perf_counter()
            result_python = pack_python(python_stacks, h, w, 0)
            python_time = time.perf_counter() - start

            # Time C implementation
            start = time.perf_counter()
            result_c = pack_c(c_stacks, h, w, 0)
            c_time = time.perf_counter() - start
            
            # Convert C result from coordinate format to bitmap format for comparison
            convert_collages_to_bitmap(result_c)

            speedup = python_time / c_time if c_time > 0 else float('inf')

            size_str = f"{h_bitmap}x{w_bitmap}"
            print(f"{size_str:<12} {num_frames:<8} {python_time:<12.6f} {c_time:<12.6f} {speedup:<10.2f}x")
            
            assert speedup >= 1, \
                f"C implementation not sufficiently faster for size={size_str}, frames={num_frames}"

            # Compare results to ensure both implementations produce same output
            assert compare_polyomino_positions(result_python, result_c, verbose=False), \
                f"Results differ for size={size_str}, frames={num_frames}"


class TestConvertCollagesToBitmap:
    """Tests for the convert_collages_to_bitmap adapter function."""

    def test_convert_collages_to_bitmap(self):
        """Test that coordinate-based shapes are correctly converted to bitmaps."""
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
        # Original coords: [(2,3), (2,4), (2,5)] -> bitmap 3x6 with line at row 2
        pos2_result = collages[0][1]
        assert pos2_result.shape.shape == (3, 6), "Line bitmap should be 3x6"
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

    def test_convert_empty_collage(self):
        """Test handling of empty collages."""
        # Test with empty collages
        result = convert_collages_to_bitmap([])
        assert result == [], "Empty input should return empty list"
        
        # Test with collage containing no polyominoes
        result = convert_collages_to_bitmap([[]])
        assert result == [[]], "Empty collage should return empty inner list"
