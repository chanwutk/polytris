#!/usr/bin/env python3
"""
Test suite for comparing C and Python implementations of pack_ffd (First Fit Decreasing).

This module tests the correctness and consistency of the C implementation (pack_ffd.pyx)
against the Python prototype implementation (pack_ffd_python.py).
"""

import pytest
import numpy as np
import time
from polyis.pack.group_tiles import group_tiles as group_tiles_cython
from polyis.pack.python.pack_ffd import pack_all as pack_all_python
from polyis.cbinpack.group_tiles import group_tiles as group_tiles_c
from polyis.cbinpack.pack_ffd import pack_all as pack_all_c


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


def get_polyominoes_for_both_implementations(bitmap):
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
    python_stack = group_tiles_cython(bitmap.copy(), 0)
    c_stack = group_tiles_c(bitmap.copy(), 0)
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
            ph, pw = shape.shape

            # Adjust for placement position
            py, px = pos.py, pos.px

            # Find actual occupied tiles in shape
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

        result_python = pack_all_python(polyominoes_stacks, h, w)
        result_c = pack_all_c(polyominoes_stacks, h, w)

        assert len(result_python) == 0
        assert len(result_c) == 0

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
        result_c = pack_all_c([c_stack], h, w)

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
        python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap)

        h, w = 10, 10

        result_python = pack_all_python([python_stack], h, w)
        result_c = pack_all_c([c_stack], h, w)

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
        python_stacks = []
        c_stacks = []
        for bitmap in bitmaps:
            python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap)
            python_stacks.append(python_stack)
            c_stacks.append(c_stack)

        h, w = 8, 8

        result_python = pack_all_python(python_stacks, h, w)
        result_c = pack_all_c(c_stacks, h, w)

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
        python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap)

        h, w = 50, 50

        result_python = pack_all_python([python_stack], h, w)
        result_c = pack_all_c([c_stack], h, w)

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
        python_stacks = []
        c_stacks = []
        for bitmap in bitmaps:
            python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap)
            python_stacks.append(python_stack)
            c_stacks.append(c_stack)

        h, w = 40, 40

        result_python = pack_all_python(python_stacks, h, w)
        result_c = pack_all_c(c_stacks, h, w)

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
            ((20, 20), 0.3, 3),   # Small: 3 frames of 20x20
            ((50, 50), 0.3, 5),   # Medium: 5 frames of 50x50
            # ((100, 100), 0.3, 3), # Large: 3 frames of 100x100
        ]

        print(f"{'Size':<12} {'Frames':<8} {'Python (s)':<12} {'C (s)':<12} {'Speedup':<10}")
        print("-" * 60)

        for (h_bitmap, w_bitmap), density, num_frames in test_cases:
            # Generate test data
            bitmaps = [generate_test_bitmap((h_bitmap, w_bitmap), density=density, seed=i*10)
                      for i in range(num_frames)]

            # Get polyominoes for both implementations for each frame
            python_stacks = []
            c_stacks = []
            for bitmap in bitmaps:
                python_stack, c_stack = get_polyominoes_for_both_implementations(bitmap)
                python_stacks.append(python_stack)
                c_stacks.append(c_stack)

            h, w = 128, 128

            # Time Python implementation
            start = time.perf_counter()
            result_python = pack_all_python(python_stacks, h, w)
            python_time = time.perf_counter() - start

            # Time C implementation
            start = time.perf_counter()
            result_c = pack_all_c(c_stacks, h, w)
            c_time = time.perf_counter() - start

            speedup = python_time / c_time if c_time > 0 else float('inf')

            size_str = f"{h_bitmap}x{w_bitmap}"
            print(f"{size_str:<12} {num_frames:<8} {python_time:<12.6f} {c_time:<12.6f} {speedup:<10.2f}x")

            # Compare results to ensure both implementations produce same output
            assert compare_polyomino_positions(result_python, result_c, verbose=False), \
                f"Results differ for size={size_str}, frames={num_frames}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
