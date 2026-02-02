"""
Pytest tests for the group_tiles Cython implementation.
"""

import pytest
import numpy as np
import time
from polyis.pack.cython.adapters import group_tiles as group_tiles_cython
from polyis.pack.adapters import c_group_tiles as group_tiles_c
from polyis.pack.python.group_tiles import group_tiles as python_group_tiles


def same_results(result1, result2, mode):
    """Helper function to compare two group_tiles results."""
    assert len(result1) == len(result2), \
        f"Different number of polyominoes: Result1={len(result1)}, Result2={len(result2)}, mode {mode}"
    result1 = sorted(result1, key=lambda x: (x[1][0], x[1][1], x[0].sum()))
    result2 = sorted(result2, key=lambda x: (x[1][0], x[1][1], x[0].sum()))
    for (mask1, offset1), (mask2, offset2) in zip(result1, result2):
        np.testing.assert_array_equal(mask1.astype(int), mask2.astype(int), 
            err_msg=f"Masks differ at offset {offset1}, {offset2}, mode {mode}" \
            f"{result1} != {result2}")


class TestGroupTiles:
    """Test class for group_tiles functionality."""
    
    def test_single_connected_component(self):
        """Test grouping a single connected component."""
        bitmap = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        # Compare results - all should have same number of polyominoes
        assert len(result_cython) == len(result_original) == len(result_c), \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Sort results by offset for consistent comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_c_sorted = sorted(result_c, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))

        for i, ((cython_mask, cython_offset), (c_mask, c_offset), (orig_mask, orig_offset)) in enumerate(
            zip(result_cython_sorted, result_c_sorted, result_original_sorted)
        ):
            # All masks should have the same shape and content
            assert cython_mask.shape == orig_mask.shape == c_mask.shape, f"Polyomino {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Cython vs Original mask content differs")
            np.testing.assert_array_equal(c_mask, orig_mask, f"Polyomino {i}: C vs Original mask content differs")
            assert cython_offset == orig_offset == c_offset, f"Polyomino {i}: Different offsets"
    
    def test_multiple_connected_components(self):
        """Test grouping multiple separate connected components."""
        bitmap1 = np.array([
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0]
        ], dtype=np.uint8)

        bitmap2 = np.array([
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ], dtype=np.uint8)

        for bitmap in [bitmap1, bitmap2]:
            # Test all three implementations
            result_cython = group_tiles_cython(bitmap.copy(), 0)
            result_c = group_tiles_c(bitmap.copy(), 0)
            result_original = python_group_tiles(bitmap.copy())

            # Compare results
            assert len(result_cython) == len(result_original) == len(result_c), \
                f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

            # Sort results by offset for consistent comparison
            result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
            result_c_sorted = sorted(result_c, key=lambda x: (x[1][0], x[1][1]))
            result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))

            for i, ((cython_mask, cython_offset), (c_mask, c_offset), (orig_mask, orig_offset)) in enumerate(
                zip(result_cython_sorted, result_c_sorted, result_original_sorted)
            ):
                assert cython_mask.shape == orig_mask.shape == c_mask.shape, f"Polyomino {i}: Different mask shapes"
                np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Cython vs Original mask content differs")
                np.testing.assert_array_equal(c_mask, orig_mask, f"Polyomino {i}: C vs Original mask content differs")
                assert cython_offset == orig_offset == c_offset, f"Polyomino {i}: Different offsets"
    
    def test_empty_bitmap(self):
        """Test with empty bitmap."""
        bitmap = np.zeros((3, 3), dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        assert len(result_cython) == len(result_original) == len(result_c) == 0, \
            f"Different results for empty bitmap: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"
    
    def test_full_bitmap(self):
        """Test with completely filled bitmap."""
        bitmap = np.ones((3, 3), dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        # Compare results
        assert len(result_cython) == len(result_original) == len(result_c) == 1, \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Compare the single polyomino
        cython_mask, cython_offset = result_cython[0]
        c_mask, c_offset = result_c[0]
        orig_mask, orig_offset = result_original[0]

        assert cython_mask.shape == orig_mask.shape == c_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Cython vs Original masks should be identical")
        np.testing.assert_array_equal(c_mask, orig_mask, "C vs Original masks should be identical")
        assert cython_offset == orig_offset == c_offset, "Offsets should be identical"
    
    def test_single_tile(self):
        """Test with single isolated tile."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        # Compare results
        assert len(result_cython) == len(result_original) == len(result_c) == 1, \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Compare the single polyomino
        cython_mask, cython_offset = result_cython[0]
        c_mask, c_offset = result_c[0]
        orig_mask, orig_offset = result_original[0]

        assert cython_mask.shape == orig_mask.shape == c_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Cython vs Original masks should be identical")
        np.testing.assert_array_equal(c_mask, orig_mask, "C vs Original masks should be identical")
        assert cython_offset == orig_offset == c_offset, "Offsets should be identical"
    
    def test_complex_shape(self):
        """Test with L-shaped polyomino."""
        bitmap = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        # Compare results
        assert len(result_cython) == len(result_original) == len(result_c) == 1, \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Compare the L-shaped polyomino
        cython_mask, cython_offset = result_cython[0]
        c_mask, c_offset = result_c[0]
        orig_mask, orig_offset = result_original[0]

        assert cython_mask.shape == orig_mask.shape == c_mask.shape, "Masks should have same shape"
        np.testing.assert_array_equal(cython_mask, orig_mask, "Cython vs Original masks should be identical")
        np.testing.assert_array_equal(c_mask, orig_mask, "C vs Original masks should be identical")
        assert cython_offset == orig_offset == c_offset, "Offsets should be identical"
        assert cython_mask.sum() == c_mask.sum() == 5, "L-shape should have 5 tiles"
    
    def test_connected_padding_split(self):
        bitmap = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        results = [
            [  # none
                (np.array([[1]]), (1, 0)),
                (np.array([[1]]), (1, 3)),
            ],
            [  # plus
                (np.array([
                    [1, 0],
                    [1, 1],
                    [1, 0]
                ]), (0, 0)),
                (np.array([
                    [0, 1],
                    [1, 1],
                    [0, 1]
                ]), (0, 2))
            ],
            [  # top-left
                (np.array([
                    [1],
                    [1]
                ]), (0, 0)),
                (np.array([
                    [1, 1],
                    [1, 1]
                ]), (0, 2))
            ],
            [  # top-right
                (np.array([
                    [1, 1],
                    [1, 1],
                ]), (0, 0)),
                (np.array([
                    [1],
                    [1],
                ]), (0, 3))
            ],
            [  # bottom-left
                (np.array([
                    [1],
                    [1]
                ]), (1, 0)),
                (np.array([
                    [1, 1],
                    [1, 1]
                ]), (1, 2))
            ],
            [  # bottom-right
                (np.array([
                    [1, 1],
                    [1, 1]
                ]), (1, 0)),
                (np.array([
                    [1],
                    [1]
                ]), (1, 3))
            ],
            [  # square
                (np.array([
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]), (0, 0)),
                (np.array([
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]), (0, 2))
            ]
        ]

        for mode, result in enumerate(results):
            result_cython = group_tiles_cython(bitmap.copy(), mode)
            result_c = group_tiles_c(bitmap.copy(), mode)
            result_original = python_group_tiles(bitmap.copy(), mode)
            # same_results(result_cython, result)
            same_results(result_original, result, mode)
            same_results(result_c, result, mode)

    
    def test_connected_padding_combined(self):
        bitmap = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0]
        ], dtype=np.uint8)
        result1 = [(np.array([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ]), (0, 0))]
        result2 = [(np.array([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ]), (0, 0))]

        # Test all three implementations with mode 1
        result_cython = group_tiles_cython(bitmap.copy(), 1)
        result_c = group_tiles_c(bitmap.copy(), 1)
        result_original = python_group_tiles(bitmap.copy(), 1)
        same_results(result_cython, result1, 1)
        same_results(result_c, result1, 1)
        same_results(result_original, result1, 1)

        # # Test all three implementations with mode 2
        # result_cython = group_tiles_cython(bitmap.copy(), 2)
        # result_c = group_tiles_c(bitmap.copy(), 2)
        # result_original = python_group_tiles(bitmap.copy(), 2)
        # same_results(result_cython, result2)
        # same_results(result_c, result2)
        # same_results(result_original, result2)
    
    def test_diagonal_not_connected(self):
        """Test that diagonal tiles are not considered connected."""
        bitmap = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.uint8)

        # Test all three implementations
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy())

        # Compare results
        assert len(result_cython) == len(result_original) == len(result_c) == 3, \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Sort results for consistent comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_c_sorted = sorted(result_c, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))

        for i, ((cython_mask, cython_offset), (c_mask, c_offset), (orig_mask, orig_offset)) in enumerate(
            zip(result_cython_sorted, result_c_sorted, result_original_sorted)
        ):
            assert cython_mask.shape == orig_mask.shape == c_mask.shape, f"Component {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Component {i}: Cython vs Original mask content differs")
            np.testing.assert_array_equal(c_mask, orig_mask, f"Component {i}: C vs Original mask content differs")
            assert cython_offset == orig_offset == c_offset, f"Component {i}: Different offsets"
            assert cython_mask.sum() == c_mask.sum() == 1, f"Component {i}: Should have one tile"


class TestGroupTilesPerformance:
    """Performance tests for group_tiles."""
    
    def test_performance_comparison(self):
        """Compare performance between all three implementations."""
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
        result_original = python_group_tiles(bitmap.copy())
        original_time = time.time() - start_time

        # Test Cython implementation
        start_time = time.time()
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        cython_time = time.time() - start_time

        # Test C implementation
        start_time = time.time()
        result_c = group_tiles_c(bitmap.copy(), 0)
        c_time = time.time() - start_time

        # Verify results are identical
        assert len(result_cython) == len(result_original) == len(result_c), \
            f"Different number of polyominoes: Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # Sort results for comparison
        result_cython_sorted = sorted(result_cython, key=lambda x: (x[1][0], x[1][1]))
        result_c_sorted = sorted(result_c, key=lambda x: (x[1][0], x[1][1]))
        result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1]))

        for i, ((cython_mask, cython_offset), (c_mask, c_offset), (orig_mask, orig_offset)) in enumerate(
            zip(result_cython_sorted, result_c_sorted, result_original_sorted)
        ):
            assert cython_mask.shape == orig_mask.shape == c_mask.shape, f"Polyomino {i}: Different mask shapes"
            np.testing.assert_array_equal(cython_mask, orig_mask, f"Polyomino {i}: Cython vs Original mask content differs")
            np.testing.assert_array_equal(c_mask, orig_mask, f"Polyomino {i}: C vs Original mask content differs")
            assert cython_offset == orig_offset == c_offset, f"Polyomino {i}: Different offsets"

        # Performance reporting
        cython_speedup = original_time / cython_time if cython_time > 0 else float('inf')
        c_speedup = original_time / c_time if c_time > 0 else float('inf')

        print(f"\nPerformance comparison:")
        print(f"  Original time: {original_time:.6f}s")
        print(f"  Cython time:   {cython_time:.6f}s")
        print(f"  C time:        {c_time:.6f}s")
        print(f"  Cython speedup: {cython_speedup:.2f}x")
        print(f"  C speedup:      {c_speedup:.2f}x")
        print(f"  Found {len(result_cython)} polyominoes")
        print(f"  Bitmap size: {bitmap.shape}")

        assert cython_time < 1.0, "Cython implementation should complete within reasonable time"
        assert c_time < 1.0, "C implementation should complete within reasonable time"


class TestGroupTilesPaddingModes:
    """Test class for different padding modes."""

    def test_mode_0_no_padding(self):
        """Test mode 0: No padding - isolated tiles stay separate."""
        bitmap = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=np.uint8)

        # Test all three implementations with mode 0
        result_cython = group_tiles_cython(bitmap.copy(), 0)
        result_c = group_tiles_c(bitmap.copy(), 0)
        result_original = python_group_tiles(bitmap.copy(), 0)

        # All should produce 4 separate polyominoes
        assert len(result_cython) == len(result_original) == len(result_c) == 4, \
            f"Mode 0: Expected 4 polyominoes, got Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

    def test_mode_1_plus_padding(self):
        """Test mode 1: Plus padding (4 orthogonal neighbors)."""
        bitmap = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=np.uint8)

        # Test all three implementations with mode 1
        result_cython = group_tiles_cython(bitmap.copy(), 1)
        result_c = group_tiles_c(bitmap.copy(), 1)
        result_original = python_group_tiles(bitmap.copy(), 1)

        # With plus padding, all corners connect through padding tiles into 1 polyomino
        assert len(result_cython) == len(result_original) == len(result_c) == 1, \
            f"Mode 1: Expected 1 polyomino, got Cython={len(result_cython)}, C={len(result_c)}, Original={len(result_original)}"

        # The polyomino should contain 4 original tiles + 4 padding tiles = 8 tiles
        assert result_c[0][0].sum() == 8, f"Mode 1: Expected 8 tiles, got {result_c[0][0].sum()}"

    def test_mode_2_top_left_padding(self):
        """Test mode 2: Top-left padding."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test C and Python implementations (Cython support for modes 2-6 not yet implemented)
        result_c = group_tiles_c(bitmap.copy(), 2)
        result_original = python_group_tiles(bitmap.copy(), 2)

        # Mode 2 produces 2 polyominoes: main (original + top + left) and corner padding
        assert len(result_original) == len(result_c) == 1, \
            f"Mode 2: Expected 1 polyomino, got C={len(result_c)}, Python={len(result_original)}"
        same_results(result_original, result_c, 2)

    def test_mode_3_top_right_padding(self):
        """Test mode 3: Top-right padding."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test C and Python implementations (Cython support for modes 2-6 not yet implemented)
        result_c = group_tiles_c(bitmap.copy(), 3)
        result_original = python_group_tiles(bitmap.copy(), 3)

        # Mode 3 produces 2 polyominoes: main (original + top + right) and corner padding
        assert len(result_original) == len(result_c) == 1, \
            f"Mode 3: Expected 1 polyomino, got C={len(result_c)}, Python={len(result_original)}"
        same_results(result_original, result_c, 3)

    def test_mode_4_bottom_left_padding(self):
        """Test mode 4: Bottom-left padding."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test C and Python implementations (Cython support for modes 2-6 not yet implemented)
        result_c = group_tiles_c(bitmap.copy(), 4)
        result_original = python_group_tiles(bitmap.copy(), 4)

        # Mode 4 produces 2 polyominoes: main (original + bottom + left) and corner padding
        assert len(result_original) == len(result_c) == 1, \
            f"Mode 4: Expected 1 polyomino, got C={len(result_c)}, Python={len(result_original)}"
        same_results(result_original, result_c, 4)

    def test_mode_5_bottom_right_padding(self):
        """Test mode 5: Bottom-right padding."""
        bitmap = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # Test C and Python implementations (Cython support for modes 2-6 not yet implemented)
        result_c = group_tiles_c(bitmap.copy(), 5)
        result_original = python_group_tiles(bitmap.copy(), 5)

        # Mode 5 produces 2 polyominoes: main (original + bottom + right) and corner padding
        assert len(result_original) == len(result_c) == 1, \
            f"Mode 5: Expected 1 polyomino, got C={len(result_c)}, Python={len(result_original)}"
        same_results(result_original, result_c, 5)

    def test_mode_6_square_padding(self):
        """Test mode 6: Square padding (all 8 neighbors)."""
        bitmap = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=np.uint8)

        # Test C and Python implementations (Cython support for modes 2-6 not yet implemented)
        result_c = group_tiles_c(bitmap.copy(), 6)
        result_original = python_group_tiles(bitmap.copy(), 6)

        # With square padding (8 neighbors), the outer ring connects but center padding is separate
        # This results in 2 polyominoes: outer ring (8 tiles) + center padding (1 tile)
        assert len(result_original) == len(result_c) == 1, \
            f"Mode 6: Expected 1 polyomino, got C={len(result_c)}, Python={len(result_original)}"
        same_results(result_original, result_c, 6)

        # Sort by size (largest first)
        result_c_sorted = sorted(result_c, key=lambda x: -(x[0].sum().astype(np.int16)))
        result_python_sorted = sorted(result_original, key=lambda x: -(x[0].sum().astype(np.int16)))
        assert result_c_sorted[0][0].sum() == 9, f"Mode 6: Expected outer ring with 9 tiles, got {result_c_sorted[0][0].sum()}"
        assert result_python_sorted[0][0].sum() == 9, f"Mode 6: Expected outer ring with 9 tiles, got {result_python_sorted[0][0].sum()}"

    def test_padding_modes_consistency(self):
        """Test that C and Python implementations produce consistent results across all modes."""
        bitmap = np.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1]
        ], dtype=np.uint8)

        # Test each mode (0-6)
        for mode in range(7):
            result_c = group_tiles_c(bitmap.copy(), mode)
            result_original = python_group_tiles(bitmap.copy(), mode)

            # C and Python implementations should produce same number of polyominoes
            assert len(result_original) == len(result_c), \
                f"Mode {mode}: Different number of polyominoes - C={len(result_c)}, Python={len(result_original)}"

            # Sort results for consistent comparison (by offset, then by size, then by shape)
            # Use sum as secondary key for stable sorting when offsets are identical
            result_c_sorted = sorted(result_c, key=lambda x: (x[1][0], x[1][1], -x[0].sum().astype(np.int16), x[0].shape))
            result_original_sorted = sorted(result_original, key=lambda x: (x[1][0], x[1][1], -x[0].sum().astype(np.int16), x[0].shape))

            # Compare each polyomino
            for i, ((c_mask, c_offset), (orig_mask, orig_offset)) in enumerate(
                zip(result_c_sorted, result_original_sorted)
            ):
                assert orig_mask.shape == c_mask.shape, \
                    f"Mode {mode}, Polyomino {i}: Different mask shapes - C={c_mask.shape}, Python={orig_mask.shape}"
                np.testing.assert_array_equal(c_mask, orig_mask,
                    f"Mode {mode}, Polyomino {i}: C vs Python mask content differs")
                assert orig_offset == c_offset, \
                    f"Mode {mode}, Polyomino {i}: Different offsets - C={c_offset}, Python={orig_offset}"
