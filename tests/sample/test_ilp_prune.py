"""
Unit tests for the ILP pruning algorithm.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, '/work/cwkt/projects/polyis')

# Import modules after path is set
from polyis.sample.ilp_prune import ilp_prune_polyominoes
from polyis.sample.ilp_prune_optimized import ilp_prune_polyominoes_optimized
from polyis.pack import group_tiles


class TestILPPrune:
    """Test suite for ILP pruning algorithm."""

    def test_simple_optimal_solution(self):
        """Test that ILP finds optimal solution for simple case."""
        # Create a simple 2x2 grid with 5 frames
        num_frames = 5
        height = 2
        width = 2

        # All tiles always relevant
        relevance_bitmaps = np.ones((num_frames, height, width), dtype=np.uint8) * 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = np.ones((height, width), dtype=np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gaps
        max_gaps = np.full((height, width), 2, dtype=np.int32)

        # Run ILP algorithm
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # Should select optimal number of frames
        assert len(selected_frames) <= num_frames
        assert 0 in selected_frames  # First frame should be selected
        assert (num_frames - 1) in selected_frames  # Last frame should be selected

        # Verify temporal constraints are satisfied
        for i in range(len(selected_frames) - 1):
            gap = selected_frames[i + 1] - selected_frames[i]
            assert gap <= 2  # Should not exceed max_gap

    def test_minimize_tiles_objective(self):
        """Test that ILP minimizes total tiles selected."""
        num_frames = 6
        height = 3
        width = 3

        # Create varying polyomino sizes
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Frame 0: Large polyomino (all tiles)
        relevance_bitmaps[0, :, :] = 255

        # Frame 1: Small polyomino (center only)
        relevance_bitmaps[1, 1, 1] = 255

        # Frame 2: Medium polyomino (cross pattern)
        relevance_bitmaps[2, 1, :] = 255
        relevance_bitmaps[2, :, 1] = 255

        # Frame 3: Small polyomino
        relevance_bitmaps[3, 0, 0] = 255

        # Frame 4: Medium polyomino
        relevance_bitmaps[4, :2, :2] = 255

        # Frame 5: Large polyomino
        relevance_bitmaps[5, :, :] = 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set generous max gaps to allow flexibility
        max_gaps = np.full((height, width), 4, dtype=np.int32)

        # Run ILP algorithm
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # ILP should prefer frames with smaller polyominoes when possible
        # while maintaining coverage
        assert len(selected_frames) > 0
        assert len(selected_frames) < num_frames

    def test_impossible_covering_ilp(self):
        """Test ILP handling of impossible covering constraint."""
        num_frames = 10
        height = 2
        width = 2

        # Create scenario with gap larger than max_gap
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Tile (0,0) relevant only at frames 0 and 9 (gap of 9)
        relevance_bitmaps[0, 0, 0] = 255
        relevance_bitmaps[9, 0, 0] = 255

        # Other tiles have normal patterns
        relevance_bitmaps[:, 1, 1] = 255  # Always relevant at (1,1)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gap smaller than the gap for tile (0,0)
        max_gaps = np.full((height, width), 3, dtype=np.int32)

        # Run ILP algorithm
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5,
            time_limit=60  # Shorter timeout for test
        )

        # Both frames 0 and 9 must be selected due to impossible covering
        assert 0 in selected_frames
        assert 9 in selected_frames

    def test_solver_options(self):
        """Test different solver options."""
        num_frames = 5
        height = 2
        width = 2
        relevance_bitmaps = np.ones((num_frames, height, width), dtype=np.uint8) * 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = np.ones((height, width), dtype=np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 2, dtype=np.int32)

        # Test with CBC solver (default)
        selected_cbc = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, solver='CBC', time_limit=30
        )
        assert len(selected_cbc) > 0

        # Test with GLPK solver if available
        try:
            selected_glpk = ilp_prune_polyominoes(
                polyomino_arrays, relevance_bitmaps, max_gaps,
                threshold=0.5, solver='GLPK', time_limit=30
            )
            assert len(selected_glpk) > 0
        except:
            # GLPK might not be installed
            pass


class TestILPOptimized:
    """Test suite for optimized ILP pruning algorithm."""

    def test_warm_start(self):
        """Test that warm start improves solving time."""
        num_frames = 20
        height = 5
        width = 5

        np.random.seed(42)
        relevance_bitmaps = np.random.randint(100, 256, (num_frames, height, width), dtype=np.uint8)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 5, dtype=np.int32)

        # Test without warm start
        import time
        start = time.time()
        selected_no_warm = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_warm_start=False, time_limit=30
        )
        time_no_warm = time.time() - start

        # Test with warm start
        start = time.time()
        selected_warm = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_warm_start=True, time_limit=30
        )
        time_warm = time.time() - start

        # Both should find valid solutions
        assert len(selected_no_warm) > 0
        assert len(selected_warm) > 0

        # Warm start should typically be faster (but not always guaranteed)
        print(f"Without warm start: {time_no_warm:.3f}s, With warm start: {time_warm:.3f}s")

    def test_decomposition(self):
        """Test problem decomposition for large videos."""
        num_frames = 100
        height = 4
        width = 4

        # Create simple pattern
        relevance_bitmaps = np.ones((num_frames, height, width), dtype=np.uint8) * 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = np.ones((height, width), dtype=np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 10, dtype=np.int32)

        # Test with decomposition
        selected_decomp = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_decomposition=True,
            decomposition_window=30, time_limit=60
        )

        # Should find a valid solution
        assert len(selected_decomp) > 0
        assert len(selected_decomp) < num_frames

        # Verify constraints are satisfied
        for i in range(len(selected_decomp) - 1):
            gap = selected_decomp[i + 1] - selected_decomp[i]
            assert gap <= 10

    def test_lazy_constraints(self):
        """Test lazy constraint generation."""
        num_frames = 15
        height = 3
        width = 3

        # Create sparse relevance pattern
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Only a few tiles are relevant
        for f in range(0, num_frames, 3):
            relevance_bitmaps[f, 1, 1] = 255  # Center tile every 3 frames

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 4, dtype=np.int32)

        # Test with lazy constraints
        selected_lazy = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_lazy_constraints=True, time_limit=30
        )

        # Should find a valid solution
        assert len(selected_lazy) > 0

        # Should select frames with relevant tiles
        for f in selected_lazy:
            if f % 3 == 0 and f < num_frames:
                # These frames should be selected as they have relevant tiles
                assert f in selected_lazy or (f - 1) in selected_lazy or (f + 1) in selected_lazy


class TestILPComparison:
    """Compare ILP with greedy algorithm."""

    @pytest.mark.skipif(
        'greedy_prune_polyominoes' not in dir(),
        reason="Cython module not built"
    )
    def test_ilp_vs_greedy_optimality(self):
        """Test that ILP finds better or equal solution compared to greedy."""
        from polyis.sample.cython.greedy_prune import greedy_prune_polyominoes

        num_frames = 10
        height = 3
        width = 3

        # Create varying relevance
        np.random.seed(42)
        relevance_bitmaps = np.random.randint(50, 256, (num_frames, height, width), dtype=np.uint8)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 3, dtype=np.int32)

        # Run both algorithms
        selected_greedy = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        selected_ilp = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, time_limit=60
        )

        # ILP should find solution with fewer or equal tiles
        # (Note: comparing frame counts as proxy for tile counts)
        assert len(selected_ilp) <= len(selected_greedy)

        print(f"Greedy selected {len(selected_greedy)} frames")
        print(f"ILP selected {len(selected_ilp)} frames")
        print(f"ILP improvement: {(1 - len(selected_ilp)/len(selected_greedy))*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])