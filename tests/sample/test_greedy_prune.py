"""
Unit tests for the greedy temporal covering algorithm.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, '/work/cwkt/projects/polyis')

# Import modules after path is set
from polyis.sample.cython.greedy_prune import greedy_prune_polyominoes
from polyis.pack import group_tiles


class TestGreedyPrune:
    """Test suite for greedy pruning algorithm."""

    def test_simple_constant_relevance(self):
        """Test with constant relevance across all frames."""
        # Create a simple 3x3 grid with 5 frames, all tiles always relevant
        num_frames = 5
        height = 3
        width = 3
        relevance_bitmaps = np.ones((num_frames, height, width), dtype=np.uint8) * 255

        # Create polyomino arrays (will be single polyomino per frame)
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = np.ones((height, width), dtype=np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set uniform max gaps
        max_gaps = np.full((height, width), 2, dtype=np.int32)

        # Run greedy algorithm
        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # Should select frames at regular intervals
        assert len(selected_frames) <= num_frames
        assert 0 in selected_frames  # First frame should be selected
        assert (num_frames - 1) in selected_frames  # Last frame should be selected

        # Check that gaps are respected
        for i in range(len(selected_frames) - 1):
            gap = selected_frames[i + 1] - selected_frames[i]
            assert gap <= 2  # Should not exceed max_gap

    def test_sparse_relevance(self):
        """Test with sparse relevance patterns."""
        num_frames = 10
        height = 4
        width = 4

        # Create sparse relevance: only some tiles relevant in some frames
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Make center tile relevant in frames 0, 3, 6, 9
        for f in [0, 3, 6, 9]:
            relevance_bitmaps[f, 1:3, 1:3] = 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gaps
        max_gaps = np.full((height, width), 3, dtype=np.int32)

        # Run greedy algorithm
        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # Should select at least the frames with relevant tiles
        assert 0 in selected_frames
        assert 9 in selected_frames

    def test_impossible_covering_constraint(self):
        """Test handling of impossible covering constraint."""
        num_frames = 10
        height = 2
        width = 2

        # Create scenario with gap larger than max_gap
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Tile (0,0) relevant only at frames 0 and 8 (gap of 8)
        relevance_bitmaps[0, 0, 0] = 255
        relevance_bitmaps[8, 0, 0] = 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gap smaller than actual gap
        max_gaps = np.full((height, width), 3, dtype=np.int32)

        # Run greedy algorithm
        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # Both frames 0 and 8 should be selected due to impossible covering
        assert 0 in selected_frames
        assert 8 in selected_frames

    def test_different_max_gaps(self):
        """Test with different max gaps for different tile positions."""
        num_frames = 10
        height = 3
        width = 3

        # All tiles always relevant
        relevance_bitmaps = np.ones((num_frames, height, width), dtype=np.uint8) * 255

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = np.ones((height, width), dtype=np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set varying max gaps
        max_gaps = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ], dtype=np.int32)

        # Run greedy algorithm
        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        # Check that the most restrictive gap (1) is respected
        assert len(selected_frames) >= num_frames // 1  # At least every other frame

    def test_threshold_effects(self):
        """Test effect of different thresholds on selection."""
        num_frames = 5
        height = 2
        width = 2

        # Create varying relevance scores
        relevance_bitmaps = np.array([
            [[100, 200], [150, 255]],  # Frame 0
            [[50, 100], [75, 127]],     # Frame 1
            [[200, 255], [225, 255]],   # Frame 2
            [[25, 50], [40, 60]],       # Frame 3
            [[255, 255], [255, 255]],   # Frame 4
        ], dtype=np.uint8)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            # This will be computed by the algorithm based on threshold
            bitmap = np.ones((height, width), dtype=np.uint8)  # Dummy
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 2, dtype=np.int32)

        # Test with low threshold (more tiles considered relevant)
        selected_low = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.2
        )

        # Test with high threshold (fewer tiles considered relevant)
        selected_high = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.8
        )

        # High threshold should select fewer or equal frames
        assert len(selected_high) <= len(selected_low)


class TestGreedyPrunePerformance:
    """Performance tests for greedy pruning algorithm."""

    def test_large_video_performance(self):
        """Test performance with a larger video."""
        num_frames = 1000
        height = 20
        width = 30

        # Create random relevance
        np.random.seed(42)
        relevance_bitmaps = np.random.randint(0, 256, (num_frames, height, width), dtype=np.uint8)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 30, dtype=np.int32)

        # Measure time
        import time
        start = time.time()
        selected_frames = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )
        end = time.time()

        runtime = end - start
        print(f"Greedy algorithm runtime for {num_frames} frames: {runtime:.3f} seconds")

        # Should complete reasonably fast (< 1 second for 1000 frames)
        assert runtime < 1.0

        # Should achieve significant reduction
        reduction = 1 - len(selected_frames) / num_frames
        print(f"Achieved {reduction:.1%} frame reduction")
        assert reduction > 0.5  # Should reduce by at least 50%


if __name__ == "__main__":
    # Note: This will fail until the Cython module is built
    print("Note: Run 'python setup.py build_ext' first to compile Cython modules")
    pytest.main([__file__, "-v"])