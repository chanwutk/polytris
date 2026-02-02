"""
Integration tests for the pruning pipeline.

Tests the complete flow from classification results to pruned frames.
"""

import pytest
import numpy as np
import json
import os
import tempfile
import shutil
import sys

# Add project root to path
sys.path.insert(0, '/work/cwkt/projects/polyis')

# Import modules after path is set
from polyis.sample import ilp_prune_polyominoes, ilp_prune_polyominoes_optimized
from polyis.pack import group_tiles, free_polyomino_array


class TestPruneIntegration:
    """End-to-end integration tests for pruning pipeline."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_mock_classification_output(self, num_frames, height, width, relevance_pattern):
        """
        Create mock classification output files similar to p020_exec_classify.py.

        Parameters:
            num_frames: Number of frames
            height: Grid height in tiles
            width: Grid width in tiles
            relevance_pattern: Function that returns relevance for (frame, row, col)
        """
        # Create directory structure
        score_dir = os.path.join(self.test_dir, '020_relevancy', 'SimpleCNN_30', 'score')
        os.makedirs(score_dir, exist_ok=True)

        score_path = os.path.join(score_dir, 'score.jsonl')

        # Write mock classification results
        with open(score_path, 'w') as f:
            for frame_idx in range(num_frames):
                # Create relevance grid
                relevance_grid = np.zeros((height, width), dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        relevance_grid[i, j] = relevance_pattern(frame_idx, i, j)

                # Create frame entry
                entry = {
                    'classification_size': [height, width],
                    'classification_hex': relevance_grid.flatten().tobytes().hex(),
                    'idx': frame_idx
                }
                f.write(json.dumps(entry) + '\n')

        return score_path

    def test_full_pipeline_constant_relevance(self):
        """Test complete pipeline with constant relevance pattern."""
        num_frames = 20
        height = 5
        width = 5

        # Create constant relevance pattern
        def relevance_pattern(f, i, j):
            return 255  # All tiles always relevant

        score_path = self.create_mock_classification_output(
            num_frames, height, width, relevance_pattern
        )

        # Load classification results
        frames_data = []
        with open(score_path, 'r') as f:
            for line in f:
                frames_data.append(json.loads(line))

        # Convert to relevance bitmaps
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i, frame_data in enumerate(frames_data):
            hex_data = frame_data['classification_hex']
            flat_scores = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
            relevance_bitmaps[i] = flat_scores.reshape(height, width)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gaps
        max_gaps = np.full((height, width), 5, dtype=np.int32)

        # Run ILP pruning
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5, time_limit=30
        )

        # Clean up polyomino arrays
        for poly_ptr in polyomino_arrays:
            free_polyomino_array(poly_ptr)

        # Verify results
        assert len(selected_frames) > 0
        assert len(selected_frames) < num_frames  # Should achieve some reduction
        assert 0 in selected_frames  # First frame should be selected
        assert (num_frames - 1) in selected_frames  # Last frame should be selected

        # Verify temporal constraints
        for i in range(len(selected_frames) - 1):
            gap = selected_frames[i + 1] - selected_frames[i]
            assert gap <= 5  # Should respect max_gap

    def test_full_pipeline_sparse_relevance(self):
        """Test complete pipeline with sparse relevance pattern."""
        num_frames = 30
        height = 6
        width = 6

        # Create sparse relevance pattern
        def relevance_pattern(f, i, j):
            # Checkerboard pattern that changes over time
            if (f + i + j) % 3 == 0:
                return 255
            return 0

        score_path = self.create_mock_classification_output(
            num_frames, height, width, relevance_pattern
        )

        # Load classification results
        frames_data = []
        with open(score_path, 'r') as f:
            for line in f:
                frames_data.append(json.loads(line))

        # Convert to relevance bitmaps
        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i, frame_data in enumerate(frames_data):
            hex_data = frame_data['classification_hex']
            flat_scores = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
            relevance_bitmaps[i] = flat_scores.reshape(height, width)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        # Set max gaps
        max_gaps = np.full((height, width), 3, dtype=np.int32)

        # Run optimized ILP pruning
        selected_frames = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_warm_start=True, time_limit=30
        )

        # Clean up polyomino arrays
        for poly_ptr in polyomino_arrays:
            free_polyomino_array(poly_ptr)

        # Verify results
        assert len(selected_frames) > 0
        assert len(selected_frames) <= num_frames

        # Calculate statistics
        reduction_percentage = 100.0 * (1 - len(selected_frames) / num_frames)
        print(f"Sparse pattern: Selected {len(selected_frames)}/{num_frames} frames")
        print(f"Reduction: {reduction_percentage:.1f}%")

    def test_output_format_compatibility(self):
        """Test that output format is compatible with downstream pipeline."""
        num_frames = 10
        height = 3
        width = 3

        # Create simple relevance pattern
        def relevance_pattern(f, i, j):
            return 200 if f % 2 == 0 else 100

        score_path = self.create_mock_classification_output(
            num_frames, height, width, relevance_pattern
        )

        # Load and process classification results
        frames_data = []
        with open(score_path, 'r') as f:
            for line in f:
                frames_data.append(json.loads(line))

        relevance_bitmaps = np.zeros((num_frames, height, width), dtype=np.uint8)
        frame_indices = []
        for i, frame_data in enumerate(frames_data):
            frame_indices.append(frame_data['idx'])
            hex_data = frame_data['classification_hex']
            flat_scores = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8)
            relevance_bitmaps[i] = flat_scores.reshape(height, width)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 4, dtype=np.int32)

        # Run pruning
        selected_frames = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5, time_limit=30
        )

        # Clean up
        for poly_ptr in polyomino_arrays:
            free_polyomino_array(poly_ptr)

        # Create output in expected format
        output_dir = os.path.join(self.test_dir, '022_pruned', 'SimpleCNN_30_none')
        os.makedirs(output_dir, exist_ok=True)

        # Save selected frames
        selected_frames_path = os.path.join(output_dir, 'selected_frames.jsonl')
        with open(selected_frames_path, 'w') as f:
            for frame_idx in selected_frames:
                entry = {
                    'frame_idx': frame_idx,
                    'original_idx': frame_indices[frame_idx]
                }
                f.write(json.dumps(entry) + '\n')

        # Save statistics
        statistics = {
            'num_frames_original': num_frames,
            'num_frames_selected': len(selected_frames),
            'reduction_percentage': 100.0 * (1 - len(selected_frames) / num_frames)
        }
        statistics_path = os.path.join(output_dir, 'statistics.json')
        with open(statistics_path, 'w') as f:
            json.dump(statistics, f, indent=2)

        # Verify output files exist and are valid
        assert os.path.exists(selected_frames_path)
        assert os.path.exists(statistics_path)

        # Verify we can read the output
        with open(selected_frames_path, 'r') as f:
            loaded_frames = [json.loads(line) for line in f]
            assert len(loaded_frames) == len(selected_frames)

        with open(statistics_path, 'r') as f:
            loaded_stats = json.load(f)
            assert loaded_stats['num_frames_selected'] == len(selected_frames)

    def test_algorithm_comparison(self):
        """Compare greedy and ILP algorithms on same input."""
        try:
            from polyis.sample.cython.greedy_prune import greedy_prune_polyominoes
        except ImportError:
            pytest.skip("Cython module not built")

        num_frames = 15
        height = 4
        width = 4

        # Create varying relevance pattern
        np.random.seed(42)
        relevance_bitmaps = np.random.randint(0, 256, (num_frames, height, width), dtype=np.uint8)

        # Create polyomino arrays
        polyomino_arrays = []
        for f in range(num_frames):
            bitmap = (relevance_bitmaps[f] >= 127).astype(np.uint8)
            poly_ptr = group_tiles(bitmap, mode=0)
            polyomino_arrays.append(poly_ptr)

        max_gaps = np.full((height, width), 4, dtype=np.int32)

        # Run both algorithms
        selected_greedy = greedy_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5
        )

        selected_ilp = ilp_prune_polyominoes(
            polyomino_arrays, relevance_bitmaps, max_gaps, threshold=0.5, time_limit=30
        )

        selected_ilp_opt = ilp_prune_polyominoes_optimized(
            polyomino_arrays, relevance_bitmaps, max_gaps,
            threshold=0.5, use_warm_start=True, time_limit=30
        )

        # Clean up
        for poly_ptr in polyomino_arrays:
            free_polyomino_array(poly_ptr)

        # All should find valid solutions
        assert len(selected_greedy) > 0
        assert len(selected_ilp) > 0
        assert len(selected_ilp_opt) > 0

        # ILP should be optimal (fewer or equal frames)
        assert len(selected_ilp) <= len(selected_greedy)
        assert len(selected_ilp_opt) <= len(selected_greedy)

        print(f"\nAlgorithm comparison:")
        print(f"Greedy: {len(selected_greedy)} frames")
        print(f"ILP: {len(selected_ilp)} frames")
        print(f"ILP Optimized: {len(selected_ilp_opt)} frames")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])