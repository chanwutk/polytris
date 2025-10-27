"""
Pytest tests for the pack_append Cython implementation.
"""

import pytest
import numpy as np
import time
from polyis.binpack.adapters import pack_append


class TestPackAppend:
    """Test class for pack_append functionality."""
    
    def test_single_square_polyomino(self, sample_polyominoes, empty_grid):
        """Test placing a single 2x2 square polyomino."""
        polyominoes = [sample_polyominoes['square']]
        h, w = 4, 4
        occupied_tiles = empty_grid.copy()
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is not None, "Should successfully place single square"
        assert len(positions) == 1, "Should return one position"
        assert positions[0][0] == 0 and positions[0][1] == 0, "Should place at (0,0)"
        
        # Check that the square was placed correctly
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[0:2, 0:2] = 1
        np.testing.assert_array_equal(occupied_tiles, expected)
    
    def test_multiple_polyominoes(self, sample_polyominoes, empty_grid):
        """Test placing multiple polyominoes."""
        polyominoes = [
            sample_polyominoes['rect_h'],  # 1x2 horizontal
            sample_polyominoes['rect_v']   # 2x1 vertical
        ]
        h, w = 4, 4
        occupied_tiles = empty_grid.copy()
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is not None, "Should successfully place multiple polyominoes"
        assert len(positions) == 2, "Should return two positions"
        
        # Verify no overlaps (sum should equal number of True cells)
        total_cells = np.sum(occupied_tiles)
        expected_cells = 2 + 2  # 1x2 + 2x1
        assert total_cells == expected_cells, f"Expected {expected_cells} cells, got {total_cells}"
    
    def test_impossible_packing(self, sample_polyominoes, small_grid):
        """Test scenario where packing is impossible."""
        # Try to place a 2x2 square in a 3x3 grid that's too constrained
        polyominoes = [sample_polyominoes['square']]
        h, w = 3, 3
        occupied_tiles = small_grid.copy()
        
        # Fill most of the grid
        occupied_tiles[0, :] = 1  # Fill top row
        occupied_tiles[1, :2] = 1  # Fill part of second row
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is None, "Should fail to place polyomino in constrained space"
    
    def test_pre_occupied_space(self, sample_polyominoes, empty_grid):
        """Test packing with pre-occupied space."""
        polyominoes = [sample_polyominoes['rect_h']]  # 1x2 horizontal
        h, w = 4, 4
        occupied_tiles = empty_grid.copy()
        occupied_tiles[0, 0] = 1  # Block top-left corner
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is not None, "Should find alternative placement"
        assert len(positions) == 1, "Should place one polyomino"
        
        # Should not place at (0,0) since it's blocked
        pos = positions[0]
        assert not (pos[0] == 0 and pos[1] == 0), "Should not place at blocked position"
    
    def test_edge_placement(self, sample_polyominoes):
        """Test placement at grid edges."""
        polyominoes = [sample_polyominoes['rect_h']]  # 1x2 horizontal
        h, w = 3, 3
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is not None, "Should successfully place at edge"
        pos = positions[0]
        # Verify placement is within bounds
        assert 0 <= pos[0] < h, "Row position should be within bounds"
        assert 0 <= pos[1] <= w - 2, "Col position should allow for 1x2 rectangle"
    
    def test_in_place_modification(self, sample_polyominoes, empty_grid):
        """Test that occupied_tiles is modified in-place."""
        polyominoes = [sample_polyominoes['square']]
        h, w = 4, 4
        occupied_tiles = empty_grid.copy()
        original_id = id(occupied_tiles)
        
        positions = pack_append(polyominoes, h, w, occupied_tiles)
        
        assert positions is not None, "Should successfully place polyomino"
        assert id(occupied_tiles) == original_id, "Should modify array in-place"
        assert np.sum(occupied_tiles) > 0, "Should have placed polyomino in the grid"
    
    def test_collision_detection(self, sample_polyominoes, empty_grid):
        """Test collision detection with existing tiles."""
        h, w = 4, 4
        occupied_tiles = empty_grid.copy()
        
        # Place first polyomino
        polyominoes1 = [sample_polyominoes['square']]
        positions1 = pack_append(polyominoes1, h, w, occupied_tiles)
        assert positions1 is not None, "Should place first polyomino"
        
        # Try to place overlapping polyomino
        polyominoes2 = [sample_polyominoes['square']]  # Same 2x2 square
        positions2 = pack_append(polyominoes2, h, w, occupied_tiles)
        
        # Should either place in a different location or fail
        if positions2 is not None:
            # If successful, should not overlap with first placement
            pos1 = positions1[0]
            pos2 = positions2[0]
            assert not (pos1[0] == pos2[0] and pos1[1] == pos2[1]), "Should not place in same location"


class TestPerformance:
    """Performance tests for pack_append."""
    
    def test_performance_basic(self, sample_polyominoes):
        """Basic performance test."""
        polyominoes = [
            sample_polyominoes['square'],
            sample_polyominoes['rect_h'],
            sample_polyominoes['rect_v'],
        ]
        h, w = 10, 10
        
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        start_time = time.time()
        result = pack_append(polyominoes, h, w, occupied_tiles)
        elapsed_time = time.time() - start_time
        
        assert result is not None, "Should successfully place polyominoes"
        assert len(result) == 3, "Should place all three polyominoes"
        assert elapsed_time < 1.0, "Should complete within reasonable time"
        
        print(f"\nBasic performance test:")
        print(f"  Time: {elapsed_time:.6f}s")
        print(f"  Placed {len(result)} polyominoes")


def test_import_success():
    """Test that the Cython module can be imported successfully."""
    try:
        from polyis.binpack.adapters import pack_append
        assert callable(pack_append), "pack_append should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import Cython implementation: {e}")


def test_return_format():
    """Test the return format of pack_append_fast."""
    # Create simple test case
    polyomino_mask = np.array([[1, 1]], dtype=np.uint8)
    polyominoes = [(polyomino_mask, (0, 0))]
    h, w = 3, 3
    occupied_tiles = np.zeros((h, w), dtype=np.uint8)
    
    result = pack_append(polyominoes, h, w, occupied_tiles)
    
    assert result is not None, "Should return positions list"
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 1, "Should return one position"
    
    position = result[0]
    assert len(position) == 4, "Position should have 4 elements: (i, j, mask, offset)"
    assert isinstance(position[0], int), "Row should be integer"
    assert isinstance(position[1], int), "Column should be integer"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])