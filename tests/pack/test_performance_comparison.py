#!/usr/bin/env python3
"""
Performance and correctness tests comparing Cython implementations 
against original Python implementations.
"""

import time
import numpy as np
import pytest
from typing import List, Tuple, Optional, Any
import sys
import os

# Import Cython modules
from polyis.pack.adapters import group_tiles as cython_group_tiles
from polyis.pack.adapters import pack_append as cython_pack_append
CYTHON_AVAILABLE = True

# Import original Python implementations from polyis.binpack
from group_tiles_original import group_tiles as python_group_tiles
from pack_append_original import pack_append as python_pack_append


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        assert self.start_time is not None
        self.duration = self.end_time - self.start_time
        print(f"{self.name}: {self.duration:.6f} seconds")
    
    def get_duration(self) -> float:
        return self.duration or 0.0


def generate_test_bitmap(shape: Tuple[int, int], density: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate a test bitmap with specified density of 1s."""
    np.random.seed(seed)
    bitmap = np.random.random(shape) < density
    return bitmap.astype(np.uint8)


def generate_test_polyominoes(num_polyominoes: int = 5, max_size: int = 3) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Generate test polyominoes for pack_append testing."""
    polyominoes = []
    
    for i in range(num_polyominoes):
        # Generate random mask
        mask_h = np.random.randint(1, max_size + 1)
        mask_w = np.random.randint(1, max_size + 1)
        mask = np.random.randint(0, 2, (mask_h, mask_w), dtype=np.uint8)
        
        # Ensure at least one tile is occupied
        if mask.sum() == 0:
            mask[0, 0] = 1
        
        # Random offset
        offset = (np.random.randint(0, 3), np.random.randint(0, 3))
        
        polyominoes.append((mask, offset))
    
    return polyominoes


def compare_group_tiles_results(cython_result: List, python_result: List) -> bool:
    """Compare results from group_tiles implementations."""
    if len(cython_result) != len(python_result):
        return False
    
    # Sort both results by offset for comparison
    cython_sorted = sorted(cython_result, key=lambda x: (x[1][0], x[1][1], x[0].sum()))
    python_sorted = sorted(python_result, key=lambda x: (x[1][0], x[1][1], x[0].sum()))
    
    for cython_item, python_item in zip(cython_sorted, python_sorted):
        # Compare masks
        if not np.array_equal(cython_item[0], python_item[0]):
            return False
        
        # Compare offsets
        if cython_item[1] != python_item[1]:
            return False
    
    return True


def compare_pack_append_results(cython_result: Optional[List], python_result: Optional[List]) -> bool:
    """Compare results from pack_append implementations."""
    # Both should be None or both should be lists
    if cython_result is None and python_result is None:
        return True
    
    if cython_result is None or python_result is None:
        return False
    
    if len(cython_result) != len(python_result):
        return False
    
    # Compare each position tuple
    for cython_pos, python_pos in zip(cython_result, python_result):
        # Compare first 2 elements (i, j)
        if cython_pos[:2] != python_pos[:2]:
            return False
        
        # Compare masks
        cython_mask = np.zeros_like(python_pos[2])
        cython_mask[:cython_pos[2].shape[0], :cython_pos[2].shape[1]] = cython_pos[2]
        if not np.array_equal(cython_mask, python_pos[2]):
            return False
        
        # Compare offsets
        if cython_pos[3] != python_pos[3]:
            return False
    
    return True


# @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestGroupTilesPerformance:
    """Test group_tiles performance and correctness."""
    
    def test_small_bitmap_correctness(self):
        """Test correctness with small bitmap."""
        bitmap = generate_test_bitmap((10, 10), density=0.4, seed=123)
        
        with PerformanceTimer("Python group_tiles (small)"):
            python_result = python_group_tiles(bitmap.copy())
        
        with PerformanceTimer("Cython group_tiles (small)"):
            cython_result = cython_group_tiles(bitmap.copy(), 0)
        
        assert compare_group_tiles_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_medium_bitmap_correctness(self):
        """Test correctness with medium bitmap."""
        bitmap = generate_test_bitmap((50, 50), density=0.3, seed=456)
        
        with PerformanceTimer("Python group_tiles (medium)"):
            python_result = python_group_tiles(bitmap.copy())
        
        with PerformanceTimer("Cython group_tiles (medium)"):
            cython_result = cython_group_tiles(bitmap.copy(), 0)
        
        assert compare_group_tiles_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_large_bitmap_correctness(self):
        """Test correctness with large bitmap."""
        bitmap = generate_test_bitmap((100, 100), density=0.2, seed=789)
        
        with PerformanceTimer("Python group_tiles (large)"):
            python_result = python_group_tiles(bitmap.copy())
        
        with PerformanceTimer("Cython group_tiles (large)"):
            cython_result = cython_group_tiles(bitmap.copy(), 0)
        
        assert compare_group_tiles_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_empty_bitmap(self):
        """Test with empty bitmap."""
        bitmap = np.zeros((10, 10), dtype=np.uint8)
        
        python_result = python_group_tiles(bitmap.copy())
        cython_result = cython_group_tiles(bitmap.copy(), 0)
        
        assert compare_group_tiles_results(cython_result, python_result)
        assert len(cython_result) == 0
    
    def test_full_bitmap(self):
        """Test with fully occupied bitmap."""
        bitmap = np.ones((5, 5), dtype=np.uint8)
        
        python_result = python_group_tiles(bitmap.copy())
        cython_result = cython_group_tiles(bitmap.copy(), 0)
        
        assert compare_group_tiles_results(cython_result, python_result)
        assert len(cython_result) == 1  # Should be one large group
    
    def test_performance_benchmark(self):
        """Benchmark performance across different sizes."""
        sizes = [(20, 20), (50, 50), (100, 100)]
        densities = [0.1, 0.3, 0.5]
        
        print("\n=== Group Tiles Performance Benchmark ===")
        print("Size\t\tDensity\tPython (s)\tCython (s)\tSpeedup")
        print("-" * 60)
        
        for h, w in sizes:
            for density in densities:
                bitmap = generate_test_bitmap((h, w), density=density)
                
                # Time Python implementation
                start = time.perf_counter()
                python_result = python_group_tiles(bitmap.copy())
                python_time = time.perf_counter() - start
                
                # Time Cython implementation
                start = time.perf_counter()
                cython_result = cython_group_tiles(bitmap.copy(), 0)
                cython_time = time.perf_counter() - start
                
                speedup = python_time / cython_time if cython_time > 0 else float('inf')
                
                print(f"{h}x{w}\t\t{density:.1f}\t{python_time:.6f}\t\t{cython_time:.6f}\t\t{speedup:.2f}x")
                
                # Verify correctness
                assert compare_group_tiles_results(cython_result, python_result)


# @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython modules not available")
class TestPackAppendPerformance:
    """Test pack_append performance and correctness."""
    
    def test_small_packing_correctness(self):
        """Test correctness with small packing scenario."""
        h, w = 20, 20
        polyominoes = generate_test_polyominoes(3, max_size=2)
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        with PerformanceTimer("Python pack_append (small)"):
            python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        with PerformanceTimer("Cython pack_append (small)"):
            cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        assert compare_pack_append_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_medium_packing_correctness(self):
        """Test correctness with medium packing scenario."""
        h, w = 50, 50
        polyominoes = generate_test_polyominoes(8, max_size=3)
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        with PerformanceTimer("Python pack_append (medium)"):
            python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        with PerformanceTimer("Cython pack_append (medium)"):
            cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        assert compare_pack_append_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_large_packing_correctness(self):
        """Test correctness with large packing scenario."""
        h, w = 100, 100
        polyominoes = generate_test_polyominoes(15, max_size=4)
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        with PerformanceTimer("Python pack_append (large)"):
            python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        with PerformanceTimer("Cython pack_append (large)"):
            cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        assert compare_pack_append_results(cython_result, python_result), \
            "Results don't match between Python and Cython implementations"
    
    def test_packing_failure(self):
        """Test packing failure scenario."""
        h, w = 5, 5
        # Create polyominoes that are too large to fit
        polyominoes = [
            (np.ones((3, 3), dtype=np.uint8), (0, 0)),
            (np.ones((3, 3), dtype=np.uint8), (0, 0))
        ]
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        assert compare_pack_append_results(cython_result, python_result)
        assert cython_result is None  # Should fail to pack
    
    def test_empty_polyominoes(self):
        """Test with empty polyominoes list."""
        h, w = 10, 10
        polyominoes = []
        occupied_tiles = np.zeros((h, w), dtype=np.uint8)
        
        python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
        
        assert compare_pack_append_results(cython_result, python_result)
        assert cython_result == []
    
    def test_performance_benchmark(self):
        """Benchmark performance across different scenarios."""
        scenarios = [
            (20, 20, 5, 2),
            (50, 50, 10, 3),
            (100, 100, 20, 4),
        ]
        
        print("\n=== Pack Append Performance Benchmark ===")
        print("Size\t\tPolyominoes\tMax Size\tPython (s)\tCython (s)\tSpeedup")
        print("-" * 80)
        
        for h, w, num_poly, max_size in scenarios:
            polyominoes = generate_test_polyominoes(num_poly, max_size)
            occupied_tiles = np.zeros((h, w), dtype=np.uint8)
            
            # Time Python implementation
            start = time.perf_counter()
            python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
            python_time = time.perf_counter() - start
            
            # Time Cython implementation
            start = time.perf_counter()
            cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
            cython_time = time.perf_counter() - start
            
            speedup = python_time / cython_time if cython_time > 0 else float('inf')
            
            print(f"{h}x{w}\t\t{num_poly}\t\t{max_size}\t\t{python_time:.6f}\t\t{cython_time:.6f}\t\t{speedup:.2f}x")
            
            # Verify correctness
            assert compare_pack_append_results(cython_result, python_result)


def run_manual_benchmark():
    """Run a manual benchmark for quick testing."""
    # if not CYTHON_AVAILABLE: # pragma: no cover
    #     print("Cython modules not available. Please build them first.")
    #     return
    
    print("Running manual benchmark...")
    
    # Test group_tiles
    print("\n=== Group Tiles Manual Test ===")
    bitmap = generate_test_bitmap((30, 30), density=0.3, seed=42)
    
    with PerformanceTimer("Python group_tiles"):
        python_result = python_group_tiles(bitmap.copy())
    
    with PerformanceTimer("Cython group_tiles"):
        cython_result = cython_group_tiles(bitmap.copy())
    
    print(f"Python found {len(python_result)} groups")
    print(f"Cython found {len(cython_result)} groups")
    print(f"Results match: {compare_group_tiles_results(cython_result, python_result)}")
    
    # Test pack_append
    print("\n=== Pack Append Manual Test ===")
    h, w = 30, 30
    polyominoes = generate_test_polyominoes(5, max_size=3)
    occupied_tiles = np.zeros((h, w), dtype=np.uint8)
    
    with PerformanceTimer("Python pack_append"):
        python_result = python_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
    
    with PerformanceTimer("Cython pack_append"):
        cython_result = cython_pack_append(polyominoes.copy(), h, w, occupied_tiles.copy())
    
    print(f"Python packed {len(python_result) if python_result else 0} polyominoes")
    print(f"Cython packed {len(cython_result) if cython_result else 0} polyominoes")
    print(f"Results match: {compare_pack_append_results(cython_result, python_result)}")


if __name__ == "__main__":
    run_manual_benchmark()
