#!/usr/bin/env python3
"""
Standalone benchmark script for comparing Python vs Cython implementations.
"""

import time
import numpy as np
import sys
import os

# Add the lib directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_test_bitmap(shape, density=0.3, seed=42):
    """Generate a test bitmap with specified density of 1s."""
    np.random.seed(seed)
    bitmap = np.random.random(shape) < density
    return bitmap.astype(np.uint8)

def generate_test_polyominoes(num_polyominoes=5, max_size=3):
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
        
        polyominoes.append((i + 1, mask, offset))
    
    return polyominoes

def benchmark_group_tiles():
    """Benchmark group_tiles implementations."""
    print("=== Group Tiles Benchmark ===")
    
    try:
        from group_tiles import group_tiles as cython_group_tiles
        from group_tiles_original import group_tiles as python_group_tiles
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return
    
    sizes = [(20, 20), (50, 50), (100, 100)]
    densities = [0.1, 0.3, 0.5]
    
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
            cython_result = cython_group_tiles(bitmap.copy())
            cython_time = time.perf_counter() - start
            
            speedup = python_time / cython_time if cython_time > 0 else float('inf')
            
            print(f"{h}x{w}\t\t{density:.1f}\t{python_time:.6f}\t\t{cython_time:.6f}\t\t{speedup:.2f}x")

def benchmark_pack_append():
    """Benchmark pack_append implementations."""
    print("\n=== Pack Append Benchmark ===")
    
    try:
        from pack_append import pack_append as cython_pack_append
        from pack_append_original import pack_append as python_pack_append
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return
    
    scenarios = [
        (20, 20, 5, 2),
        (50, 50, 10, 3),
        (100, 100, 20, 4),
    ]
    
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

def main():
    """Run all benchmarks."""
    print("Performance Comparison: Python vs Cython")
    print("=" * 50)
    
    benchmark_group_tiles()
    benchmark_pack_append()
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
