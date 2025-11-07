#!/usr/bin/env python3
"""
Benchmark script for pack_append implementations.
"""

import time
import numpy as np

from . import generate_test_polyominoes
from polyis.pack.cython.adapters import get_polyominoes


def benchmark_pack_append():
    """Benchmark pack_append implementations."""
    print("\n=== Pack Append Benchmark ===")
    
    try:
        from polyis.pack.cython.pack_append import pack_append as cython_pack_append
        from polyis.pack.python.pack_append import pack_append as python_pack_append
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
        polyominoes_stack = get_polyominoes(polyominoes.copy())
        start = time.perf_counter()
        cython_result = cython_pack_append(polyominoes_stack, h, w, occupied_tiles.copy())
        cython_time = time.perf_counter() - start
        
        speedup = python_time / cython_time if cython_time > 0 else float('inf')
        
        print(f"{h}x{w}\t\t{num_poly}\t\t{max_size}\t\t{python_time:.6f}\t\t{cython_time:.6f}\t\t{speedup:.2f}x")
