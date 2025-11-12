#!/usr/bin/env python3
"""
Benchmark script for group_tiles implementations.
"""

import time
import numpy as np

from . import generate_test_bitmap


def benchmark_group_tiles():
    """Benchmark group_tiles implementations."""
    print("=== Group Tiles Benchmark ===")

    from polyis.pack.cython.group_tiles import group_tiles as group_tiles_cython
    from polyis.pack.group_tiles import group_tiles as group_tiles_c
    from polyis.pack.python.group_tiles import group_tiles as python_group_tiles

    sizes = [(20, 20), (50, 50), (100, 100)]
    densities = [0.1, 0.3, 0.5]

    print("Size\t\tDensity\tPython (s)\tCython (s)\tC (s)\t\tCy/Py\tC/Py\tC/Cy")
    print("-" * 100)

    for h, w in sizes:
        for density in densities:
            bitmap = generate_test_bitmap((h, w), density=density)

            # Time Python implementation
            start = time.perf_counter()
            python_result = python_group_tiles(bitmap.copy())
            python_time = time.perf_counter() - start

            # Time Cython implementation
            start = time.perf_counter()
            cython_result = group_tiles_cython(bitmap.copy(), 0)
            cython_time = time.perf_counter() - start

            # Time C implementation
            start = time.perf_counter()
            c_result = group_tiles_c(bitmap.copy(), 0)
            c_time = time.perf_counter() - start

            # Verify results match
            # assert len(python_result) == len(cython_result) == len(c_result), \
            #     f"Result mismatch: Python={len(python_result)}, Cython={len(cython_result)}, C={len(c_result)}"

            speedup_cy = python_time / cython_time if cython_time > 0 else float('inf')
            speedup_c = python_time / c_time if c_time > 0 else float('inf')
            speedup_c_vs_cy = cython_time / c_time if c_time > 0 else float('inf')

            print(f"{h}x{w}\t\t{density:.1f}\t{python_time:.6f}\t{cython_time:.6f}\t{c_time:.6f}\t{speedup_cy:.2f}x\t{speedup_c:.2f}x\t{speedup_c_vs_cy:.2f}x")
