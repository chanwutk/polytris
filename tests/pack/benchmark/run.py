#!/usr/bin/env python3
"""
Standalone benchmark script for comparing Python vs Cython implementations.
"""

# Import benchmark functions from split modules
import sys
import os
# Add workspace root to path for imports
_workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from tests.pack.benchmark.group_tiles import benchmark_group_tiles
from tests.pack.benchmark.pack_append import benchmark_pack_append
from tests.pack.benchmark.pack_ffd import benchmark_pack_ffd
from tests.pack.benchmark.compress import benchmark_compress


def main():
    """Run all benchmarks."""
    print("Performance Comparison: Python vs Cython vs C")
    print("=" * 50)

    benchmark_group_tiles()
    benchmark_pack_append()
    benchmark_pack_ffd()
    benchmark_compress()

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
