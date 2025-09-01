"""
Polyis Performance Library

This package provides optimized implementations of performance-critical functions
from the Polyis project.

Available optimizations:
- pack_append: Cython-optimized polyomino packing (54x speedup)
- group_tiles: Cython-optimized tile grouping for connected components
"""

from .pack_append import pack_append
from .group_tiles import group_tiles

__version__ = "0.1.0"
__all__ = ["pack_append", "group_tiles"]
