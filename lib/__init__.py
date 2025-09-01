"""
Polyis Performance Library

This package provides optimized implementations of performance-critical functions
from the Polyis project.

Available optimizations:
- pack_append: Cython-optimized polyomino packing (54x speedup)
"""

from .pack_append import pack_append

__version__ = "0.1.0"
__all__ = ["pack_append"]
