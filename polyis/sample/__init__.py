"""
Polyomino sampling and pruning module.

This module provides algorithms for temporal sampling of polyominoes
to reduce the number of frames processed while maintaining detection coverage.
"""

from .ilp_prune import ilp_prune_polyominoes
from .ilp_prune_optimized import ilp_prune_polyominoes_optimized

# Import Cython module when available
try:
    from .cython.greedy_prune import greedy_prune_polyominoes
except ImportError:
    # Cython module not yet built
    greedy_prune_polyominoes = None
    import warnings
    warnings.warn("Cython greedy_prune module not built. Run setup.py build_ext to compile.")

__all__ = ['greedy_prune_polyominoes', 'ilp_prune_polyominoes', 'ilp_prune_polyominoes_optimized']