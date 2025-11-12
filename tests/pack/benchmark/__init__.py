#!/usr/bin/env python3
"""
Benchmark utilities for comparing Python vs Cython vs C implementations.
"""

import numpy as np


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
        
        polyominoes.append((mask, offset))
    
    return polyominoes
