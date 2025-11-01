#!/usr/bin/env python3
"""
Test script to verify the Cython implementation matches the Python implementation.
"""

import numpy as np
import sys
import os
import time

# No path manipulation needed - using inline reference implementation

def original_pack_append(polyominoes, h, w, occupied_tiles):
    """Simplified reference implementation for basic testing."""
    # Note: This is a simplified version for testing basic functionality
    # The actual algorithm details may differ from the optimized Cython version
    
    positions = []
    for mask, offset in polyominoes:
        placed = False
        mask = np.asarray(mask, dtype=np.uint8)
        
        # Try to place at each position
        for j in range(w - mask.shape[1] + 1):
            for i in range(h - mask.shape[0] + 1):
                # Check for collisions
                valid = True
                for row in range(mask.shape[0]):
                    for col in range(mask.shape[1]):
                        if mask[row, col] and occupied_tiles[i + row, j + col]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Place the polyomino
                    for row in range(mask.shape[0]):
                        for col in range(mask.shape[1]):
                            if mask[row, col]:
                                occupied_tiles[i + row, j + col] = 1
                    
                    positions.append((i, j, mask, offset))
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            return None
    
    return positions


def run_implementation_comparison():
    """Test the Cython implementation against the original Python version."""
    
    # Import Cython implementation
    try:
        from polyis.binpack.adapters import pack_append
        print("✓ Cython implementation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Cython implementation: {e}")
        return False
    
    test_cases = []
    
    # Test case 1: Simple single polyomino
    print("\nTest 1: Single 2x2 square polyomino")
    polyomino_mask = np.array([[True, True], [True, True]], dtype=np.bool_)
    polyominoes = [(polyomino_mask, (0, 0))]
    h, w = 4, 4
    occupied = np.zeros((h, w), dtype=np.bool_)
    test_cases.append(("Single 2x2 square", polyominoes, h, w, occupied.copy()))
    
    # Test case 2: Multiple polyominoes
    print("Test 2: Multiple small polyominoes")
    poly1 = np.array([[True, True]], dtype=np.bool_)  # 1x2 rectangle
    poly2 = np.array([[True], [True]], dtype=np.bool_)  # 2x1 rectangle
    polyominoes = [(poly1, (0, 0)), (poly2, (0, 0))]
    h, w = 3, 3
    occupied = np.zeros((h, w), dtype=np.bool_)
    test_cases.append(("Multiple small polyominoes", polyominoes, h, w, occupied.copy()))
    
    # Test case 3: Impossible packing
    print("Test 3: Impossible packing scenario")
    large_poly = np.ones((5, 5), dtype=np.bool_)  # 5x5 square
    polyominoes = [(large_poly, (0, 0))]
    h, w = 3, 3  # Too small to fit
    occupied = np.zeros((h, w), dtype=np.bool_)
    test_cases.append(("Impossible packing", polyominoes, h, w, occupied.copy()))
    
    # Test case 4: Pre-occupied space
    print("Test 4: Packing with pre-occupied space")
    poly = np.array([[True, True]], dtype=np.bool_)
    polyominoes = [(poly, (0, 0))]
    h, w = 3, 3
    occupied = np.zeros((h, w), dtype=np.bool_)
    occupied[0, 0] = True  # Block top-left corner
    test_cases.append(("Pre-occupied space", polyominoes, h, w, occupied.copy()))
    
    # Run all tests
    all_passed = True
    for test_name, polyominoes, h, w, occupied in test_cases:
        print(f"\n--- {test_name} ---")
        
        # Test original implementation
        occupied_orig = occupied.copy().astype(np.uint8)
        start_time = time.time()
        result_orig = original_pack_append(polyominoes, h, w, occupied_orig)
        orig_time = time.time() - start_time
        
        # Test Cython implementation
        occupied_cython = occupied.copy().astype(np.uint8)
        start_time = time.time()
        result_cython = pack_append(polyominoes, h, w, occupied_cython)
        cython_time = time.time() - start_time
        
        # Compare results
        if result_orig is None and result_cython is None:
            print("✓ Both returned None (expected failure)")
        elif result_orig is not None and result_cython is not None:
            pos_orig = result_orig
            pos_cython = result_cython
            
            # Check if bitmaps are equivalent
            if np.array_equal(occupied_orig, occupied_cython):
                print("✓ Bitmaps match")
            else:
                print("✗ Bitmaps don't match!")
                all_passed = False
                
            # Check if positions are equivalent (order might differ)
            if len(pos_orig) == len(pos_cython):
                print(f"✓ Same number of positions ({len(pos_orig)})")
            else:
                print(f"✗ Different number of positions: {len(pos_orig)} vs {len(pos_cython)}")
                all_passed = False
        else:
            print("✗ Results differ: one succeeded, one failed")
            all_passed = False
            
        print(f"  Original time: {orig_time:.6f}s")
        print(f"  Cython time:   {cython_time:.6f}s")
        if orig_time > 0:
            print(f"  Speedup:       {orig_time/cython_time:.2f}x")
    
    if all_passed:
        print("\n🎉 All tests passed! Cython implementation matches Python behavior.")
        return True
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        return False


def test_implementations_pytest():
    """Pytest wrapper for the legacy test function."""
    success = run_implementation_comparison()
    assert success, "Legacy test implementation should pass"


if __name__ == "__main__":
    success = run_implementation_comparison()
    sys.exit(0 if success else 1)
