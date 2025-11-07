#!/usr/bin/env python3
"""
Benchmark script for pack_ffd implementations with correctness validation.
"""

import time
import numpy as np
import os
import json
import tqdm
import statistics


def compare_polyomino_positions(python_result, c_result, verbose=False):
    """Compare the results from Python and C implementations.

    Args:
        python_result: List of lists of PolyominoPosition from Python implementation
        c_result: List of lists of PolyominoPosition from C implementation
        verbose: If True, print detailed comparison information

    Returns:
        bool: True if results match (same number of collages and polyominoes)
    """
    # Compare number of collages
    if len(python_result) != len(c_result):
        if verbose:
            print(f"❌ Number of collages differs: Python={len(python_result)}, C={len(c_result)}")
        return False

    # Compare each collage
    for i, (py_collage, c_collage) in enumerate(zip(python_result, c_result)):
        if len(py_collage) != len(c_collage):
            if verbose:
                print(f"❌ Collage {i}: Number of polyominoes differs: Python={len(py_collage)}, C={len(c_collage)}")
            return False

    if verbose:
        print(f"✓ Results match: {len(python_result)} collages")
        for i, collage in enumerate(python_result):
            print(f"  Collage {i}: {len(collage)} polyominoes")

    return True


def print_packing_diagnostics(python_result, c_result, frame_idx, max_polyominoes=5):
    """Print detailed diagnostics about packing results when they differ.

    Args:
        python_result: List of lists of PolyominoPosition from Python implementation
        c_result: List of lists of PolyominoPosition from C implementation
        frame_idx: Frame index for reference
        max_polyominoes: Maximum number of polyominoes to show per collage
    """
    print(f"\n{'='*80}")
    print(f"MISMATCH DIAGNOSTICS - Frame {frame_idx}")
    print(f"{'='*80}")

    print(f"\n📊 High-Level Comparison:")
    print(f"  Python: {len(python_result)} collages, {sum(len(c) for c in python_result)} total polyominoes")
    print(f"  C:      {len(c_result)} collages, {sum(len(c) for c in c_result)} total polyominoes")

    # Compare collage structure
    max_collages = max(len(python_result), len(c_result))
    print(f"\n📦 Collage-by-Collage Breakdown:")
    print(f"  {'Collage':<10} {'Python':<15} {'C':<15} {'Status':<10}")
    print(f"  {'-'*50}")

    for i in range(max_collages):
        py_count = len(python_result[i]) if i < len(python_result) else 0
        c_count = len(c_result[i]) if i < len(c_result) else 0
        status = "✓ Match" if py_count == c_count else "✗ Differ"

        py_str = f"{py_count} polyominoes" if i < len(python_result) else "N/A"
        c_str = f"{c_count} polyominoes" if i < len(c_result) else "N/A"

        print(f"  {i:<10} {py_str:<15} {c_str:<15} {status:<10}")

    # Show detailed polyomino info for first few collages
    print(f"\n🔍 Detailed Polyomino Information (first {max_polyominoes} per collage):")

    for collage_idx in range(min(3, max_collages)):  # Show first 3 collages
        print(f"\n  --- Collage {collage_idx} ---")

        # Python implementation
        if collage_idx < len(python_result):
            print(f"  Python ({len(python_result[collage_idx])} total):")
            for poly_idx, poly in enumerate(python_result[collage_idx][:max_polyominoes]):
                shape_h, shape_w = poly.shape.shape
                tiles = np.sum(poly.shape)
                print(f"    [{poly_idx}] pos=({poly.py:3d},{poly.px:3d}) offset=({poly.oy:3d},{poly.ox:3d}) "
                      f"shape={shape_h}x{shape_w} tiles={tiles} frame={poly.frame}")
                # Print bitmap representation
                for row in poly.shape:
                    bitmap_str = ''.join('█' if cell else '·' for cell in row)
                    print(f"      {bitmap_str}")
        else:
            print(f"  Python: N/A")

        # C implementation
        if collage_idx < len(c_result):
            print(f"  C ({len(c_result[collage_idx])} total):")
            for poly_idx, poly in enumerate(c_result[collage_idx][:max_polyominoes]):
                shape_h, shape_w = poly.shape.shape
                tiles = np.sum(poly.shape)
                print(f"    [{poly_idx}] pos=({poly.py:3d},{poly.px:3d}) offset=({poly.oy:3d},{poly.ox:3d}) "
                      f"shape={shape_h}x{shape_w} tiles={tiles} frame={poly.frame}")
                # Print bitmap representation
                for row in poly.shape:
                    bitmap_str = ''.join('█' if cell else '·' for cell in row)
                    print(f"      {bitmap_str}")
        else:
            print(f"  C: N/A")

    print(f"\n{'='*80}\n")


def verify_packing_properties(result, h, w):
    """Verify that the packing result has valid properties.

    Args:
        result: List of lists of PolyominoPosition
        h: Collage height
        w: Collage width

    Returns:
        dict: Statistics about the packing including validity
    """
    stats = {
        'num_collages': len(result),
        'total_polyominoes': sum(len(collage) for collage in result),
        'collage_fills': [],
        'valid': True,
        'errors': []
    }

    # Check each collage
    for collage_idx, collage in enumerate(result):
        # Create occupancy grid for this collage (stores polyomino index + 1, 0 = empty)
        occupancy = np.zeros((h, w), dtype=np.int32)

        # Track error information for visualization
        has_error = False
        overlap_positions = []  # List of (y, x, poly_idx1, poly_idx2)
        oob_positions = []  # List of (y, x, poly_idx)

        for poly_idx, pos in enumerate(collage):
            # Check bounds
            shape = pos.shape
            ph, pw = shape.shape

            # Adjust for placement position
            py, px = pos.py, pos.px

            # Find actual occupied tiles in shape
            occupied_coords = np.argwhere(shape == 1)

            for coord in occupied_coords:
                y = py + coord[0]
                x = px + coord[1]

                # Check bounds
                if y < 0 or y >= h or x < 0 or x >= w:
                    stats['valid'] = False
                    stats['errors'].append(f"Collage {collage_idx}: Position out of bounds at ({y}, {x})")
                    has_error = True
                    oob_positions.append((y, x, poly_idx))
                    continue

                # Check overlap
                if occupancy[y, x] != 0:
                    stats['valid'] = False
                    overlapping_poly_idx = occupancy[y, x] - 1
                    stats['errors'].append(f"Collage {collage_idx}: Overlap detected at ({y}, {x}) between polyomino {overlapping_poly_idx} and {poly_idx}")
                    has_error = True
                    overlap_positions.append((y, x, overlapping_poly_idx, poly_idx))

                occupancy[y, x] = poly_idx + 1  # Store 1-indexed to distinguish from empty

        # Calculate fill percentage
        fill_pct = np.sum(occupancy > 0) / (h * w) * 100
        stats['collage_fills'].append(fill_pct)

        # If errors detected, visualize the collage
        if has_error:
            print(f"\n{'='*100}")
            print(f"VERIFICATION ERROR VISUALIZATION - Collage {collage_idx}")
            print(f"{'='*100}")

            # Print error summary
            if overlap_positions:
                print(f"\n⚠️  OVERLAP ERRORS ({len(overlap_positions)} overlaps detected):")
                for y, x, poly_idx1, poly_idx2 in overlap_positions[:10]:  # Show first 10
                    print(f"  Position ({y}, {x}): Polyomino {poly_idx1} overlaps with Polyomino {poly_idx2}")
                if len(overlap_positions) > 10:
                    print(f"  ... and {len(overlap_positions) - 10} more overlaps")

            if oob_positions:
                print(f"\n⚠️  OUT-OF-BOUNDS ERRORS ({len(oob_positions)} tiles out of bounds):")
                for y, x, poly_idx in oob_positions[:10]:  # Show first 10
                    print(f"  Position ({y}, {x}): Polyomino {poly_idx} is out of bounds (collage size: {h}x{w})")
                if len(oob_positions) > 10:
                    print(f"  ... and {len(oob_positions) - 10} more out-of-bounds tiles")

            # Visualize the full collage with polyominoes
            print(f"\n📊 Full Collage Visualization ({h}x{w}):")
            print("  Legend: '·' = empty, numbers = polyomino ID, '█' = overlap, 'X' = out-of-bounds")

            # Create error map
            error_map = np.zeros((h, w), dtype=np.int32)  # 0 = no error, 1 = overlap, 2 = oob
            for y, x, _, _ in overlap_positions:
                if 0 <= y < h and 0 <= x < w:
                    error_map[y, x] = 1
            for y, x, _ in oob_positions:
                # Mark in extended visualization if needed
                pass

            # Print a compact visualization for large collages
            if h <= 40 and w <= 80:
                # Print full collage
                for y in range(h):
                    row_str = "  "
                    for x in range(w):
                        if error_map[y, x] == 1:  # Overlap
                            row_str += "█"
                        elif occupancy[y, x] == 0:
                            row_str += "·"
                        else:
                            # Show polyomino ID (use modulo to fit in single char)
                            poly_id = (occupancy[y, x] - 1) % 36
                            if poly_id < 10:
                                row_str += str(poly_id)
                            else:
                                row_str += chr(ord('A') + poly_id - 10)
                    print(row_str)
            else:
                print(f"  (Collage too large to visualize completely: {h}x{w})")

            # Print detailed polyomino information for those involved in errors
            error_poly_indices = set()
            for _, _, poly_idx1, poly_idx2 in overlap_positions:
                error_poly_indices.add(poly_idx1)
                error_poly_indices.add(poly_idx2)
            for _, _, poly_idx in oob_positions:
                error_poly_indices.add(poly_idx)

            if error_poly_indices:
                print(f"\n🔍 Detailed Polyomino Information (polyominoes involved in errors):")
                for poly_idx in sorted(error_poly_indices)[:10]:  # Show first 10
                    if poly_idx >= len(collage):
                        continue
                    pos = collage[poly_idx]
                    shape = pos.shape
                    shape_h, shape_w = shape.shape
                    tiles = np.sum(shape)
                    print(f"\n  Polyomino {poly_idx}:")
                    print(f"    Position: ({pos.py}, {pos.px})")
                    print(f"    Offset: ({pos.oy}, {pos.ox})")
                    print(f"    Shape: {shape_h}x{shape_w}, {tiles} tiles")
                    print(f"    Frame: {pos.frame}")
                    print(f"    Bitmap representation:")

                    # Show polyomino in its placement context
                    for local_y in range(shape_h):
                        row_str = "      "
                        for local_x in range(shape_w):
                            global_y = pos.py + local_y
                            global_x = pos.px + local_x

                            if shape[local_y, local_x] == 1:
                                # Check if this tile is involved in an error
                                is_overlap = any(y == global_y and x == global_x for y, x, _, _ in overlap_positions)
                                is_oob = any(y == global_y and x == global_x for y, x, _ in oob_positions)

                                if is_overlap:
                                    row_str += "█"  # Overlap
                                elif is_oob or global_y < 0 or global_y >= h or global_x < 0 or global_x >= w:
                                    row_str += "X"  # Out of bounds
                                else:
                                    row_str += "█"  # Normal tile
                            else:
                                row_str += "·"  # Empty
                        print(row_str)

                    # Show where this polyomino is placed in the collage
                    print(f"    Placement context (showing area around polyomino):")
                    min_y = max(0, pos.py - 1)
                    max_y = min(h, pos.py + shape_h + 1)
                    min_x = max(0, pos.px - 1)
                    max_x = min(w, pos.px + shape_w + 1)

                    for y in range(min_y, max_y):
                        row_str = "      "
                        for x in range(min_x, max_x):
                            if error_map[y, x] == 1:  # Overlap
                                row_str += "█"
                            elif occupancy[y, x] == 0:
                                row_str += "·"
                            elif occupancy[y, x] == poly_idx + 1:
                                row_str += "#"  # Current polyomino
                            else:
                                # Other polyomino
                                other_id = (occupancy[y, x] - 1) % 36
                                if other_id < 10:
                                    row_str += str(other_id)
                                else:
                                    row_str += chr(ord('A') + other_id - 10)
                        print(row_str)

                if len(error_poly_indices) > 10:
                    print(f"\n  ... and {len(error_poly_indices) - 10} more polyominoes involved in errors")

            print(f"\n{'='*100}\n")

    return stats


def benchmark_pack_ffd():
    """Benchmark pack_ffd implementations (C vs Python) with correctness validation."""
    print("\n=== Pack FFD Benchmark ===")

    # Import implementations
    from polyis.pack.pack_ffd import pack_all as c_pack_all
    from polyis.pack.python.pack_ffd import pack_all as python_pack_all
    from polyis.pack.group_tiles import group_tiles as group_tiles_cython

    # Test data files
    test_files = [
        ('caldot1_te03', '/polyis/tests/pack/data/caldot1_te03.jsonl'),
        ('caldot2_te05', '/polyis/tests/pack/data/caldot2_te05.jsonl'),
        ('jnc0_te04', '/polyis/tests/pack/data/jnc0_te04.jsonl'),
    ]

    # Configuration
    threshold = 0.5
    tilepadding = 0  # No padding for this benchmark

    # Storage for results
    results_summary = {}

    for test_name, test_file in test_files:
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}, skipping...")
            continue

        print(f"\n📁 Processing: {test_name}")
        print("-" * 80)

        # Read all frames from JSONL file
        with open(test_file, 'r') as f:
            lines = f.readlines()

        # Get frame dimensions from first frame
        first_frame = json.loads(lines[0])
        frame_size = first_frame['frame_size']
        tile_size = first_frame['tile_size']
        grid_height = frame_size[0] // tile_size
        grid_width = frame_size[1] // tile_size

        # Storage for timing results and correctness checks
        python_times = []
        c_times = []
        correctness_matches = 0
        correctness_mismatches = 0
        python_valid_count = 0
        c_valid_count = 0
        total_validation_runs = 0  # Track total number of validation runs

        # Sample frames (every 10th frame to keep benchmark manageable)
        sample_indices = list(range(0, len(lines), max(1, len(lines) // 50)))

        print(f"  Sampling {len(sample_indices)} frames out of {len(lines)} total frames")

        # Step 1: Collect all bitmaps from all frames first
        all_bitmaps = []
        for frame_idx in tqdm.tqdm(sample_indices, desc=f"  Loading bitmaps for {test_name}"):
            frame_result = json.loads(lines[frame_idx])

            # Parse classification hex to bitmap
            classifications: str = frame_result['classification_hex']
            classification_size: tuple[int, int] = frame_result['classification_size']

            # Convert hex to bitmap
            bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
            bitmap_frame = bitmap_frame > (threshold * 255)
            bitmap_frame = bitmap_frame.astype(np.uint8)
            all_bitmaps.append(bitmap_frame)

        # Step 2: Run the full pipeline multiple times to measure performance
        for tilepadding in [0, 1, 2]:
            num_runs = 20
            for run_idx in tqdm.tqdm(range(num_runs), desc=f"  Benchmarking {test_name}"):
                # Python implementation: group_tiles + pack_all
                start = time.perf_counter()
                # Group all bitmaps
                all_polyominoes_python = []
                for bitmap in all_bitmaps:
                    polyominoes = group_tiles_cython(bitmap.copy(), tilepadding)
                    all_polyominoes_python.append(polyominoes)
                # Pack all polyominoes
                python_result = python_pack_all(all_polyominoes_python, grid_height, grid_width)
                python_time = (time.perf_counter() - start) * 1e6  # Convert to microseconds
                python_times.append(python_time)

                # C implementation: group_tiles + pack_all
                start = time.perf_counter()
                # Group all bitmaps
                all_polyominoes_c = []
                for bitmap in all_bitmaps:
                    polyominoes = group_tiles_cython(bitmap.copy(), tilepadding)
                    all_polyominoes_c.append(polyominoes)
                # Pack all polyominoes
                c_result = c_pack_all(all_polyominoes_c, grid_height, grid_width)
                c_time = (time.perf_counter() - start) * 1e6  # Convert to microseconds
                c_times.append(c_time)

                # Only validate correctness on the first run to avoid spam
                if run_idx == 0:
                    # Correctness validation: compare Python vs C results
                    if compare_polyomino_positions(python_result, c_result, verbose=False):
                        correctness_matches += 1
                    else:
                        correctness_mismatches += 1
                        print(f"\n⚠️  Results mismatch detected!")
                        # Print detailed diagnostics about the mismatch
                        print_packing_diagnostics(python_result, c_result, 0, max_polyominoes=1000)

                    # Verify packing properties for both implementations
                    # Track if all collages in the frame are valid
                    python_frame_valid = True
                    c_frame_valid = True
                    python_frame_errors = []
                    c_frame_errors = []

                    for collage_idx, (py_collage, c_collage) in enumerate(zip(python_result, c_result)):
                        python_stats = verify_packing_properties([py_collage], grid_height, grid_width)
                        c_stats = verify_packing_properties([c_collage], grid_height, grid_width)

                        if not python_stats['valid']:
                            python_frame_valid = False
                            python_frame_errors.extend(python_stats['errors'][:3])

                        if not c_stats['valid']:
                            c_frame_valid = False
                            c_frame_errors.extend(c_stats['errors'][:3])

                    # Print summary of validation results for this run
                    if not python_frame_valid or not c_frame_valid:
                        print(f"\n⚠️  Validation failed for tilepadding={tilepadding}:")
                        if not python_frame_valid:
                            print(f"     Python errors: {python_frame_errors[:3]}")
                        if not c_frame_valid:
                            print(f"     C errors: {c_frame_errors[:3]}")

                    # Increment frame-level valid counts
                    if python_frame_valid:
                        python_valid_count += 1
                    if c_frame_valid:
                        c_valid_count += 1

                    # Increment total validation runs counter
                    total_validation_runs += 1

            # Store results
            results_summary[test_name] = {
                'python': python_times,
                'c': c_times,
                'correctness': {
                    'matches': correctness_matches,
                    'mismatches': correctness_mismatches,
                    'python_valid': python_valid_count,
                    'c_valid': c_valid_count,
                    'total': total_validation_runs  # Use actual validation run count
                }
            }

    # Print summary
    print("\n" + "=" * 100)
    print("PACK FFD PERFORMANCE SUMMARY")
    print("=" * 100)
    print("\nTest File".ljust(20) + " | " + "Python Avg".ljust(12) + " | " + "C Avg".ljust(12) + " | " + "Speedup".ljust(10) + " | " + "Samples")
    print("-" * 100)

    for test_name, results in results_summary.items():
        python_times = results['python']
        c_times = results['c']

        if python_times and c_times:
            python_avg = statistics.mean(python_times)
            c_avg = statistics.mean(c_times)
            speedup = python_avg / c_avg if c_avg > 0 else float('inf')
            samples = len(python_times)

            print(f"{test_name:20} | {python_avg:10.2f}μs | {c_avg:10.2f}μs | {speedup:8.2f}x | {samples:7d}")

    # Print correctness summary
    print("\n" + "=" * 100)
    print("PACK FFD CORRECTNESS SUMMARY")
    print("=" * 100)
    print("\nTest File".ljust(20) + " | " + "Matches".ljust(10) + " | " + "Mismatches".ljust(12) + " | " + "Python Valid".ljust(14) + " | " + "C Valid".ljust(10) + " | " + "Match Rate")
    print("-" * 100)

    total_matches = 0
    total_mismatches = 0
    total_python_valid = 0
    total_c_valid = 0
    total_samples = 0

    for test_name, results in results_summary.items():
        correctness = results['correctness']
        matches = correctness['matches']
        mismatches = correctness['mismatches']
        python_valid = correctness['python_valid']
        c_valid = correctness['c_valid']
        total = correctness['total']
        match_rate = (matches / total * 100) if total > 0 else 0

        print(f"{test_name:20} | {matches:10d} | {mismatches:12d} | {python_valid:14d} | {c_valid:10d} | {match_rate:8.2f}%")

        total_matches += matches
        total_mismatches += mismatches
        total_python_valid += python_valid
        total_c_valid += c_valid
        total_samples += total

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    all_python_times = []
    all_c_times = []
    for results in results_summary.values():
        all_python_times.extend(results['python'])
        all_c_times.extend(results['c'])

    if all_python_times and all_c_times:
        python_avg = statistics.mean(all_python_times)
        c_avg = statistics.mean(all_c_times)
        speedup = python_avg / c_avg if c_avg > 0 else float('inf')
        samples = len(all_python_times)

        print(f"\nPerformance:")
        print(f"  Overall Python Average: {python_avg:10.2f}μs")
        print(f"  Overall C Average:      {c_avg:10.2f}μs")
        print(f"  Overall Speedup:        {speedup:8.2f}x")
        print(f"  Total Samples:          {samples:7d}")

        print(f"\nCorrectness:")
        print(f"  Total Matches:          {total_matches:7d}")
        print(f"  Total Mismatches:       {total_mismatches:7d}")
        print(f"  Match Rate:             {(total_matches / total_samples * 100):6.2f}%")
        print(f"  Python Valid Packings:  {total_python_valid:7d} / {total_samples:7d} ({total_python_valid / total_samples * 100:6.2f}%)")
        print(f"  C Valid Packings:       {total_c_valid:7d} / {total_samples:7d} ({total_c_valid / total_samples * 100:6.2f}%)")

    print("\n" + "=" * 100)
