#!/usr/bin/env python3
"""
Standalone benchmark script for comparing Python vs Cython implementations.
"""

import time
import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
import cv2
import json
import tqdm
import statistics


# Import adapters with fallback
try:
    from adapters import get_polyominoes, format_positions  # type: ignore
except ImportError:
    # Fallback if adapters not available
    def get_polyominoes(polyominoes):
        return polyominoes
    def format_positions(positions):
        return positions

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
        
        polyominoes.append((mask, offset))
    
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
        polyominoes_stack = get_polyominoes(polyominoes.copy())
        start = time.perf_counter()
        cython_result = cython_pack_append(polyominoes_stack, h, w, occupied_tiles.copy())
        cython_time = time.perf_counter() - start
        
        speedup = python_time / cython_time if cython_time > 0 else float('inf')
        
        print(f"{h}x{w}\t\t{num_poly}\t\t{max_size}\t\t{python_time:.6f}\t\t{cython_time:.6f}\t\t{speedup:.2f}x")


def benchmark_compress():
    """Benchmark compress implementations."""
    print("\n=== Compress Benchmark ===")

    try:
        from group_tiles import group_tiles as cython_group_tiles
        try:
            from group_tiles import cleanup_polyomino_stack  # type: ignore
        except ImportError:
            # Fallback if cleanup function not available
            def cleanup_polyomino_stack(polyominoes):
                pass
        from group_tiles_original import group_tiles as python_group_tiles
        from pack_append import pack_append as cython_pack_append
        from pack_append_original import pack_append as python_pack_append
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return
    
    add_margin = torch.tensor(
        [[[[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]]],
        dtype=torch.uint8,
        requires_grad=False,
    )
    
    dataset_video = [
        ('caldot1', 'caldot1-1.mp4'),
        ('caldot1', 'caldot1-3.mp4'),
        ('caldot1', 'caldot1-5.mp4'),
        ('caldot1', 'caldot1-7.mp4'),
        ('caldot2', 'caldot2-1.mp4'),
        ('caldot2', 'caldot2-3.mp4'),
        ('caldot2', 'caldot2-5.mp4'),
        ('caldot2', 'caldot2-7.mp4'),
        ('b3d-jnc00', 'jnc00.mp4'),
        ('b3d-jnc02', 'jnc02.mp4'),
        ('b3d-jnc06', 'jnc06.mp4'),
        ('b3d-jnc07', 'jnc07.mp4'),
    ]
    fns = [
        (True, cython_group_tiles, cython_pack_append),
        (False, python_group_tiles, python_pack_append),
    ]
    tilepaddings = [True, False]
    threshold = 0.5
    classifier = 'Perfect'
    tilesize = 60

    # Store results for summary
    results_summary = {}

    for dataset, video in dataset_video:
        video_file = os.path.basename(os.path.join('/polyis-data/video-datasets', dataset, video))
        
        # Initialize results for this dataset/video combination
        if (dataset, video) not in results_summary:
            results_summary[(dataset, video)] = {}
        
        for tilepadding in tilepaddings:
            if tilepadding not in results_summary[(dataset, video)]:
                results_summary[(dataset, video)][tilepadding] = {
                    'python': {'group_tiles': [], 'pack_append': [], 'pack_append_retry': []},
                    'cython': {'group_tiles': [], 'pack_append': [], 'pack_append_retry': []}
                }
            
            for is_cython, group_tiles, pack_append in fns:
                print(f"Dataset: {dataset}, Video: {video}, Tilepadding: {tilepadding}, Cython: {is_cython}")

                cap = cv2.VideoCapture(video_file)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Calculate grid dimensions
                grid_height = height // tilesize
                grid_width = width // tilesize
                def init_compression_variables():
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    # assert dtypes.is_np_image(canvas), canvas.shape
                    occupied_tiles = np.zeros((grid_height, grid_width), dtype=np.uint8)
                    # assert dtypes.is_bitmap(occupied_tiles), occupied_tiles.shape
                    index_map = np.zeros((grid_height, grid_width), dtype=np.uint16)
                    # assert dtypes.is_index_map(index_map), index_map.shape
                    offset_lookup: list[tuple[tuple[int, int], tuple[int, int], int]] = []
                    return canvas, occupied_tiles, index_map, offset_lookup, True, False

                _, occupied_tiles, _, _, _, full = init_compression_variables()

                result_file = os.path.join('/polyis-cache', dataset, 'execution', video_file,
                                           '020_relevancy', f'{classifier}_{tilesize}', 'score', 'score.jsonl')

                # Collect all timing data for this configuration
                all_group_tiles_times = []
                all_pack_append_times = []
                all_pack_append_retry_times = []

                with open(result_file, 'r') as f:
                    lines = list(f.readlines())
                    for line in tqdm.tqdm(lines[::len(lines)//500]):
                        frame_result = json.loads(line)

                        
                        step_start = (time.time_ns() / 1e6)
                        classifications: str = frame_result['classification_hex']
                        classification_size: tuple[int, int] = frame_result['classification_size']
                        
                        bitmap_frame = np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8).reshape(classification_size)
                        bitmap_frame = bitmap_frame > (threshold * 255)
                        bitmap_frame = bitmap_frame.astype(np.uint8)
                        if tilepadding:
                            bitmap_frame = F.conv2d(
                                torch.from_numpy(np.array([[bitmap_frame]])), add_margin, padding='same').numpy()[0, 0]
                        
                        step_times = {
                            'cython': is_cython,
                            'group_tiles': [],
                            'pack_append': [],
                            'pack_append_retry': [],
                        }
                        for _ in range(50):
                            occupied_tiles_copy = occupied_tiles.copy()
                            # Profile: Group connected tiles into polyominoes
                            step_start = (time.time_ns() / 1e6)
                            polyominoes = group_tiles(bitmap_frame.copy())
                            step_times['group_tiles'].append((time.time_ns() / 1e6) - step_start)
                            
                            if not is_cython:
                                polyominoes = sorted(polyominoes, key=lambda x: x[0].sum(), reverse=True)
                            
                            # Profile: Try compressing polyominoes
                            step_start = (time.time_ns() / 1e6)
                            positions = None if full else pack_append(polyominoes, grid_height, grid_width, occupied_tiles_copy)
                            step_times['pack_append'].append((time.time_ns() / 1e6) - step_start)

                            if positions is None:
                                _, occupied_tiles, _, _, _, full = init_compression_variables()

                                # Profile: Retry compression for current frame
                                step_start = (time.time_ns() / 1e6)
                                positions = pack_append(polyominoes, grid_height, grid_width, occupied_tiles_copy)
                                step_times['pack_append_retry'].append((time.time_ns() / 1e6) - step_start)
                            
                            if is_cython:
                                cleanup_polyomino_stack(polyominoes)

                        # Update occupied_tiles if copy was made
                        try:
                            occupied_tiles = occupied_tiles_copy
                        except NameError:
                            pass  # occupied_tiles_copy not available
                        # Get median times, handling cases where lists might be empty
                        group_tiles_times = sorted(step_times['group_tiles'])
                        pack_append_times = sorted(step_times['pack_append'])
                        pack_append_retry_times = sorted(step_times['pack_append_retry'])
                        
                        group_tiles_time = group_tiles_times[len(group_tiles_times)//2] * 1000  # Convert to microseconds
                        pack_append_time = pack_append_times[len(pack_append_times)//2] * 1000  # Convert to microseconds
                        pack_append_retry_time = pack_append_retry_times[len(pack_append_retry_times)//2] * 1000 if pack_append_retry_times else 0  # Convert to microseconds
                        
                        # Store times for this frame
                        all_group_tiles_times.append(group_tiles_time)
                        all_pack_append_times.append(pack_append_time)
                        all_pack_append_retry_times.append(pack_append_retry_time)

                # Store results for this configuration
                impl_type = 'cython' if is_cython else 'python'
                results_summary[(dataset, video)][tilepadding][impl_type]['group_tiles'].extend(all_group_tiles_times)
                results_summary[(dataset, video)][tilepadding][impl_type]['pack_append'].extend(all_pack_append_times)
                results_summary[(dataset, video)][tilepadding][impl_type]['pack_append_retry'].extend(all_pack_append_retry_times)

    # Print comprehensive summary
    print("\n" + "=" * 100)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 100)
    
    for (dataset, video), video_results in results_summary.items():
        print(f"\n📊 Dataset: {dataset}, Video: {video}")
        print("-" * 80)
        
        for tilepadding, padding_results in video_results.items():
            print(f"\n  🔧 Tilepadding: {tilepadding}")
            print("  " + "-" * 60)
            
            # Get Python and Cython results
            python_results = padding_results['python']
            cython_results = padding_results['cython']
            
            # Calculate statistics for each function
            functions = ['group_tiles', 'pack_append', 'pack_append_retry']
            
            for func in functions:
                python_times = python_results[func]
                cython_times = cython_results[func]
                
                if python_times and cython_times:
                    python_avg = statistics.mean(python_times)
                    cython_avg = statistics.mean(cython_times)
                    speedup = python_avg / cython_avg if cython_avg > 0 else float('inf')
                    
                    print(f"    {func:20} | Python: {python_avg:8.2f}μs | Cython: {cython_avg:8.2f}μs | Speedup: {speedup:6.2f}x")
                else:
                    print(f"    {func:20} | No data available")
    
    # Overall summary across all datasets
    print("\n" + "=" * 100)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 100)
    
    overall_stats = {
        'group_tiles': {'python': [], 'cython': []},
        'pack_append': {'python': [], 'cython': []},
        'pack_append_retry': {'python': [], 'cython': []}
    }
    
    # Aggregate all results
    for (dataset, video), video_results in results_summary.items():
        for tilepadding, padding_results in video_results.items():
            for impl_type in ['python', 'cython']:
                for func in ['group_tiles', 'pack_append', 'pack_append_retry']:
                    overall_stats[func][impl_type].extend(padding_results[impl_type][func])
    
    print("\nFunction".ljust(20) + " | " + "Python Avg".ljust(12) + " | " + "Cython Avg".ljust(12) + " | " + "Speedup".ljust(10) + " | " + "Samples")
    print("-" * 80)
    
    for func in ['group_tiles', 'pack_append', 'pack_append_retry']:
        python_times = overall_stats[func]['python']
        cython_times = overall_stats[func]['cython']
        
        if python_times and cython_times:
            python_avg = statistics.mean(python_times)
            cython_avg = statistics.mean(cython_times)
            speedup = python_avg / cython_avg if cython_avg > 0 else float('inf')
            samples = len(python_times)
            
            print(f"{func:20} | {python_avg:10.2f}μs | {cython_avg:10.2f}μs | {speedup:8.2f}x | {samples:7d}")
        else:
            print(f"{func:20} | No data available")
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETED")
    print("=" * 100)

def main():
    """Run all benchmarks."""
    print("Performance Comparison: Python vs Cython")
    print("=" * 50)
    
    # benchmark_group_tiles()
    # benchmark_pack_append()
    benchmark_compress()
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
