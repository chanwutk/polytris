#!/usr/local/bin/python

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
from pathlib import Path

CACHE_DIR = '/polyis-cache'
OUTPUT_DIR = 'pipeline-stages/track-speed-results'
# TILE_SIZES = [32, 64, 128]
TILE_SIZES = [64]


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (str): Tile size to use for evaluation (choices: '64', '128', 'all')
            - output_dir (str): Output directory for results (default: 'pipeline-stages/track-speed-results')
            - create_plots (bool): Whether to create visualization plots (default: False)
    """
    parser = argparse.ArgumentParser(description='Analyze tracking speed and performance statistics from runtime data')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--tile_size', type=str, choices=['64', '128', 'all'], default='all',
                        help='Tile size to use for evaluation (or "all" for all tile sizes)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--create_plots', action='store_true', default=False,
                        help='Whether to create visualization plots')
    return parser.parse_args()


def find_tracking_results(cache_dir: str, dataset: str, tile_size: str) -> List[Tuple[str, int]]:
    """
    Find all video files with tracking results and runtime data for the specified dataset and tile size.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        tile_size (str): Tile size to look for ('64', '128', or 'all')
        
    Returns:
        List[Tuple[str, int]]: List of (video_name, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset)
    if not os.path.exists(dataset_cache_dir):
        print(f"Dataset cache directory {dataset_cache_dir} does not exist")
        return []
    
    video_tile_combinations = []
    
    # Determine which tile sizes to process
    tile_sizes_to_process = TILE_SIZES if tile_size == 'all' else [int(tile_size)]
    
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            for ts in tile_sizes_to_process:
                tracking_path = os.path.join(item_path, 'uncompressed_tracking', f'proxy_{ts}', 'tracking.jsonl')
                runtime_path = os.path.join(item_path, 'uncompressed_tracking', f'proxy_{ts}', 'runtimes.jsonl')
                
                if os.path.exists(tracking_path) and os.path.exists(runtime_path):
                    video_tile_combinations.append((item, ts))
                    print(f"Found tracking results with runtime data: {item} with tile size {ts}")
    
    return video_tile_combinations


def load_runtime_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load runtime data from JSONL file.
    
    Args:
        file_path (str): Path to the runtime JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of runtime data for each frame
    """
    runtime_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                runtime_data.append(data)
    
    return runtime_data


def analyze_runtime_performance(video_name: str, tile_size: int, runtime_path: str, 
                               output_dir: str) -> Dict[str, Any]:
    """
    Analyze runtime performance for a single video.
    
    Args:
        video_name (str): Name of the video
        tile_size (int): Tile size used
        runtime_path (str): Path to runtime data
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Runtime performance analysis results
    """
    print(f"Analyzing runtime performance for {video_name} with tile size {tile_size}")
    
    try:
        # Load runtime data
        runtime_data = load_runtime_data(runtime_path)
        
        if not runtime_data:
            return {
                'video_name': video_name,
                'tile_size': tile_size,
                'error': 'No runtime data found',
                'success': False
            }
        
        # Extract timing data
        frame_indices = []
        total_frame_times = []
        step_times = {
            'convert_detections': [],
            'tracker_update': [],
            'process_results': []
        }
        num_detections = []
        num_tracks = []
        
        for frame_data in runtime_data:
            frame_indices.append(frame_data['frame_idx'])
            total_frame_times.append(frame_data['step_times']['total_frame_time'])
            num_detections.append(frame_data['num_detections'])
            num_tracks.append(frame_data['num_tracks'])
            
            # Extract step times
            for step in step_times.keys():
                if step in frame_data['step_times']:
                    step_times[step].append(frame_data['step_times'][step])
                else:
                    step_times[step].append(0.0)
        
        # Calculate performance statistics
        performance_stats = {
            'video_name': video_name,
            'tile_size': tile_size,
            'total_frames': len(frame_indices),
            'total_processing_time': sum(total_frame_times),
            'avg_frame_time': np.mean(total_frame_times),
            'std_frame_time': np.std(total_frame_times),
            'min_frame_time': np.min(total_frame_times),
            'max_frame_time': np.max(total_frame_times),
            'median_frame_time': np.median(total_frame_times),
            'fps': 1.0 / np.mean(total_frame_times) if np.mean(total_frame_times) > 0 else 0,
            'total_detections': sum(num_detections),
            'total_tracks': sum(num_tracks),
            'avg_detections_per_frame': np.mean(num_detections),
            'avg_tracks_per_frame': np.mean(num_tracks),
            'success': True
        }
        
        # Add step-by-step timing analysis
        for step_name, step_times_list in step_times.items():
            if step_times_list:
                performance_stats[f'{step_name}_avg'] = np.mean(step_times_list)
                performance_stats[f'{step_name}_std'] = np.std(step_times_list)
                performance_stats[f'{step_name}_min'] = np.min(step_times_list)
                performance_stats[f'{step_name}_max'] = np.max(step_times_list)
                performance_stats[f'{step_name}_total'] = sum(step_times_list)
        
        # Add percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            performance_stats[f'frame_time_p{p}'] = np.percentile(total_frame_times, p)
        
        # Add detailed frame data for visualization
        performance_stats['frame_data'] = {
            'frame_indices': frame_indices,
            'total_frame_times': total_frame_times,
            'step_times': step_times,
            'num_detections': num_detections,
            'num_tracks': num_tracks
        }
        
        return performance_stats
        
    except Exception as e:
        print(f"Error analyzing runtime performance for {video_name} with tile size {tile_size}: {str(e)}")
        return {
            'video_name': video_name,
            'tile_size': tile_size,
            'error': str(e),
            'success': False
        }


def create_speed_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create a simple text summary of the speed performance results.
    
    Args:
        results (List[Dict[str, Any]]): List of performance analysis results
        output_dir (str): Output directory for results
    """
    print("Creating speed performance summary...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful performance analysis results to summarize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to JSON
    with open(os.path.join(output_dir, 'detailed_speed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create text summary
    summary_file = os.path.join(output_dir, 'speed_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Tracking Speed Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Individual Video Results:\n")
        f.write("-" * 30 + "\n")
        for result in successful_results:
            f.write(f"Video: {result['video_name']}\n")
            f.write(f"  Tile Size: {result['tile_size']}\n")
            f.write(f"  Total Frames: {result['total_frames']}\n")
            f.write(f"  Total Processing Time: {result['total_processing_time']:.4f}s\n")
            f.write(f"  Average Frame Time: {result['avg_frame_time']:.4f}s\n")
            f.write(f"  FPS: {result['fps']:.2f}\n")
            f.write(f"  Total Detections: {result['total_detections']}\n")
            f.write(f"  Total Tracks: {result['total_tracks']}\n")
            f.write(f"  Avg Detections/Frame: {result['avg_detections_per_frame']:.2f}\n")
            f.write(f"  Avg Tracks/Frame: {result['avg_tracks_per_frame']:.2f}\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-" * 20 + "\n")
        
        # Calculate overall statistics
        fps_values = [r['fps'] for r in successful_results]
        frame_times = [r['avg_frame_time'] for r in successful_results]
        total_times = [r['total_processing_time'] for r in successful_results]
        
        f.write(f"FPS - Mean: {np.mean(fps_values):.2f}, Std: {np.std(fps_values):.2f}\n")
        f.write(f"      Min: {np.min(fps_values):.2f}, Max: {np.max(fps_values):.2f}\n")
        f.write(f"Frame Time - Mean: {np.mean(frame_times):.4f}s, Std: {np.std(frame_times):.4f}s\n")
        f.write(f"            Min: {np.min(frame_times):.4f}s, Max: {np.max(frame_times):.4f}s\n")
        f.write(f"Total Time - Mean: {np.mean(total_times):.2f}s, Std: {np.std(total_times):.2f}s\n")
        f.write(f"            Min: {np.min(total_times):.2f}s, Max: {np.max(total_times):.2f}s\n")
        
        # Tile size analysis if multiple tile sizes
        if len(set([r['tile_size'] for r in successful_results])) > 1:
            f.write("\nTile Size Analysis:\n")
            f.write("-" * 20 + "\n")
            for tile_size in sorted(set([r['tile_size'] for r in successful_results])):
                tile_results = [r for r in successful_results if r['tile_size'] == tile_size]
                tile_fps = [r['fps'] for r in tile_results]
                tile_frame_times = [r['avg_frame_time'] for r in tile_results]
                
                f.write(f"Tile Size {tile_size}:\n")
                f.write(f"  FPS: Mean={np.mean(tile_fps):.2f}, Std={np.std(tile_fps):.2f}\n")
                f.write(f"  Frame Time: Mean={np.mean(tile_frame_times):.4f}s, Std={np.std(tile_frame_times):.4f}s\n\n")
    
    print(f"Summary saved to {summary_file}")


def create_speed_visualizations(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for tracking speed performance results using matplotlib.
    
    Args:
        results (List[Dict[str, Any]]): List of performance analysis results
        output_dir (str): Output directory for visualizations
    """
    print("Creating matplotlib speed visualizations...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful performance analysis results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    video_names = []
    tile_sizes = []
    fps_values = []
    frame_times = []
    total_times = []
    total_frames = []
    
    for result in successful_results:
        video_names.append(result['video_name'])
        tile_sizes.append(result['tile_size'])
        fps_values.append(result['fps'])
        frame_times.append(result['avg_frame_time'])
        total_times.append(result['total_processing_time'])
        total_frames.append(result['total_frames'])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Video': video_names,
        'Tile_Size': tile_sizes,
        'FPS': fps_values,
        'Frame_Time': frame_times,
        'Total_Time': total_times,
        'Total_Frames': total_frames
    })
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'speed_results.csv'), index=False)
    
    # Set matplotlib style
    plt.style.use('default')
    
    # 1. Performance overview dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tracking Speed Performance Overview', fontsize=16, fontweight='bold')
    
    # FPS comparison
    bars1 = axes[0, 0].bar(range(len(video_names)), fps_values, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    axes[0, 0].set_title('Frames Per Second (FPS)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Video', fontsize=12)
    axes[0, 0].set_ylabel('FPS', fontsize=12)
    axes[0, 0].set_xticks(range(len(video_names)))
    axes[0, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Frame time comparison
    bars2 = axes[0, 1].bar(range(len(video_names)), frame_times, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    axes[0, 1].set_title('Average Frame Processing Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Video', fontsize=12)
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 1].set_xticks(range(len(video_names)))
    axes[0, 1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Total processing time
    bars3 = axes[1, 0].bar(range(len(video_names)), total_times, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    axes[1, 0].set_title('Total Processing Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Video', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_xticks(range(len(video_names)))
    axes[1, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Performance heatmap
    performance_matrix = np.array([fps_values, frame_times, total_times])
    im = axes[1, 1].imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
    axes[1, 1].set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Video', fontsize=12)
    axes[1, 1].set_ylabel('Metric', fontsize=12)
    axes[1, 1].set_xticks(range(len(video_names)))
    axes[1, 1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(['FPS', 'Frame Time', 'Total Time'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Performance Value', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed timing analysis for each video
    for result in successful_results:
        if 'frame_data' not in result:
            continue
            
        frame_data = result['frame_data']
        frame_indices = frame_data['frame_indices']
        total_frame_times = frame_data['total_frame_times']
        step_times = frame_data['step_times']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detailed Timing Analysis: {result["video_name"]} (Tile {result["tile_size"]})', 
                     fontsize=16, fontweight='bold')
        
        # Frame processing time over time
        axes[0, 0].plot(frame_indices, total_frame_times, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Frame Processing Time Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Frame Index', fontsize=12)
        axes[0, 0].set_ylabel('Processing Time (seconds)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Step-by-step timing breakdown
        step_names = list(step_times.keys())
        step_data = [step_times[step] for step in step_names]
        
        bp = axes[0, 1].boxplot(step_data, labels=step_names, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 1].set_title('Step-by-Step Timing Breakdown', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Cumulative processing time
        cumulative_time = np.cumsum(total_frame_times)
        axes[1, 0].plot(frame_indices, cumulative_time, 'g-', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('Cumulative Processing Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Frame Index', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Time (seconds)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # FPS over time (rolling average)
        if len(total_frame_times) > 1:
            fps_over_time = [1.0/t if t > 0 else 0 for t in total_frame_times]
            # Simple rolling average
            window_size = min(10, len(fps_over_time) // 10)
            if window_size > 1:
                rolling_fps = []
                for i in range(len(fps_over_time)):
                    start = max(0, i - window_size // 2)
                    end = min(len(fps_over_time), i + window_size // 2 + 1)
                    rolling_fps.append(np.mean(fps_over_time[start:end]))
            else:
                rolling_fps = fps_over_time
            
            axes[1, 1].plot(frame_indices, rolling_fps, 'r-', alpha=0.7, linewidth=1)
            axes[1, 1].set_title('FPS Over Time (Rolling Average)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Frame Index', fontsize=12)
            axes[1, 1].set_ylabel('FPS', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        safe_filename = result["video_name"].replace('.', '_')
        plt.savefig(os.path.join(output_dir, f'detailed_timing_{safe_filename}_tile{result["tile_size"]}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Tile size comparison (if multiple tile sizes exist)
    if len(set(tile_sizes)) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Speed Performance Comparison by Tile Size', fontsize=16, fontweight='bold')
        
        for i, (metric_name, values) in enumerate([('FPS', fps_values), ('Frame Time', frame_times), ('Total Time', total_times)]):
            # Group by tile size
            tile_size_data = {}
            for tile_size, value in zip(tile_sizes, values):
                if tile_size not in tile_size_data:
                    tile_size_data[tile_size] = []
                tile_size_data[tile_size].append(value)
            
            # Create box plot
            bp = axes[i].boxplot([tile_size_data[ts] for ts in sorted(tile_size_data.keys())], 
                                labels=[f'Tile {ts}' for ts in sorted(tile_size_data.keys())],
                                patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'{metric_name} by Tile Size', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tile_size_speed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Matplotlib speed visualizations saved to {output_dir}")
    print("Generated files:")
    print("  - speed_overview.png")
    print("  - detailed_timing_[video]_tile[size].png (for each video)")
    if len(set(tile_sizes)) > 1:
        print("  - tile_size_speed_comparison.png")
    print("  - speed_results.csv")


def main(args):
    """
    Main function that orchestrates the tracking speed performance analysis process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with tracking results and runtime data for the specified dataset and tile size(s)
    2. Analyzes runtime performance metrics from the tracking execution
    3. Creates comprehensive performance summaries and visualizations
    4. Saves results in multiple formats (JSON, TXT, CSV, PNG)
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects runtime data from 060_exec_track.py in:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/proxy_{tile_size}/runtimes.jsonl
        - Results are saved to the specified output directory
        - Performance metrics include FPS, frame processing time, and step-by-step timing breakdown
    """
    print(f"Starting tracking speed performance analysis for dataset: {args.dataset}")
    print(f"Tile size(s): {args.tile_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Find tracking results with runtime data
    video_tile_combinations = find_tracking_results(CACHE_DIR, args.dataset, args.tile_size)
    
    if not video_tile_combinations:
        print("No tracking results with runtime data found. Please ensure 060_exec_track.py has been run first.")
        return
    
    print(f"Found {len(video_tile_combinations)} video-tile size combinations to analyze")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze runtime performance
    results = []
    
    # Prepare arguments for parallel processing
    analysis_args = []
    for video_name, tile_size in video_tile_combinations:
        runtime_path = os.path.join(CACHE_DIR, args.dataset, video_name, 
                                    'uncompressed_tracking', f'proxy_{tile_size}', 'runtimes.jsonl')
        analysis_args.append((video_name, tile_size, runtime_path, args.output_dir))
    
    # Run analysis in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(analyze_runtime_performance, analysis_args)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nAnalysis completed:")
    print(f"  Successful analyses: {len(successful_results)}")
    print(f"  Failed analyses: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed analyses:")
        for result in failed_results:
            print(f"  {result['video_name']} (tile size {result['tile_size']}): {result['error']}")
    
    if successful_results:
        # Create summary
        create_speed_summary(successful_results, args.output_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        fps_values = [r['fps'] for r in successful_results]
        frame_times = [r['avg_frame_time'] for r in successful_results]
        total_times = [r['total_processing_time'] for r in successful_results]
        
        print(f"  FPS: Mean={np.mean(fps_values):.2f}, Std={np.std(fps_values):.2f}")
        print(f"  Frame Time: Mean={np.mean(frame_times):.4f}s, Std={np.std(frame_times):.4f}s")
        print(f"  Total Time: Mean={np.mean(total_times):.2f}s, Std={np.std(total_times):.2f}s")
        
        # Create visualizations if requested
        if args.create_plots:
            print("\nCreating speed performance visualizations...")
            create_speed_visualizations(successful_results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main(parse_args())
