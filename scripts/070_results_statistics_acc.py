#!/usr/local/bin/python

import argparse
import json
import os
import sys
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
import tempfile

# Add TrackEval to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules', 'TrackEval')))
import trackeval
from trackeval.datasets import B3D
from trackeval.metrics import HOTA, CLEAR, Identity

# Optional imports for visualization
import matplotlib.pyplot as plt
import pandas as pd
VISUALIZATION_AVAILABLE = True

CACHE_DIR = '/polyis-cache'
OUTPUT_DIR = 'pipeline-stages/track-accuracy-results'


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - tile_size (str): Tile size to use for evaluation (choices: '64', '128', 'all')
            - metrics (str): Comma-separated list of metrics to evaluate (default: 'HOTA,CLEAR,Identity')
            - output_dir (str): Output directory for results (default: 'pipeline-stages/track-accuracy-results')
            - parallel (bool): Whether to use parallel processing (default: True)
            - num_cores (int): Number of parallel cores to use (default: 8)
            - create_plots (bool): Whether to create visualization plots (default: False)
    """
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval and create visualizations')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--tile_size', type=str, choices=['64', '128', 'all'], default='all',
                        help='Tile size to use for evaluation (or "all" for all tile sizes)')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR,Identity',
                        help='Comma-separated list of metrics to evaluate')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Whether to use parallel processing')
    parser.add_argument('--num_cores', type=int, default=8,
                        help='Number of parallel cores to use')
    parser.add_argument('--create_plots', action='store_true', default=False,
                        help='Whether to create visualization plots (requires matplotlib/seaborn)')
    return parser.parse_args()


def find_tracking_results(cache_dir: str, dataset: str, tile_size: str) -> List[Tuple[str, int]]:
    """
    Find all video files with tracking results for the specified dataset and tile size.
    
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
    if tile_size == 'all':
        tile_sizes_to_process = [64, 128]
    else:
        tile_sizes_to_process = [int(tile_size)]
    
    for item in os.listdir(dataset_cache_dir):
        item_path = os.path.join(dataset_cache_dir, item)
        if os.path.isdir(item_path):
            for ts in tile_sizes_to_process:
                tracking_path = os.path.join(item_path, 'uncompressed_tracking', f'proxy_{ts}', 'tracking.jsonl')
                groundtruth_path = os.path.join(item_path, 'groundtruth', 'tracking.jsonl')
                
                if os.path.exists(tracking_path) and os.path.exists(groundtruth_path):
                    video_tile_combinations.append((item, ts))
                    print(f"Found tracking results: {item} with tile size {ts}")
    
    return video_tile_combinations


def load_tracking_data(file_path: str) -> Dict[int, List[List[float]]]:
    """
    Load tracking data from JSONL file.
    
    Args:
        file_path (str): Path to the tracking JSONL file
        
    Returns:
        Dict[int, List[List[float]]]: Dictionary mapping frame indices to list of detections
    """
    frame_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                frame_idx = data['frame_idx']
                tracks = data['tracks']
                frame_data[frame_idx] = tracks
    
    return frame_data


def load_groundtruth_data(file_path: str) -> Dict[int, List[List[float]]]:
    """
    Load groundtruth data from JSONL file.
    
    Args:
        file_path (str): Path to the groundtruth JSONL file
        
    Returns:
        Dict[int, List[List[float]]]: Dictionary mapping frame indices to list of groundtruth detections
    """
    frame_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                frame_idx = data['frame_idx']
                detections = data['detections'] if 'detections' in data else data.get('tracks', [])
                frame_data[frame_idx] = detections
    
    return frame_data


def convert_to_trackeval_format(frame_data: Dict[int, List[List[float]]], is_gt: bool = False) -> str:
    """
    Convert frame data to TrackEval format and save to temporary file.
    
    Args:
        frame_data (Dict[int, List[List[float]]]): Frame data dictionary
        is_gt (bool): Whether this is groundtruth data
        
    Returns:
        str: Path to temporary file in TrackEval format
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    # Sort frames and write data
    for frame_idx in sorted(frame_data.keys()):
        detections = frame_data[frame_idx]
        
        if is_gt:
            # Groundtruth format: [track_id, x1, y1, x2, y2]
            formatted_dets = detections
        else:
            # Tracking format: [track_id, x1, y1, x2, y2]
            formatted_dets = detections
        
        # Write in TrackEval format: [frame_idx, detections]
        line_data = [frame_idx, formatted_dets]
        temp_file.write(json.dumps(line_data) + '\n')
    
    temp_file.close()
    return temp_file.name


def evaluate_tracking_accuracy(video_name: str, tile_size: int, tracking_path: str, 
                              groundtruth_path: str, metrics_list: List[str], 
                              output_dir: str) -> Dict[str, Any]:
    """
    Evaluate tracking accuracy for a single video using TrackEval.
    
    Args:
        video_name (str): Name of the video
        tile_size (int): Tile size used
        tracking_path (str): Path to tracking results
        groundtruth_path (str): Path to groundtruth data
        metrics_list (List[str]): List of metrics to evaluate
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"Evaluating {video_name} with tile size {tile_size}")
    
    try:
        # Load data
        tracking_data = load_tracking_data(tracking_path)
        groundtruth_data = load_groundtruth_data(groundtruth_path)
        
        # Convert to TrackEval format
        temp_tracking_file = convert_to_trackeval_format(tracking_data, is_gt=False)
        temp_groundtruth_file = convert_to_trackeval_format(groundtruth_data, is_gt=True)
        
        # Create dataset configuration
        dataset_config = {
            'output_fol': output_dir,
            'output_sub_fol': f'{video_name}_tile{tile_size}',
            'input_gt': temp_groundtruth_file,
            'input_track': temp_tracking_file,
            'skip': 1,  # Process every frame
            'tracker': f'tile{tile_size}'
        }
        
        # Create evaluator configuration
        eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'BREAK_ON_ERROR': False,
            'PRINT_RESULTS': False,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': False
        }
        
        # Create metrics
        metrics = []
        for metric_name in metrics_list:
            if metric_name == 'HOTA':
                metrics.append(HOTA({'THRESHOLD': 0.5}))
            elif metric_name == 'CLEAR':
                metrics.append(CLEAR({'THRESHOLD': 0.5}))
            elif metric_name == 'Identity':
                metrics.append(Identity({'THRESHOLD': 0.5}))
        
        # Create dataset and evaluator
        dataset = B3D(dataset_config)
        evaluator = trackeval.Evaluator(eval_config)
        
        # Run evaluation
        results = evaluator.evaluate([dataset], metrics)
        
        # Clean up temporary files
        os.unlink(temp_tracking_file)
        os.unlink(temp_groundtruth_file)
        
        # Extract summary results
        summary_results = {}
        for metric in metrics:
            metric_name = metric.get_name()
            if metric_name in results:
                metric_results = results[metric_name]
                if 'COMBINED_SEQ' in metric_results:
                    combined = metric_results['COMBINED_SEQ']
                    if 'car' in combined:  # B3D uses 'car' class
                        summary_results[metric_name] = combined['car']
        
        return {
            'video_name': video_name,
            'tile_size': tile_size,
            'metrics': summary_results,
            'success': True
        }
        
    except Exception as e:
        print(f"Error evaluating {video_name} with tile size {tile_size}: {str(e)}")
        return {
            'video_name': video_name,
            'tile_size': tile_size,
            'error': str(e),
            'success': False
        }


def create_simple_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create a simple text summary of the accuracy results.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for results
    """
    print("Creating accuracy summary...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful evaluation results to summarize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for summary
    video_names = []
    tile_sizes = []
    hota_scores = []
    clear_scores = []
    identity_scores = []
    
    for result in successful_results:
        video_names.append(result['video_name'])
        tile_sizes.append(result['tile_size'])
        
        metrics = result['metrics']
        hota_scores.append(metrics.get('HOTA', {}).get('HOTA(0)', 0.0))
        clear_scores.append(metrics.get('CLEAR', {}).get('MOTA', 0.0))
        identity_scores.append(metrics.get('Identity', {}).get('IDF1', 0.0))
    
    # Save detailed results to JSON
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create text summary
    summary_file = os.path.join(output_dir, 'accuracy_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Tracking Accuracy Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Individual Video Results:\n")
        f.write("-" * 30 + "\n")
        for i, video_name in enumerate(video_names):
            f.write(f"Video: {video_name}\n")
            f.write(f"  Tile Size: {tile_sizes[i]}\n")
            f.write(f"  HOTA: {hota_scores[i]:.4f}\n")
            f.write(f"  MOTA: {clear_scores[i]:.4f}\n")
            f.write(f"  IDF1: {identity_scores[i]:.4f}\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"HOTA - Mean: {np.mean(hota_scores):.4f}, Std: {np.std(hota_scores):.4f}\n")
        f.write(f"      Min: {np.min(hota_scores):.4f}, Max: {np.max(hota_scores):.4f}\n")
        f.write(f"MOTA - Mean: {np.mean(clear_scores):.4f}, Std: {np.std(clear_scores):.4f}\n")
        f.write(f"      Min: {np.min(clear_scores):.4f}, Max: {np.max(clear_scores):.4f}\n")
        f.write(f"IDF1 - Mean: {np.mean(identity_scores):.4f}, Std: {np.std(identity_scores):.4f}\n")
        f.write(f"      Min: {np.min(identity_scores):.4f}, Max: {np.max(identity_scores):.4f}\n")
        
        # Tile size analysis if multiple tile sizes
        if len(set(tile_sizes)) > 1:
            f.write("\nTile Size Analysis:\n")
            f.write("-" * 20 + "\n")
            for tile_size in sorted(set(tile_sizes)):
                tile_indices = [i for i, ts in enumerate(tile_sizes) if ts == tile_size]
                tile_hota = [hota_scores[i] for i in tile_indices]
                tile_mota = [clear_scores[i] for i in tile_indices]
                tile_idf1 = [identity_scores[i] for i in tile_indices]
                
                f.write(f"Tile Size {tile_size}:\n")
                f.write(f"  HOTA: Mean={np.mean(tile_hota):.4f}, Std={np.std(tile_hota):.4f}\n")
                f.write(f"  MOTA: Mean={np.mean(tile_mota):.4f}, Std={np.std(tile_mota):.4f}\n")
                f.write(f"  IDF1: Mean={np.mean(tile_idf1):.4f}, Std={np.std(tile_idf1):.4f}\n\n")
    
    print(f"Summary saved to {summary_file}")


def main(args):
    """
    Main function that orchestrates the tracking accuracy evaluation process.
    
    This function serves as the entry point for the script. It:
    1. Finds all videos with tracking results for the specified dataset and tile size(s)
    2. Runs accuracy evaluation using TrackEval's B3D evaluation methods
    3. Creates summary reports of the accuracy results
    4. Optionally creates visualizations if requested and libraries are available
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tracking results from 060_exec_track.py in:
          {CACHE_DIR}/{dataset}/{video_file}/uncompressed_tracking/proxy_{tile_size}/tracking.jsonl
        - Groundtruth data should be in:
          {CACHE_DIR}/{dataset}/{video_file}/groundtruth/tracking.jsonl
        - Results are saved to the specified output directory
        - Multiple metrics are evaluated: HOTA, CLEAR (MOTA), and Identity (IDF1)
    """
    print(f"Starting tracking accuracy evaluation for dataset: {args.dataset}")
    print(f"Tile size(s): {args.tile_size}")
    print(f"Metrics: {args.metrics}")
    print(f"Output directory: {args.output_dir}")
    
    # Parse metrics
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    print(f"Evaluating metrics: {metrics_list}")
    
    # Find tracking results
    video_tile_combinations = find_tracking_results(CACHE_DIR, args.dataset, args.tile_size)
    
    if not video_tile_combinations:
        print("No tracking results found. Please ensure 060_exec_track.py has been run first.")
        return
    
    print(f"Found {len(video_tile_combinations)} video-tile size combinations to evaluate")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate tracking accuracy
    results = []
    
    if args.parallel and len(video_tile_combinations) > 1:
        print(f"Using parallel processing with {args.num_cores} cores")
        
        # Prepare arguments for parallel processing
        eval_args = []
        for video_name, tile_size in video_tile_combinations:
            tracking_path = os.path.join(CACHE_DIR, args.dataset, video_name, 
                                       'uncompressed_tracking', f'proxy_{tile_size}', 'tracking.jsonl')
            groundtruth_path = os.path.join(CACHE_DIR, args.dataset, video_name, 
                                          'groundtruth', 'tracking.jsonl')
            
            eval_args.append((video_name, tile_size, tracking_path, groundtruth_path, 
                            metrics_list, args.output_dir))
        
        # Run evaluation in parallel
        with mp.Pool(processes=args.num_cores) as pool:
            results = pool.starmap(evaluate_tracking_accuracy, eval_args)
    else:
        print("Using sequential processing")
        
        # Run evaluation sequentially
        for video_name, tile_size in video_tile_combinations:
            tracking_path = os.path.join(CACHE_DIR, args.dataset, video_name, 
                                       'uncompressed_tracking', f'proxy_{tile_size}', 'tracking.jsonl')
            groundtruth_path = os.path.join(CACHE_DIR, args.dataset, video_name, 
                                          'groundtruth', 'tracking.jsonl')
            
            result = evaluate_tracking_accuracy(video_name, tile_size, tracking_path, 
                                             groundtruth_path, metrics_list, args.output_dir)
            results.append(result)
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nEvaluation completed:")
    print(f"  Successful evaluations: {len(successful_results)}")
    print(f"  Failed evaluations: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed evaluations:")
        for result in failed_results:
            print(f"  {result['video_name']} (tile size {result['tile_size']}): {result['error']}")
    
    if successful_results:
        # Create summary
        create_simple_summary(successful_results, args.output_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for metric_name in metrics_list:
            scores = []
            for result in successful_results:
                if metric_name in result['metrics']:
                    if metric_name == 'HOTA':
                        scores.append(result['metrics'][metric_name].get('HOTA(0)', 0.0))
                    elif metric_name == 'CLEAR':
                        scores.append(result['metrics'][metric_name].get('MOTA', 0.0))
                    elif metric_name == 'Identity':
                        scores.append(result['metrics'][metric_name].get('IDF1', 0.0))
            
            if scores:
                print(f"  {metric_name}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
        
        # Optionally create plots if requested
        if args.create_plots:
            print("\nAttempting to create visualization plots...")
            if VISUALIZATION_AVAILABLE:
                print("Visualization libraries available. Creating plots...")
                create_matplotlib_visualizations(successful_results, args.output_dir)
            else:
                print("Visualization libraries not available.")
                print("Install matplotlib and pandas to enable plotting.")
                print("Required: pip install matplotlib pandas")
    
    print(f"\nResults saved to: {args.output_dir}")


def create_matplotlib_visualizations(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for tracking accuracy results using matplotlib.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for visualizations
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping plots.")
        return
        
    print("Creating matplotlib visualizations...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful evaluation results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    video_names = []
    tile_sizes = []
    hota_scores = []
    clear_scores = []
    identity_scores = []
    
    for result in successful_results:
        video_names.append(result['video_name'])
        tile_sizes.append(result['tile_size'])
        
        metrics = result['metrics']
        hota_scores.append(metrics.get('HOTA', {}).get('HOTA(0)', 0.0))
        clear_scores.append(metrics.get('CLEAR', {}).get('MOTA', 0.0))
        identity_scores.append(metrics.get('Identity', {}).get('IDF1', 0.0))
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Video': video_names,
        'Tile_Size': tile_sizes,
        'HOTA': hota_scores,
        'MOTA': clear_scores,
        'IDF1': identity_scores
    })
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'accuracy_results.csv'), index=False)
    
    # Set matplotlib style
    plt.style.use('default')
    
    # 1. Overall accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tracking Accuracy Comparison Across Videos', fontsize=16, fontweight='bold')
    
    # HOTA scores
    bars1 = axes[0, 0].bar(range(len(video_names)), hota_scores, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    axes[0, 0].set_title('HOTA Scores', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Video', fontsize=12)
    axes[0, 0].set_ylabel('HOTA Score', fontsize=12)
    axes[0, 0].set_xticks(range(len(video_names)))
    axes[0, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MOTA scores
    bars2 = axes[0, 1].bar(range(len(video_names)), clear_scores, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    axes[0, 1].set_title('MOTA Scores', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Video', fontsize=12)
    axes[0, 1].set_ylabel('MOTA Score', fontsize=12)
    axes[0, 1].set_xticks(range(len(video_names)))
    axes[0, 1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # IDF1 scores
    bars3 = axes[1, 0].bar(range(len(video_names)), identity_scores, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    axes[1, 0].set_title('IDF1 Scores', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Video', fontsize=12)
    axes[1, 0].set_ylabel('IDF1 Score', fontsize=12)
    axes[1, 0].set_xticks(range(len(video_names)))
    axes[1, 0].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Combined scores heatmap
    score_matrix = np.array([hota_scores, clear_scores, identity_scores])
    im = axes[1, 1].imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('Score Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Video', fontsize=12)
    axes[1, 1].set_ylabel('Metric', fontsize=12)
    axes[1, 1].set_xticks(range(len(video_names)))
    axes[1, 1].set_xticklabels(video_names, rotation=45, ha='right')
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(['HOTA', 'MOTA', 'IDF1'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Score', fontsize=12)
    
    # Add text annotations to heatmap
    for i in range(3):
        for j in range(len(video_names)):
            text = axes[1, 1].text(j, i, f'{score_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tile size comparison (if multiple tile sizes exist)
    if len(set(tile_sizes)) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Accuracy Comparison by Tile Size', fontsize=16, fontweight='bold')
        
        for i, (metric_name, scores) in enumerate([('HOTA', hota_scores), ('MOTA', clear_scores), ('IDF1', identity_scores)]):
            # Group by tile size
            tile_size_data = {}
            for tile_size, score in zip(tile_sizes, scores):
                if tile_size not in tile_size_data:
                    tile_size_data[tile_size] = []
                tile_size_data[tile_size].append(score)
            
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
            axes[i].set_ylabel(f'{metric_name} Score', fontsize=12)
            axes[i].grid(True, alpha=0.3, linestyle='--')
            axes[i].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tile_size_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Metric correlation analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Metric Correlation Analysis', fontsize=16, fontweight='bold')
    
    # HOTA vs MOTA
    scatter1 = axes[0].scatter(hota_scores, clear_scores, alpha=0.7, s=100, c='blue', edgecolors='navy', linewidth=1)
    axes[0].set_xlabel('HOTA Score', fontsize=12)
    axes[0].set_ylabel('MOTA Score', fontsize=12)
    axes[0].set_title('HOTA vs MOTA', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(0, 1.1)
    axes[0].set_ylim(0, 1.1)
    
    # Add correlation coefficient
    corr = np.corrcoef(hota_scores, clear_scores)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # HOTA vs IDF1
    scatter2 = axes[1].scatter(hota_scores, identity_scores, alpha=0.7, s=100, c='green', edgecolors='darkgreen', linewidth=1)
    axes[1].set_xlabel('HOTA Score', fontsize=12)
    axes[1].set_ylabel('IDF1 Score', fontsize=12)
    axes[1].set_title('HOTA vs IDF1', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(0, 1.1)
    axes[1].set_ylim(0, 1.1)
    
    # Add correlation coefficient
    corr = np.corrcoef(hota_scores, identity_scores)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1].transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # MOTA vs IDF1
    scatter3 = axes[2].scatter(clear_scores, identity_scores, alpha=0.7, s=100, c='red', edgecolors='darkred', linewidth=1)
    axes[2].set_xlabel('MOTA Score', fontsize=12)
    axes[2].set_ylabel('IDF1 Score', fontsize=12)
    axes[2].set_title('MOTA vs IDF1', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim(0, 1.1)
    axes[2].set_ylim(0, 1.1)
    
    # Add correlation coefficient
    corr = np.corrcoef(clear_scores, identity_scores)[0, 1]
    axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[2].transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary statistics table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate summary statistics
    summary_data = []
    for metric_name, scores in [('HOTA', hota_scores), ('MOTA', clear_scores), ('IDF1', identity_scores)]:
        summary_data.append([
            metric_name,
            f"{np.mean(scores):.4f}",
            f"{np.std(scores):.4f}",
            f"{np.min(scores):.4f}",
            f"{np.max(scores):.4f}",
            f"{np.median(scores):.4f}"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(summary_data) + 1):
        for j in range(len(summary_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Accuracy Metrics Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib visualizations saved to {output_dir}")
    print("Generated files:")
    print("  - accuracy_comparison.png")
    if len(set(tile_sizes)) > 1:
        print("  - tile_size_comparison.png")
    print("  - metric_correlation.png")
    print("  - summary_statistics.png")
    print("  - accuracy_results.csv")


if __name__ == '__main__':
    main(parse_args())
