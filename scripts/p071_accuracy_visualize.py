#!/usr/local/bin/python

import argparse
from functools import partial
import json
import os
import multiprocessing as mp
import tempfile
import sys
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import csv

from rich.progress import track

sys.path.append('/polyis/modules/TrackEval')
import trackeval
from trackeval.datasets import B3D
from trackeval.metrics import HOTA, CLEAR, Identity

from polyis.utilities import CACHE_DIR


TILE_SIZES = [30, 60]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - metrics (str): Comma-separated list of metrics to evaluate (default: 'HOTA,CLEAR,Identity')
            - parallel (bool): Whether to use parallel processing (default: True)
            - no_recompute (bool): Whether to use saved results instead of recomputing (default: False)
    """
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval and create visualizations')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR',  #,Identity',
                        help='Comma-separated list of metrics to evaluate')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Whether to use parallel processing')
    parser.add_argument('--no_recompute', action='store_true', default=False,
                        help='Use saved accuracy results from detailed_results.json instead of recomputing')
    return parser.parse_args()


def load_saved_results(dataset: str) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from detailed_results.json.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    results_file = os.path.join(CACHE_DIR, 'summary', dataset, 'accuracy', 'detailed_results.json')
    
    if not os.path.exists(results_file):
        print(f"No saved results found at {results_file}")
        return []
    
    print(f"Loading saved results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} saved evaluation results")
    return results


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
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
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


def get_results(eval_task: Callable[[], dict], res_queue: "mp.Queue[tuple[int, dict]]", worker_id: int):
    result = eval_task()
    res_queue.put((worker_id, result))


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
        - Multiple metrics are evaluated: HOTA, CLEAR (MOTA), and Identity (IDF1)
    """
    print(f"Starting tracking accuracy evaluation for dataset: {args.dataset}")
    
    # Parse metrics (needed for both compute and no_recompute modes)
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    
    # Check if we should use saved results instead of recomputing
    print("Using saved accuracy results...")
    results = load_saved_results(args.dataset)
    if not results:
        print("No saved results found. Please run without --no_recompute first to generate results.")
        return
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    

    print(f"\nEvaluation completed:")
    print(f"  Successful evaluations: {len(successful_results)}")
    print(f"  Failed evaluations: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed evaluations:")
        for result in failed_results:
            error_msg = result.get('error', 'Unknown error')
            print(f"  {result['video_name']} (tile size {result['tile_size']}): {error_msg}")
    

    output_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'accuracy')

    # Create summary
    create_simple_summary(successful_results, output_dir)
    
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
    visualize_tracking_accuracy(successful_results, output_dir)
    print(f"\nResults saved to: {output_dir}")


def visualize_tracking_accuracy(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for tracking accuracy results using matplotlib.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for visualizations
    """
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
    classifiers = []
    combined_labels = []
    hota_scores = []
    clear_scores = []
    identity_scores = []
    
    for result in successful_results:
        video_names.append(result['video_name'])
        tile_sizes.append(result['tile_size'])
        classifiers.append(result['classifier'])
        
        # Create combined label with video name, classifier, and tile size
        combined_label = f"{result['video_name']}\n{result['classifier']}"
        combined_labels.append(combined_label)
        
        metrics = result['metrics']
        hota_scores.append(metrics.get('HOTA', {}).get('HOTA(0)', 0.0))
        clear_scores.append(metrics.get('CLEAR', {}).get('MOTA', 0.0))
        identity_scores.append(metrics.get('Identity', {}).get('IDF1', 0.0))
    
    # Save results to CSV using native Python csv module
    csv_file_path = os.path.join(output_dir, 'accuracy_results.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Video', 'Classifier', 'Tile_Size', 'HOTA', 'MOTA', 'IDF1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(video_names)):
            writer.writerow({
                'Video': video_names[i],
                'Classifier': classifiers[i],
                'Tile_Size': tile_sizes[i],
                'HOTA': hota_scores[i],
                'MOTA': clear_scores[i],
                'IDF1': identity_scores[i]
            })
    
    # Set matplotlib style
    plt.style.use('default')
    
    # 1. Overall accuracy comparison - arranged vertically with flipped axes
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle('Tracking Accuracy Comparison Across Configurations', fontsize=16, fontweight='bold')
    
    # HOTA scores
    bars1 = axes[0].barh(range(len(combined_labels)), hota_scores, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    # axes[0].set_title('HOTA Scores', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('HOTA Score', fontsize=12)
    axes[0].set_ylabel('Video / Classifier / Tile Size', fontsize=12)
    axes[0].set_yticks(range(len(combined_labels)))
    axes[0].set_yticklabels(combined_labels, fontsize=9, ha='left')
    axes[0].tick_params(axis='y', length=0)  # Remove y-axis ticks
    axes[0].yaxis.set_tick_params(pad=-5)    # Add 10 padding to y-axis labels
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(0, 1.0)  # Stretch x-axis so 1.0 is at the end
    
    # Add value labels inside bars
    for i, (bar, score) in enumerate(zip(bars1, hota_scores)):
        width = bar.get_width()
        axes[0].text(width * .98, bar.get_y() + bar.get_height()/2.,
                        f'{score:.3f}', ha='right', va='center', fontsize=10, fontweight='bold', color='navy')
    
    # MOTA scores
    bars2 = axes[1].barh(range(len(combined_labels)), clear_scores, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    # axes[1].set_title('MOTA Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('MOTA Score', fontsize=12)
    axes[1].set_ylabel('Video / Classifier / Tile Size', fontsize=12)
    axes[1].set_yticks(range(len(combined_labels)))
    axes[1].set_yticklabels(combined_labels, fontsize=9, ha='left')
    axes[1].tick_params(axis='y', length=0)  # Remove y-axis ticks
    axes[1].yaxis.set_tick_params(pad=-5)    # Add 10 padding to y-axis labels
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(0, 1.0)  # Stretch x-axis so 1.0 is at the end
    
    # Add value labels inside bars
    for i, (bar, score) in enumerate(zip(bars2, clear_scores)):
        width = bar.get_width()
        axes[1].text(width * .98, bar.get_y() + bar.get_height()/2.,
                        f'{score:.3f}', ha='right', va='center', fontsize=10, fontweight='bold', color='darkred')
    
    # IDF1 scores
    bars3 = axes[2].barh(range(len(combined_labels)), identity_scores, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    # axes[2].set_title('IDF1 Scores', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('IDF1 Score', fontsize=12)
    axes[2].set_ylabel('Video / Classifier / Tile Size', fontsize=12)
    axes[2].set_yticks(range(len(combined_labels)))
    axes[2].set_yticklabels(combined_labels, fontsize=9, ha='left')
    axes[2].tick_params(axis='y', length=0)  # Remove y-axis ticks
    axes[2].yaxis.set_tick_params(pad=-5)    # Add 10 padding to y-axis labels
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim(0, 1.0)  # Stretch x-axis so 1.0 is at the end
    
    # Add value labels inside bars
    for i, (bar, score) in enumerate(zip(bars3, identity_scores)):
        width = bar.get_width()
        axes[2].text(width * .98, bar.get_y() + bar.get_height()/2.,
                        f'{score:.3f}', ha='right', va='center', fontsize=10, fontweight='bold', color='darkgreen')
    
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
    
    print(f"Matplotlib visualizations saved to {output_dir}")
    print("Generated files:")
    print("  - accuracy_comparison.png")
    if len(set(tile_sizes)) > 1:
        print("  - tile_size_comparison.png")
    print("  - accuracy_results.csv")


if __name__ == '__main__':
    main(parse_args())
