#!/usr/local/bin/python

import argparse
import json
import os
import multiprocessing as mp
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    """
    parser = argparse.ArgumentParser(description='Evaluate tracking accuracy using TrackEval and create visualizations')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR',  #,Identity',
                        help='Comma-separated list of metrics to evaluate')
    return parser.parse_args()


def find_saved_results(cache_dir: str, dataset: str) -> List[Tuple[str, str, int]]:
    """
    Find all video files with saved accuracy results for the specified dataset.
    
    Args:
        cache_dir (str): Cache directory path
        dataset (str): Dataset name
        
    Returns:
        List[Tuple[str, str, int]]: List of (video_name, classifier, tile_size) tuples
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset)
    if not os.path.exists(dataset_cache_dir):
        print(f"Dataset cache directory {dataset_cache_dir} does not exist")
        return []
    
    video_tile_combinations: list[tuple[str, str, int]] = []
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        assert os.path.isdir(video_dir)
            
        evaluation_dir = os.path.join(video_dir, 'evaluation')
        assert os.path.exists(evaluation_dir)

        for classifier_tilesize in os.listdir(evaluation_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            results_path = os.path.join(evaluation_dir, f'{classifier}_{ts}',
                                        'accuracy', 'detailed_results.json')
            
            if os.path.exists(results_path):
                video_tile_combinations.append((video_filename, classifier, ts))
    
    return video_tile_combinations


def load_saved_results(dataset: str) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from individual detailed_results.json files.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    # Find all saved results
    video_tile_combinations = find_saved_results(CACHE_DIR, dataset)
    
    if not video_tile_combinations:
        print("No saved results found. Please run p070_accuracy_compute.py first to generate results.")
        return []
    
    results = []
    for video_name, classifier, tile_size in video_tile_combinations:
        results_path = os.path.join(CACHE_DIR, dataset, video_name,
                                    'evaluation', f'{classifier}_{tile_size}',
                                    'accuracy', 'detailed_results.json')
        
        print(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            results.append(json.load(f))
    
    print(f"Loaded {len(results)} saved evaluation results")
    return results


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
    assert len(results) > 0
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"  Successful evaluations: {len(successful_results)}")
    print(f"  Failed evaluations: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed evaluations:")
        for result in failed_results:
            error_msg = result.get('error', 'Unknown error')
            print(f"  {result['video_name']} (tile size {result['tile_size']}): {error_msg}")
    assert len(failed_results) == 0
    
    output_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'accuracy')
    
    # Optionally create plots if requested
    visualize_tracking_accuracy(successful_results, output_dir)
    print(f"Results saved to: {output_dir}")


def visualize_compared_accuracy_bar(video_tile_groups: Dict[str, Dict[int, Dict[str, List]]], 
                            sorted_videos: List[str], sorted_tile_sizes: List[int],
                            num_videos: int, num_tile_sizes: int, score_field: str,
                            metric_name: str, xlabel: str, output_path: str) -> None:
    """
    Create a comparison plot for tracking accuracy scores by video and tile size.
    
    Args:
        video_tile_groups: Grouped data by video and tile size
        sorted_videos: List of video names in sorted order
        sorted_tile_sizes: List of tile sizes in sorted order
        num_videos: Number of videos
        num_tile_sizes: Number of tile sizes
        score_field: Field name for scores ('hota_scores' or 'clear_scores')
        metric_name: Display name for the metric ('HOTA' or 'MOTA')
        xlabel: Label for x-axis
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(num_videos, num_tile_sizes, figsize=(4 * num_tile_sizes, 3 * num_videos))
    if num_videos == 1 and num_tile_sizes == 1:
        axes = [[axes]]  # Make it a 2D array for single subplot
    elif num_videos == 1:
        axes = [axes]  # Make it a 2D array for single row
    elif num_tile_sizes == 1:
        axes = [[ax] for ax in axes]  # Make it a 2D array for single column
    
    fig.suptitle(f'{metric_name} Scores Comparison by Video and Tile Size')
    
    for row, video_name in enumerate(sorted_videos):
        for col, tile_size in enumerate(sorted_tile_sizes):
            ax = axes[row][col]
            
            # Check if this video has data for this tile size
            if tile_size in video_tile_groups[video_name]:
                group_data = video_tile_groups[video_name][tile_size]
                
                # Sort by scores (descending)
                sorted_indices = sorted(range(len(group_data[score_field])), 
                                       key=lambda x: group_data[score_field][x], reverse=True)
                
                sorted_labels = [group_data['labels'][idx] for idx in sorted_indices]
                sorted_scores = [group_data[score_field][idx] for idx in sorted_indices]
                
                bars = ax.barh(range(len(sorted_labels)), sorted_scores, color='cornflowerblue')
                
                # Add labels inside bars (to the right) with dark blue text
                for bar, label, score in zip(bars, sorted_labels, sorted_scores):
                    # Label inside the bar (to the right)
                    ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2.,
                           label, ha='left', va='center', color='darkblue', fontweight='bold')
                    
                    # Score at the end of the bar (only if score is not 0)
                    if score != 0:
                        ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height()/2.,
                               f'{score:.2f}', ha='right', va='center', color='darkblue', fontweight='bold')
                
                # Remove y-axis ticks
                ax.set_yticks([])
            else:
                # No data for this video/tile size combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
            
            # Set titles and labels
            if row == 0:
                ax.set_title(f'Tile Size {tile_size}')
            if col == 0:
                ax.set_ylabel(video_name)
            if row == num_videos - 1:
                ax.set_xlabel(xlabel)
            
            ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_tracking_accuracy(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations for tracking accuracy results using seaborn and matplotlib.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Output directory for visualizations
    """
    print("Creating visualizations...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful evaluation results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier data handling
    data = []
    for result in successful_results:
        metrics = result['metrics']
        data.append({
            'Video': result['video_name'],
            'Classifier': result['classifier'],
            'Tile_Size': result['tile_size'],
            'HOTA': metrics.get('HOTA', {}).get('HOTA(0)', 0.0),
            'MOTA': metrics.get('CLEAR', {}).get('MOTA', 0.0)
        })
    
    df = pd.DataFrame(data)
    
    # Save results to CSV
    csv_file_path = os.path.join(output_dir, 'accuracy_results.csv')
    df.to_csv(csv_file_path, index=False)
    
    # Group data by video and tile size for bar plots
    video_tile_groups = {}
    for _, row in df.iterrows():
        video_name = row['Video']
        tile_size = row['Tile_Size']
        if video_name not in video_tile_groups:
            video_tile_groups[video_name] = {}
        if tile_size not in video_tile_groups[video_name]:
            video_tile_groups[video_name][tile_size] = {
                'labels': [],
                'hota_scores': [],
                'clear_scores': []
            }
        video_tile_groups[video_name][tile_size]['labels'].append(row['Classifier'])
        video_tile_groups[video_name][tile_size]['hota_scores'].append(row['HOTA'])
        video_tile_groups[video_name][tile_size]['clear_scores'].append(row['MOTA'])
    
    # Sort videos and tile sizes for consistent ordering
    sorted_videos = sorted(video_tile_groups.keys())
    sorted_tile_sizes = sorted(df['Tile_Size'].unique())
    num_videos = len(sorted_videos)
    num_tile_sizes = len(sorted_tile_sizes)
    
    # Create comparison plots for both HOTA and MOTA scores
    visualize_compared_accuracy_bar(
        video_tile_groups, sorted_videos, sorted_tile_sizes, num_videos, num_tile_sizes,
        'hota_scores', 'HOTA', 'HOTA Score',
        os.path.join(output_dir, 'hota_comparison.png')
    )
    
    visualize_compared_accuracy_bar(
        video_tile_groups, sorted_videos, sorted_tile_sizes, num_videos, num_tile_sizes,
        'clear_scores', 'MOTA', 'MOTA Score',
        os.path.join(output_dir, 'mota_comparison.png')
    )
    
    # Tile size comparison using seaborn (if multiple tile sizes exist)
    if len(df['Tile_Size'].unique()) > 1:
        _, axes = plt.subplots(1, 2, figsize=(5, 5))
        
        # HOTA comparison
        sns.boxplot(data=df, x='Tile_Size', y='HOTA', ax=axes[0])
        axes[0].set_title('HOTA by Tile Size')
        axes[0].set_ylabel('HOTA Score')
        
        # MOTA comparison
        sns.boxplot(data=df, x='Tile_Size', y='MOTA', ax=axes[1])
        axes[1].set_title('MOTA by Tile Size')
        axes[1].set_ylabel('MOTA Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tile_size_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main(parse_args())

