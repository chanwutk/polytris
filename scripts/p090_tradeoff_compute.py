#!/usr/local/bin/python

import argparse
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from polyis.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST


def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset (str): Dataset name to process (default: 'b3d')
            - metrics (str): Comma-separated list of metrics to evaluate (default: 'HOTA,CLEAR')
    """
    parser = argparse.ArgumentParser(description='Visualize accuracy-throughput tradeoffs')
    parser.add_argument('--dataset', required=False, default='b3d',
                        help='Dataset name to process')
    parser.add_argument('--metrics', type=str, default='HOTA,CLEAR',
                        help='Comma-separated list of metrics to evaluate')
    return parser.parse_args()


def load_accuracy_results(dataset: str) -> List[Dict[str, Any]]:
    """
    Load saved accuracy results from individual detailed_results.json files.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    dataset_cache_dir = os.path.join(CACHE_DIR, dataset)
    if not os.path.exists(dataset_cache_dir):
        print(f"Dataset cache directory {dataset_cache_dir} does not exist")
        return []
    
    results = []
    for video_filename in os.listdir(dataset_cache_dir):
        video_dir = os.path.join(dataset_cache_dir, video_filename)
        if not os.path.isdir(video_dir):
            continue
            
        evaluation_dir = os.path.join(video_dir, 'evaluation')
        if not os.path.exists(evaluation_dir):
            continue

        for classifier_tilesize in os.listdir(evaluation_dir):
            classifier, tilesize = classifier_tilesize.split('_')
            ts = int(tilesize)
            results_path = os.path.join(evaluation_dir, f'{classifier}_{ts}',
                                        'accuracy', 'detailed_results.json')
            
            if os.path.exists(results_path):
                print(f"Loading accuracy results from {results_path}")
                with open(results_path, 'r') as f:
                    results.append(json.load(f))
    
    print(f"Loaded {len(results)} accuracy evaluation results")
    return results


def load_throughput_results(dataset: str) -> Dict[str, Any]:
    """
    Load throughput results from the measurements directory.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        Dict[str, Any]: Throughput measurement data
    """
    measurements_dir = os.path.join(CACHE_DIR, 'summary', dataset, 'throughput', 'measurements')
    
    if not os.path.exists(measurements_dir):
        print(f"Throughput measurements directory {measurements_dir} does not exist")
        return {}
    
    # Load query execution summaries (aggregated timing data)
    query_summaries_file = os.path.join(measurements_dir, 'query_execution_summaries.json')
    if not os.path.exists(query_summaries_file):
        print(f"Query execution summaries file {query_summaries_file} does not exist")
        return {}
    
    with open(query_summaries_file, 'r') as f:
        query_summaries = json.load(f)
    
    # Load metadata
    metadata_file = os.path.join(measurements_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded throughput data with {len(query_summaries)} stage summaries")
    return {
        'summaries': query_summaries,
        'metadata': metadata
    }


def calculate_naive_runtime(video_name: str, query_summaries: Dict[str, Any]) -> float:
    """
    Calculate naive runtime for a specific video.
    
    This matches the naive calculation in p082_throughput_visualize.py,
    which sums the preprocessing stages (detection and tracking).
    
    Args:
        video_name: Name of the video
        query_summaries: Query execution timing data
        
    Returns:
        float: Naive runtime in seconds
    """
    config_key = f"b3d/{video_name}_groundtruth_0"  # Naive uses groundtruth with tile size 0
    naive_runtime = 0.0
    
    # Add preprocessing time (naive approach)
    preprocessing_stages = ['001_preprocess_groundtruth_detection', '002_preprocess_groundtruth_tracking']
    for stage_name, stage_summaries in query_summaries.items():
        if stage_name in preprocessing_stages and config_key in stage_summaries and stage_summaries[config_key]:
            stage_total = np.mean(stage_summaries[config_key])  # Use mean like in p082
            naive_runtime += stage_total
    
    return naive_runtime


def calculate_query_execution_runtime(video_name: str, classifier: str, tile_size: int, 
                                    query_summaries: Dict[str, Any]) -> float:
    """
    Calculate query execution runtime for a specific configuration.
    
    This matches the query execution portion of the bars in p082_throughput_visualize.py,
    ignoring index construction time.
    
    Args:
        video_name: Name of the video
        classifier: Classifier used
        tile_size: Tile size used
        query_summaries: Query execution timing data
        
    Returns:
        float: Query execution runtime in seconds
    """
    config_key = f"b3d/{video_name}_{classifier}_{tile_size}"
    query_runtime = 0.0
    
    # Add query execution time (specific to this tile size)
    query_stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    for stage_name, stage_summaries in query_summaries.items():
        if stage_name in query_stages and config_key in stage_summaries and stage_summaries[config_key]:
            stage_total = np.mean(stage_summaries[config_key])  # Use mean like in p082
            query_runtime += stage_total
    
    return query_runtime


def match_accuracy_throughput_data(accuracy_results: List[Dict[str, Any]], 
                                 throughput_data: Dict[str, Any], dataset: str = 'b3d') -> List[Dict[str, Any]]:
    """
    Match accuracy and throughput data by video/classifier/tilesize combination.
    
    Args:
        accuracy_results: List of accuracy evaluation results
        throughput_data: Throughput measurement data
        
    Returns:
        List[Dict[str, Any]]: Matched data with both accuracy and throughput metrics
    """
    matched_data = []
    query_summaries = throughput_data.get('summaries', {})
    query_timings = throughput_data.get('timings', {})
    
    # Key stages for query execution throughput
    query_stages = ['020_exec_classify', '030_exec_compress', '040_exec_detect', '060_exec_track']
    
    for result in accuracy_results:
        if not result.get('success', False):
            continue
            
        video_name = result['video_name']
        classifier = result['classifier']
        tile_size = result['tile_size']
        
        # Only include classifiers from CLASSIFIERS_TO_TEST
        if classifier not in CLASSIFIERS_TO_TEST:
            continue
        
        # Create config key for throughput lookup (include dataset prefix)
        config_key = f"b3d/{video_name}_{classifier}_{tile_size}"
        
        # Extract accuracy metrics
        metrics = result['metrics']
        hota_score = metrics.get('HOTA', {}).get('HOTA(0)', 0.0)
        mota_score = metrics.get('CLEAR', {}).get('MOTA', 0.0)
        
        # Calculate total query execution time (for reference)
        total_query_time = 0.0
        stage_times = {}
        
        for stage in query_stages:
            if stage in query_summaries and config_key in query_summaries[stage]:
                # Get the first (and typically only) timing for this config
                stage_time = query_summaries[stage][config_key][0] if query_summaries[stage][config_key] else 0.0
                stage_times[stage] = stage_time
                total_query_time += stage_time
        
        # Calculate query execution runtime only
        query_runtime = calculate_query_execution_runtime(video_name, classifier, tile_size, 
                                                        query_summaries)
        
        matched_entry = {
            'video_name': video_name,
            'classifier': classifier,
            'tile_size': tile_size,
            'hota_score': hota_score,
            'mota_score': mota_score,
            'total_query_time': total_query_time,
            'query_runtime': query_runtime,
            'stage_times': stage_times
        }
        
        matched_data.append(matched_entry)
    
    print(f"Matched {len(matched_data)} accuracy-throughput data points")
    return matched_data


def create_tradeoff_visualizations(matched_data: List[Dict[str, Any]], output_dir: str, 
                                 metrics_list: List[str], query_summaries: Dict[str, Any]) -> None:
    """
    Create visualizations showing accuracy-throughput tradeoffs.
    
    Args:
        matched_data: List of matched accuracy-throughput data points
        output_dir: Output directory for visualizations
        metrics_list: List of metrics to visualize
    """
    print("Creating tradeoff visualizations...")
    
    if not matched_data:
        print("No matched data available for visualization")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier data handling
    df = pd.DataFrame(matched_data)
    
    # Save matched data to CSV
    csv_file_path = os.path.join(output_dir, 'accuracy_query_runtime_tradeoff.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Saved matched data to: {csv_file_path}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("tab20")
    
    # Create scatter plots for each metric
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'hota_score'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'mota_score'
            metric_name = 'MOTA'
        else:
            continue
        
        # Create faceted layout - one subplot per video
        unique_videos = sorted(df['video_name'].unique())
        n_videos = len(unique_videos)
        
        # Calculate grid dimensions
        n_cols = min(2, n_videos)  # Max 3 columns
        n_rows = (n_videos + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'{metric_name} vs Query Execution Runtime Tradeoff Analysis (By Video)', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_videos == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Create size mapping for tile sizes
        unique_tile_sizes = sorted(df['tile_size'].unique())
        tile_size_sizes = {ts: 50 + (ts - min(unique_tile_sizes)) * 30 for ts in unique_tile_sizes}
        
        # Create color mapping for classifiers (filter to only include CLASSIFIERS_TO_TEST)
        unique_classifiers = [cls for cls in df['classifier'].unique() if cls in CLASSIFIERS_TO_TEST]
        colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_classifiers)))
        classifier_colors = {cls: colors[i] for i, cls in enumerate(unique_classifiers)}
        
        # Calculate naive runtime for each video
        naive_runtimes = {}
        for video in unique_videos:
            naive_runtime = calculate_naive_runtime(video, query_summaries)
            naive_runtimes[video] = naive_runtime
        
        # Create one subplot per video
        for i, video in enumerate(unique_videos):
            ax = axes[i]
            video_data = df[df['video_name'] == video]
            
            # Plot each classifier with different colors and tile sizes
            for classifier in unique_classifiers:
                classifier_data = video_data[video_data['classifier'] == classifier]
                for tile_size in unique_tile_sizes:
                    tile_classifier_data = classifier_data[classifier_data['tile_size'] == tile_size]
                    if len(tile_classifier_data) > 0:
                        ax.scatter(tile_classifier_data['query_runtime'], tile_classifier_data[accuracy_col], 
                                   c=[classifier_colors[classifier]], 
                                   s=tile_size_sizes[tile_size],
                                   alpha=0.7)
            
            # Add vertical red line for naive runtime
            naive_runtime = naive_runtimes[video]
            ax.axvline(x=naive_runtime, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                      label=f'Naive Runtime ({naive_runtime:.1f}s)' if i == 0 else '')
            
            ax.set_xlabel('Query Execution Runtime (seconds)')
            ax.set_ylabel(f'{metric_name} Score')
            ax.set_title(f'Video: {video}')
            ax.grid(True, alpha=0.3)
            
            # Add separate legends only to the first subplot to avoid clutter
            if i == 0:
                # Create classifier legend
                classifier_legend_elements = []
                for classifier in unique_classifiers:
                    classifier_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                               markerfacecolor=classifier_colors[classifier], 
                                                               markersize=8, label=classifier))
                
                # Create tile size legend
                tile_size_legend_elements = []
                for tile_size in unique_tile_sizes:
                    tile_size_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                               markerfacecolor='gray', 
                                                               markersize=np.sqrt(tile_size_sizes[tile_size])/2, 
                                                               label=f'Tile Size {tile_size}'))
                
                # Add legends to the plot
                legend1 = ax.legend(handles=classifier_legend_elements, title='Classifiers', 
                                  bbox_to_anchor=(1.05, 1), loc='upper left')
                legend2 = ax.legend(handles=tile_size_legend_elements, title='Tile Sizes', 
                                  bbox_to_anchor=(1.05, 0.3), loc='upper left')
                
                # Add the first legend back to the plot (second one overwrites it)
                ax.add_artist(legend1)
        
        # Hide unused subplots if any
        for i in range(n_videos, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric.lower()}_query_runtime_tradeoff.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} tradeoff plot to: {plot_path}")
    
    # Create combined metric comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Accuracy Metrics vs Query Execution Runtime Tradeoff Comparison', fontsize=16, fontweight='bold')
    
    # Create size mapping for tile sizes
    unique_tile_sizes = sorted(df['tile_size'].unique())
    tile_size_sizes = {ts: 50 + (ts - min(unique_tile_sizes)) * 30 for ts in unique_tile_sizes}
    
    # Calculate naive runtime for each video (for combined plot)
    unique_videos = sorted(df['video_name'].unique())
    naive_runtimes = {}
    for video in unique_videos:
        naive_runtime = calculate_naive_runtime(video, query_summaries)
        naive_runtimes[video] = naive_runtime
    
    # HOTA vs Query Runtime
    ax1 = axes[0]
    for tile_size in unique_tile_sizes:
        tile_data = df[df['tile_size'] == tile_size]
        ax1.scatter(tile_data['query_runtime'], tile_data['hota_score'], 
                   s=tile_size_sizes[tile_size], label=f'Tile Size {tile_size}', 
                   alpha=0.7, c='steelblue')
    
    # Add naive runtime lines for each video
    for video, naive_runtime in naive_runtimes.items():
        ax1.axvline(x=naive_runtime, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Naive ({video})' if video == list(naive_runtimes.keys())[0] else '')
    
    ax1.set_xlabel('Query Execution Runtime (seconds)')
    ax1.set_ylabel('HOTA Score')
    ax1.set_title('HOTA vs Query Runtime')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # MOTA vs Query Runtime
    ax2 = axes[1]
    for tile_size in unique_tile_sizes:
        tile_data = df[df['tile_size'] == tile_size]
        ax2.scatter(tile_data['query_runtime'], tile_data['mota_score'], 
                   s=tile_size_sizes[tile_size], label=f'Tile Size {tile_size}', 
                   alpha=0.7, c='steelblue')
    
    # Add naive runtime lines for each video
    for video, naive_runtime in naive_runtimes.items():
        ax2.axvline(x=naive_runtime, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Naive ({video})' if video == list(naive_runtimes.keys())[0] else '')
    
    ax2.set_xlabel('Query Execution Runtime (seconds)')
    ax2.set_ylabel('MOTA Score')
    ax2.set_title('MOTA vs Query Runtime')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'combined_accuracy_query_runtime_tradeoff.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined tradeoff plot to: {combined_plot_path}")
    
    # Create correlation analysis
    create_correlation_analysis(df, output_dir)


def create_correlation_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create correlation analysis between accuracy and throughput metrics.
    
    Args:
        df: DataFrame with matched accuracy-throughput data
        output_dir: Output directory for analysis results
    """
    print("Creating correlation analysis...")
    
    # Calculate correlations using numpy for better type safety
    correlations = {
        'HOTA_vs_QueryRuntime': float(np.corrcoef(df['hota_score'], df['query_runtime'])[0, 1]),
        'MOTA_vs_QueryRuntime': float(np.corrcoef(df['mota_score'], df['query_runtime'])[0, 1]),
        'HOTA_vs_TotalQueryTime': float(np.corrcoef(df['hota_score'], df['total_query_time'])[0, 1]),
        'MOTA_vs_TotalQueryTime': float(np.corrcoef(df['mota_score'], df['total_query_time'])[0, 1]),
        'HOTA_vs_MOTA': float(np.corrcoef(df['hota_score'], df['mota_score'])[0, 1])
    }
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare correlation matrix using numpy
    data_cols = ['hota_score', 'mota_score', 'query_runtime', 'total_query_time']
    corr_matrix = np.corrcoef(df[data_cols].T)
    corr_data = pd.DataFrame(corr_matrix)
    corr_data.columns = data_cols
    corr_data.index = data_cols
    
    # Create heatmap
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, fmt='.3f')
    ax.set_title('Correlation Matrix: Accuracy vs Query Execution Runtime Metrics')
    
    plt.tight_layout()
    corr_plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {corr_plot_path}")
    
    # Save correlation statistics
    corr_stats = {
        'correlations': correlations,
        'summary_statistics': {
            'total_data_points': len(df),
            'videos': df['video_name'].unique().tolist(),
            'classifiers': df['classifier'].unique().tolist(),
            'tile_sizes': sorted(df['tile_size'].unique().tolist()),
            'hota_mean': float(df['hota_score'].mean()),
            'hota_std': float(df['hota_score'].std()),
            'mota_mean': float(df['mota_score'].mean()),
            'mota_std': float(df['mota_score'].std()),
            'query_runtime_mean': float(df['query_runtime'].mean()),
            'query_runtime_std': float(df['query_runtime'].std())
        }
    }
    
    corr_file_path = os.path.join(output_dir, 'correlation_analysis.json')
    with open(corr_file_path, 'w') as f:
        json.dump(corr_stats, f, indent=2)
    print(f"Saved correlation analysis to: {corr_file_path}")
    
    # Print summary
    print("\nCorrelation Analysis Summary:")
    print("=" * 50)
    for metric, corr in correlations.items():
        print(f"{metric:<25}: {corr:>8.3f}")
    
    print(f"\nSummary Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Videos: {df['video_name'].unique().tolist()}")
    print(f"Classifiers: {df['classifier'].unique().tolist()}")
    print(f"Tile sizes: {sorted(df['tile_size'].unique().tolist())}")


def main(args):
    """
    Main function that orchestrates the accuracy-query execution runtime tradeoff analysis.
    
    This function serves as the entry point for the script. It:
    1. Loads accuracy results from p070_accuracy_compute.py
    2. Loads throughput results from p081_throughput_compute.py
    3. Matches the data by video/classifier/tilesize combination
    4. Creates visualizations showing accuracy vs query execution runtime tradeoffs
    5. Performs correlation analysis
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects accuracy results from p070_accuracy_compute.py in:
          {CACHE_DIR}/{dataset}/{video_file}/evaluation/{classifier}_{tile_size}/accuracy/detailed_results.json
        - The script expects throughput results from p081_throughput_compute.py in:
          {CACHE_DIR}/summary/{dataset}/throughput/measurements/query_execution_summaries.json
        - Only query execution runtime is used (index construction time is ignored)
    """
    print(f"Starting accuracy-query execution runtime tradeoff analysis for dataset: {args.dataset}")
    
    # Parse metrics
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    print(f"Analyzing metrics: {metrics_list}")
    
    # Load accuracy results
    print("Loading accuracy results...")
    accuracy_results = load_accuracy_results(args.dataset)
    
    if not accuracy_results:
        print("No accuracy results found. Please run p070_accuracy_compute.py first.")
        return
    
    # Load throughput results
    print("Loading throughput results...")
    throughput_data = load_throughput_results(args.dataset)
    
    if not throughput_data:
        print("No throughput results found. Please run p080_throughput_gather.py and p081_throughput_compute.py first.")
        return
    
    # Match accuracy and throughput data
    print("Matching accuracy and throughput data...")
    matched_data = match_accuracy_throughput_data(accuracy_results, throughput_data, args.dataset)
    
    if not matched_data:
        print("No matching data points found between accuracy and throughput results.")
        return
    
    # Create visualizations
    output_dir = os.path.join(CACHE_DIR, 'summary', args.dataset, 'tradeoff')
    create_tradeoff_visualizations(matched_data, output_dir, metrics_list, throughput_data['summaries'])
    
    print(f"\nAccuracy-query execution runtime tradeoff analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main(parse_args())
