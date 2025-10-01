#!/usr/local/bin/python

import argparse
import os
from typing import List

from rich.progress import track
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, load_tradeoff_data, tradeoff_scatter_and_naive_baseline


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize accuracy-throughput tradeoffs')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def visualize_tradeoff(df_individual: pd.DataFrame, df_aggregated: pd.DataFrame,
                       output_dir: str, metrics_list: List[str], 
                       x_column: str, x_title: str,
                       naive_column: str, plot_suffix: str):
    """
    Create a single tradeoff visualization with configurable x-axis,
    including dataset-wide aggregated subplot.
    
    Args:
        df_individual: DataFrame with individual video data (already includes naive values)
        df_aggregated: DataFrame with aggregated dataset data (already includes naive values)
        output_dir: Output directory for visualizations
        metrics_list: List of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for plot filename
    """
    print(f"Creating {plot_suffix} tradeoff visualizations...")
    
    assert len(df_individual) > 0, \
        f"No individual data available for {plot_suffix} visualization"
    assert len(df_aggregated) > 0, \
        f"No aggregated data available for {plot_suffix} visualization"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base charts
    base_individual = alt.Chart(df_individual)
    base_aggregated = alt.Chart(df_aggregated)
    
    # Create scatter plots for each metric using Altair
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'hota_score'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'mota_score'
            metric_name = 'MOTA'
        else:
            continue
        
        # Create individual video scatter plot and baseline using combined function
        individual_scatter, naive_lines_individual = tradeoff_scatter_and_naive_baseline(
            base_individual, x_column, x_title, accuracy_col, metric_name, naive_column,
        )
        
        # Combine individual video charts
        individual_chart = (individual_scatter + naive_lines_individual).facet(
            facet=alt.Facet('video_name:N', title=None,
                            header=alt.Header(labelExpr="'Video: ' + datum.value")),
            columns=3
        ).resolve_scale(
            x='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff (By Video)',
        )
        
        # Create dataset-wide scatter plot and baseline using combined function
        aggregated_scatter, naive_line_aggregated = tradeoff_scatter_and_naive_baseline(
            base_aggregated, x_column, x_title, accuracy_col, metric_name, naive_column,
        )
        
        # Create dataset-wide chart
        dataset_chart = (aggregated_scatter + naive_line_aggregated).properties(
            title=f'{metric_name} vs {x_title} Tradeoff (Dataset Average)',
            width=400,
            height=300
        )
        
        # Combine individual and dataset charts vertically
        combined_chart = (individual_chart | dataset_chart).resolve_scale(x='independent')
        
        # Save the chart
        plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_tradeoff.png')
        combined_chart.save(plot_path, scale_factor=2)
        print(f"Saved {metric_name} {plot_suffix} tradeoff plot to: {plot_path}")


def visualize_tradeoffs(dataset: str):
    """
    Create both runtime and throughput tradeoff visualizations for a dataset.
    
    Args:
        dataset: Dataset name
    """
    print(f"Creating tradeoff visualizations for dataset: {dataset}")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, dataset, 'evaluation', '091_tradeoff')
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Create runtime visualization
    df_individual_runtime, df_aggregated_runtime = load_tradeoff_data(dataset, 'runtime')
    visualize_tradeoff(
        df_individual_runtime, df_aggregated_runtime, output_dir, metrics_list,
        x_column='query_runtime',
        x_title='Query Execution Runtime (seconds)',
        naive_column='naive_runtime',
        plot_suffix='runtime'
    )
    
    # Create throughput visualization
    df_individual_throughput, df_aggregated_throughput = load_tradeoff_data(dataset, 'throughput')
    visualize_tradeoff(
        df_individual_throughput, df_aggregated_throughput, output_dir, metrics_list,
        x_column='throughput_fps',
        x_title='Throughput (frames/second)',
        naive_column='naive_throughput',
        plot_suffix='throughput'
    )


def main(args):
    """
    Main function that orchestrates the accuracy-throughput tradeoff visualization.
    
    This function serves as the entry point for the script. It loads pre-computed 
    tradeoff data from CSV files created by p090_tradeoff_compute.py and creates 
    visualizations showing the tradeoff between accuracy and query execution runtime/throughput.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tradeoff data from p090_tradeoff_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/accuracy_{suffix}_tradeoff.csv
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/accuracy_{suffix}_tradeoff_combined.csv
        - Results are saved to: {CACHE_DIR}/{dataset}/evaluation/091_tradeoff/
        - Please run p090_tradeoff_compute.py first to generate the required CSV files
        - Metrics are automatically detected from the CSV files
    """
    print(f"Processing datasets: {args.datasets}")
    
    # Process datasets
    for dataset in track(args.datasets, description="Processing datasets"):
        visualize_tradeoffs(dataset)


if __name__ == '__main__':
    main(parse_args())