#!/usr/local/bin/python

import argparse
from functools import partial
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, load_all_datasets_tradeoff_data, print_best_data_points, tradeoff_scatter_and_naive_baseline


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize all dataset-wide accuracy-throughput tradeoffs')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, metrics_list: list[str], 
                                    x_column: str, x_title: str, 
                                    plot_suffix: str, output_dir: str):
    """
    Create visualization showing all dataset-wide trade-offs for a specific metric and axis.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets (already merged with naive data)
        metrics_list: list of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")
    
    # Print best data points tables first
    print_best_data_points(df_combined, metrics_list, x_column, plot_suffix)
    
    # Create base chart
    base_chart = alt.Chart(df_combined)
    
    # Create visualizations for each metric
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'HOTA_HOTA'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'MOTA_MOTA'
            metric_name = 'MOTA'
        elif metric == 'Count':
            accuracy_col = 'Count_TracksMAPE'
            metric_name = 'Count'
        else:
            continue
        
        # Create scatter plot and baseline using shared function
        scatter, baseline = tradeoff_scatter_and_naive_baseline(
            base_chart, x_column, x_title, accuracy_col, metric_name,
            size_range=(50, 300), scatter_opacity=0.8,
            baseline_stroke_width=3, baseline_opacity=0.9,
            size_field='tilepadding'
        )
        
        # Add dataset to tooltip for all datasets visualization
        scatter = scatter.encode(tooltip=['dataset', 'video', 'classifier', 'tilesize', 'tilepadding', x_column, accuracy_col])
        
        # Create the combined chart with dataset facets
        combined_chart = (scatter + baseline).facet(
            facet=alt.Facet('dataset:N', title=None,
                            header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
            columns=3
        ).resolve_scale(
            x='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff',
        )
        
        # Save the chart
        plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_tradeoff.png')
        combined_chart.save(plot_path, scale_factor=4)
        print(f"Saved all datasets {metric_name} {plot_suffix} tradeoff plot to: {plot_path}")


def visualize_all_datasets_tradeoffs(datasets: list[str]):
    """
    Create both runtime and throughput tradeoff visualizations for all datasets.
    
    Args:
        datasets: list of dataset names
    """
    print(f"Creating all datasets tradeoff visualizations for {len(datasets)} datasets...")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, 'SUMMARY', '092_tradeoff_all')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Load tradeoff data for all datasets
    combined_df, naive_df = load_all_datasets_tradeoff_data(datasets, system_name=None)
    
    # Ensure naive_combined has 'video' column set to 'dataset_level' (matching p091)
    # Each dataset's combined data has video='dataset_level', so we merge on dataset only
    if 'video' not in naive_df.columns:
        naive_df['video'] = 'dataset_level'
    
    # Merge naive data into combined data
    # Since combined data has video='dataset_level' for each dataset, merge on 'dataset'
    combined_with_naive = combined_df.merge(naive_df, on='dataset', how='left', suffixes=('', '_naive'))

    visualize = partial(visualize_all_datasets_tradeoff, combined_with_naive, metrics_list, output_dir=output_dir)
    visualize(x_column='time', x_title='Query Execution Runtime (seconds)', plot_suffix='runtime')
    visualize(x_column='throughput_fps', x_title='Throughput (frames/second)', plot_suffix='throughput')


def main(args):
    """
    Main function that orchestrates the all datasets accuracy-throughput tradeoff visualization.
    
    This function serves as the entry point for the script. It loads pre-computed 
    tradeoff data from CSV files created by p090_tradeoff_compute.py for all datasets
    and creates visualizations showing the tradeoff between accuracy and query execution 
    runtime/throughput across all datasets.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tradeoff data from p090_tradeoff_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/accuracy_{suffix}_tradeoff_combined.csv
        - Results are saved to: {CACHE_DIR}/evaluation/092_tradeoff_all/
        - Please run p090_tradeoff_compute.py first to generate the required CSV files
    """
    print(f"Processing datasets: {args.datasets}")
    
    # Create visualizations for all datasets
    visualize_all_datasets_tradeoffs(args.datasets)


if __name__ == '__main__':
    main(parse_args())