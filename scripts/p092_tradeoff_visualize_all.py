#!/usr/local/bin/python

import argparse
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, load_tradeoff_data, tradeoff_scatter_and_naive_baseline


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize all dataset-wide accuracy-throughput tradeoffs')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_all_datasets_tradeoff_data(datasets: list[str], csv_suffix: str) -> pd.DataFrame:
    """
    Load tradeoff data from all datasets and combine into a single DataFrame.
    
    Args:
        datasets: list of dataset names
        csv_suffix: Suffix for CSV files ('runtime' or 'throughput')
        
    Returns:
        pd.DataFrame: Combined tradeoff data from all datasets
    """
    all_data = []
    
    for dataset in datasets:
        # Use the load_tradeoff_data function from polyis.utilities
        _, df_aggregated = load_tradeoff_data(dataset, csv_suffix)
        
        # Add dataset column to identify the source
        df_aggregated['dataset'] = dataset
        
        all_data.append(df_aggregated)
        print(f"Loaded tradeoff data for {dataset}: {len(df_aggregated)} rows")
    
    assert len(all_data) > 0, \
        f"No tradeoff data found for any dataset. " \
        "Please run p090_tradeoff_compute.py first."
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined tradeoff data from {len(datasets)} datasets: {len(combined_df)} total rows")
    
    return combined_df


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, metrics_list: list[str], 
                                    x_column: str, x_title: str, naive_column: str, 
                                    plot_suffix: str, output_dir: str):
    """
    Create visualization showing all dataset-wide trade-offs for a specific metric and axis.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets
        metrics_list: list of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")
    
    # Create base chart
    base_chart = alt.Chart(df_combined)
    
    # Create visualizations for each metric
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'hota_score'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'mota_score'
            metric_name = 'MOTA'
        else:
            continue
        
        # Create scatter plot and baseline using shared function
        scatter, baseline = tradeoff_scatter_and_naive_baseline(
            base_chart, x_column, x_title, accuracy_col, metric_name, naive_column,
            size_range=(50, 300), scatter_opacity=0.8, size=300,
            baseline_stroke_width=3, baseline_opacity=0.9
        )
        
        # Add dataset to tooltip for all datasets visualization
        scatter = scatter.encode(tooltip=['dataset', 'video_name', 'classifier', 'tile_size', x_column, accuracy_col])
        
        # Create the combined chart with dataset facets
        combined_chart = (scatter + baseline).facet(
            facet=alt.Facet('dataset:N', title=None,
                            header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
            columns=3
        ).resolve_scale(
            x='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff (All Datasets)',
        )
        
        # Save the chart
        plot_path = os.path.join(output_dir, f'all_datasets_{metric.lower()}_{plot_suffix}_tradeoff.png')
        combined_chart.save(plot_path, scale_factor=2)
        print(f"Saved all datasets {metric_name} {plot_suffix} tradeoff plot to: {plot_path}")


def visualize_all_datasets_tradeoffs(datasets: list[str]):
    """
    Create both runtime and throughput tradeoff visualizations for all datasets.
    
    Args:
        datasets: list of dataset names
    """
    print(f"Creating all datasets tradeoff visualizations for {len(datasets)} datasets...")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, 'evaluation', '092_tradeoff_all')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Create runtime visualization
    df_combined_runtime = load_all_datasets_tradeoff_data(datasets, 'runtime')
    visualize_all_datasets_tradeoff(
        df_combined_runtime, metrics_list,
        x_column='query_runtime',
        x_title='Query Execution Runtime (seconds)',
        naive_column='naive_runtime',
        plot_suffix='runtime',
        output_dir=output_dir
    )
    
    # Create throughput visualization
    df_combined_throughput = load_all_datasets_tradeoff_data(datasets, 'throughput')
    visualize_all_datasets_tradeoff(
        df_combined_throughput, metrics_list,
        x_column='throughput_fps',
        x_title='Throughput (frames/second)',
        naive_column='naive_throughput',
        plot_suffix='throughput',
        output_dir=output_dir
    )


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