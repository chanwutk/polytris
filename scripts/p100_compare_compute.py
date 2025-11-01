#!/usr/local/bin/python

import argparse
from functools import partial
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, STR_NA, load_all_datasets_tradeoff_data, print_best_data_points, tradeoff_scatter_and_naive_baseline


def parse_args():
    parser = argparse.ArgumentParser(description='Compare our tradeoff results with SOTA results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--sota-dir', required=False,
                        default='/sota-results',
                        help='Directory containing SOTA results CSV files')
    return parser.parse_args()


def load_sota_data(sota_dir: str) -> pd.DataFrame:
    """
    Load SOTA tradeoff data from CSV files in the specified directory.
    SOTA data is available for caldot1 and caldot2, excluding split data.

    Args:
        sota_dir: Directory containing SOTA results CSV files

    Returns:
        pd.DataFrame: Combined SOTA tradeoff data for caldot1 and caldot2
    """
    sota_data = []

    # Process each CSV file in the SOTA directory
    for filename in os.listdir(sota_dir):
        if filename.endswith('.csv'):
            # Include both full and split data files
            # Skip other types of split data files if any
            if 'split' in filename and 'otif' not in filename:
                print(f"Skipping non-OTIF split data file: {filename}")
                continue

            filepath = os.path.join(sota_dir, filename)

            # Extract system name and dataset from filename
            # Examples: 'otif_caldot1_full.csv' -> system='otif_full', dataset='caldot1'
            #          'otif_caldot2_full.csv' -> system='otif_full', dataset='caldot2'
            #          'otif_caldot1_split.csv' -> system='otif_split', dataset='caldot1'
            base_name = filename.replace('.csv', '')
            if '_caldot1_' in base_name:
                system_name = base_name.replace('_caldot1', '')
                dataset_name = 'caldot1'
            elif '_caldot2_' in base_name:
                system_name = base_name.replace('_caldot2', '')
                dataset_name = 'caldot2'
            else:
                print(f"  Warning: Cannot determine dataset from filename {filename}, skipping")
                continue

            print(f"Loading SOTA data from: {filename} (system: {system_name}, dataset: {dataset_name})")

            # Read CSV file - the format appears to be: 0,1,2,3,4,5,6,hota,fps
            # The first row is the header, but the data contains JSON strings
            df = pd.read_csv(filepath, header=0)

            # check if column 'runtime' exists
            runtime_col = '3'
            if 'runtime' in df.columns:
                runtime_col = 'runtime'

            # Handle different column names for HOTA score
            if 'hota' in df.columns:
                hota_col = 'hota'
            elif 'avg_hota' in df.columns:
                hota_col = 'avg_hota'
            else:
                raise ValueError(f"No HOTA column found in {filename}, available columns: {df.columns.tolist()}")

            # Map system names to display names using a dictionary
            system_name_map = {
                'otif_full': 'OTIF-Full',
                'otif_split': 'OTIF'
            }
            display_system_name = system_name_map.get(system_name, system_name)

            # Create SOTA data for the appropriate dataset using DataFrame
            clean_df = pd.DataFrame({
                'system': display_system_name,
                'dataset': dataset_name,
                'video': 'dataset_level',  # SOTA data appears to be aggregated
                'classifier': STR_NA,  # SOTA systems don't use our classifier system
                'tilesize': 0,  # SOTA data doesn't have explicit tile size
                'tilepadding': STR_NA,  # SOTA data doesn't have explicit tile padding
                'HOTA_HOTA': df[hota_col] / 100.0,  # Convert percentage to decimal
                'MOTA_MOTA': -1,  # Use HOTA as MOTA approximation for SOTA, convert percentage to decimal
                'time': df[runtime_col],  # Use actual runtime from column 3
                'throughput_fps': df['fps'],
                'time_mspf': 1000 / df['fps'],
            })

            sota_data.append(clean_df)
            print(f"  Loaded {len(df)} data points for {system_name} on {dataset_name}")

    assert len(sota_data) > 0, f"No SOTA data found in {sota_dir}"

    # Combine all SOTA data
    combined_sota_df = pd.concat(sota_data, ignore_index=True)
    datasets_covered = combined_sota_df['dataset'].unique()
    print(f"Combined SOTA data: {len(combined_sota_df)} total rows from {len(sota_data)} systems (datasets: {list(datasets_covered)})")

    return combined_sota_df


def merge_sota_with_naive_baselines(df_combined: pd.DataFrame, df_sota: pd.DataFrame,
                                    x_column: str) -> pd.DataFrame:
    """
    Merge SOTA data with our system's naive baselines for consistent comparison.

    Args:
        df_combined: Combined DataFrame with our system's data (already merged with naive data)
        df_sota: DataFrame with SOTA data
        x_column: Column name for the metric being compared (e.g., 'time', 'throughput_fps')

    Returns:
        pd.DataFrame: Combined data with SOTA results added, using our naive baselines
    """
    assert not df_sota.empty, "No SOTA data found"

    # Naive column is automatically created from merge with suffix '_naive'
    naive_column = f'{x_column}_naive'

    # Get unique datasets from SOTA data
    sota_datasets = df_sota['dataset'].unique()
    print(f"Adding SOTA data for datasets: {list(sota_datasets)}")

    # Get naive baselines from our data for each dataset
    our_naive_baselines = df_combined.groupby('dataset')[naive_column].first().reset_index()
    our_naive_baselines.columns = ['dataset', naive_column]

    # Filter SOTA data to only include datasets we have data for
    df_sota_filtered = df_sota[df_sota['dataset'].isin(our_naive_baselines['dataset'])]

    # Warn about any skipped datasets
    skipped_datasets = set(sota_datasets) - set(our_naive_baselines['dataset'])
    for dataset in skipped_datasets:
        print(f"  Warning: No our data found for dataset {dataset}, skipping SOTA data")

    # Merge SOTA data with our naive baselines using pandas merge
    assert not df_sota_filtered.empty, f"No SOTA data found for datasets: {list(sota_datasets)}"

    df_sota_with_naive = df_sota_filtered.merge(our_naive_baselines, on='dataset', how='left')

    # Combine with existing data
    return df_sota_with_naive


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, df_sota: pd.DataFrame,
                                   metrics_list: list[str], x_column: str, x_title: str,
                                   plot_suffix: str, output_dir: str):
    """
    Create visualization showing all dataset-wide trade-offs for a specific metric and axis.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets (already merged with naive data)
        df_sota: DataFrame with SOTA data
        metrics_list: list of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")

    # Print best data points tables first
    print_best_data_points(df_combined, metrics_list, x_column, plot_suffix, include_system=True)

    # Add SOTA data to appropriate datasets if available
    sota_with_naive = merge_sota_with_naive_baselines(df_combined, df_sota, x_column)
    df_combined = pd.concat([df_combined, sota_with_naive], ignore_index=True)
    
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
        plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_comparison.png')
        combined_chart.save(plot_path, scale_factor=4)
        print(f"Saved all datasets {metric_name} {plot_suffix} comparison plot to: {plot_path}")


def visualize_all_datasets_tradeoffs(datasets: list[str], sota_dir: str):
    """
    Create both runtime and throughput tradeoff visualizations for all datasets.
    
    Args:
        datasets: list of dataset names
        sota_dir: Directory containing SOTA results CSV files
    """
    print(f"Creating all datasets tradeoff visualizations for {len(datasets)} datasets...")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, 'SUMMARY', '100_compare_compute')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use metrics from utilities
    metrics_list = METRICS
    print(f"Using metrics: {metrics_list}")
    
    # Load SOTA data
    print("Loading SOTA data...")
    df_sota = load_sota_data(sota_dir)
    
    # Load tradeoff data for all datasets
    combined_df, naive_df = load_all_datasets_tradeoff_data(datasets, system_name='Polytris')
    combined_df['time_mspf'] = combined_df['time'] * 1000 / combined_df['frame_count']
    naive_df['time_mspf'] = naive_df['time'] * 1000 / naive_df['frame_count']
    
    # Ensure naive_combined has 'video' column set to 'dataset_level' (matching p091)
    # Each dataset's combined data has video='dataset_level', so we merge on dataset only
    if 'video' not in naive_df.columns:
        naive_df['video'] = 'dataset_level'
    
    # Merge naive data into combined data
    # Since combined data has video='dataset_level' for each dataset, merge on 'dataset'
    combined_with_naive = combined_df.merge(naive_df, on='dataset', how='left', suffixes=('', '_naive'))

    visualize = partial(visualize_all_datasets_tradeoff, combined_with_naive, df_sota, metrics_list, output_dir=output_dir)
    visualize(x_column='time', x_title='Query Execution Runtime (seconds)', plot_suffix='runtime')
    visualize(x_column='time_mspf', x_title='Query Execution Runtime (milliseconds per frame)', plot_suffix='runtime_mspf')
    visualize(x_column='throughput_fps', x_title='Throughput (frames/second)', plot_suffix='throughput')


def main(args):
    """
    Main function that orchestrates the comparison between our tradeoff results and SOTA results.
    
    This function serves as the entry point for the script. It loads pre-computed 
    tradeoff data from CSV files created by p090_tradeoff_compute.py for our system
    and SOTA results from CSV files in the specified directory, then creates 
    comparison visualizations showing both systems' performance.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tradeoff data from p090_tradeoff_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/tradeoff_combined.csv
        - The script expects SOTA results in CSV files in the specified SOTA directory
          - Supports caldot1 and caldot2 datasets
          - Expected format: otif_caldot1_full.csv, otif_caldot2_full.csv
        - Results are saved to: {CACHE_DIR}/SUMMARY/100_compare_compute/
        - Please run p090_tradeoff_compute.py first to generate the required CSV files
    """
    print(f"Processing datasets: {args.datasets}")
    print(f"SOTA directory: {args.sota_dir}")
    
    # Create comparison visualizations
    visualize_all_datasets_tradeoffs(args.datasets, args.sota_dir)


if __name__ == '__main__':
    main(parse_args())