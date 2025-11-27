#!/usr/local/bin/python

import argparse
from functools import partial
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, STR_NA, load_all_datasets_tradeoff_data, print_best_data_points


def parse_args():
    parser = argparse.ArgumentParser(description='Compare our tradeoff results with OTIF tradeoff results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    return parser.parse_args()


def load_otif_tradeoff_data(datasets: list[str]) -> pd.DataFrame:
    """
    Load OTIF tradeoff data from tradeoff.csv files created by p142_otif_tradeoff.py.
    
    Each dataset has a tradeoff.csv file at {CACHE_DIR}/SOTA/otif/{dataset}/tradeoff.csv
    containing param_id, runtime, and accuracy metrics (HOTA_HOTA, etc.).

    Args:
        datasets: List of dataset names to load tradeoff data for

    Returns:
        pd.DataFrame: Combined OTIF tradeoff data for all specified datasets
    """
    otif_data = []

    # Process each dataset
    for dataset_name in datasets:
        # Construct path to tradeoff.csv file
        tradeoff_csv_path = os.path.join(CACHE_DIR, 'SOTA', 'otif', dataset_name, 'tradeoff.csv')
        
        # Skip if tradeoff.csv doesn't exist
        if not os.path.exists(tradeoff_csv_path):
            print(f"  Warning: tradeoff.csv not found for dataset {dataset_name}, skipping")
            continue
        
        print(f"Loading OTIF tradeoff data from: {tradeoff_csv_path} (dataset: {dataset_name})")
        
        # Read tradeoff.csv file
        df = pd.read_csv(tradeoff_csv_path)
        
        # Validate required columns exist
        required_columns = ['param_id', 'runtime', 'HOTA_HOTA']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {tradeoff_csv_path}: {missing_columns}")
        
        # Create OTIF data with consistent structure for visualization
        clean_df = pd.DataFrame({
            'system': 'OTIF',  # Display name for OTIF system
            'dataset': dataset_name,
            'video': 'dataset_level',  # OTIF data is aggregated at dataset level
            'classifier': STR_NA,  # OTIF doesn't use our classifier system
            'tilesize': 0,  # OTIF doesn't have explicit tile size
            'tilepadding': STR_NA,  # OTIF doesn't have explicit tile padding
            'HOTA_HOTA': df['HOTA_HOTA'],  # Already in decimal format (0-1)
            'MOTA_MOTA': -1,  # Not available for OTIF
            'time': df['runtime'],  # Runtime in seconds
            'throughput_fps': float('nan'),  # Not available in tradeoff.csv
            'time_mspf': float('nan'),  # Not available without frame count
        })
        
        otif_data.append(clean_df)
        print(f"  Loaded {len(df)} data points for OTIF on {dataset_name}")

    assert len(otif_data) > 0, f"No OTIF tradeoff data found for datasets: {datasets}"

    # Combine all OTIF data
    combined_otif_df = pd.concat(otif_data, ignore_index=True)
    datasets_covered = combined_otif_df['dataset'].unique()
    print(f"Combined OTIF data: {len(combined_otif_df)} total rows (datasets: {list(datasets_covered)})")

    return combined_otif_df


def merge_otif_with_naive_baselines(df_combined: pd.DataFrame, df_otif: pd.DataFrame,
                                    x_column: str) -> pd.DataFrame:
    """
    Merge OTIF data with our system's naive baselines for consistent comparison.

    Args:
        df_combined: Combined DataFrame with our system's data (already merged with naive data)
        df_otif: DataFrame with OTIF tradeoff data
        x_column: Column name for the metric being compared (e.g., 'time', 'throughput_fps')

    Returns:
        pd.DataFrame: Combined data with OTIF results added, using our naive baselines
    """
    assert not df_otif.empty, "No OTIF data found"

    # Naive column is automatically created from merge with suffix '_naive'
    naive_column = f'{x_column}_naive'

    # Get unique datasets from OTIF data
    otif_datasets = df_otif['dataset'].unique()
    print(f"Adding OTIF data for datasets: {list(otif_datasets)}")

    # Get naive baselines from our data for each dataset
    our_naive_baselines = df_combined.groupby('dataset')[naive_column].first().reset_index()
    our_naive_baselines.columns = ['dataset', naive_column]

    # Filter OTIF data to only include datasets we have data for
    df_otif_filtered = df_otif[df_otif['dataset'].isin(our_naive_baselines['dataset'])]

    # Warn about any skipped datasets
    skipped_datasets = set(otif_datasets) - set(our_naive_baselines['dataset'])
    for dataset in skipped_datasets:
        print(f"  Warning: No our data found for dataset {dataset}, skipping OTIF data")

    # Merge OTIF data with our naive baselines using pandas merge
    assert not df_otif_filtered.empty, f"No OTIF data found for datasets: {list(otif_datasets)}"

    df_otif_with_naive = df_otif_filtered.merge(our_naive_baselines, on='dataset', how='left')

    # Combine with existing data
    return df_otif_with_naive


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, df_otif: pd.DataFrame,
                                   metrics_list: list[str], x_column: str, x_title: str,
                                   plot_suffix: str, output_dir: str):
    """
    Create visualization showing all dataset-wide trade-offs for a specific metric and axis.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets (already merged with naive data)
        df_otif: DataFrame with OTIF tradeoff data
        metrics_list: list of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")

    # Print best data points tables first
    print_best_data_points(df_combined, metrics_list, x_column, plot_suffix, include_system=True)

    # Add OTIF data to appropriate datasets if available
    # Skip if x_column is not available in OTIF data (e.g., throughput_fps, time_mspf)
    if x_column in df_otif.columns and not df_otif[x_column].isna().all():
        otif_with_naive = merge_otif_with_naive_baselines(df_combined, df_otif, x_column)
        df_combined = pd.concat([df_combined, otif_with_naive], ignore_index=True)
    else:
        print(f"  Warning: {x_column} not available in OTIF data, skipping OTIF for this visualization")
    
    # Filter out rows where classifier == 'Perfect'
    df_combined = df_combined[df_combined['classifier'] != 'Perfect']
    
    # Convert tilepadding to string for shape encoding
    # Handle both numeric and string values, and STR_NA
    if 'tilepadding' in df_combined.columns:
        df_combined['tilepadding'] = df_combined['tilepadding'].astype(str)
    
    # Update system column: rows with classifier=='Groundtruth' should have system='Groundtruth'
    # This identifies the naive baseline points in our results
    df_combined.loc[df_combined['classifier'] == 'Groundtruth', 'system'] = 'Groundtruth'
    
    # Create base chart
    base_chart = alt.Chart(df_combined)
    
    # Create visualizations for each metric - focus on HOTA_HOTA
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'HOTA_HOTA'
            metric_name = 'HOTA'
            y_scale = {'scale': alt.Scale(domain=[0, 1])}
        elif metric == 'CLEAR':
            accuracy_col = 'MOTA_MOTA'
            metric_name = 'MOTA'
            y_scale = {'scale': alt.Scale(domain=[0, 1])}
        elif metric == 'Count':
            accuracy_col = 'Count_TracksMAPE'
            metric_name = 'Count'
            y_scale = {}
        else:
            continue
        
        # Create scatter plot with color by system and shape by tilepadding
        # Use conditional encoding to handle groundtruth points differently
        scatter = base_chart.mark_point(
            opacity=0.8
        ).encode(
            x=alt.X(f'{x_column}:Q', title=x_title),
            y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score', **y_scale),
            color=alt.Color('system:N', title='System', 
                          scale=alt.Scale(domain=['Polytris', 'OTIF', 'Groundtruth'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
            size=alt.condition(
                alt.datum.system == 'Groundtruth',
                alt.value(100),  # Larger size for groundtruth
                alt.value(50)   # Normal size for others
            ),
            shape=alt.condition(
                (alt.datum.system == 'Polytris') & (alt.datum.classifier != 'Groundtruth'),
                alt.Shape('classifier:N', title='Classifier', scale=alt.Scale(domain=['MobileNetS', 'ShuffleNet05'], range=['square', 'triangle'])),
                alt.value('circle')  # Circle for non-Polytris points (OTIF, Groundtruth)
            ),
            tooltip=['system', 'dataset', 'classifier', 'tilepadding', x_column, accuracy_col]
        ).properties(
            width=150,
            height=150
        )
        
        # Create the combined chart with dataset facets
        combined_chart = scatter.facet(
            facet=alt.Facet('dataset:N', title=None,
                            header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
            columns=3
        ).resolve_scale(
            x='independent',
            y='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff',
        )
        
        # Save the chart
        plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_comparison.png')
        combined_chart.save(plot_path, scale_factor=4)
        print(f"Saved all datasets {metric_name} {plot_suffix} comparison plot to: {plot_path}")


def visualize_all_datasets_tradeoffs(datasets: list[str]):
    """
    Create runtime tradeoff visualizations for all datasets comparing with OTIF.
    
    Args:
        datasets: list of dataset names
    """
    print(f"Creating all datasets tradeoff visualizations for {len(datasets)} datasets...")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, 'SUMMARY', '100_compare_compute')
    os.makedirs(output_dir, exist_ok=True)
    
    # Focus on HOTA metric for visualization
    metrics_list = ['HOTA']
    print(f"Using metrics: {metrics_list}")
    
    # Load OTIF tradeoff data
    print("Loading OTIF tradeoff data...")
    df_otif = load_otif_tradeoff_data(datasets)
    
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

    # Visualize HOTA_HOTA vs runtime (OTIF has runtime data)
    visualize = partial(visualize_all_datasets_tradeoff, combined_with_naive, df_otif, metrics_list, output_dir=output_dir)
    visualize(x_column='time', x_title='Query Execution Runtime (seconds)', plot_suffix='runtime')


def main(args):
    """
    Main function that orchestrates the comparison between our tradeoff results and OTIF results.
    
    This function serves as the entry point for the script. It loads pre-computed 
    tradeoff data from CSV files created by p090_tradeoff_compute.py for our system
    and OTIF tradeoff results from tradeoff.csv files created by p142_otif_tradeoff.py,
    then creates comparison visualizations showing both systems' performance.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Note:
        - The script expects tradeoff data from p090_tradeoff_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/tradeoff_combined.csv
        - The script expects OTIF tradeoff results from p142_otif_tradeoff.py in:
          {CACHE_DIR}/SOTA/otif/{dataset}/tradeoff.csv
        - Results are saved to: {CACHE_DIR}/SUMMARY/100_compare_compute/
        - Please run p090_tradeoff_compute.py and p142_otif_tradeoff.py first to generate the required CSV files
    """
    print(f"Processing datasets: {args.datasets}")
    
    # Create comparison visualizations
    visualize_all_datasets_tradeoffs(args.datasets)


if __name__ == '__main__':
    main(parse_args())
