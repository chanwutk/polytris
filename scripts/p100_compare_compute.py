#!/usr/local/bin/python

import argparse
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, DATASETS_TO_TEST, METRICS, OPTIMAL_PARAMS, load_tradeoff_data, tradeoff_scatter_and_naive_baseline


def parse_args():
    parser = argparse.ArgumentParser(description='Compare our tradeoff results with SOTA results')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--sota-dir', required=False,
                        default='/sota-results',
                        help='Directory containing SOTA results CSV files')
    parser.add_argument('--best', action='store_true',
                        help='Only visualize optimal parameters from OPTIMAL_PARAMS')
    parser.add_argument('--no_perfect', action='store_true',
                        help='Exclude perfect classifier from visualization')
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
            # Skip split data files
            if 'split' in filename:
                print(f"Skipping split data file: {filename}")
                continue
                
            filepath = os.path.join(sota_dir, filename)
            
            # Extract system name and dataset from filename
            # Examples: 'otif_caldot1_full.csv' -> system='otif_full', dataset='caldot1'
            #          'otif_caldot2_full.csv' -> system='otif_full', dataset='caldot2'
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
            
            # Convert data to numeric, handling any string values
            runtime_numeric = pd.to_numeric(df[runtime_col], errors='coerce')  # Column 3 is runtime
            fps_numeric = pd.to_numeric(df['fps'], errors='coerce')
            
            # Handle different column names for HOTA score
            if 'hota' in df.columns:
                hota_col = 'hota'
            elif 'avg_hota' in df.columns:
                hota_col = 'avg_hota'
            else:
                print(f"  Warning: No HOTA column found in {filename}, available columns: {df.columns.tolist()}")
                continue
                
            hota_numeric = pd.to_numeric(df[hota_col], errors='coerce')
            
            # Filter out rows with invalid data
            valid_mask = (runtime_numeric.notna() & hota_numeric.notna() & 
                         (runtime_numeric > 0) & (fps_numeric.notna()) & (fps_numeric > 0))
            df_valid = df[valid_mask]
            runtime_valid = runtime_numeric[valid_mask]
            fps_valid = fps_numeric[valid_mask]
            hota_valid = hota_numeric[valid_mask]
            
            if len(df_valid) == 0:
                print(f"  Warning: No valid data found in {filename}")
                continue
            
            # Create SOTA data for the appropriate dataset
            clean_df = pd.DataFrame({
                'system': system_name,
                'dataset': dataset_name,  # Use extracted dataset name
                'video_name': 'Dataset Average',  # SOTA data appears to be aggregated
                'classifier': 'N/A',  # SOTA systems don't use our classifier system
                'tilesize': 'N/A',  # SOTA data doesn't have explicit tile size
                'tilepadding': 'N/A',  # SOTA data doesn't have explicit tile padding
                'hota_score': hota_valid / 100.0,  # Convert percentage to decimal
                'mota_score': hota_valid / 100.0,  # Use HOTA as MOTA approximation for SOTA, convert percentage to decimal
                'query_runtime': runtime_valid,  # Use actual runtime from column 3
                'throughput_fps': fps_valid,
                'frame_count': 1000,  # Approximate frame count for normalization
                'naive_runtime': runtime_valid.max(),  # Use slowest as naive baseline
                'naive_throughput': fps_valid.max()  # Use fastest as naive baseline
            })
            
            sota_data.append(clean_df)
            print(f"  Loaded {len(df_valid)} data points for {system_name} on {dataset_name}")
    
    if not sota_data:
        print(f"No SOTA data found in {sota_dir}")
        return pd.DataFrame()
    
    # Combine all SOTA data
    combined_sota_df = pd.concat(sota_data, ignore_index=True)
    datasets_covered = combined_sota_df['dataset'].unique()
    print(f"Combined SOTA data: {len(combined_sota_df)} total rows from {len(sota_data)} systems (datasets: {list(datasets_covered)})")
    
    return combined_sota_df


def load_all_datasets_tradeoff_data(datasets: list[str], csv_suffix: str, best_only: bool = False, no_perfect: bool = False) -> pd.DataFrame:
    """
    Load tradeoff data from all datasets and combine into a single DataFrame.
    This follows the same pattern as p092_tradeoff_visualize_all.py.
    
    Args:
        datasets: list of dataset names
        csv_suffix: Suffix for CSV files ('runtime' or 'throughput')
        best_only: If True, only include optimal parameters from OPTIMAL_PARAMS
        no_perfect: If True, exclude perfect classifier from the data
        
    Returns:
        pd.DataFrame: Combined tradeoff data from all datasets
    """
    all_data = []
    
    for dataset in datasets:
        # Use the load_tradeoff_data function from polyis.utilities
        _, df_aggregated = load_tradeoff_data(dataset, csv_suffix)
        
        # Filter for optimal parameters if best_only is True
        if best_only and dataset in OPTIMAL_PARAMS:
            optimal_params = OPTIMAL_PARAMS[dataset]
            print(f"Filtering {dataset} for optimal parameters: {optimal_params}")
            
            # Filter the dataframe for optimal parameters
            mask = (
                (df_aggregated['classifier'] == optimal_params['classifier']) &
                (df_aggregated['tilesize'] == optimal_params['tilesize']) &
                (df_aggregated['tilepadding'] == optimal_params['tilepadding'])
            )
            df_aggregated = df_aggregated[mask]
            
            if len(df_aggregated) == 0:
                print(f"  Warning: No data found for optimal parameters in {dataset}")
                continue
            else:
                print(f"  Filtered to {len(df_aggregated)} rows with optimal parameters")
        elif best_only:
            print(f"  Warning: No optimal parameters defined for {dataset}, skipping")
            continue
        
        # Filter out perfect classifier if no_perfect is True
        if no_perfect:
            initial_count = len(df_aggregated)
            df_aggregated = df_aggregated[df_aggregated['classifier'] != 'Perfect']
            filtered_count = len(df_aggregated)
            if initial_count > filtered_count:
                print(f"  Filtered out perfect classifier: {initial_count - filtered_count} rows removed")
        
        # Add dataset and system columns to identify the source
        df_aggregated['dataset'] = dataset
        df_aggregated['system'] = 'Our System'  # Our system name
        
        all_data.append(df_aggregated)
        print(f"Loaded tradeoff data for {dataset}: {len(df_aggregated)} rows")
    
    assert len(all_data) > 0, \
        f"No tradeoff data found for any dataset. " \
        "Please run p090_tradeoff_compute.py first."
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined tradeoff data from {len(datasets)} datasets: {len(combined_df)} total rows")

    return combined_df


def print_best_data_points(df_combined: pd.DataFrame, metrics_list: list[str], 
                          x_column: str, naive_column: str, plot_suffix: str):
    """
    Print the best data point (highest accuracy, faster than baseline) for each dataset and metric as tables.
    This follows the same pattern as p092_tradeoff_visualize_all.py.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets
        metrics_list: list of metrics to analyze
        x_column: Column name for x-axis data (runtime or throughput)
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for the analysis type ('runtime' or 'throughput')
    """
    print(f"\n=== Best Data Points Analysis ({plot_suffix.upper()}) ===")
    
    for metric in metrics_list:
        if metric == 'HOTA':
            accuracy_col = 'hota_score'
            metric_name = 'HOTA'
        elif metric == 'CLEAR':
            accuracy_col = 'mota_score'
            metric_name = 'MOTA'
        else:
            continue
            
        print(f"\n--- {metric_name} Analysis ---")
        
        # Collect results for this metric
        results = []
        
        for dataset in df_combined['dataset'].unique():
            dataset_data = df_combined[df_combined['dataset'] == dataset]
            
            # Filter data points that are faster than baseline for this dataset
            faster_than_baseline = dataset_data[dataset_data[x_column] < dataset_data[naive_column]]
            
            if len(faster_than_baseline) == 0:
                # If no points are faster than baseline, use the fastest point
                assert isinstance(dataset_data, pd.DataFrame), \
                    f"dataset_data should be a DataFrame, got {type(dataset_data)}"
                best_point = dataset_data.loc[dataset_data[x_column].idxmin()]
            else:
                # Find the point with highest accuracy among those faster than baseline
                assert isinstance(faster_than_baseline, pd.DataFrame), \
                    f"faster_than_baseline should be a DataFrame, got {type(faster_than_baseline)}"
                best_point = faster_than_baseline.loc[faster_than_baseline[accuracy_col].idxmax()]
            
            # Calculate speed improvement
            naive_runtime = best_point[naive_column]
            best_runtime = best_point[x_column]
            speedup = naive_runtime / best_runtime if best_runtime > 0 else 0
            
            results.append({
                'Dataset': dataset,
                'System': best_point['system'],
                'HOTA Score': f"{best_point[accuracy_col]:.2f}",
                'Speedup': f"{speedup:.2f}"
            })
        
        # Create and print table for this metric
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        else:
            print("No results found.")


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, df_sota: pd.DataFrame, 
                                   metrics_list: list[str], x_column: str, x_title: str, 
                                   naive_column: str, plot_suffix: str, output_dir: str, best_only: bool = False):
    """
    Create visualization showing all dataset-wide trade-offs for a specific metric and axis.
    This follows the same pattern as p092_tradeoff_visualize_all.py but adds SOTA data to caldot1.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets
        df_sota: DataFrame with SOTA data (only for caldot1)
        metrics_list: list of metrics to visualize
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        naive_column: Column name for naive baseline data
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
        best_only: If True, use system color encoding and remove size encoding
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")
    
    # Print best data points tables first
    print_best_data_points(df_combined, metrics_list, x_column, naive_column, plot_suffix)
    
    # Add SOTA data to appropriate datasets if available
    if not df_sota.empty:
        # Get unique datasets from SOTA data
        sota_datasets = df_sota['dataset'].unique()
        print(f"Adding SOTA data for datasets: {list(sota_datasets)}")
        
        # Process each dataset that has SOTA data
        for dataset in sota_datasets:
            # Filter our data to the current dataset
            df_dataset = df_combined[df_combined['dataset'] == dataset].copy()
            
            if len(df_dataset) == 0:
                print(f"  Warning: No our data found for dataset {dataset}, skipping SOTA data")
                continue
            
            # Get the naive baseline from our system's data for this dataset
            our_naive_baseline = df_dataset[naive_column].iloc[0]  # Should be the same for all rows
            
            # Filter SOTA data for this dataset
            df_sota_dataset = df_sota[df_sota['dataset'] == dataset].copy()
            
            # Update SOTA data to use our system's naive baseline for consistent baseline line
            df_sota_dataset[naive_column] = our_naive_baseline
            
            # Add SOTA data to this dataset
            df_dataset_with_sota = pd.concat([df_dataset, df_sota_dataset], ignore_index=True)
            
            # Replace dataset data in combined dataframe
            df_combined = pd.concat([
                df_combined[df_combined['dataset'] != dataset],
                df_dataset_with_sota
            ], ignore_index=True)
            print(f"  Added SOTA data to {dataset}: {len(df_sota_dataset)} additional data points")
            print(f"  Using our system's naive baseline for {dataset}: {our_naive_baseline}")
    
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
            size_range=(50, 300), scatter_opacity=0.8,
            baseline_stroke_width=3, baseline_opacity=0.9,
            size_field='tilepadding', size=100
        )
        
        # Update encoding based on best_only flag
        if best_only:
            # For best mode: color by system only, no size or shape encoding in legend
            scatter = scatter.mark_point(opacity=0.8).encode(
                x=alt.X(f'{x_column}:Q', title=x_title),
                y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score',
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('system:N', title='System'),
                # shape=alt.Shape('system:N',
                #               scale=alt.Scale(domain=['Our System', 'otif_full'],
                #                             range=['circle', 'square'])),
                tooltip=['dataset', 'video_name', 'system', 'classifier', 'tilesize', 'tilepadding', x_column, accuracy_col]
            ).properties(
                width=200,
                height=200
            )
        else:
            # For normal mode: color by classifier, size by tilepadding
            scatter = scatter.mark_point(opacity=0.8).encode(
                x=alt.X(f'{x_column}:Q', title=x_title),
                y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score',
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('classifier:N', title='Classifier'),
                shape=alt.Shape('system:N', title='System',
                              scale=alt.Scale(domain=['Our System', 'otif_full'],
                                            range=['circle', 'square'])),
                size=alt.Size('tilepadding:O', title='Tile Padding',
                             scale=alt.Scale(range=[50, 300])),
                tooltip=['dataset', 'video_name', 'system', 'classifier', 'tilesize', 'tilepadding', x_column, accuracy_col]
            ).properties(
                width=200,
                height=200
            )
        
        # Create the combined chart with dataset facets
        combined_chart = (scatter + baseline).facet(
            facet=alt.Facet('dataset:N', title=None,
                            header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
            columns=3
        ).resolve_scale(
            x='independent'
        ).properties(
            title=f'{metric_name} vs {x_title} Tradeoff Comparison',
        )
        
        # Save the chart with appropriate filename
        if plot_suffix == 'runtime_per_frame':
            plot_path = os.path.join(output_dir, f'{metric.lower()}_runtime_per_frame_comparison.png')
        else:
            plot_path = os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_comparison.png')
        combined_chart.save(plot_path, scale_factor=4)
        print(f"Saved comparison {metric_name} {plot_suffix} tradeoff plot to: {plot_path}")


def create_runtime_per_frame_data(df_throughput: pd.DataFrame, df_sota: pd.DataFrame) -> pd.DataFrame:
    """
    Create runtime per frame data by calculating 1/fps from throughput data and converting to milliseconds.
    
    Args:
        df_throughput: DataFrame with throughput data
        df_sota: DataFrame with SOTA data
        
    Returns:
        pd.DataFrame: DataFrame with runtime per frame data in milliseconds
    """
    # Create a copy of throughput data
    df_runtime_per_frame = df_throughput.copy()
    
    # Calculate runtime per frame (1 / fps) and convert to milliseconds
    df_runtime_per_frame['runtime_per_frame'] = (1.0 / df_runtime_per_frame['throughput_fps']) * 1000.0
    df_runtime_per_frame['naive_runtime_per_frame'] = (1.0 / df_runtime_per_frame['naive_throughput']) * 1000.0
    
    # Update SOTA data if available
    if not df_sota.empty:
        df_sota_runtime_per_frame = df_sota.copy()
        df_sota_runtime_per_frame['runtime_per_frame'] = (1.0 / df_sota_runtime_per_frame['throughput_fps']) * 1000.0
        df_sota_runtime_per_frame['naive_runtime_per_frame'] = (1.0 / df_sota_runtime_per_frame['naive_throughput']) * 1000.0
        
        # Get unique datasets from SOTA data
        sota_datasets = df_sota_runtime_per_frame['dataset'].unique()
        print(f"Adding SOTA runtime per frame data for datasets: {list(sota_datasets)}")
        
        # Process each dataset that has SOTA data
        for dataset in sota_datasets:
            # Filter our data to the current dataset
            df_dataset = df_runtime_per_frame[df_runtime_per_frame['dataset'] == dataset].copy()
            
            if len(df_dataset) == 0:
                print(f"  Warning: No our data found for dataset {dataset}, skipping SOTA data")
                continue
            
            # Get the naive baseline from our system's data for this dataset
            our_naive_baseline = df_dataset['naive_runtime_per_frame'].iloc[0]
            
            # Filter SOTA data for this dataset
            df_sota_dataset = df_sota_runtime_per_frame[df_sota_runtime_per_frame['dataset'] == dataset].copy()
            df_sota_dataset['naive_runtime_per_frame'] = our_naive_baseline
            
            # Add SOTA data to this dataset
            df_dataset_with_sota = pd.concat([df_dataset, df_sota_dataset], ignore_index=True)
            
            # Replace dataset data in combined dataframe
            df_runtime_per_frame = pd.concat([
                df_runtime_per_frame[df_runtime_per_frame['dataset'] != dataset],
                df_dataset_with_sota
            ], ignore_index=True)
            
            print(f"  Added SOTA runtime per frame data to {dataset}: {len(df_sota_dataset)} additional data points")
        
        print(f"Created runtime per frame data: {len(df_runtime_per_frame)} total rows")
    
    return df_runtime_per_frame


def visualize_all_datasets_tradeoffs(datasets: list[str], sota_dir: str, best_only: bool = False, no_perfect: bool = False):
    """
    Create runtime, throughput, and runtime per frame tradeoff visualizations for all datasets.
    This follows the same pattern as p092_tradeoff_visualize_all.py but adds SOTA comparison.
    
    Args:
        datasets: list of dataset names
        sota_dir: Directory containing SOTA results CSV files
        best_only: If True, only include optimal parameters from OPTIMAL_PARAMS
        no_perfect: If True, exclude perfect classifier from the data
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
    
    if df_sota.empty:
        print("No SOTA data found. Creating visualizations with only our data.")
        df_sota = pd.DataFrame()
    
    # Create runtime visualization
    df_combined_runtime = load_all_datasets_tradeoff_data(datasets, 'runtime', best_only, no_perfect)
    visualize_all_datasets_tradeoff(
        df_combined_runtime, df_sota, metrics_list,
        x_column='query_runtime',
        x_title='Query Execution Runtime (seconds)',
        naive_column='naive_runtime',
        plot_suffix='runtime',
        output_dir=output_dir,
        best_only=best_only
    )
    
    # Create throughput visualization
    df_combined_throughput = load_all_datasets_tradeoff_data(datasets, 'throughput', best_only, no_perfect)
    visualize_all_datasets_tradeoff(
        df_combined_throughput, df_sota, metrics_list,
        x_column='throughput_fps',
        x_title='Throughput (frames/second)',
        naive_column='naive_throughput',
        plot_suffix='throughput',
        output_dir=output_dir,
        best_only=best_only
    )
    
    # Create runtime per frame visualization
    print("Creating runtime per frame visualization...")
    df_combined_runtime_per_frame = create_runtime_per_frame_data(df_combined_throughput, df_sota)
    visualize_all_datasets_tradeoff(
        df_combined_runtime_per_frame, df_sota, metrics_list,
        x_column='runtime_per_frame',
        x_title='Runtime per Frame (milliseconds)',
        naive_column='naive_runtime_per_frame',
        plot_suffix='runtime_per_frame',
        output_dir=output_dir,
        best_only=best_only
    )


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
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/accuracy_{suffix}_tradeoff_combined.csv
        - The script expects SOTA results in CSV files in the specified SOTA directory
          - Supports caldot1 and caldot2 datasets
          - Excludes split data files (files containing 'split' in filename)
          - Expected format: otif_caldot1_full.csv, otif_caldot2_full.csv
        - Results are saved to: {CACHE_DIR}/SUMMARY/100_compare_compute/
        - Use --best flag to only visualize optimal parameters from OPTIMAL_PARAMS
        - Please run p090_tradeoff_compute.py first to generate the required CSV files
    """
    print(f"Processing datasets: {args.datasets}")
    print(f"SOTA directory: {args.sota_dir}")
    print(f"Best only mode: {args.best}")
    print(f"No perfect classifier mode: {args.no_perfect}")
    
    # Create comparison visualizations
    visualize_all_datasets_tradeoffs(args.datasets, args.sota_dir, args.best, args.no_perfect)


if __name__ == '__main__':
    main(parse_args())