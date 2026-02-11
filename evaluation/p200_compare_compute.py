#!/usr/local/bin/python

from functools import partial
import os
import pandas as pd
import altair as alt

from polyis.utilities import CACHE_DIR, STR_NA, load_all_datasets_tradeoff_data, print_best_data_points, get_config
from evaluation.utilities import ColorScheme


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS = config['EXEC']['DATASETS']


def load_sota_tradeoff_data(datasets: list[str], system: str) -> pd.DataFrame:
    """
    Load SOTA (OTIF or LEAP) tradeoff data from tradeoff.csv files created by p142_otif_tradeoff.py.
    
    Each dataset has a tradeoff.csv file at {CACHE_DIR}/SOTA/{system}/{dataset}/tradeoff.csv
    containing param_id, runtime, and accuracy metrics (HOTA_HOTA, etc.).

    Args:
        datasets: List of dataset names to load tradeoff data for
        system: System name ('otif' or 'leap')

    Returns:
        pd.DataFrame: Combined SOTA tradeoff data for all specified datasets
    """
    sota_data = []

    # Process each dataset
    for dataset_name in datasets:
        # Construct path to tradeoff.csv file
        tradeoff_csv_path = os.path.join(CACHE_DIR, 'SOTA', system, dataset_name, 'tradeoff.csv')
        
        # Skip if tradeoff.csv doesn't exist
        if not os.path.exists(tradeoff_csv_path):
            print(f"  Warning: {system.upper()} tradeoff.csv not found for dataset {dataset_name}, skipping")
            continue
        
        print(f"Loading {system.upper()} tradeoff data from: {tradeoff_csv_path} (dataset: {dataset_name})")
        
        # Read tradeoff.csv file
        df = pd.read_csv(tradeoff_csv_path)
        
        # Validate required columns exist
        required_columns = ['param_id', 'runtime', 'HOTA_HOTA']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {tradeoff_csv_path}: {missing_columns}")
        
        # Define all possible submetrics that might be in the CSV
        submetric_columns = ['HOTA_HOTA', 'HOTA_AssA', 'HOTA_DetA', 'MOTA_MOTA', 
                            'Count_DetsMAPE', 'Count_TracksMAPE', 'sample_rate']
        
        # Create base DataFrame with required columns
        clean_df = pd.DataFrame({
            'system': system.upper(),  # Display name for SOTA system
            'dataset': dataset_name,
            'video': 'dataset_level',  # SOTA data is aggregated at dataset level
            'classifier': STR_NA,  # SOTA doesn't use our classifier system
            'tilesize': 0,  # SOTA doesn't have explicit tile size
            'tilepadding': STR_NA,  # SOTA doesn't have explicit tile padding
            'sample_rate': df['sample_rate'],
            'tracker': STR_NA,  # SOTA doesn't use our tracker system
            'time': df['runtime'],  # Runtime in seconds
            'throughput_fps': float('nan'),  # Not available in tradeoff.csv
            'time_mspf': float('nan'),  # Not available without frame count
        })
        
        # Add all available submetrics from the CSV
        for submetric in submetric_columns:
            if submetric in df.columns:
                clean_df[submetric] = df[submetric]
            else:
                # Set to NaN if not available (for consistency)
                clean_df[submetric] = float('nan')
        
        sota_data.append(clean_df)
        print(f"  Loaded {len(df)} data points for {system.upper()} on {dataset_name}")

    if not sota_data:
        return pd.DataFrame()

    # Combine all SOTA data
    combined_sota_df = pd.concat(sota_data, ignore_index=True)
    datasets_covered = combined_sota_df['dataset'].unique()
    print(f"Combined {system.upper()} data: {len(combined_sota_df)} total rows (datasets: {list(datasets_covered)})")

    return combined_sota_df


def merge_sota_with_naive_baselines(df_combined: pd.DataFrame, df_sota: pd.DataFrame,
                                    x_column: str, system: str) -> pd.DataFrame:
    """
    Merge SOTA (OTIF or LEAP) data with our system's naive baselines for consistent comparison.

    Args:
        df_combined: Combined DataFrame with our system's data (already merged with naive data)
        df_sota: DataFrame with SOTA tradeoff data
        x_column: Column name for the metric being compared (e.g., 'time', 'throughput_fps')
        system: System name ('otif' or 'leap')

    Returns:
        pd.DataFrame: Combined data with SOTA results added, using our naive baselines
    """
    assert not df_sota.empty, f"No {system.upper()} data found"

    # Naive column is automatically created from merge with suffix '_naive'
    naive_column = f'{x_column}_naive'

    # Get unique datasets from SOTA data
    sota_datasets = df_sota['dataset'].unique()
    print(f"Adding {system.upper()} data for datasets: {list(sota_datasets)}")

    # Get naive baselines from our data for each dataset
    our_naive_baselines = df_combined.groupby('dataset')[naive_column].first().reset_index()
    our_naive_baselines.columns = ['dataset', naive_column]

    # Filter SOTA data to only include datasets we have data for
    df_sota_filtered = df_sota[df_sota['dataset'].isin(our_naive_baselines['dataset'])]

    # Warn about any skipped datasets
    skipped_datasets = set(sota_datasets) - set(our_naive_baselines['dataset'])
    for dataset in skipped_datasets:
        print(f"  Warning: No our data found for dataset {dataset}, skipping {system.upper()} data")

    # Merge SOTA data with our naive baselines using pandas merge
    assert not df_sota_filtered.empty, f"No {system.upper()} data found for datasets: {list(sota_datasets)}"

    df_sota_with_naive = df_sota_filtered.merge(our_naive_baselines, on='dataset', how='left')

    # Combine with existing data
    return df_sota_with_naive


def visualize_all_datasets_tradeoff(df_combined: pd.DataFrame, df_sota_dict: dict[str, pd.DataFrame],
                                   metrics_list: list[str], x_column: str, x_title: str,
                                   plot_suffix: str, output_dir: str):
    """
    Create visualization showing all dataset-wide trade-offs for all submetrics.
    
    Args:
        df_combined: Combined DataFrame with data from all datasets (already merged with naive data)
        df_sota_dict: Dictionary mapping system names ('otif', 'leap') to their tradeoff DataFrames
        metrics_list: list of main metrics to visualize (e.g., ['HOTA', 'Count'])
        x_column: Column name for x-axis data
        x_title: Title for x-axis
        plot_suffix: Suffix for plot filename
        output_dir: Output directory for visualizations
    """
    print(f"Creating all datasets {plot_suffix} tradeoff visualizations...")

    # Print best data points tables first
    print_best_data_points(df_combined, metrics_list, x_column, plot_suffix, include_system=True)

    # Add SOTA data (OTIF and LEAP) to appropriate datasets if available
    # Skip if x_column is not available in SOTA data (e.g., throughput_fps, time_mspf)
    for system, df_sota in df_sota_dict.items():
        if df_sota.empty:
            continue
        x_column_is_na = df_sota[x_column].isna().all()
        assert not isinstance(x_column_is_na, pd.Series), f"x_column_is_na is a Series, not a bool: {x_column_is_na}"
        if x_column in df_sota.columns and not x_column_is_na:
            sota_with_naive = merge_sota_with_naive_baselines(df_combined, df_sota, x_column, system)
            df_combined = pd.concat([df_combined, sota_with_naive], ignore_index=True)
        else:
            print(f"  Warning: {x_column} not available in {system.upper()} data, skipping {system.upper()} for this visualization")
    
    # Filter out rows where classifier == 'Perfect'
    df_combined = df_combined.query("classifier != 'Perfect'")
    
    # Convert tilepadding to string for shape encoding
    # Handle both numeric and string values, and STR_NA
    if 'tilepadding' in df_combined.columns:
        df_combined['tilepadding'] = df_combined['tilepadding'].astype(str)
    
    # Update system column: rows with classifier=='Groundtruth' should have system='Groundtruth'
    # This identifies the naive baseline points in our results
    df_combined.loc[df_combined['classifier'] == 'Groundtruth', 'system'] = 'Naive'
    
    # Define mapping of submetrics to their display names and y-axis scales
    # Each main metric maps to a list of (submetric_column, display_name, y_scale_dict) tuples
    submetrics_map = {
        'HOTA': [
            ('HOTA_HOTA', 'HOTA', {'scale': alt.Scale(domain=[0, 1])}),
            ('HOTA_AssA', 'AssA', {'scale': alt.Scale(domain=[0, 1])}),
            ('HOTA_DetA', 'DetA', {'scale': alt.Scale(domain=[0, 1])}),
        ],
        'CLEAR': [
            ('MOTA_MOTA', 'MOTA', {'scale': alt.Scale(domain=[0, 1])}),
        ],
        'Count': [
            ('Count_DetsMAPE', 'Dets MAPE', {}),
            ('Count_TracksMAPE', 'Tracks MAPE', {}),
        ],
    }
    
    # Create base chart
    base_chart = alt.Chart(df_combined)
    
    # Create visualizations for each submetric
    for main_metric in metrics_list:
        if main_metric not in submetrics_map:
            print(f"  Warning: Unknown metric '{main_metric}', skipping")
            continue
        
        # Iterate over all submetrics for this main metric
        for accuracy_col, metric_name, y_scale in submetrics_map[main_metric]:
            # Check if this submetric exists in the data
            if accuracy_col not in df_combined.columns:
                print(f"  Warning: Submetric '{accuracy_col}' not found in data, skipping")
                continue
            
            # Check if there's any non-NaN data for this submetric
            if bool(df_combined[accuracy_col].isna().all()):
                print(f"  Warning: Submetric '{accuracy_col}' has no valid data, skipping")
                continue
            
            # Create scatter plot with color by system and shape by tilepadding
            # Use conditional encoding to handle groundtruth points differently
            color_scale=alt.Scale(domain=['Polytris', 'OTIF', 'LEAP', 'Naive'], range=ColorScheme.CarbonDark)
            base_point = base_chart.mark_point(
                fillOpacity=1,
                stroke=None
            ).encode(
                fill=alt.Fill('system:N', title='System', scale=color_scale),
                size=alt.condition(
                    alt.datum.system == 'Naive',
                    alt.value(100),  # Larger size for groundtruth
                    alt.value(50)   # Normal size for others
                ),
                # shape=alt.condition(
                #     (alt.datum.system == 'Polytris') & (alt.datum.classifier != 'Naive'),
                #     alt.Shape('classifier:N', title='Polytris\' Classifier', scale=alt.Scale(domain=['MobileNetS', 'ShuffleNet05'], range=['triangle', 'diamond'])),
                #     alt.value('circle')  # Circle for non-Polytris points (OTIF, Naive)
                # ),
                tooltip=['system', 'dataset', 'classifier', 'sample_rate', 'tilepadding', 'tracker', x_column, accuracy_col]
            )
            
            base_line = base_chart.mark_line(
                strokeWidth=1.5,
                strokeOpacity=1
            ).encode(stroke=alt.Stroke('system:N', title='System', scale=color_scale))
            
            scatter = (base_point + base_line).encode(
                x=alt.X(f'{x_column}:Q', title=x_title),
                y=alt.Y(f'{accuracy_col}:Q', title=f'{metric_name} Score', **y_scale),
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
            
            # Save the chart with submetric name in filename in multiple formats
            # Save PNG format (raster image)
            plot_path_png = os.path.join(output_dir, f'{accuracy_col.lower()}_{plot_suffix}_comparison.png')
            combined_chart.save(plot_path_png, scale_factor=4)
            print(f"Saved all datasets {metric_name} {plot_suffix} comparison plot (PNG) to: {plot_path_png}")
            
            # Save SVG format (static vector image)
            plot_path_svg = os.path.join(output_dir, f'{accuracy_col.lower()}_{plot_suffix}_comparison.svg')
            combined_chart.save(plot_path_svg)
            print(f"Saved all datasets {metric_name} {plot_suffix} comparison plot (SVG) to: {plot_path_svg}")
            
            # Save HTML format (interactive SVG with tooltips)
            plot_path_html = os.path.join(output_dir, f'{accuracy_col.lower()}_{plot_suffix}_comparison.html')
            combined_chart.save(plot_path_html)
            print(f"Saved all datasets {metric_name} {plot_suffix} comparison plot (HTML with tooltips) to: {plot_path_html}")


def visualize_all_datasets_tradeoffs(datasets: list[str]):
    """
    Create runtime tradeoff visualizations for all datasets comparing with OTIF and LEAP.
    
    Args:
        datasets: list of dataset names
    """
    print(f"Creating all datasets tradeoff visualizations for {len(datasets)} datasets...")
    
    # Create output directory
    output_dir = os.path.join(CACHE_DIR, 'SUMMARY', '100_compare_compute')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use all available metrics for visualization (will visualize all submetrics)
    metrics_list = ['HOTA', 'Count']
    print(f"Using metrics: {metrics_list} (will visualize all submetrics)")
    
    # Load SOTA tradeoff data (OTIF and LEAP)
    print("Loading SOTA tradeoff data...")
    df_sota_dict = {}
    for system in ['otif', 'leap']:
        df_sota = load_sota_tradeoff_data(datasets, system)
        if not df_sota.empty:
            df_sota_dict[system] = df_sota
    
    if not df_sota_dict:
        print("  Warning: No SOTA tradeoff data found, proceeding without SOTA comparison")
    
    # Load tradeoff data for all datasets
    combined_df, naive_df = load_all_datasets_tradeoff_data(datasets, system_name='Polytris')
    
    # Handle backward compatibility: add sample_rate and tracker if missing
    if 'sample_rate' not in combined_df.columns:
        combined_df['sample_rate'] = 1
    if 'tracker' not in combined_df.columns:
        combined_df['tracker'] = 'unknown'
    if 'sample_rate' not in naive_df.columns:
        naive_df['sample_rate'] = 1
    if 'tracker' not in naive_df.columns:
        naive_df['tracker'] = 'unknown'
    
    combined_df['time_mspf'] = combined_df['time'] * 1000 / combined_df['frame_count']
    naive_df['time_mspf'] = naive_df['time'] * 1000 / naive_df['frame_count']
    
    # Ensure naive_combined has 'video' column set to 'dataset_level' (matching p091)
    # Each dataset's combined data has video='dataset_level', so we merge on dataset only
    if 'video' not in naive_df.columns:
        naive_df['video'] = 'dataset_level'
    
    # Merge naive data into combined data
    # Since combined data has video='dataset_level' for each dataset, merge on 'dataset'
    combined_with_naive = combined_df.merge(naive_df, on='dataset', how='left', suffixes=('', '_naive'))

    # Visualize HOTA_HOTA vs runtime (SOTA has runtime data)
    visualize = partial(visualize_all_datasets_tradeoff, combined_with_naive, df_sota_dict, metrics_list, output_dir=output_dir)
    visualize(x_column='time', x_title='Query Execution Runtime (seconds)', plot_suffix='runtime')


def main():
    """
    Main function that orchestrates the comparison between our tradeoff results and SOTA (OTIF/LEAP) results.
    
    This function serves as the entry point for the script. It loads pre-computed 
    tradeoff data from CSV files created by p090_tradeoff_compute.py for our system
    and SOTA tradeoff results from tradeoff.csv files created by p142_otif_tradeoff.py,
    then creates comparison visualizations showing all systems' performance.
    
    Note:
        - The script expects tradeoff data from p130_tradeoff_compute.py in:
          {CACHE_DIR}/{dataset}/evaluation/090_tradeoff/tradeoff_combined.csv
        - The script expects OTIF tradeoff results from p142_otif_tradeoff.py in:
          {CACHE_DIR}/SOTA/otif/{dataset}/tradeoff.csv
        - The script expects LEAP tradeoff results from p142_otif_tradeoff.py in:
          {CACHE_DIR}/SOTA/leap/{dataset}/tradeoff.csv
        - Results are saved to: {CACHE_DIR}/SUMMARY/100_compare_compute/
        - Please run p130_tradeoff_compute.py and p142_otif_tradeoff.py first to generate the required CSV files
        - Supports sample_rate and tracker dimensions with backward compatibility
    """
    print(f"Processing datasets: {DATASETS}")
    
    # Create comparison visualizations
    visualize_all_datasets_tradeoffs(DATASETS)


if __name__ == '__main__':
    main()
