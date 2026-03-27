#!/usr/local/bin/python

"""
Compare system performance using Pareto fronts and relative comparisons.

This script extends p200_compare_compute.py by:
1. Computing and displaying Pareto front lines instead of all data points for Polytris
2. Adding visualizations showing speedup ratios at accuracy increments
3. Adding visualizations showing accuracy gains at runtime increments
"""

import argparse
import os
import numpy as np
import pandas as pd
import altair as alt

from polyis.io import cache
from polyis.pareto import compute_pareto_front
from polyis.utilities import get_config, load_tradeoff_data, split_tradeoff_variants
from evaluation.utilities import ColorScheme
from evaluation.p200_compare_compute import load_sota_tradeoff_data


config = get_config()
DATASETS = config['EXEC']['DATASETS']
CLASSIFIERS = config['EXEC']['CLASSIFIERS']
TILEPADDING_MODES = config['EXEC']['TILEPADDING_MODES']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']
TRACKERS = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS = config['EXEC']['TRACKING_ACCURACY_THRESHOLDS']

# Define the chart size multiplier for all rendered comparison charts.
CHART_SIZE_SCALE = 1.5

# Define fixed system-to-color categories for deterministic chart encoding.
SYSTEM_COLOR_DOMAIN = ['Polytris', 'Naive', 'OTIF', 'LEAP']

# Define the color for each system category.
SYSTEM_COLOR_RANGE = ColorScheme.CarbonDark[:len(SYSTEM_COLOR_DOMAIN)]



def load_polytris_tradeoff_split_data(datasets: list[str]) -> pd.DataFrame:
    """
    Load the canonical split-level Polytris tradeoff table for all datasets.

    Args:
        datasets: Datasets to load

    Returns:
        Combined split-level tradeoff DataFrame
    """
    all_tradeoff_rows = []

    # Load the canonical split-level tradeoff table for each configured dataset.
    for dataset in datasets:
        tradeoff_df = load_tradeoff_data(dataset)
        if 'dataset' not in tradeoff_df.columns:
            tradeoff_df['dataset'] = dataset
        all_tradeoff_rows.append(tradeoff_df)

    if not all_tradeoff_rows:
        return pd.DataFrame()

    return pd.concat(all_tradeoff_rows, ignore_index=True)




def compute_speedup_at_accuracy_levels(df_polytris: pd.DataFrame,
                                       df_sota_dict: dict[str, pd.DataFrame],
                                       accuracy_col: str,
                                       time_col: str = 'time',
                                       increment: float = 0.02) -> pd.DataFrame:
    """
    Compute speedup ratio (time_other/time_polytris) at accuracy increments.

    Args:
        df_polytris: DataFrame with Polytris Pareto front data
        df_sota_dict: Dictionary mapping system names to their Pareto front DataFrames
        accuracy_col: Column name for accuracy metric (e.g., 'HOTA_HOTA')
        time_col: Column name for runtime
        increment: Accuracy increment step size

    Returns:
        DataFrame with columns: dataset, accuracy_level, comparison_system, speedup_ratio
    """
    results = []

    # Get all datasets from Polytris data
    datasets = df_polytris['dataset'].unique()

    for dataset in datasets:
        # Get Polytris data for this dataset
        polytris_data = df_polytris[df_polytris['dataset'] == dataset]

        if polytris_data.empty:
            continue

        # Use data directly (already Pareto-optimal from p022).
        polytris_by_acc = polytris_data.dropna(subset=[time_col, accuracy_col]).sort_values(accuracy_col)

        if polytris_by_acc.empty:
            continue

        # Define accuracy query points based on data range
        min_acc = 0.0
        max_acc = min(1.0, polytris_by_acc[accuracy_col].max())
        accuracy_levels = np.arange(min_acc, max_acc + increment, increment)

        # For speedup comparison, interpolate time at given accuracy levels
        # by inverting the sorted data: given accuracy, find time

        # Interpolate Polytris time at each accuracy level
        polytris_times = np.interp(
            accuracy_levels,
            polytris_by_acc[accuracy_col].values,
            polytris_by_acc[time_col].values
        )

        # Compare with each SOTA system
        for system_name, df_sota in df_sota_dict.items():
            # Get SOTA data for this dataset
            sota_data = df_sota[df_sota['dataset'] == dataset]

            if sota_data.empty:
                continue

            # Use data directly.
            sota_by_acc = sota_data.dropna(subset=[time_col, accuracy_col]).sort_values(accuracy_col)

            if sota_by_acc.empty:
                continue

            # Interpolate SOTA time at each accuracy level
            sota_times = np.interp(
                accuracy_levels,
                sota_by_acc[accuracy_col].values,
                sota_by_acc[time_col].values
            )

            # Compute speedup ratio (SOTA time / Polytris time)
            # > 1 means Polytris is faster
            speedup_ratios = sota_times / polytris_times

            # Mask values where interpolation is outside valid range
            valid_mask = (
                (accuracy_levels >= polytris_by_acc[accuracy_col].min()) &
                (accuracy_levels <= polytris_by_acc[accuracy_col].max()) &
                (accuracy_levels >= sota_by_acc[accuracy_col].min()) &
                (accuracy_levels <= sota_by_acc[accuracy_col].max())
            )
            speedup_ratios[~valid_mask] = np.nan

            # Create result DataFrame for this comparison
            comparison_df = pd.DataFrame({
                'dataset': dataset,
                'accuracy_level': accuracy_levels,
                'comparison_system': system_name.upper(),
                'speedup_ratio': speedup_ratios,
                'polytris_time': polytris_times,
                'other_time': sota_times
            })

            results.append(comparison_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def compute_accuracy_gain_at_runtime_levels(df_polytris: pd.DataFrame,
                                            df_sota_dict: dict[str, pd.DataFrame],
                                            accuracy_col: str,
                                            time_col: str = 'time',
                                            increment: float = 50.0) -> pd.DataFrame:
    """
    Compute accuracy gain (acc_polytris - acc_other) at runtime increments.

    Args:
        df_polytris: DataFrame with Polytris Pareto front data
        df_sota_dict: Dictionary mapping system names to their Pareto front DataFrames
        accuracy_col: Column name for accuracy metric (e.g., 'HOTA_HOTA')
        time_col: Column name for runtime
        increment: Runtime increment step size in seconds

    Returns:
        DataFrame with columns: dataset, runtime_level, comparison_system, accuracy_gain
    """
    results = []

    # Get all datasets from Polytris data
    datasets = df_polytris['dataset'].unique()

    for dataset in datasets:
        # Get Polytris data for this dataset
        polytris_data = df_polytris[df_polytris['dataset'] == dataset]

        if polytris_data.empty:
            continue

        # Use data directly (already Pareto-optimal from p022).
        polytris_by_time = polytris_data.dropna(subset=[time_col, accuracy_col]).sort_values(time_col)

        if polytris_by_time.empty:
            continue

        # Determine runtime range based on all systems
        max_time = polytris_by_time[time_col].max()

        # Check SOTA systems for max time
        for df_sota in df_sota_dict.values():
            sota_data = df_sota[df_sota['dataset'] == dataset]
            if not sota_data.empty:
                max_time = max(max_time, sota_data[time_col].max())

        # Define runtime query points
        runtime_levels = np.arange(0, max_time + increment, increment)

        # Interpolate Polytris accuracy at each runtime level
        polytris_accuracies = np.interp(
            runtime_levels,
            polytris_by_time[time_col].values,
            polytris_by_time[accuracy_col].values
        )

        # Mask values outside Polytris range
        valid_polytris = (
            (runtime_levels >= polytris_by_time[time_col].min()) &
            (runtime_levels <= polytris_by_time[time_col].max())
        )

        # Compare with each SOTA system
        for system_name, df_sota in df_sota_dict.items():
            # Get SOTA data for this dataset
            sota_data = df_sota[df_sota['dataset'] == dataset]

            if sota_data.empty:
                continue

            # Use data directly.
            sota_by_time = sota_data.dropna(subset=[time_col, accuracy_col]).sort_values(time_col)

            if sota_by_time.empty:
                continue

            # Interpolate SOTA accuracy at each runtime level
            sota_accuracies = np.interp(
                runtime_levels,
                sota_by_time[time_col].values,
                sota_by_time[accuracy_col].values
            )

            # Compute accuracy gain (Polytris - SOTA)
            # > 0 means Polytris is more accurate
            accuracy_gains = polytris_accuracies - sota_accuracies

            # Mask values where interpolation is outside valid range
            valid_sota = (
                (runtime_levels >= sota_by_time[time_col].min()) &
                (runtime_levels <= sota_by_time[time_col].max())
            )
            valid_mask = valid_polytris & valid_sota
            accuracy_gains[~valid_mask] = np.nan

            # Create result DataFrame for this comparison
            comparison_df = pd.DataFrame({
                'dataset': dataset,
                'runtime_level': runtime_levels,
                'comparison_system': system_name.upper(),
                'accuracy_gain': accuracy_gains,
                'polytris_accuracy': polytris_accuracies,
                'other_accuracy': sota_accuracies
            })

            results.append(comparison_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def create_speedup_chart(df_speedup: pd.DataFrame, accuracy_col_name: str) -> alt.Chart:
    """
    Create faceted line chart showing speedup ratio vs accuracy level.

    Args:
        df_speedup: DataFrame with speedup data
        accuracy_col_name: Display name for the accuracy metric

    Returns:
        Altair Chart object
    """
    # Drop NaN values for visualization
    df_clean = df_speedup.dropna(subset=['speedup_ratio'])

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Color scale for comparison systems
    color_scale = alt.Scale(
        domain=df_clean['comparison_system'].unique().tolist(),
        range=ColorScheme.CarbonDark[:len(df_clean['comparison_system'].unique())]
    )

    # Base chart
    base = alt.Chart(df_clean)

    # Line chart showing speedup ratio (no tooltip - lines are not easily hoverable)
    line = base.mark_line(strokeWidth=2).encode(
        x=alt.X('accuracy_level:Q', title=f'{accuracy_col_name} Level'),
        y=alt.Y('speedup_ratio:Q', title='Speedup Ratio (Other/Polytris)'),
        color=alt.Color('comparison_system:N', title='Compared To', scale=color_scale),
    )

    # Add points for better visibility (with tooltip for interactivity)
    points = base.mark_point(size=30, filled=True).encode(
        x=alt.X('accuracy_level:Q'),
        y=alt.Y('speedup_ratio:Q'),
        color=alt.Color('comparison_system:N', scale=color_scale),
        tooltip=['dataset', 'accuracy_level', 'comparison_system', 'speedup_ratio',
                 'polytris_time', 'other_time']
    )

    # Horizontal rule at y=1 (parity line)
    rule = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1
    ).encode(y='y:Q')

    # Combine layers
    chart = (line + points + rule).properties(
        width=200 * CHART_SIZE_SCALE,
        height=150 * CHART_SIZE_SCALE
    )

    # Facet by dataset
    faceted_chart = chart.facet(
        facet=alt.Facet('dataset:N', title=None,
                        header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
        columns=3
    ).resolve_scale(
        x='independent',
        y='independent'
    ).properties(
        title=f'Speedup Ratio at {accuracy_col_name} Levels (>1 = Polytris faster)'
    )

    return faceted_chart


def create_accuracy_gain_chart(df_accuracy_gain: pd.DataFrame, accuracy_col_name: str,
                               log_scale: bool = False) -> alt.Chart:
    """
    Create faceted line chart showing accuracy gain vs runtime level.

    Args:
        df_accuracy_gain: DataFrame with accuracy gain data
        accuracy_col_name: Display name for the accuracy metric

    Returns:
        Altair Chart object
    """
    # Drop NaN values for visualization
    df_clean = df_accuracy_gain.dropna(subset=['accuracy_gain'])

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Color scale for comparison systems
    color_scale = alt.Scale(
        domain=df_clean['comparison_system'].unique().tolist(),
        range=ColorScheme.CarbonDark[:len(df_clean['comparison_system'].unique())]
    )

    # Base chart
    base = alt.Chart(df_clean)

    # X-axis uses log scale for runtime (seconds) if enabled
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X('runtime_level:Q', title='Runtime (seconds)', scale=x_scale)
    # Line chart showing accuracy gain (no tooltip - lines are not easily hoverable)
    line = base.mark_line(strokeWidth=2).encode(
        x=x_enc,
        y=alt.Y('accuracy_gain:Q', title=f'{accuracy_col_name} Gain (Polytris - Other)'),
        color=alt.Color('comparison_system:N', title='Compared To', scale=color_scale),
    )

    # Add points for better visibility (with tooltip for interactivity)
    points = base.mark_point(size=30, filled=True).encode(
        x=x_enc,
        y=alt.Y('accuracy_gain:Q'),
        color=alt.Color('comparison_system:N', scale=color_scale),
        tooltip=['dataset', 'runtime_level', 'comparison_system', 'accuracy_gain',
                 'polytris_accuracy', 'other_accuracy']
    )

    # Horizontal rule at y=0 (parity line)
    rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1
    ).encode(y='y:Q')

    # Combine layers
    chart = (line + points + rule).properties(
        width=200 * CHART_SIZE_SCALE,
        height=150 * CHART_SIZE_SCALE
    )

    # Facet by dataset
    faceted_chart = chart.facet(
        facet=alt.Facet('dataset:N', title=None,
                        header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
        columns=3
    ).resolve_scale(
        x='independent',
        y='independent'
    ).properties(
        title=f'{accuracy_col_name} Gain at Runtime Levels (>0 = Polytris more accurate)'
    )

    return faceted_chart


def create_pareto_comparison_chart(df_combined: pd.DataFrame, accuracy_col: str,
                                   accuracy_col_name: str, time_col: str = 'time',
                                   log_scale: bool = False) -> alt.Chart:
    """
    Create faceted line chart showing Pareto fronts for all systems.

    Args:
        df_combined: DataFrame with Pareto front data from all systems
        accuracy_col: Column name for accuracy metric
        accuracy_col_name: Display name for the accuracy metric
        time_col: Column name for runtime

    Returns:
        Altair Chart object
    """
    # Drop rows with NaN in key columns
    df_clean = df_combined.dropna(subset=[time_col, accuracy_col])

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Define consistent color scale for systems.
    color_scale = alt.Scale(
        domain=SYSTEM_COLOR_DOMAIN,
        range=SYSTEM_COLOR_RANGE
    )

    # Base chart for Pareto fronts (with lines)
    base_pareto = alt.Chart(df_clean)

    # X-axis uses log scale for runtime (seconds) if enabled
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X(f'{time_col}:Q', title='Runtime (seconds)', scale=x_scale)
    # Opacity scale: full opacity for Polytris, reduced for other systems.
    opacity_scale = alt.Scale(
        domain=SYSTEM_COLOR_DOMAIN,
        range=[1.0, 0.2, 0.2, 0.2]
    )

    # Line chart showing Pareto fronts (no tooltip - lines are not easily hoverable)
    line = base_pareto.mark_line(strokeWidth=2).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q', title=f'{accuracy_col_name} Score',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('system:N', title='System', scale=color_scale),
        opacity=alt.Opacity('system:N', title='System', scale=opacity_scale),
        # Group lines by the columns that define a single Pareto front.
        # Polytris fronts are computed per (dataset, classifier, canvas_scale);
        # SOTA fronts are computed per (dataset).  The chart is already faceted
        # by dataset, so only system/classifier/canvas_scale are needed here.
        # Including varying parameters (sample_rate, tilepadding, etc.) would
        # split each front into isolated single-point "lines".
        detail=['system:N', 'classifier:N', 'canvas_scale:N']
    )

    # Shape scale: square for Polytris, trangle for Naive, circle for others.
    shape_scale = alt.Scale(
        domain=SYSTEM_COLOR_DOMAIN,
        range=['square', 'triangle', 'circle', 'circle']
    )

    # Add points for Pareto fronts (with tooltip for interactivity)
    points_pareto = base_pareto.mark_point(size=50, filled=True).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q'),
        color=alt.Color('system:N', title='System', scale=color_scale),
        opacity=alt.Opacity('system:N', title='System', scale=opacity_scale),
        shape=alt.Shape('system:N', title='System', scale=shape_scale),
        tooltip=['system', 'dataset', 'classifier', 'sample_rate',
                 'tracking_accuracy_threshold', 'tilepadding', 'canvas_scale',
                 'tracker', time_col, accuracy_col]
    )

    # Combine layers
    chart = (line + points_pareto).properties(
        width=200 * CHART_SIZE_SCALE,
        height=150 * CHART_SIZE_SCALE
    )

    # Facet by dataset
    faceted_chart = chart.facet(
        facet=alt.Facet('dataset:N', title=None,
                        header=alt.Header(labelExpr="'Dataset: ' + datum.value")),
        columns=3
    ).resolve_scale(
        x='independent',
        y='independent'
    ).properties(
        title=f'{accuracy_col_name} vs Runtime Pareto Fronts'
    )

    return faceted_chart


def filter_by_config(df: pd.DataFrame,
                     classifiers: list[str] | None = None,
                     tilepadding_modes: list[str] | None = None,
                     sample_rates: list[int] | None = None,
                     tracking_accuracy_thresholds: list[float | None] | None = None,
                     trackers: list[str] | None = None) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows matching configuration.

    Args:
        df: DataFrame to filter
        classifiers: List of allowed classifier names (if column exists)
        tilepadding_modes: List of allowed tilepadding modes (if column exists)
        sample_rates: List of allowed sample rates (if column exists)
        tracking_accuracy_thresholds: List of allowed thresholds (None means no pruning)
        trackers: List of allowed tracker names (if column exists)

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Filter by classifier (if column exists and filter is specified)
    if classifiers is not None and 'classifier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['classifier'].isin(classifiers)]
        print(f"  Filtered by classifiers: {len(filtered_df)} rows remain")

    # Filter by tilepadding (if column exists and filter is specified)
    if tilepadding_modes is not None and 'tilepadding' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['tilepadding'].isin(tilepadding_modes)]
        print(f"  Filtered by tilepadding: {len(filtered_df)} rows remain")

    # Filter by sample_rate (if column exists and filter is specified)
    if sample_rates is not None and 'sample_rate' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sample_rate'].isin(sample_rates)]
        print(f"  Filtered by sample_rates: {len(filtered_df)} rows remain")

    # Filter by pruning threshold (if column exists and filter is specified).
    if tracking_accuracy_thresholds is not None and 'tracking_accuracy_threshold' in filtered_df.columns:
        allowed_thresholds = [th for th in tracking_accuracy_thresholds if th is not None]
        include_no_pruning = any(th is None for th in tracking_accuracy_thresholds)
        threshold_mask = filtered_df['tracking_accuracy_threshold'].isin(allowed_thresholds)
        if include_no_pruning:
            threshold_mask = threshold_mask | filtered_df['tracking_accuracy_threshold'].isna()
        filtered_df = filtered_df[threshold_mask]
        print(f"  Filtered by tracking_accuracy_thresholds: {len(filtered_df)} rows remain")

    # Filter by tracker (if column exists and filter is specified)
    if trackers is not None and 'tracker' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['tracker'].isin(trackers)]
        print(f"  Filtered by trackers: {len(filtered_df)} rows remain")

    return filtered_df


def save_chart(chart: alt.Chart, output_dir: str, base_name: str):
    """Save chart in multiple formats (PNG, SVG, HTML)."""
    # Save PNG format
    png_path = os.path.join(output_dir, f'{base_name}.png')
    chart.save(png_path, scale_factor=4)
    print(f"  Saved PNG: {png_path}")

    # Save SVG format
    svg_path = os.path.join(output_dir, f'{base_name}.svg')
    chart.save(svg_path)
    print(f"  Saved SVG: {svg_path}")

    # Save HTML format (interactive)
    html_path = os.path.join(output_dir, f'{base_name}.html')
    chart.save(html_path)
    print(f"  Saved HTML: {html_path}")


def _filter_pareto_per_dataset(df: pd.DataFrame, time_col: str,
                               accuracy_col: str) -> pd.DataFrame:
    """
    Filter DataFrame to only Pareto-optimal points per dataset.

    Keeps only points on the Pareto front (minimize time, maximize accuracy)
    computed independently for each dataset.

    Args:
        df: DataFrame with 'dataset' column and the specified metric columns
        time_col: Column name for runtime (minimized)
        accuracy_col: Column name for accuracy (maximized)

    Returns:
        DataFrame containing only Pareto-optimal points across all datasets
    """
    pareto_groups = []
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        # Compute Pareto front: minimize time (x), maximize accuracy (y).
        pareto_df = compute_pareto_front(dataset_df, time_col, accuracy_col)
        pareto_groups.append(pareto_df)

    if not pareto_groups:
        return pd.DataFrame()
    return pd.concat(pareto_groups, ignore_index=True)


def visualize_all_datasets_tradeoffs_pareto(datasets: list[str], log_scale: bool = False):
    """
    Create Pareto front comparison visualizations for all datasets.

    Args:
        datasets: List of dataset names to process
    """
    print(f"Creating Pareto front visualizations for {len(datasets)} datasets...")

    # Create output directory
    output_dir = cache.summary('101_compare_pareto')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load the canonical split-level Polytris tradeoff rows.
    print("\nLoading Polytris tradeoff data...")
    tradeoff_df = load_polytris_tradeoff_split_data(datasets)

    if tradeoff_df.empty:
        print("  Warning: No Polytris tradeoff rows found")
        return

    # Split the canonical table into Polytris and naive rows.
    polytris_tradeoff_df, naive_tradeoff_df = split_tradeoff_variants(tradeoff_df)

    # Filter out non-data points before valid/test selection.
    polytris_tradeoff_df = polytris_tradeoff_df.query("classifier != 'Perfect'")

    # Work directly on the canonical split-level table emitted by p130.
    polytris_split_df = polytris_tradeoff_df.copy()
    if polytris_split_df.empty:
        print("  Warning: No split-level Polytris rows found")
        return

    # Print a quick split distribution for sanity checking.
    split_counts = polytris_split_df['videoset'].value_counts(dropna=False).to_dict()
    print(f"  Aggregated Polytris rows: {len(polytris_split_df)} (videoset counts: {split_counts})")

    # Extract test rows only; valid split is no longer needed.
    test_all_df = polytris_split_df[polytris_split_df['videoset'] == 'test'].copy()
    print(f"  Test rows: {len(test_all_df)}")
    if test_all_df.empty:
        print("  Warning: No test rows found after videoset separation")

    # Extract test naive baseline rows before config filtering.
    naive_df = naive_tradeoff_df[naive_tradeoff_df['videoset'] == 'test'].copy()
    naive_df['system'] = 'Naive'
    print(f"\nExtracted {len(naive_df)} naive baseline rows from test split")

    # Filter test Polytris rows by configured parameter dimensions.
    print("\nFiltering Polytris data by configuration settings...")
    print(f"  Test rows before filtering: {len(test_all_df)}")
    filtered_test_non_naive_df = filter_by_config(
        test_all_df,
        classifiers=CLASSIFIERS,
        tilepadding_modes=TILEPADDING_MODES,
        sample_rates=SAMPLE_RATES,
        tracking_accuracy_thresholds=TRACKING_ACCURACY_THRESHOLDS,
        trackers=TRACKERS
    )
    print(f"  Test rows after filtering: {len(filtered_test_non_naive_df)}")

    # Load SOTA tradeoff data
    print("\nLoading SOTA tradeoff data...")
    df_sota_dict = {}
    for system in ['otif', 'leap']:
        df_sota = load_sota_tradeoff_data(datasets, system)
        if not df_sota.empty:
            df_sota_dict[system] = df_sota
            print(f"  Loaded {len(df_sota)} rows for {system.upper()}")

    if not df_sota_dict:
        print("  Warning: No SOTA tradeoff data found")

    # Define metrics to visualize
    metrics_map = {
        'HOTA_HOTA': 'HOTA',
        'HOTA_AssA': 'AssA',
        'HOTA_DetA': 'DetA',
    }

    # Process each metric
    for accuracy_col, accuracy_name in metrics_map.items():
        print(f"\n{'='*60}")
        print(f"Processing metric: {accuracy_name} ({accuracy_col})")
        print('='*60)

        # Check if metric exists in data
        if accuracy_col not in polytris_split_df.columns:
            print(f"  Warning: {accuracy_col} not found in Polytris data, skipping")
            continue

        # Skip if all NaN
        if polytris_split_df[accuracy_col].isna().all():
            print(f"  Warning: {accuracy_col} has no valid data, skipping")
            continue

        # Filter test rows to valid metric data.
        polytris_df = filtered_test_non_naive_df.dropna(subset=['time', accuracy_col]).copy()
        if polytris_df.empty:
            print(f"  Warning: No Polytris test rows found for {accuracy_name}, skipping")
            continue
        print(f"  Polytris test rows: {len(polytris_df)}")

        # Filter each system's data to Pareto-optimal points per dataset
        # (minimize time, maximize accuracy).
        polytris_df = _filter_pareto_per_dataset(polytris_df, 'time', accuracy_col)
        print(f"  Polytris Pareto-optimal points: {len(polytris_df)}")

        # Filter Naive baseline to Pareto-optimal points per dataset.
        if accuracy_col in naive_df.columns:
            pareto_naive_df = _filter_pareto_per_dataset(
                naive_df.dropna(subset=['time', accuracy_col]), 'time', accuracy_col
            )
        else:
            pareto_naive_df = pd.DataFrame()

        # Filter each SOTA system to Pareto-optimal points per dataset.
        pareto_sota_dict: dict[str, pd.DataFrame] = {}
        for system_name, df_sota in df_sota_dict.items():
            if accuracy_col not in df_sota.columns:
                continue
            filtered_sota = _filter_pareto_per_dataset(
                df_sota.dropna(subset=['time', accuracy_col]), 'time', accuracy_col
            )
            if not filtered_sota.empty:
                pareto_sota_dict[system_name] = filtered_sota
                print(f"  {system_name.upper()} Pareto-optimal points: {len(filtered_sota)}")

        # 2. Collect Pareto-optimal data for visualization
        print(f"\n2. Collecting data for {accuracy_name}...")

        # Columns to keep for tooltip display.
        tooltip_cols = ['system', 'dataset', 'videoset', 'classifier', 'sample_rate',
                        'tracking_accuracy_threshold', 'tilepadding', 'canvas_scale',
                        'tracker', 'time', accuracy_col]

        # Collect Pareto-optimal rows from Polytris, Naive, and SOTA systems.
        pareto_data_list = []

        # Append Polytris Pareto-optimal points.
        polytris_metric_df = polytris_df.copy()
        if not polytris_metric_df.empty:
            polytris_metric_df['system'] = 'Polytris'
            polytris_cols = [c for c in tooltip_cols if c in polytris_metric_df.columns]
            pareto_data_list.append(polytris_metric_df[polytris_cols])

        # Append Naive Pareto-optimal points.
        if not pareto_naive_df.empty:
            naive_point_df = pareto_naive_df.copy()
            naive_point_df['system'] = 'Naive'
            naive_cols = [c for c in tooltip_cols if c in naive_point_df.columns]
            pareto_data_list.append(naive_point_df[naive_cols])

        # Append SOTA Pareto-optimal points.
        sota_pareto_list = []
        for system_name, df_sota in pareto_sota_dict.items():
            sota_metric_df = df_sota.copy()
            sota_metric_df['system'] = system_name.upper()
            sota_cols = [c for c in tooltip_cols if c in sota_metric_df.columns]
            sota_pareto_list.append(sota_metric_df[sota_cols])

        if sota_pareto_list:
            pareto_data_list.extend(sota_pareto_list)

        if not pareto_data_list:
            print(f"  No data available for {accuracy_name}")
            continue

        df_pareto_combined = pd.concat(pareto_data_list, ignore_index=True)
        print(f"  Combined Pareto-optimal data: {len(df_pareto_combined)} points")

        # 3. Create comparison chart
        print(f"\n3. Creating comparison chart for {accuracy_name}...")
        pareto_chart = create_pareto_comparison_chart(
            df_pareto_combined, accuracy_col, accuracy_name, log_scale=log_scale
        )
        save_chart(pareto_chart, output_dir,
                   f'{accuracy_col.lower()}_runtime_pareto_comparison')

        # 4. Compute and visualize speedup at accuracy levels
        print(f"\n4. Computing speedup at accuracy levels for {accuracy_name}...")
        df_speedup = compute_speedup_at_accuracy_levels(
            polytris_df, pareto_sota_dict, accuracy_col, 'time', increment=0.005
        )

        if not df_speedup.empty:
            print(f"  Speedup data: {len(df_speedup)} comparison points")
            speedup_chart = create_speedup_chart(df_speedup, accuracy_name)
            save_chart(speedup_chart, output_dir,
                       f'{accuracy_col.lower()}_speedup_at_accuracy')
        else:
            print("  No speedup data available")

        # 5. Compute and visualize accuracy gain at runtime levels
        print(f"\n5. Computing accuracy gain at runtime levels for {accuracy_name}...")
        df_accuracy_gain = compute_accuracy_gain_at_runtime_levels(
            polytris_df, pareto_sota_dict, accuracy_col, 'time', increment=5.0
        )

        if not df_accuracy_gain.empty:
            print(f"  Accuracy gain data: {len(df_accuracy_gain)} comparison points")
            accuracy_gain_chart = create_accuracy_gain_chart(df_accuracy_gain, accuracy_name,
                                                               log_scale=log_scale)
            save_chart(accuracy_gain_chart, output_dir,
                       f'{accuracy_col.lower()}_accuracy_gain_at_runtime')
        else:
            print("  No accuracy gain data available")

    print(f"\n{'='*60}")
    print("Pareto front visualizations complete!")
    print(f"Output directory: {output_dir}")
    print('='*60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true',
                        help='Use log scale for time x-axes')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Processing datasets: {DATASETS}")

    # Resolve the single videoset from the mutually exclusive CLI flags.
    videoset = 'test' if args.test else 'valid'

    # This script compares test-split results; skip when test was not requested.
    if videoset != 'test':
        print("Skipping: Pareto comparison visualization requires --test.")
        return

    visualize_all_datasets_tradeoffs_pareto(DATASETS, log_scale=args.log)


if __name__ == '__main__':
    main()
