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

# Parameter columns used to transfer valid-Pareto parameter sets to test points.
PARETO_PARAM_COLS = [
    'variant_id',
    'classifier',
    'tilesize',
    'sample_rate',
    'tracking_accuracy_threshold',
    'tilepadding',
    'canvas_scale',
    'tracker',
]

def val_gte(val: float, best_val: float) -> bool:
    return val >= best_val


def val_lte(val: float, best_val: float) -> bool:
    return val <= best_val


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str,
                         minimize_x: bool = False, maximize_y: bool = True) -> pd.DataFrame:
    """
    Compute Pareto-optimal points from DataFrame.

    For minimize_x=True and maximize_y=True (the default):
    A point is Pareto-optimal if no other point has both lower x AND higher y.

    Args:
        df: DataFrame with data points
        x_col: Column name for x-axis (e.g., 'time')
        y_col: Column name for y-axis (e.g., 'HOTA_HOTA')
        minimize_x: If True, lower x is better; if False, higher x is better
        maximize_y: If True, higher y is better; if False, lower y is better

    Returns:
        DataFrame containing only Pareto-optimal points, sorted by x_col
    """
    # Drop rows with NaN in x or y columns
    df_clean = df.dropna(subset=[x_col, y_col]).copy()

    if df_clean.empty:
        return df_clean

    # Sort by x_col (ascending if minimizing x, descending if maximizing)
    df_sorted = df_clean.sort_values(x_col, ascending=minimize_x).reset_index(drop=True)

    # Build Pareto front using cumulative max/min approach
    # Track the best y value seen so far from the "expensive" end (high x if minimizing x)
    pareto_indices = []

    better_y = val_gte if maximize_y else val_lte
    best_y = float('-inf') if maximize_y else float('inf')
    for idx in reversed(df_sorted.index):
        y_val = df_sorted.loc[idx, y_col]
        if better_y(y_val, best_y):
            pareto_indices.append(idx)
            best_y = y_val

    # Reverse to maintain sorted order by x
    pareto_indices = list(reversed(pareto_indices))

    # Return Pareto-optimal points
    return df_sorted.loc[pareto_indices].reset_index(drop=True)


def interpolate_pareto_line(pareto_df: pd.DataFrame, x_col: str, y_col: str,
                            query_points: np.ndarray,
                            extrapolate: bool = False) -> pd.DataFrame:
    """
    Linearly interpolate Pareto front at specified query points.

    Args:
        pareto_df: DataFrame with Pareto front points (sorted by x_col)
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values to interpolate
        query_points: Array of x values at which to interpolate
        extrapolate: If False, return NaN for points outside the data range

    Returns:
        DataFrame with 'query_point' and 'interpolated_value' columns
    """
    if pareto_df.empty:
        return pd.DataFrame({
            'query_point': query_points,
            'interpolated_value': [np.nan] * len(query_points)
        })

    # Get x and y values from Pareto front
    x_vals = pareto_df[x_col].values
    y_vals = pareto_df[y_col].values

    # Interpolate using numpy
    if extrapolate:
        # Allow extrapolation outside data range
        interpolated = np.interp(query_points, x_vals, y_vals)
    else:
        # Return NaN for points outside the data range
        interpolated = np.interp(query_points, x_vals, y_vals)
        # Mask values outside the range
        mask_below = query_points < x_vals.min()
        mask_above = query_points > x_vals.max()
        interpolated[mask_below | mask_above] = np.nan

    return pd.DataFrame({
        'query_point': query_points,
        'interpolated_value': interpolated
    })


def compute_pareto_fronts_by_group(df: pd.DataFrame, group_cols: list[str],
                                   x_col: str, y_col: str) -> pd.DataFrame:
    """
    Compute Pareto fronts for each group in the DataFrame.

    Args:
        df: DataFrame with data points
        group_cols: Columns to group by (e.g., ['dataset', 'system'])
        x_col: Column name for x-axis
        y_col: Column name for y-axis

    Returns:
        DataFrame with Pareto-optimal points for each group
    """
    # Apply Pareto front computation to each group
    pareto_groups = (
        df.groupby(group_cols, group_keys=True, dropna=False)
        .apply(lambda g: compute_pareto_front(g, x_col, y_col), include_groups=False)
    )

    # Restore grouping keys from the MultiIndex as regular columns.
    pareto_groups = pareto_groups.reset_index(level=group_cols)

    return pareto_groups.reset_index(drop=True)


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


def select_test_points_from_valid_pareto(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    accuracy_col: str,
    time_col: str = 'time',
) -> pd.DataFrame:
    """
    Select test points whose parameter sets appear on the valid Pareto front.

    Args:
        valid_df: Videoset-level Polytris rows for valid split
        test_df: Videoset-level Polytris rows for test split
        accuracy_col: Accuracy metric column used for Pareto computation
        time_col: Runtime metric column used for Pareto computation

    Returns:
        Test rows matched to valid Pareto parameter sets
    """
    # Short-circuit when no test rows are available.
    if test_df.empty:
        print("  Warning: No test rows found; nothing to select")
        return pd.DataFrame(columns=test_df.columns)

    # Fall back to all test rows when no valid rows are available.
    if valid_df.empty:
        print("  Warning: No valid rows found; using all test rows")
        return test_df.copy()

    # Keep only parameter columns available in both valid and test DataFrames.
    match_cols = [col for col in PARETO_PARAM_COLS if col in valid_df.columns and col in test_df.columns]
    if not match_cols:
        print("  Warning: No shared parameter columns between valid and test; using all test rows")
        return test_df.copy()

    # Keep valid rows with metric/time values required by Pareto computation.
    valid_metric_df = valid_df.dropna(subset=[time_col, accuracy_col]).copy()
    if valid_metric_df.empty:
        print("  Warning: Valid rows have no metric/time values; using all test rows")
        return test_df.copy()

    # Compute valid Pareto points per dataset using DataFrame groupby+apply.
    valid_pareto_df = compute_pareto_fronts_by_group(
        valid_metric_df,
        ['dataset'],
        time_col,
        accuracy_col,
    )
    if valid_pareto_df.empty:
        print("  Warning: Valid Pareto is empty; using all test rows")
        return test_df.copy()

    # Build unique valid Pareto parameter sets per dataset.
    merge_cols = ['dataset', *match_cols]
    valid_param_sets = valid_pareto_df[merge_cols].drop_duplicates().copy()

    # Build merge keys that treat NaN values as equal for parameter matching.
    def build_merge_keys(df_in: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
        df_out = df_in.copy()
        key_cols: list[str] = []
        for col in cols:
            key_col = f'__{prefix}_{col}'
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[key_col] = df_out[col].fillna(-1_000_000_000.0)
            else:
                df_out[key_col] = df_out[col].astype('object').where(df_out[col].notna(), '__NA__')
            key_cols.append(key_col)
        return df_out, key_cols

    # Construct merge-key DataFrames for valid-parameter sets and test rows.
    valid_param_sets_keyed, merge_key_cols = build_merge_keys(valid_param_sets, merge_cols, 'valid')
    test_df_keyed, _ = build_merge_keys(test_df, merge_cols, 'valid')
    # Select test rows by matching merge keys against valid Pareto parameter sets.
    matched_test_df = test_df_keyed.merge(
        valid_param_sets_keyed[merge_key_cols].drop_duplicates(),
        on=merge_key_cols,
        how='inner'
    )
    # Drop temporary merge-key columns after match selection.
    matched_test_df = matched_test_df.drop(columns=merge_key_cols)

    # Fall back per-dataset to all test rows where no parameter match exists.
    matched_datasets = matched_test_df['dataset'].dropna().unique().tolist() \
        if 'dataset' in matched_test_df.columns else []
    fallback_test_df = test_df[~test_df['dataset'].isin(matched_datasets)].copy()
    selected_df = pd.concat([matched_test_df, fallback_test_df], ignore_index=True)

    # Print a compact per-dataset summary using DataFrame joins.
    summary_df = (
        test_df.groupby('dataset').size().rename('all_test_rows').to_frame()
        .join(valid_pareto_df.groupby('dataset').size().rename('valid_pareto_points'), how='left')
        .join(valid_param_sets.groupby('dataset').size().rename('valid_param_sets'), how='left')
        .join(matched_test_df.groupby('dataset').size().rename('matched_test_rows'), how='left')
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    print(summary_df.to_string(index=False))

    # Warn once when fallback rows were required for unmatched datasets.
    if not fallback_test_df.empty:
        fallback_datasets = sorted(fallback_test_df['dataset'].dropna().astype(str).unique().tolist())
        print(f"  Warning: No valid-parameter match for datasets {fallback_datasets}; used all test rows")

    return selected_df


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

        # Compute Pareto front for Polytris (minimize time, maximize accuracy)
        polytris_pareto = compute_pareto_front(polytris_data, time_col, accuracy_col)

        if polytris_pareto.empty:
            continue

        # Define accuracy query points based on data range
        min_acc = 0.0
        max_acc = min(1.0, polytris_pareto[accuracy_col].max())
        accuracy_levels = np.arange(min_acc, max_acc + increment, increment)

        # For speedup comparison, we need to interpolate time at given accuracy levels
        # This requires inverting the Pareto front: given accuracy, find time
        # Sort Pareto by accuracy for interpolation
        polytris_by_acc = polytris_pareto.sort_values(accuracy_col)

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

            # Compute Pareto front for SOTA
            sota_pareto = compute_pareto_front(sota_data, time_col, accuracy_col)

            if sota_pareto.empty:
                continue

            # Sort by accuracy for interpolation
            sota_by_acc = sota_pareto.sort_values(accuracy_col)

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

        # Compute Pareto front for Polytris
        polytris_pareto = compute_pareto_front(polytris_data, time_col, accuracy_col)

        if polytris_pareto.empty:
            continue

        # Sort Pareto by time for interpolation
        polytris_by_time = polytris_pareto.sort_values(time_col)

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

            # Compute Pareto front for SOTA
            sota_pareto = compute_pareto_front(sota_data, time_col, accuracy_col)

            if sota_pareto.empty:
                continue

            # Sort by time for interpolation
            sota_by_time = sota_pareto.sort_values(time_col)

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
    # Line chart showing Pareto fronts (no tooltip - lines are not easily hoverable)
    line = base_pareto.mark_line(strokeWidth=2).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q', title=f'{accuracy_col_name} Score',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('system:N', title='System', scale=color_scale),
        # Group lines by the columns that define a single Pareto front.
        # Polytris fronts are computed per (dataset, classifier, canvas_scale);
        # SOTA fronts are computed per (dataset).  The chart is already faceted
        # by dataset, so only system/classifier/canvas_scale are needed here.
        # Including varying parameters (sample_rate, tilepadding, etc.) would
        # split each front into isolated single-point "lines".
        detail=['system:N', 'classifier:N', 'canvas_scale:N']
    )

    # Add points for Pareto fronts (with tooltip for interactivity)
    points_pareto = base_pareto.mark_point(size=50, filled=True).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q'),
        color=alt.Color('system:N', title='System', scale=color_scale),
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

    # Separate valid and test rows for the valid-to-test Pareto transfer workflow.
    valid_all_df = polytris_split_df[polytris_split_df['videoset'] == 'valid'].copy()
    test_all_df = polytris_split_df[polytris_split_df['videoset'] == 'test'].copy()
    print(f"  Valid rows: {len(valid_all_df)}")
    print(f"  Test rows: {len(test_all_df)}")
    if valid_all_df.empty:
        print("  Warning: No valid rows found; test-only fallback will be used")
    if test_all_df.empty:
        print("  Warning: No test rows found after videoset separation")

    # Extract test naive baseline rows before config filtering.
    naive_df = naive_tradeoff_df[naive_tradeoff_df['videoset'] == 'test'].copy()
    naive_df['system'] = 'Naive'
    print(f"\nExtracted {len(naive_df)} naive baseline rows from test split")

    # Filter valid/test Polytris rows by configured parameter dimensions.
    print("\nFiltering Polytris data by configuration settings...")
    valid_non_naive_df = valid_all_df.copy()
    test_non_naive_df = test_all_df.copy()

    print(f"  Valid rows before filtering: {len(valid_non_naive_df)}")
    filtered_valid_non_naive_df = filter_by_config(
        valid_non_naive_df,
        classifiers=CLASSIFIERS,
        tilepadding_modes=TILEPADDING_MODES,
        sample_rates=SAMPLE_RATES,
        tracking_accuracy_thresholds=TRACKING_ACCURACY_THRESHOLDS,
        trackers=TRACKERS
    )
    print(f"  Valid rows after filtering: {len(filtered_valid_non_naive_df)}")

    print(f"  Test rows before filtering: {len(test_non_naive_df)}")
    filtered_test_non_naive_df = filter_by_config(
        test_non_naive_df,
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

        # Select test rows based on parameter sets from the valid Pareto front.
        print(f"\n1. Computing valid Pareto and selecting matched test rows for {accuracy_name}...")
        polytris_df = select_test_points_from_valid_pareto(
            filtered_valid_non_naive_df,
            filtered_test_non_naive_df,
            accuracy_col,
            'time'
        )
        if polytris_df.empty:
            print(f"  Warning: No Polytris test rows selected for {accuracy_name}, skipping")
            continue
        print(f"  Selected Polytris test rows: {len(polytris_df)}")

        # 2. Compute Pareto fronts for visualization
        print(f"\n2. Computing Pareto fronts for {accuracy_name}...")

        # Columns to keep for tooltip display
        tooltip_cols = ['system', 'dataset', 'videoset', 'classifier', 'sample_rate',
                        'tracking_accuracy_threshold', 'tilepadding', 'canvas_scale',
                        'tracker', 'time', accuracy_col]

        # Collect Pareto rows from Polytris, Naive, and SOTA systems.
        pareto_data_list = []

        # Compute Polytris Pareto fronts per dataset/classifier/canvas_scale using groupby.
        polytris_group_cols = [col for col in ['dataset', 'classifier', 'canvas_scale'] if col in polytris_df.columns]
        polytris_metric_df = polytris_df.dropna(subset=['time', accuracy_col]).copy()
        if polytris_group_cols and not polytris_metric_df.empty:
            polytris_pareto_df = compute_pareto_fronts_by_group(
                polytris_metric_df,
                polytris_group_cols,
                'time',
                accuracy_col
            )
            if not polytris_pareto_df.empty:
                polytris_pareto_df['system'] = 'Polytris'
                polytris_cols = [c for c in tooltip_cols if c in polytris_pareto_df.columns]
                pareto_data_list.append(polytris_pareto_df[polytris_cols])

        # Append Naive test points directly (Naive is a baseline point, not a front).
        if accuracy_col in naive_df.columns:
            naive_point_df = naive_df.dropna(subset=['time', accuracy_col]).copy()
            if not naive_point_df.empty:
                naive_cols = [c for c in tooltip_cols if c in naive_point_df.columns]
                pareto_data_list.append(naive_point_df[naive_cols])

        # Compute SOTA Pareto fronts per dataset using groupby.
        sota_pareto_list = []
        for system_name, df_sota in df_sota_dict.items():
            if accuracy_col not in df_sota.columns:
                continue
            sota_metric_df = df_sota.dropna(subset=['time', accuracy_col]).copy()
            if sota_metric_df.empty:
                continue
            sota_pareto_df = compute_pareto_fronts_by_group(
                sota_metric_df,
                ['dataset'],
                'time',
                accuracy_col
            )
            if sota_pareto_df.empty:
                continue
            sota_pareto_df['system'] = system_name.upper()
            sota_cols = [c for c in tooltip_cols if c in sota_pareto_df.columns]
            sota_pareto_list.append(sota_pareto_df[sota_cols])

        if sota_pareto_list:
            pareto_data_list.extend(sota_pareto_list)

        if not pareto_data_list:
            print(f"  No Pareto data available for {accuracy_name}")
            continue

        df_pareto_combined = pd.concat(pareto_data_list, ignore_index=True)
        print(f"  Combined Pareto data: {len(df_pareto_combined)} points")

        # 3. Create Pareto front comparison chart
        print(f"\n3. Creating Pareto comparison chart for {accuracy_name}...")
        pareto_chart = create_pareto_comparison_chart(
            df_pareto_combined, accuracy_col, accuracy_name, log_scale=log_scale
        )
        save_chart(pareto_chart, output_dir,
                   f'{accuracy_col.lower()}_runtime_pareto_comparison')

        # 4. Compute and visualize speedup at accuracy levels
        print(f"\n4. Computing speedup at accuracy levels for {accuracy_name}...")
        df_speedup = compute_speedup_at_accuracy_levels(
            polytris_df, df_sota_dict, accuracy_col, 'time', increment=0.005
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
            polytris_df, df_sota_dict, accuracy_col, 'time', increment=5.0
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
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    print(f"Processing datasets: {DATASETS}")
    visualize_all_datasets_tradeoffs_pareto(DATASETS, log_scale=args.log)


if __name__ == '__main__':
    main()
