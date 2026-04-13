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
from collections.abc import Callable
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

# Keep facet layout dimensions explicit so all comparison charts stay aligned.
FACET_COLUMNS = 4
FACET_SUBPLOT_WIDTH = 225
FACET_SUBPLOT_HEIGHT = 170
ONE_ROW_FACET_SUBPLOT_WIDTH = FACET_SUBPLOT_WIDTH // 2

# Approximate spacing used by Altair between facet cells and for header labels.
_FACET_COL_SPACING = 60
_FACET_ROW_SPACING = 40
_FACET_HEADER_HEIGHT = 40

# Legend placement: position in the empty bottom-right cell of a 4-column grid
# (assumes the last row has fewer subplots than FACET_COLUMNS).
LEGEND_X = (FACET_COLUMNS - 1) * (FACET_SUBPLOT_WIDTH + _FACET_COL_SPACING)
LEGEND_Y = FACET_SUBPLOT_HEIGHT + _FACET_HEADER_HEIGHT + _FACET_ROW_SPACING

# Define fixed system-to-color categories for deterministic chart encoding.
SYSTEM_COLOR_DOMAIN = ['Polytris', 'Naive', 'OTIF', 'LEAP']

# Define the color for each system category.
SYSTEM_COLOR_RANGE = ColorScheme.CarbonDark[:len(SYSTEM_COLOR_DOMAIN)]
SYSTEM_COLOR_LOOKUP = dict(zip(SYSTEM_COLOR_DOMAIN, SYSTEM_COLOR_RANGE))

# Map internal dataset identifiers to the display names used in charts.
DATASET_NAME_MAP = {
    'caldot1-y05': 'CalDoT 1',
    'caldot2-y05': 'CalDoT 2',
    'ams-y05': 'Amsterdam',
    'jnc0': 'B3D 1',
    'jnc2': 'B3D 2',
    'jnc6': 'B3D 3',
    'jnc7': 'B3D 4',
}


def _add_dataset_display_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable dataset label column for chart facets and tooltips.

    Args:
        df: DataFrame that may include a dataset column

    Returns:
        Copy of the input DataFrame with dataset_display when dataset exists
    """
    labeled_df = df.copy()

    # Only attach labels when the source table actually has dataset ids.
    if 'dataset' not in labeled_df.columns:
        return labeled_df

    # Fall back to the raw dataset id when no friendly label is configured.
    labeled_df['dataset_display'] = (
        labeled_df['dataset']
        .map(DATASET_NAME_MAP)
        .fillna(labeled_df['dataset'])
    )
    return labeled_df


def _get_dataset_display_sort(df: pd.DataFrame) -> list[str] | None:
    """
    Preserve configured dataset ordering after mapping to display names.

    Args:
        df: DataFrame with dataset ids and display names

    Returns:
        Ordered list of dataset display labels, or None if unavailable
    """
    if 'dataset' not in df.columns or 'dataset_display' not in df.columns:
        return None

    # Keep configured datasets first so facet order remains stable across charts.
    present_datasets = set(df['dataset'].dropna().unique())
    ordered_dataset_ids = [dataset for dataset in DATASETS if dataset in present_datasets]

    # Append unexpected datasets in appearance order rather than dropping them.
    extra_dataset_ids = [
        dataset
        for dataset in df['dataset'].dropna().unique()
        if dataset not in ordered_dataset_ids
    ]

    return [DATASET_NAME_MAP.get(dataset, dataset) for dataset in ordered_dataset_ids + extra_dataset_ids]


def _get_system_color_scale(systems: list[str]) -> alt.Scale:
    """
    Build a deterministic system color scale without showing absent systems.

    Args:
        systems: System labels present in the chart data

    Returns:
        Altair scale that preserves the canonical system-to-color mapping
    """
    ordered_systems = []
    seen_systems = set()

    # Keep the canonical order for known systems so colors stay stable.
    for system in SYSTEM_COLOR_DOMAIN:
        if system in systems and system not in seen_systems:
            ordered_systems.append(system)
            seen_systems.add(system)

    # Append any unexpected systems after the known set.
    for system in systems:
        if pd.isna(system) or system in seen_systems:
            continue
        ordered_systems.append(system)
        seen_systems.add(system)

    # Fall back to unused palette entries if an unexpected system appears.
    color_range = []
    used_colors = set()
    palette_index = 0
    for system in ordered_systems:
        if system in SYSTEM_COLOR_LOOKUP:
            color = SYSTEM_COLOR_LOOKUP[system]
        else:
            while ColorScheme.CarbonDark[palette_index % len(ColorScheme.CarbonDark)] in used_colors:
                palette_index += 1
            color = ColorScheme.CarbonDark[palette_index % len(ColorScheme.CarbonDark)]
            palette_index += 1
        color_range.append(color)
        used_colors.add(color)

    return alt.Scale(domain=ordered_systems, range=color_range)


def _facet_chart(chart: alt.Chart, df: pd.DataFrame, title: str, *,
                 single_row: bool = False,
                 apply_padding: bool = True,
                 apply_legend_config: bool = True) -> alt.Chart:
    """
    Apply the shared dataset facet layout used by all comparison charts.

    Args:
        chart: Layered chart to facet
        df: Data backing the chart, including dataset_display
        title: Figure title
        single_row: Whether to emit the compact one-row layout

    Returns:
        Faceted chart with consistent ordering and legend placement
    """
    # Compact exports halve each subplot width and force all facets into one row.
    subplot_width = ONE_ROW_FACET_SUBPLOT_WIDTH if single_row else FACET_SUBPLOT_WIDTH
    facet_columns = max(1, df['dataset_display'].nunique()) if single_row else FACET_COLUMNS

    # Share the y-axis only in the compact one-row export so the left-most axis
    # carries the labels and title once for the whole row.
    y_scale_resolution = 'shared' if single_row else 'independent'

    faceted_chart = chart.properties(
        width=subplot_width,
        height=FACET_SUBPLOT_HEIGHT,
    ).facet(
        facet=alt.Facet(
            'dataset_display:N',
            title=None,
            sort=_get_dataset_display_sort(df),
            header=alt.Header(labelExpr="'Dataset: ' + datum.value")
        ),
        columns=facet_columns,
        spacing=0,
    ).resolve_scale(
        x='independent',
        y=y_scale_resolution
    ).properties(
        title=title,
    )

    # Keep zero outer padding for standalone charts, but omit it when this
    # faceted chart will be nested inside a larger concat composition.
    if apply_padding:
        faceted_chart = faceted_chart.properties(padding=0)

    # The default layout keeps the legend in the empty grid cell; the one-row
    # export has no spare cell, so move the legend above the chart.
    if not apply_legend_config:
        return faceted_chart

    if single_row:
        return faceted_chart.configure_legend(orient='top')

    return faceted_chart.configure_legend(
        orient='none',
        legendX=LEGEND_X,
        legendY=LEGEND_Y,
    )



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
                'system': system_name.upper(),
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
                'system': system_name.upper(),
                'comparison_system': system_name.upper(),
                'accuracy_gain': accuracy_gains,
                'polytris_accuracy': polytris_accuracies,
                'other_accuracy': sota_accuracies
            })

            results.append(comparison_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def create_speedup_chart(df_speedup: pd.DataFrame, accuracy_col_name: str, *,
                         single_row: bool = False,
                         legend_title: str = 'Compared To',
                         show_legend: bool = True,
                         apply_padding: bool = True,
                         apply_legend_config: bool = True) -> alt.Chart:
    """
    Create faceted line chart showing speedup ratio vs accuracy level.

    Args:
        df_speedup: DataFrame with speedup data
        accuracy_col_name: Display name for the accuracy metric

    Returns:
        Altair Chart object
    """
    # Drop NaN values for visualization and attach display labels once.
    df_clean = _add_dataset_display_names(df_speedup.dropna(subset=['speedup_ratio']))

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Use the canonical system colors so combined charts stay visually aligned.
    color_scale = _get_system_color_scale(df_clean['system'].dropna().unique().tolist())
    legend = alt.Legend(title=legend_title) if show_legend else None

    # Base chart
    base = alt.Chart(df_clean)

    # Line chart showing speedup ratio (no tooltip - lines are not easily hoverable)
    line = base.mark_line(strokeWidth=2).encode(
        x=alt.X('accuracy_level:Q', title=f'{accuracy_col_name} Level'),
        y=alt.Y('speedup_ratio:Q', title='Speedup Ratio (Other/Polytris)'),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
    )

    # Add points for better visibility (with tooltip for interactivity)
    points = base.mark_point(size=30, filled=True).encode(
        x=alt.X('accuracy_level:Q'),
        y=alt.Y('speedup_ratio:Q'),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        tooltip=[
            alt.Tooltip('dataset_display:N', title='Dataset'),
            'accuracy_level',
            'system',
            'speedup_ratio',
            'polytris_time',
            'other_time',
        ]
    )

    # Horizontal rule at y=1 (parity line)
    rule = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1
    ).encode(y='y:Q')

    # Apply the shared facet configuration so both export variants stay aligned.
    chart = line + points + rule
    return _facet_chart(
        chart,
        df_clean,
        f'Speedup Ratio at {accuracy_col_name} Levels (>1 = Polytris faster)',
        single_row=single_row,
        apply_padding=apply_padding,
        apply_legend_config=apply_legend_config,
    )


def create_accuracy_gain_chart(df_accuracy_gain: pd.DataFrame, accuracy_col_name: str,
                               log_scale: bool = False,
                               *,
                               single_row: bool = False,
                               legend_title: str = 'Compared To',
                               show_legend: bool = True,
                               apply_padding: bool = True,
                               apply_legend_config: bool = True) -> alt.Chart:
    """
    Create faceted line chart showing accuracy gain vs runtime level.

    Args:
        df_accuracy_gain: DataFrame with accuracy gain data
        accuracy_col_name: Display name for the accuracy metric

    Returns:
        Altair Chart object
    """
    # Drop NaN values for visualization and attach display labels once.
    df_clean = _add_dataset_display_names(df_accuracy_gain.dropna(subset=['accuracy_gain']))

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Use the canonical system colors so combined charts stay visually aligned.
    color_scale = _get_system_color_scale(df_clean['system'].dropna().unique().tolist())
    legend = alt.Legend(title=legend_title) if show_legend else None

    # Base chart
    base = alt.Chart(df_clean)

    # X-axis uses log scale for runtime (seconds) if enabled
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X('runtime_level:Q', title='Runtime (seconds)', scale=x_scale)
    # Line chart showing accuracy gain (no tooltip - lines are not easily hoverable)
    line = base.mark_line(strokeWidth=2).encode(
        x=x_enc,
        y=alt.Y('accuracy_gain:Q', title=f'{accuracy_col_name} Gain (Polytris - Other)'),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
    )

    # Add points for better visibility (with tooltip for interactivity)
    points = base.mark_point(size=30, filled=True).encode(
        x=x_enc,
        y=alt.Y('accuracy_gain:Q'),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        tooltip=[
            alt.Tooltip('dataset_display:N', title='Dataset'),
            'runtime_level',
            'system',
            'accuracy_gain',
            'polytris_accuracy',
            'other_accuracy',
        ]
    )

    # Horizontal rule at y=0 (parity line)
    rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1
    ).encode(y='y:Q')

    # Apply the shared facet configuration so both export variants stay aligned.
    chart = line + points + rule
    return _facet_chart(
        chart,
        df_clean,
        f'{accuracy_col_name} Gain at Runtime Levels (>0 = Polytris more accurate)',
        single_row=single_row,
        apply_padding=apply_padding,
        apply_legend_config=apply_legend_config,
    )


def create_pareto_comparison_chart(df_combined: pd.DataFrame, accuracy_col: str,
                                   accuracy_col_name: str, time_col: str = 'time',
                                   log_scale: bool = False,
                                   x_title: str = 'Runtime (seconds)',
                                   *,
                                   single_row: bool = False,
                                   legend_title: str = 'System',
                                   show_legend: bool = True,
                                   apply_padding: bool = True,
                                   apply_legend_config: bool = True) -> alt.Chart:
    """
    Create faceted line chart showing Pareto fronts for all systems.

    Args:
        df_combined: DataFrame with Pareto front data from all systems
        accuracy_col: Column name for accuracy metric
        accuracy_col_name: Display name for the accuracy metric
        time_col: Column name for runtime
        x_title: Display label for the x-axis

    Returns:
        Altair Chart object
    """
    # Drop rows with NaN in key columns and attach display labels once.
    df_clean = _add_dataset_display_names(df_combined.dropna(subset=[time_col, accuracy_col]))

    if df_clean.empty:
        return alt.Chart().mark_text().encode(text=alt.value('No data available'))

    # Define a consistent color scale for the systems present in this chart.
    color_scale = _get_system_color_scale(df_clean['system'].dropna().unique().tolist())
    legend = alt.Legend(title=legend_title) if show_legend else None

    # Base chart for Pareto fronts (with lines)
    base_pareto = alt.Chart(df_clean)

    # X-axis uses log scale if enabled.
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X(f'{time_col}:Q', title=x_title, scale=x_scale)
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
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        opacity=alt.Opacity('system:N', scale=opacity_scale, legend=legend),
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
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        opacity=alt.Opacity('system:N', scale=opacity_scale, legend=legend),
        shape=alt.Shape('system:N', scale=shape_scale, legend=legend),
        tooltip=[
            'system',
            alt.Tooltip('dataset_display:N', title='Dataset'),
            'classifier',
            'sample_rate',
            'tracking_accuracy_threshold',
            'tilepadding',
            'canvas_scale',
            'tracker',
            time_col,
            accuracy_col,
        ]
    )

    # Apply the shared facet configuration so both export variants stay aligned.
    chart = line + points_pareto
    return _facet_chart(
        chart,
        df_clean,
        f'{accuracy_col_name} vs Runtime Pareto Fronts',
        single_row=single_row,
        apply_padding=apply_padding,
        apply_legend_config=apply_legend_config,
    )


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
    """Save chart in multiple formats (PNG, PDF, HTML)."""
    # Save PNG format
    png_path = os.path.join(output_dir, f'{base_name}.png')
    chart.save(png_path, scale_factor=4)
    print(f"  Saved PNG: {png_path}")

    # Save PDF format
    pdf_path = os.path.join(output_dir, f'{base_name}.pdf')
    chart.save(pdf_path)
    print(f"  Saved PDF: {pdf_path}")

    # Save HTML format (interactive)
    html_path = os.path.join(output_dir, f'{base_name}.html')
    chart.save(html_path)
    print(f"  Saved HTML: {html_path}")


def save_chart_variants(chart_factory: Callable[..., alt.Chart], output_dir: str,
                        base_name: str, *args, **kwargs):
    """
    Save the default chart plus a compact one-row variant.

    Args:
        chart_factory: Chart builder that accepts single_row=
        output_dir: Destination directory
        base_name: Base filename without extension
        *args: Positional args forwarded to the chart factory
        **kwargs: Keyword args forwarded to the chart factory
    """
    # Preserve the existing multi-row export names for backward compatibility.
    save_chart(chart_factory(*args, **kwargs), output_dir, base_name)

    # Emit the additional compact export with half-width facets in one row.
    save_chart(
        chart_factory(*args, single_row=True, **kwargs),
        output_dir,
        f'{base_name}_one_row',
    )


def create_hota_summary_one_row_chart(df_throughput: pd.DataFrame,
                                      df_speedup: pd.DataFrame,
                                      df_accuracy_gain: pd.DataFrame,
                                      *,
                                      log_scale: bool = False) -> alt.Chart:
    """
    Create the combined one-row HOTA summary by stacking three compact charts.

    Args:
        df_throughput: Throughput Pareto data
        df_speedup: Speedup-at-accuracy data
        df_accuracy_gain: Accuracy-gain-at-runtime data
        log_scale: Whether runtime-based charts should use a log x-axis

    Returns:
        Vertically concatenated Altair chart
    """
    throughput_chart = create_pareto_comparison_chart(
        df_throughput,
        'HOTA_HOTA',
        'HOTA',
        time_col='throughput_fps',
        log_scale=True,
        x_title='Throughput (frames/sec)',
        single_row=True,
        legend_title='System',
        show_legend=True,
        apply_padding=False,
        apply_legend_config=False,
    )
    speedup_chart = create_speedup_chart(
        df_speedup,
        'HOTA',
        single_row=True,
        legend_title='System',
        show_legend=False,
        apply_padding=False,
        apply_legend_config=False,
    )
    accuracy_gain_chart = create_accuracy_gain_chart(
        df_accuracy_gain,
        'HOTA',
        log_scale=log_scale,
        single_row=True,
        legend_title='System',
        show_legend=False,
        apply_padding=False,
        apply_legend_config=False,
    )

    return alt.vconcat(
        throughput_chart,
        speedup_chart,
        accuracy_gain_chart,
        spacing=0,
    ).resolve_scale(
        color='shared'
    ).properties(
        padding=0,
    ).configure_legend(
        orient='top',
    )


def _filter_pareto_per_dataset(df: pd.DataFrame, time_col: str,
                               accuracy_col: str, *,
                               minx: bool, miny: bool) -> pd.DataFrame:
    """
    Filter DataFrame to only Pareto-optimal points per dataset.

    Keeps only points on the Pareto front computed independently for each
    dataset.

    Args:
        df: DataFrame with 'dataset' column and the specified metric columns
        time_col: Column name for x-axis metric (e.g. 'time' or 'throughput_fps')
        accuracy_col: Column name for y-axis metric (e.g. 'HOTA_HOTA')
        minx: If True, lower x is better; if False, higher x is better
        miny: If True, lower y is better; if False, higher y is better

    Returns:
        DataFrame containing only Pareto-optimal points across all datasets
    """
    pareto_groups = []
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        # Compute Pareto front per dataset.
        pareto_df = compute_pareto_front(dataset_df, time_col, accuracy_col,
                                         minx=minx, miny=miny)
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

    # Derive throughput_fps for SOTA systems from Polytris frame counts.
    # Frame count is a property of the dataset/split, not the system.
    if 'frame_count' in tradeoff_df.columns:
        frame_count_lookup = (
            tradeoff_df
            .dropna(subset=['frame_count'])
            .groupby(['dataset', 'videoset'])['frame_count']
            .first()
        )
        for system_name, df_sota in df_sota_dict.items():
            # Merge frame_count from the lookup into SOTA rows.
            merged_fc = df_sota.set_index(['dataset', 'videoset']).index.map(
                lambda idx: frame_count_lookup.get(idx, float('nan'))
            )
            df_sota = df_sota.copy()
            df_sota['frame_count'] = merged_fc.values
            df_sota['throughput_fps'] = df_sota['frame_count'] / df_sota['time']
            df_sota_dict[system_name] = df_sota
            n_valid = df_sota['throughput_fps'].notna().sum()
            print(f"  Computed throughput_fps for {system_name.upper()}: {n_valid}/{len(df_sota)} rows")
    else:
        print("  Warning: frame_count not available in tradeoff data; SOTA throughput_fps will be NaN")

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
        polytris_df = _filter_pareto_per_dataset(polytris_df, 'time', accuracy_col,
                                                 minx=True, miny=False)
        print(f"  Polytris Pareto-optimal points: {len(polytris_df)}")

        # Filter Naive baseline to Pareto-optimal points per dataset.
        if accuracy_col in naive_df.columns:
            pareto_naive_df = _filter_pareto_per_dataset(
                naive_df.dropna(subset=['time', accuracy_col]), 'time', accuracy_col,
                minx=True, miny=False,
            )
        else:
            pareto_naive_df = pd.DataFrame()

        # Filter each SOTA system to Pareto-optimal points per dataset.
        pareto_sota_dict: dict[str, pd.DataFrame] = {}
        for system_name, df_sota in df_sota_dict.items():
            if accuracy_col not in df_sota.columns:
                continue
            filtered_sota = _filter_pareto_per_dataset(
                df_sota.dropna(subset=['time', accuracy_col]), 'time', accuracy_col,
                minx=True, miny=False,
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
        save_chart_variants(
            create_pareto_comparison_chart,
            output_dir,
            f'{accuracy_col.lower()}_runtime_pareto_comparison',
            df_pareto_combined,
            accuracy_col,
            accuracy_name,
            log_scale=log_scale,
        )

        # 3b. Create throughput Pareto comparison chart (accuracy vs frames/sec).
        print(f"\n3b. Creating throughput comparison chart for {accuracy_name}...")

        # Re-filter Pareto fronts for throughput (maximize both axes).
        throughput_polytris_df = _filter_pareto_per_dataset(
            polytris_df.dropna(subset=['throughput_fps', accuracy_col]),
            'throughput_fps', accuracy_col, minx=False, miny=False,
        ) if 'throughput_fps' in polytris_df.columns else pd.DataFrame()

        throughput_naive_df = _filter_pareto_per_dataset(
            naive_df.dropna(subset=['throughput_fps', accuracy_col]),
            'throughput_fps', accuracy_col, minx=False, miny=False,
        ) if ('throughput_fps' in naive_df.columns
              and not naive_df['throughput_fps'].isna().all()) else pd.DataFrame()

        throughput_sota_dict: dict[str, pd.DataFrame] = {}
        for system_name, df_sota in df_sota_dict.items():
            if accuracy_col not in df_sota.columns or 'throughput_fps' not in df_sota.columns:
                continue
            filtered = _filter_pareto_per_dataset(
                df_sota.dropna(subset=['throughput_fps', accuracy_col]),
                'throughput_fps', accuracy_col, minx=False, miny=False,
            )
            if not filtered.empty:
                throughput_sota_dict[system_name] = filtered

        # Collect throughput Pareto-optimal rows from all systems.
        throughput_tooltip_cols = ['system', 'dataset', 'videoset', 'classifier',
                                  'sample_rate', 'tracking_accuracy_threshold',
                                  'tilepadding', 'canvas_scale', 'tracker',
                                  'throughput_fps', 'time', accuracy_col]
        tp_data_list: list[pd.DataFrame] = []

        if not throughput_polytris_df.empty:
            tp_polytris = throughput_polytris_df.copy()
            tp_polytris['system'] = 'Polytris'
            tp_cols = [c for c in throughput_tooltip_cols if c in tp_polytris.columns]
            tp_data_list.append(tp_polytris[tp_cols])

        if not throughput_naive_df.empty:
            tp_naive = throughput_naive_df.copy()
            tp_naive['system'] = 'Naive'
            tp_cols = [c for c in throughput_tooltip_cols if c in tp_naive.columns]
            tp_data_list.append(tp_naive[tp_cols])

        for system_name, df_tp_sota in throughput_sota_dict.items():
            tp_sota = df_tp_sota.copy()
            tp_sota['system'] = system_name.upper()
            tp_cols = [c for c in throughput_tooltip_cols if c in tp_sota.columns]
            tp_data_list.append(tp_sota[tp_cols])

        df_tp_combined = pd.DataFrame()
        if tp_data_list:
            df_tp_combined = pd.concat(tp_data_list, ignore_index=True)
            print(f"  Combined throughput Pareto-optimal data: {len(df_tp_combined)} points")
            save_chart_variants(
                create_pareto_comparison_chart,
                output_dir,
                f'{accuracy_col.lower()}_throughput_pareto_comparison',
                df_tp_combined,
                accuracy_col,
                accuracy_name,
                time_col='throughput_fps',
                log_scale=True,
                x_title='Throughput (frames/sec)',
            )
        else:
            print("  No throughput data available")

        # 4. Compute and visualize speedup at accuracy levels
        print(f"\n4. Computing speedup at accuracy levels for {accuracy_name}...")
        df_speedup = compute_speedup_at_accuracy_levels(
            polytris_df, pareto_sota_dict, accuracy_col, 'time', increment=0.005
        )

        if not df_speedup.empty:
            print(f"  Speedup data: {len(df_speedup)} comparison points")
            save_chart_variants(
                create_speedup_chart,
                output_dir,
                f'{accuracy_col.lower()}_speedup_at_accuracy',
                df_speedup,
                accuracy_name,
            )
        else:
            print("  No speedup data available")

        # 5. Compute and visualize accuracy gain at runtime levels
        print(f"\n5. Computing accuracy gain at runtime levels for {accuracy_name}...")
        df_accuracy_gain = compute_accuracy_gain_at_runtime_levels(
            polytris_df, pareto_sota_dict, accuracy_col, 'time', increment=5.0
        )

        if not df_accuracy_gain.empty:
            print(f"  Accuracy gain data: {len(df_accuracy_gain)} comparison points")
            save_chart_variants(
                create_accuracy_gain_chart,
                output_dir,
                f'{accuracy_col.lower()}_accuracy_gain_at_runtime',
                df_accuracy_gain,
                accuracy_name,
                log_scale=log_scale,
            )
        else:
            print("  No accuracy gain data available")

        if (accuracy_col == 'HOTA_HOTA'
                and not df_tp_combined.empty
                and not df_speedup.empty
                and not df_accuracy_gain.empty):
            print(f"\n6. Creating combined one-row summary chart for {accuracy_name}...")
            save_chart(
                create_hota_summary_one_row_chart(
                    df_tp_combined,
                    df_speedup,
                    df_accuracy_gain,
                    log_scale=log_scale,
                ),
                output_dir,
                f'{accuracy_col.lower()}_summary_one_row',
            )
        elif accuracy_col == 'HOTA_HOTA':
            print("  Skipping combined one-row summary chart due to missing source data")

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
