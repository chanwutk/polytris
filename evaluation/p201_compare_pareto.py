#!/usr/local/bin/python

"""
Compare system performance using Pareto fronts and relative comparisons.

This script extends p200_compare_compute.py by:
1. Computing and displaying Pareto front lines instead of all data points for Polytris
2. Adding visualizations showing speedup ratios at each comparison-system
   Pareto point (x = that system's accuracy; y = Polytris speedup under a
   higher-accuracy pairing rule)
3. Adding visualizations showing accuracy gains at each comparison-system
   Pareto point (x = anchor throughput in FPS when frame counts exist; y = Polytris HOTA gain
   under a higher-throughput pairing rule)
"""

import argparse
import os
import shutil
from collections.abc import Callable
import numpy as np
import pandas as pd
import altair as alt

from evaluation.ablation import ABLATION_CONDITIONS, filter_by_ablation_condition
from polyis.io import cache
from polyis.pareto import compute_pareto_front
from polyis.utilities import get_config, load_tradeoff_data, split_tradeoff_variants
from evaluation.utilities import SYSTEM_COLOR_DOMAIN
from evaluation.p200_compare_compute import load_sota_tradeoff_data


config = get_config()
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_FIGURES_GENERATED_DIR = os.path.join(REPO_ROOT, 'paper', 'figures', 'generated')
DATASETS = config['EXEC']['DATASETS']
CLASSIFIERS = config['EXEC']['CLASSIFIERS']
TILEPADDING_MODES = config['EXEC']['TILEPADDING_MODES']
SAMPLE_RATES = config['EXEC']['SAMPLE_RATES']
TRACKERS = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS = config['EXEC']['TRACKING_ACCURACY_THRESHOLDS']
RELEVANCE_THRESHOLDS = config['EXEC']['RELEVANCE_THRESHOLDS']

# Keep facet layout dimensions explicit so all comparison charts stay aligned.
FACET_COLUMNS = 4
FACET_SUBPLOT_WIDTH = 250
FACET_SUBPLOT_HEIGHT = 175
ONE_ROW_FACET_SUBPLOT_WIDTH = int(round(FACET_SUBPLOT_WIDTH * 0.6))
COMBINED_ONE_ROW_SUBPLOT_HEIGHT = max(1, int(round(FACET_SUBPLOT_HEIGHT * 0.7)))

# Approximate spacing used by Altair between facet cells and for header labels.
_FACET_COL_SPACING = 60
_FACET_ROW_SPACING = 40
_FACET_HEADER_HEIGHT = 40

# Legend placement: position in the empty bottom-right cell of a 4-column grid
# (assumes the last row has fewer subplots than FACET_COLUMNS).
LEGEND_X = (FACET_COLUMNS - 1) * (FACET_SUBPLOT_WIDTH + _FACET_COL_SPACING)
LEGEND_Y = FACET_SUBPLOT_HEIGHT + _FACET_HEADER_HEIGHT + _FACET_ROW_SPACING

SYSTEM_MARK_OPACITY = 0.6
STANDARD_POINT_SIZE = 30
PARETO_POINT_SIZE = 35

# Vega expression that splits legend labels on spaces so each token renders on
# its own line -- keeps long ablation labels readable in narrow legends.
LEGEND_LABEL_BREAK_ON_SPACE = "split(datum.label, ' ')"

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


def _ordered_systems_for_chart(systems: list[str]) -> list[str]:
    """Canonical order first, then extras; used so shape scale domain matches ``system``."""
    ordered: list[str] = []
    seen: set[str] = set()
    present = {s for s in systems if not pd.isna(s)}
    for system in SYSTEM_COLOR_DOMAIN:
        if system in present and system not in seen:
            ordered.append(system)
            seen.add(system)
    for system in systems:
        if pd.isna(system) or system in seen:
            continue
        ordered.append(system)
        seen.add(system)
    return ordered


def _default_system_color_scale(systems: list[str]) -> alt.Scale:
    """
    Nominal colors in canonical ``SYSTEM_COLOR_DOMAIN`` order (then extras).

    No ``range``: Vega-Lite uses the default categorical palette, assigning
    colors by domain index so the mapping stays stable across charts.
    """
    return alt.Scale(domain=_ordered_systems_for_chart(systems))


def _facet_chart(chart: alt.Chart, df: pd.DataFrame, title: str, *,
                 single_row: bool = False,
                 apply_padding: bool = True,
                 apply_legend_config: bool = True,
                 subplot_height: int | None = None,
                 title_orient: str = 'top',
                 title_text: str | list[str] | None = None,
                 title_angle: float | None = None,
                 title_align: str = 'center',
                 title_anchor: str = 'middle',
                 title_dy: float | None = None,
                 title_dx: float | None = None,
                 title_offset: float | None = None) -> alt.Chart:
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
    subplot_height = FACET_SUBPLOT_HEIGHT if subplot_height is None else subplot_height
    facet_columns = max(1, df['dataset_display'].nunique()) if single_row else FACET_COLUMNS

    # Share the y-axis only in the compact one-row export so the left-most axis
    # carries the labels and title once for the whole row.
    y_scale_resolution = 'shared' if single_row else 'independent'
    title_spec = alt.TitleParams(
        text=title if title_text is None else title_text,
        orient=title_orient,
        anchor=title_anchor,
        align=title_align,
        angle=title_angle if title_angle is not None else alt.Undefined,
        dy=title_dy if title_dy is not None else alt.Undefined,
        dx=title_dx if title_dx is not None else alt.Undefined,
        offset=title_offset if title_offset is not None else alt.Undefined,
    )

    faceted_chart = chart.properties(
        width=subplot_width,
        height=subplot_height,
    ).facet(
        facet=alt.Facet(
            'dataset_display:N',
            title=None,
            sort=_get_dataset_display_sort(df),
            header=alt.Header(
                labelExpr="'Dataset: ' + datum.value",
                labelPadding=-12,
            )
        ),
        columns=facet_columns,
        spacing=0,
    ).resolve_scale(
        x='independent',
        y=y_scale_resolution
    ).properties(
        title=title_spec,
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
                                       time_col: str = 'time') -> pd.DataFrame:
    """
    One row per comparison-system Pareto point (anchor).

    For each anchor (other_time, other accuracy on the x-axis as
    ``accuracy_level``), pick the Polytris point with strictly higher accuracy
    that minimizes Polytris runtime (maximizes speedup_ratio =
    other_time / polytris_time). Tie-break: lowest time, then highest accuracy.

    Args:
        df_polytris: DataFrame with Polytris Pareto front data
        df_sota_dict: Dictionary mapping system names to their Pareto front DataFrames
        accuracy_col: Column name for accuracy metric (e.g., 'HOTA_HOTA')
        time_col: Column name for runtime

    Returns:
        DataFrame with columns: dataset, accuracy_level, system, comparison_system,
        speedup_ratio, polytris_time, other_time
    """
    rows: list[dict] = []

    datasets = df_polytris['dataset'].unique()

    for dataset in datasets:
        polytris_data = df_polytris[df_polytris['dataset'] == dataset]
        poly = polytris_data.dropna(subset=[time_col, accuracy_col])
        if poly.empty:
            continue

        for system_name, df_sota in df_sota_dict.items():
            sota_data = df_sota[df_sota['dataset'] == dataset]
            sota = sota_data.dropna(subset=[time_col, accuracy_col])
            if sota.empty:
                continue

            label = system_name.upper()
            for _, sota_row in sota.iterrows():
                t_other = float(sota_row[time_col])
                acc_other = float(sota_row[accuracy_col])
                feasible = poly[poly[accuracy_col] > acc_other]
                if feasible.empty or t_other <= 0:
                    rows.append({
                        'dataset': dataset,
                        'accuracy_level': acc_other,
                        'system': label,
                        'comparison_system': label,
                        'speedup_ratio': np.nan,
                        'polytris_time': np.nan,
                        'other_time': t_other,
                    })
                    continue

                best = feasible.sort_values(
                    [time_col, accuracy_col], ascending=[True, False],
                ).iloc[0]
                t_poly = float(best[time_col])
                if t_poly <= 0:
                    rows.append({
                        'dataset': dataset,
                        'accuracy_level': acc_other,
                        'system': label,
                        'comparison_system': label,
                        'speedup_ratio': np.nan,
                        'polytris_time': t_poly,
                        'other_time': t_other,
                    })
                    continue

                rows.append({
                    'dataset': dataset,
                    'accuracy_level': acc_other,
                    'system': label,
                    'comparison_system': label,
                    'speedup_ratio': t_other / t_poly,
                    'polytris_time': t_poly,
                    'other_time': t_other,
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _throughput_fps_from_row(row: pd.Series, time_col: str) -> float:
    """Frames per second implied by ``time_col``, or NaN if unavailable."""
    t = float(row[time_col])
    if not np.isfinite(t) or t <= 0:
        return float('nan')
    if 'throughput_fps' in row.index and pd.notna(row['throughput_fps']):
        return float(row['throughput_fps'])
    if 'frame_count' in row.index and pd.notna(row['frame_count']):
        fc = float(row['frame_count'])
        if fc > 0:
            return fc / t
    return float('nan')


def _frame_count_for_dataset(poly_df: pd.DataFrame, naive_row: pd.Series,
                             sota_row: pd.Series) -> float | None:
    """Single workload size (frames) for the dataset, from any available row."""
    if 'frame_count' in poly_df.columns:
        fc_ser = poly_df['frame_count'].dropna()
        if not fc_ser.empty:
            v = float(fc_ser.iloc[0])
            if np.isfinite(v) and v > 0:
                return v
    for row in (naive_row, sota_row):
        if 'frame_count' in row.index and pd.notna(row['frame_count']):
            v = float(row['frame_count'])
            if v > 0:
                return v
    return None


def _anchor_throughput_fps(poly_df: pd.DataFrame, naive_row: pd.Series, sota_row: pd.Series,
                           time_col: str, naive_time: float, t_other: float) -> float:
    """
    Throughput (FPS) of the comparison-system anchor at ``t_other``.

    Uses ``frame_count / t_other`` when a frame count is available for the dataset;
    otherwise derives FPS from explicit ``throughput_fps`` / ``frame_count`` on the
    naive or anchor rows, or falls back to ``naive_time / t_other`` (dimensionless
    speed vs naive) when no frame data exists.
    """
    if not np.isfinite(t_other) or t_other <= 0 or not np.isfinite(naive_time) or naive_time <= 0:
        return float('nan')
    fc = _frame_count_for_dataset(poly_df, naive_row, sota_row)
    if fc is not None:
        return fc / t_other
    other_tp = _throughput_fps_from_row(sota_row, time_col)
    if np.isfinite(other_tp):
        return other_tp
    naive_tp = _throughput_fps_from_row(naive_row, time_col)
    if np.isfinite(naive_tp):
        return naive_tp * (naive_time / t_other)
    return naive_time / t_other


def compute_accuracy_gain_at_naive_speedup_levels(df_polytris: pd.DataFrame,
                                                  df_sota_dict: dict[str, pd.DataFrame],
                                                  df_naive: pd.DataFrame,
                                                  accuracy_col: str,
                                                  time_col: str = 'time') -> pd.DataFrame:
    """
    One row per comparison-system Pareto point (anchor).

    x-axis ``throughput_fps`` is the anchor system's throughput (frames per second)
    when ``frame_count`` (or explicit ``throughput_fps``) is available; otherwise it
    falls back to the naive-time ratio ``naive_time / time_other`` at that anchor.
    Among Polytris points strictly faster than the anchor (lower runtime),
    pick the one that maximizes HOTA gain (Polytris accuracy - anchor accuracy).
    Tie-break: highest Polytris accuracy, then lowest Polytris time.

    Args:
        df_polytris: DataFrame with Polytris Pareto front data
        df_sota_dict: Dictionary mapping system names to their Pareto front DataFrames
        df_naive: DataFrame with one naive baseline row per dataset
        accuracy_col: Column name for accuracy metric (e.g., 'HOTA_HOTA')
        time_col: Column name for runtime

    Returns:
        DataFrame with columns: dataset, throughput_fps, system, comparison_system,
        accuracy_gain, naive_time, polytris_accuracy, other_accuracy
    """
    rows: list[dict] = []

    datasets = df_polytris['dataset'].unique()

    for dataset in datasets:
        polytris_data = df_polytris[df_polytris['dataset'] == dataset]
        poly = polytris_data.dropna(subset=[time_col, accuracy_col])
        if poly.empty:
            continue

        naive_dataset_df = df_naive[df_naive['dataset'] == dataset].dropna(subset=[time_col])
        if naive_dataset_df.empty:
            continue

        naive_time = float(naive_dataset_df[time_col].iloc[0])
        if naive_time <= 0:
            continue

        naive_row = naive_dataset_df.iloc[0]

        for system_name, df_sota in df_sota_dict.items():
            sota_data = df_sota[df_sota['dataset'] == dataset]
            sota = sota_data.dropna(subset=[time_col, accuracy_col])
            if sota.empty:
                continue

            label = system_name.upper()
            for _, sota_row in sota.iterrows():
                t_other = float(sota_row[time_col])
                acc_other = float(sota_row[accuracy_col])
                if t_other <= 0:
                    continue

                tp_anchor = _anchor_throughput_fps(
                    poly, naive_row, sota_row, time_col, naive_time, t_other,
                )
                feasible = poly[poly[time_col] < t_other]
                if feasible.empty:
                    rows.append({
                        'dataset': dataset,
                        'throughput_fps': tp_anchor,
                        'system': label,
                        'comparison_system': label,
                        'accuracy_gain': np.nan,
                        'naive_time': naive_time,
                        'polytris_accuracy': np.nan,
                        'other_accuracy': acc_other,
                    })
                    continue

                best = feasible.sort_values(
                    [accuracy_col, time_col], ascending=[False, True],
                ).iloc[0]
                acc_poly = float(best[accuracy_col])
                rows.append({
                    'dataset': dataset,
                    'throughput_fps': tp_anchor,
                    'system': label,
                    'comparison_system': label,
                    'accuracy_gain': acc_poly - acc_other,
                    'naive_time': naive_time,
                    'polytris_accuracy': acc_poly,
                    'other_accuracy': acc_other,
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def create_speedup_chart(df_speedup: pd.DataFrame, accuracy_col_name: str, *,
                         single_row: bool = False,
                         legend_title: str = 'Compared To',
                         show_legend: bool = True,
                         apply_padding: bool = True,
                         apply_legend_config: bool = True,
                         subplot_height: int | None = None,
                         title_orient: str = 'top',
                         title_text: str | list[str] | None = None,
                         title_angle: float | None = None,
                         title_align: str = 'center',
                         title_anchor: str = 'middle',
                         title_dy: float | None = None,
                         title_dx: float | None = None,
                         title_offset: float | None = None) -> alt.Chart:
    """
    Create faceted chart of speedup ratio vs comparison-system accuracy (discrete anchors).

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

    # Sort so each (facet, comparison system) line follows increasing anchor HOTA on x.
    df_plot = df_clean.sort_values(
        ['dataset_display', 'system', 'accuracy_level'],
        kind='mergesort',
    ).reset_index(drop=True)

    color_scale = _default_system_color_scale(df_plot['system'].dropna().unique().tolist())
    legend = alt.Legend(title=legend_title, labelExpr=LEGEND_LABEL_BREAK_ON_SPACE) if show_legend else None

    base = alt.Chart(df_plot)

    x_enc = alt.X('accuracy_level:Q', title=f'{accuracy_col_name} Level')
    hota_speedup_y = accuracy_col_name == 'HOTA'
    y_enc = alt.Y(
        'speedup_ratio:Q',
        title='Speedup (Other/Ours)',
        scale=alt.Scale(domain=[0, 20]) if hota_speedup_y else alt.Undefined,
    )

    line_kw: dict = {'strokeWidth': 2, 'opacity': SYSTEM_MARK_OPACITY}
    point_kw: dict = {'size': STANDARD_POINT_SIZE, 'filled': True, 'opacity': SYSTEM_MARK_OPACITY}
    if hota_speedup_y:
        line_kw['clip'] = True
        point_kw['clip'] = True

    line = base.mark_line(
        **line_kw,
    ).encode(
        x=x_enc,
        y=y_enc,
        color=alt.Color('system:N', scale=color_scale, legend=None),
    )

    points = base.mark_point(
        **point_kw,
    ).encode(
        x=x_enc,
        y=y_enc,
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

    # Horizontal rule at y=1 (parity line). Use the same data as ``points`` so
    # layered + facet specs satisfy Altair v6 (one top-level dataset per layer).
    rule = base.mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1,
    ).encode(y=alt.datum(1))

    chart = line + points + rule
    return _facet_chart(
        chart,
        df_plot,
        f'Speedup Ratio at {accuracy_col_name} Levels (>1 = Polytris faster)',
        single_row=single_row,
        apply_padding=apply_padding,
        apply_legend_config=apply_legend_config,
        subplot_height=subplot_height,
        title_orient=title_orient,
        title_text=title_text,
        title_angle=title_angle,
        title_align=title_align,
        title_anchor=title_anchor,
        title_dy=title_dy,
        title_dx=title_dx,
        title_offset=title_offset,
    )


def create_accuracy_gain_chart(df_accuracy_gain: pd.DataFrame, accuracy_col_name: str,
                               log_scale: bool = False,
                               *,
                               single_row: bool = False,
                               legend_title: str = 'Compared To',
                               show_legend: bool = True,
                               apply_padding: bool = True,
                               apply_legend_config: bool = True,
                               subplot_height: int | None = None,
                               title_orient: str = 'top',
                               title_text: str | list[str] | None = None,
                               title_angle: float | None = None,
                               title_align: str = 'center',
                               title_anchor: str = 'middle',
                               title_dy: float | None = None,
                               title_dx: float | None = None,
                               title_offset: float | None = None) -> alt.Chart:
    """
    Create faceted chart of accuracy gain vs throughput at discrete comparison anchors.

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

    # Sort so each (facet, comparison system) line follows increasing x (throughput).
    df_plot = df_clean.sort_values(
        ['dataset_display', 'system', 'throughput_fps'],
        kind='mergesort',
    ).reset_index(drop=True)

    color_scale = _default_system_color_scale(df_plot['system'].dropna().unique().tolist())
    legend = alt.Legend(title=legend_title, labelExpr=LEGEND_LABEL_BREAK_ON_SPACE) if show_legend else None

    base = alt.Chart(df_plot)

    # X-axis uses log scale for the throughput metric if enabled.
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X('throughput_fps:Q', title='Throughput (FPS)', scale=x_scale)
    hota_gain_y = accuracy_col_name == 'HOTA'
    y_enc = alt.Y(
        'accuracy_gain:Q',
        title=f'{accuracy_col_name} Gain (Ours - Other)',
        scale=alt.Scale(domain=[0, 0.6]) if hota_gain_y else alt.Undefined,
    )

    line_kw: dict = {'strokeWidth': 2, 'opacity': SYSTEM_MARK_OPACITY}
    point_kw: dict = {'size': STANDARD_POINT_SIZE, 'filled': True, 'opacity': SYSTEM_MARK_OPACITY}
    if hota_gain_y:
        line_kw['clip'] = True
        point_kw['clip'] = True

    line = base.mark_line(
        **line_kw,
    ).encode(
        x=x_enc,
        y=y_enc,
        color=alt.Color('system:N', scale=color_scale, legend=None),
    )

    points = base.mark_point(
        **point_kw,
    ).encode(
        x=x_enc,
        y=y_enc,
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        tooltip=[
            alt.Tooltip('dataset_display:N', title='Dataset'),
            alt.Tooltip('throughput_fps:Q', title='Throughput (FPS)'),
            'naive_time',
            'system',
            'accuracy_gain',
            'polytris_accuracy',
            'other_accuracy',
        ]
    )

    # Horizontal rule at y=0 (parity line). Same data as ``points`` for Altair v6.
    rule = base.mark_rule(
        strokeDash=[4, 4], color='gray', strokeWidth=1,
    ).encode(y=alt.datum(0))

    chart = line + points + rule
    return _facet_chart(
        chart,
        df_plot,
        f'{accuracy_col_name} Gain at Throughput (>0 = Polytris more accurate)',
        single_row=single_row,
        apply_padding=apply_padding,
        apply_legend_config=apply_legend_config,
        subplot_height=subplot_height,
        title_orient=title_orient,
        title_text=title_text,
        title_angle=title_angle,
        title_align=title_align,
        title_anchor=title_anchor,
        title_dy=title_dy,
        title_dx=title_dx,
        title_offset=title_offset,
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
                                   apply_legend_config: bool = True,
                                   subplot_height: int | None = None,
                                   title_orient: str = 'top',
                                   title_text: str | list[str] | None = None,
                                   title_angle: float | None = None,
                                   title_align: str = 'center',
                                   title_anchor: str = 'middle',
                                   title_dy: float | None = None,
                                   title_dx: float | None = None,
                                   title_offset: float | None = None) -> alt.Chart:
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

    systems_list = df_clean['system'].dropna().unique().tolist()
    ordered_systems = _ordered_systems_for_chart(systems_list)
    color_scale = alt.Scale(domain=ordered_systems)
    legend = alt.Legend(title=legend_title, labelExpr=LEGEND_LABEL_BREAK_ON_SPACE) if show_legend else None

    # Base chart for Pareto fronts (with lines)
    base_pareto = alt.Chart(df_clean)

    # X-axis uses log scale if enabled.
    x_scale = alt.Scale(type='log') if log_scale else alt.Undefined
    x_enc = alt.X(f'{time_col}:Q', title=x_title, scale=x_scale)

    # Line chart showing Pareto fronts (no tooltip - lines are not easily hoverable)
    line = base_pareto.mark_line(
        strokeWidth=2,
        opacity=SYSTEM_MARK_OPACITY,
    ).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q', title=f'{accuracy_col_name} Score',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        # Group lines by the columns that define a single Pareto front.
        # Polytris fronts are computed per (dataset, classifier, canvas_scale);
        # SOTA fronts are computed per (dataset).  The chart is already faceted
        # by dataset, so only system/classifier/canvas_scale are needed here.
        # Including varying parameters (sample_rate, tilepadding, etc.) would
        # split each front into isolated single-point "lines".
        detail=['system:N', 'classifier:N', 'canvas_scale:N']
    )

    # Shape scale domain must match color domain for legend merge (same ordering).
    shape_range = [
        'diamond' if s.startswith('Polytris') else ('triangle' if s == 'Oracle' else 'circle')
        for s in ordered_systems
    ]
    shape_scale = alt.Scale(domain=ordered_systems, range=shape_range)

    # Add points for Pareto fronts (with tooltip for interactivity)
    points_pareto = base_pareto.mark_point(
        size=PARETO_POINT_SIZE,
        filled=True,
        opacity=SYSTEM_MARK_OPACITY,
    ).encode(
        x=x_enc,
        y=alt.Y(f'{accuracy_col}:Q'),
        color=alt.Color('system:N', scale=color_scale, legend=legend),
        shape=alt.Shape('system:N', scale=shape_scale, legend=legend),
        tooltip=[
            'system',
            alt.Tooltip('dataset_display:N', title='Dataset'),
            'classifier',
            'sample_rate',
            'tracking_accuracy_threshold',
            'relevance_threshold',
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
        subplot_height=subplot_height,
        title_orient=title_orient,
        title_text=title_text,
        title_angle=title_angle,
        title_align=title_align,
        title_anchor=title_anchor,
        title_dy=title_dy,
        title_dx=title_dx,
        title_offset=title_offset,
    )


def filter_by_config(df: pd.DataFrame,
                     classifiers: list[str] | None = None,
                     tilepadding_modes: list[str] | None = None,
                     sample_rates: list[int] | None = None,
                     tracking_accuracy_thresholds: list[float | None] | None = None,
                     relevance_thresholds: list[float] | None = None,
                     trackers: list[str] | None = None) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows matching configuration.

    Args:
        df: DataFrame to filter
        classifiers: List of allowed classifier names (if column exists)
        tilepadding_modes: List of allowed tilepadding modes (if column exists)
        sample_rates: List of allowed sample rates (if column exists)
        tracking_accuracy_thresholds: List of allowed thresholds (None means no pruning)
        relevance_thresholds: List of allowed T_r values (if column exists)
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

    # Filter by relevance threshold T_r (if column exists and filter is specified).
    if relevance_thresholds is not None and 'relevance_threshold' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['relevance_threshold'].isin(relevance_thresholds)]
        print(f"  Filtered by relevance_thresholds: {len(filtered_df)} rows remain")

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


def copy_chart_outputs(source_dir: str, base_name: str, destination_dir: str):
    """Copy the paper-targeted chart format into another directory."""
    os.makedirs(destination_dir, exist_ok=True)

    # The paper directory only needs the publication-ready PDF.
    source_path = os.path.join(source_dir, f'{base_name}.pdf')
    destination_path = os.path.join(destination_dir, f'{base_name}.pdf')
    if not os.path.exists(source_path):
        return
    shutil.copy2(source_path, destination_path)
    print(f"  Copied PDF to: {destination_path}")


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
        df_accuracy_gain: Accuracy-gain-at-throughput data
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
        x_title='Throughput (FPS)',
        single_row=True,
        legend_title='System',
        show_legend=True,
        apply_padding=False,
        apply_legend_config=False,
        subplot_height=COMBINED_ONE_ROW_SUBPLOT_HEIGHT,
        title_orient='right',
        title_text=['HOTA vs', 'Throughput', 'Pareto', 'Fronts'],
        title_angle=0,
        title_align='left',
        title_anchor='start',
        title_dy=5,
        title_dx=-66,
        title_offset=0,
    )
    speedup_chart = create_speedup_chart(
        df_speedup,
        'HOTA',
        single_row=True,
        legend_title='System',
        show_legend=False,
        apply_padding=False,
        apply_legend_config=False,
        subplot_height=COMBINED_ONE_ROW_SUBPLOT_HEIGHT,
        title_orient='right',
        title_text=['Comparison', 'to other', 'systems:', 'Speedup at', 'HOTA Level'],
        title_angle=0,
        title_align='left',
        title_anchor='start',
        title_dy=5,
        title_dx=-70,
        title_offset=0,
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
        subplot_height=COMBINED_ONE_ROW_SUBPLOT_HEIGHT,
        title_orient='right',
        title_text=['Comparison', 'to other', 'systems:', 'Accuracy Gain', 'at Throughput'],
        title_angle=0,
        title_align='left',
        title_anchor='start',
        title_dy=5,
        title_dx=-85,
        title_offset=0,
    )

    return alt.vconcat(
        throughput_chart,
        speedup_chart,
        accuracy_gain_chart,
        spacing=1,
    ).resolve_scale(
        color='shared'
    ).properties(
        padding=0,
    ).configure_legend(
        orient='left',
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
    naive_df['system'] = 'Oracle'
    print(f"\nExtracted {len(naive_df)} naive baseline rows from test split")

    # Filter test Polytris rows by non-ablation parameter dimensions.
    # Sample_rate and tracking_accuracy_threshold are filtered per ablation
    # condition inside the metric loop below.
    print("\nFiltering Polytris data by configuration settings...")
    print(f"  Test rows before filtering: {len(test_all_df)}")
    base_filtered_test_df = filter_by_config(
        test_all_df,
        classifiers=CLASSIFIERS,
        tilepadding_modes=TILEPADDING_MODES,
        relevance_thresholds=RELEVANCE_THRESHOLDS,
        trackers=TRACKERS,
    )
    print(f"  Test rows after base filtering: {len(base_filtered_test_df)}")

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

        # Compute Pareto fronts for each ablation condition.
        # Each condition restricts sample_rate and/or tracking_accuracy_threshold
        # before computing its own Pareto front.
        print(f"\n1b. Computing ablation Pareto fronts for {accuracy_name}...")

        # Columns to keep for tooltip display.
        tooltip_cols = ['system', 'dataset', 'videoset', 'classifier', 'sample_rate',
                        'tracking_accuracy_threshold', 'relevance_threshold', 'tilepadding', 'canvas_scale',
                        'tracker', 'time', accuracy_col]

        # Collect Pareto-optimal rows from all systems.
        pareto_data_list: list[pd.DataFrame] = []

        # Track the full-system Pareto front for speedup/gain charts below.
        polytris_full_pareto_df = pd.DataFrame()

        for condition in ABLATION_CONDITIONS:
            # Apply condition-specific parameter restrictions.
            condition_df = filter_by_ablation_condition(base_filtered_test_df, condition)
            condition_df = condition_df.dropna(subset=['time', accuracy_col]).copy()

            if condition_df.empty:
                print(f"  [{condition.label}] No test rows after filtering; skipping")
                continue

            # Compute Pareto front per dataset (minimize time, maximize accuracy).
            pareto_df = _filter_pareto_per_dataset(
                condition_df, 'time', accuracy_col, minx=True, miny=False,
            )
            print(f"  [{condition.label}] {len(pareto_df)} Pareto-optimal points")

            if pareto_df.empty:
                continue

            # Tag with the condition's display label as the system name.
            pareto_df = pareto_df.copy()
            pareto_df['system'] = condition.label
            pareto_cols = [c for c in tooltip_cols if c in pareto_df.columns]
            pareto_data_list.append(pareto_df[pareto_cols])

            # Keep the full-system front for speedup/accuracy-gain comparisons.
            if condition.name == 'full':
                polytris_full_pareto_df = pareto_df.copy()

        # Filter Oracle (naive) baseline to Pareto-optimal points per dataset.
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

        # Append Oracle (naive) Pareto-optimal points.
        if not pareto_naive_df.empty:
            naive_point_df = pareto_naive_df.copy()
            naive_point_df['system'] = 'Oracle'
            naive_cols = [c for c in tooltip_cols if c in naive_point_df.columns]
            pareto_data_list.append(naive_point_df[naive_cols])

        # Append SOTA Pareto-optimal points.
        for system_name, df_sota in pareto_sota_dict.items():
            sota_metric_df = df_sota.copy()
            sota_metric_df['system'] = system_name.upper()
            sota_cols = [c for c in tooltip_cols if c in sota_metric_df.columns]
            pareto_data_list.append(sota_metric_df[sota_cols])

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

        # Compute throughput Pareto fronts for each ablation condition (maximize both axes).
        throughput_tooltip_cols = ['system', 'dataset', 'videoset', 'classifier',
                                  'sample_rate', 'tracking_accuracy_threshold',
                                  'relevance_threshold',
                                  'tilepadding', 'canvas_scale', 'tracker',
                                  'throughput_fps', 'time', accuracy_col]
        tp_data_list: list[pd.DataFrame] = []

        for condition in ABLATION_CONDITIONS:
            condition_df = filter_by_ablation_condition(base_filtered_test_df, condition)
            condition_df = condition_df.dropna(subset=['throughput_fps', accuracy_col])
            if condition_df.empty or 'throughput_fps' not in condition_df.columns:
                continue
            tp_pareto = _filter_pareto_per_dataset(
                condition_df, 'throughput_fps', accuracy_col, minx=False, miny=False,
            )
            if not tp_pareto.empty:
                tp_pareto = tp_pareto.copy()
                tp_pareto['system'] = condition.label
                tp_cols = [c for c in throughput_tooltip_cols if c in tp_pareto.columns]
                tp_data_list.append(tp_pareto[tp_cols])

        # Append Oracle (naive) throughput Pareto.
        throughput_naive_df = _filter_pareto_per_dataset(
            naive_df.dropna(subset=['throughput_fps', accuracy_col]),
            'throughput_fps', accuracy_col, minx=False, miny=False,
        ) if ('throughput_fps' in naive_df.columns
              and not naive_df['throughput_fps'].isna().all()) else pd.DataFrame()

        if not throughput_naive_df.empty:
            tp_naive = throughput_naive_df.copy()
            tp_naive['system'] = 'Oracle'
            tp_cols = [c for c in throughput_tooltip_cols if c in tp_naive.columns]
            tp_data_list.append(tp_naive[tp_cols])

        # Append SOTA throughput Pareto.
        for system_name, df_sota in df_sota_dict.items():
            if accuracy_col not in df_sota.columns or 'throughput_fps' not in df_sota.columns:
                continue
            filtered = _filter_pareto_per_dataset(
                df_sota.dropna(subset=['throughput_fps', accuracy_col]),
                'throughput_fps', accuracy_col, minx=False, miny=False,
            )
            if not filtered.empty:
                tp_sota = filtered.copy()
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
        # Uses only the full-system Polytris Pareto front for comparison.
        print(f"\n4. Computing speedup at accuracy levels for {accuracy_name}...")
        df_speedup = compute_speedup_at_accuracy_levels(
            polytris_full_pareto_df, pareto_sota_dict, accuracy_col, 'time',
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

        # 5. Compute and visualize accuracy gain at throughput levels
        # Uses only the full-system Polytris Pareto front for comparison.
        print(f"\n5. Computing accuracy gain at throughput for {accuracy_name}...")
        df_accuracy_gain = compute_accuracy_gain_at_naive_speedup_levels(
            polytris_full_pareto_df, pareto_sota_dict, pareto_naive_df, accuracy_col, 'time',
        )

        if not df_accuracy_gain.empty:
            print(f"  Accuracy gain data: {len(df_accuracy_gain)} comparison points")
            save_chart_variants(
                create_accuracy_gain_chart,
                output_dir,
                f'{accuracy_col.lower()}_accuracy_gain_at_throughput',
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
            combined_chart_base_name = f'{accuracy_col.lower()}_summary_one_row'
            save_chart(
                create_hota_summary_one_row_chart(
                    df_tp_combined,
                    df_speedup,
                    df_accuracy_gain,
                    log_scale=log_scale,
                ),
                output_dir,
                combined_chart_base_name,
            )
            copy_chart_outputs(output_dir, combined_chart_base_name, PAPER_FIGURES_GENERATED_DIR)
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
