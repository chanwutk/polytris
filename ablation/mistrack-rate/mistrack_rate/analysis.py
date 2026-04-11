from __future__ import annotations

import glob as _glob
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd

from polyis.pareto import compute_pareto_front

from .common import GRID_COLS, GRID_ROWS, ensure_dir, plots_dir, results_csv_path


@dataclass(frozen=True)
class PairDefinition:
    slug: str
    x_col: str
    y_col: str
    x_label: str
    y_label: str
    x_minimize: bool
    y_maximize: bool
    # When False, the y-axis does not start at zero (useful for metrics like HOTA
    # that are clustered far from 0).
    zero_y_axis: bool = True
    # When True, each facet subplot gets its own independent y-axis scale.
    resolve_y_independent: bool = False


PAIR_DEFINITIONS = (
    PairDefinition(
        slug='mistrack_vs_hota',
        x_col='mistrack_rate',
        y_col='HOTA_HOTA',
        x_label='Mistrack Rate',
        y_label='HOTA',
        x_minimize=True,
        y_maximize=True,
        zero_y_axis=False,
        resolve_y_independent=True,
    ),
    PairDefinition(
        slug='mistrack_vs_pruning',
        x_col='mistrack_rate',
        y_col='pruning_ratio',
        x_label='Mistrack Rate',
        y_label='Pruning Ratio (%)',
        x_minimize=True,
        # Higher pruning ratio = more aggressive pruning; equivalent to the old
        # y_maximize=False on retention_rate (minimize retention ↔ maximize pruning).
        y_maximize=True,
    ),
    PairDefinition(
        slug='pruning_vs_hota',
        x_col='pruning_ratio',
        y_col='HOTA_HOTA',
        x_label='Pruning Ratio (%)',
        y_label='HOTA',
        # Prefer high pruning (right side) with high HOTA; equivalent to the old
        # x_minimize=True on retention_rate (minimize retention ↔ maximize pruning).
        x_minimize=False,
        y_maximize=True,
        zero_y_axis=False,
        resolve_y_independent=True,
    ),
)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived display columns to a results DataFrame.

    Computes pruning_ratio = (1 - retention_rate) * 100 so that downstream
    chart code can reference it directly without touching the raw CSV columns.
    Safe to call on DataFrames that already contain the column (overwrites).
    """
    df = df.copy()
    df['pruning_ratio'] = (1 - df['retention_rate']) * 100
    return df


def annotate_pareto_flags(results_df: pd.DataFrame) -> pd.DataFrame:
    # Add derived columns so PairDefinitions referencing pruning_ratio work.
    flagged_df = _prepare_df(results_df)
    exhaustive_df = flagged_df[flagged_df['method'] == 'exhaustive'].copy()

    for pair in PAIR_DEFINITIONS:
        flag_col = f'is_pareto_{pair.slug}'
        flagged_df[flag_col] = False

        if exhaustive_df.empty:
            continue

        frontier_input = exhaustive_df.dropna(subset=[pair.x_col, pair.y_col]).copy()
        if frontier_input.empty:
            continue

        frontier_y_col = pair.y_col
        if not pair.y_maximize:
            frontier_y_col = f'__neg_{pair.slug}'
            frontier_input[frontier_y_col] = -frontier_input[pair.y_col]

        frontier_df = compute_pareto_front(
            frontier_input,
            x_col=pair.x_col,
            y_col=frontier_y_col,
            minx=pair.x_minimize,
            miny=False,
            num_points=None,
        )
        # Only flag exhaustive rows as Pareto; heuristic points are plotted separately.
        frontier_grid_keys = set(frontier_df['grid_key'])
        flagged_df[flag_col] = (
            (flagged_df['method'] == 'exhaustive')
            & flagged_df['grid_key'].isin(frontier_grid_keys)
        )

    return flagged_df


def save_results(results_df: pd.DataFrame, dataset: str, tracker_name: str) -> Path:
    output_path = results_csv_path(dataset, tracker_name)
    ensure_dir(output_path.parent)
    results_df.to_csv(output_path, index=False)
    return output_path


def _build_chart_layers(
    base: alt.Chart,
    pair: PairDefinition,
) -> list[alt.Chart]:
    """Return ordered Altair layers for one metric pair (back to front):
      1. Exhaustive background scatter — light gray, semi-transparent.
      2. Pareto frontier connecting line — black; order encoding sorts left-to-right.
      3. Pareto frontier point markers — black.
      4. Heuristic diamond markers — steelblue.
      5. Whole-frame baseline diamonds — firebrick.
      6. Whole-frame rate labels — derived via Vega expression, no Python iteration.
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'
    x_enc = alt.X(f'{pair.x_col}:Q', title=pair.x_label)
    y_enc = alt.Y(
        f'{pair.y_col}:Q',
        title=pair.y_label,
        scale=alt.Scale(zero=pair.zero_y_axis),
    )

    exhaustive_tooltip = [
        alt.Tooltip('dataset:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('grid_key:N', title='Grid'),
    ]
    heuristic_tooltip = [
        alt.Tooltip('dataset:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('heuristic_threshold:Q', title='Threshold'),
        alt.Tooltip('grid_key:N', title='Grid'),
    ]
    whole_frame_tooltip = [
        alt.Tooltip('dataset:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('grid_key:N', title='Frame rate'),
    ]

    # Layer 1: exhaustive background scatter.
    bg_layer = (
        base
        .transform_filter("datum.method === 'exhaustive'")
        .mark_point(size=14, opacity=0.35, filled=True, color='lightgray')
        .encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)
    )

    pareto_filter = f"datum.method === 'exhaustive' && datum.{pareto_flag_col}"

    # Layer 2: Pareto frontier line; order encoding ensures left-to-right rendering.
    frontier_line = (
        base
        .transform_filter(pareto_filter)
        .mark_line(strokeWidth=2.0, color='black')
        .encode(
            x=x_enc,
            y=y_enc,
            order=alt.Order(f'{pair.x_col}:Q', sort='ascending'),
            tooltip=exhaustive_tooltip,
        )
    )

    # Layer 3: Pareto frontier point markers.
    frontier_pts = (
        base
        .transform_filter(pareto_filter)
        .mark_point(size=32, filled=True, color='black')
        .encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)
    )

    # Layer 4: heuristic diamond markers.
    heuristic_layer = (
        base
        .transform_filter("datum.method === 'heuristic'")
        .mark_point(size=80, shape='diamond', filled=True, color='steelblue', stroke='white', strokeWidth=0.5)
        .encode(x=x_enc, y=y_enc, tooltip=heuristic_tooltip)
    )

    # Layer 5: whole-frame baseline diamonds.
    whole_frame_layer = (
        base
        .transform_filter("datum.method === 'whole_frame'")
        .mark_point(size=80, shape='diamond', filled=True, color='firebrick', stroke='white', strokeWidth=0.5)
        .encode(x=x_enc, y=y_enc, tooltip=whole_frame_tooltip)
    )

    # Layer 6: whole-frame rate labels via Vega expression.
    # split(grid_key, '_')[2] extracts the numeric rate from 'whole_frame_{rate}'.
    whole_frame_text = (
        base
        .transform_filter("datum.method === 'whole_frame'")
        .transform_calculate(
            label="datum.grid_key === 'whole_frame_1' ? 'no skip' : '1/' + split(datum.grid_key, '_')[2] + ' frames'",
        )
        .mark_text(dx=6, dy=-8, fontSize=9, color='firebrick')
        .encode(x=x_enc, y=y_enc, text='label:N')
    )

    return [bg_layer, frontier_line, frontier_pts, heuristic_layer, whole_frame_layer, whole_frame_text]


def _make_single_chart(
    results_df: pd.DataFrame,
    pair: PairDefinition,
    title: str,
) -> alt.LayerChart:
    """Build a single-dataset Altair chart for one metric pair."""
    pareto_flag_col = f'is_pareto_{pair.slug}'
    plot_df = results_df.dropna(subset=[pair.x_col, pair.y_col]).copy()
    plot_df[pareto_flag_col] = (
        plot_df[pareto_flag_col].fillna(False).astype(bool)
        if pareto_flag_col in plot_df.columns
        else False
    )
    layers = _build_chart_layers(alt.Chart(plot_df), pair)
    return alt.layer(*layers).properties(title=title, width=100, height=200)


def plot_results(dataset: str, tracker_name: str) -> list[Path]:
    results_path = results_csv_path(dataset, tracker_name)
    if not results_path.exists():
        raise FileNotFoundError(f'Results CSV not found: {results_path}')

    results_df = _prepare_df(pd.read_csv(results_path))
    output_dir = ensure_dir(plots_dir(dataset, tracker_name))
    written_paths: list[Path] = []

    for pair in PAIR_DEFINITIONS:
        title = f'{dataset} {tracker_name}: {pair.x_label} vs {pair.y_label}'
        chart = _make_single_chart(results_df, pair, title)
        for suffix in ('.png', '.html'):
            out_path = output_dir / f'{pair.slug}{suffix}'
            chart.save(str(out_path))
            written_paths.append(out_path)

    return written_paths


def _collect_results(tracker_name: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Glob every results.csv for *tracker_name* across all datasets and concatenate."""
    from polyis.io import cache as _polyis_cache

    # Resolve base cache directory (injectable for testing).
    base = Path(cache_dir) if cache_dir is not None else Path(_polyis_cache.CACHE_DIR)

    # Match the path layout produced by evaluation_dir().
    pattern = str(
        base / '*' / 'ablation' / 'mistrack-rate'
        / f'{GRID_ROWS}x{GRID_COLS}' / tracker_name / 'test' / 'results.csv'
    )
    csv_paths = sorted(_glob.glob(pattern))
    if not csv_paths:
        return pd.DataFrame()

    # Load each CSV, concatenate, then add derived display columns.
    frames = [pd.read_csv(p) for p in csv_paths]
    return _prepare_df(pd.concat(frames, ignore_index=True))


def _make_altair_chart(
    combined_df: pd.DataFrame,
    pair: PairDefinition,
    tracker_name: str,
) -> alt.FacetChart:
    """Build a faceted Altair chart for one metric pair; one subplot per dataset.

    All layers share a single DataFrame so Altair's facet requirement (uniform
    data source across layers) is satisfied. Row filtering is done via
    Vega-Lite transform_filter expressions inside _build_chart_layers.
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'
    # Sort by (dataset, x) so the Pareto frontier line renders left-to-right.
    plot_df = (
        combined_df
        .dropna(subset=[pair.x_col, pair.y_col])
        .sort_values(['dataset', pair.x_col])
        .copy()
    )
    plot_df[pareto_flag_col] = (
        plot_df[pareto_flag_col].fillna(False).astype(bool)
        if pareto_flag_col in plot_df.columns
        else False
    )
    layers = _build_chart_layers(alt.Chart(plot_df), pair)
    return (
        alt.layer(*layers)
        .properties(width=100, height=200)
        .facet(
            facet=alt.Facet('dataset:N', title='Dataset'),
            # columns=3,
        )
        .resolve_scale(
            x='independent',
            y='independent' if pair.resolve_y_independent else 'shared',
        )
        .properties(title=f'{tracker_name}: {pair.x_label} vs {pair.y_label}')
    )


def combine_visualize(
    tracker_name: str,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Combine per-dataset results and write Altair summary charts to SUMMARY.

    For each of the three metric pairs (mistrack/HOTA, mistrack/retention,
    retention/HOTA) the function writes both a .png and an .html file under
    {cache_dir}/SUMMARY/ablation/mistrack-rate/{tracker_name}/.

    Returns the list of written paths, or an empty list when no cached
    results exist for *tracker_name*.
    """
    from polyis.io import cache as _polyis_cache

    # Load all available per-dataset results for this tracker.
    combined_df = _collect_results(tracker_name, cache_dir=cache_dir)
    if combined_df.empty:
        return []

    # Resolve output directory under SUMMARY.
    base = Path(cache_dir) if cache_dir is not None else Path(_polyis_cache.CACHE_DIR)
    output_dir = ensure_dir(base / 'SUMMARY' / 'ablation' / 'mistrack-rate' / tracker_name)

    written_paths: list[Path] = []

    for pair in PAIR_DEFINITIONS:
        chart = _make_altair_chart(combined_df, pair, tracker_name)

        # Write PNG (requires vl-convert-python) and self-contained HTML.
        for suffix in ('.png', '.html'):
            out_path = output_dir / f'{pair.slug}{suffix}'
            chart.save(str(out_path))
            written_paths.append(out_path)

    return written_paths

