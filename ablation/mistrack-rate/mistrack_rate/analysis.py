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
        # HOTA y-axis is synced (shared) across facets so datasets are directly
        # comparable on the same HOTA scale.
        resolve_y_independent=False,
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
        # HOTA y-axis is synced (shared) across facets so datasets are directly
        # comparable on the same HOTA scale.
        resolve_y_independent=False,
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
    include_background: bool = True,
) -> list[alt.Chart]:
    """Return ordered Altair layers for one metric pair (back to front):
      1. Exhaustive background scatter — light gray, semi-transparent. Omitted
         when include_background=False to produce a "Pareto + ours +
         whole-frame only" view without the exhaustive cloud.
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
        .mark_point(size=14, opacity=0.6, filled=True, color='gray')
        .encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)
    )

    pareto_filter = f"datum.method === 'exhaustive' && datum.{pareto_flag_col}"

    # Layer 2: Pareto frontier line; order encoding ensures left-to-right rendering.
    frontier_line = (
        base
        .transform_filter(pareto_filter)
        .mark_line(strokeWidth=1.0, color='black')
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
            label="datum.grid_key === 'whole_frame_1' ? 'no skip' : '1/' + split(datum.grid_key, '_')[2]",
        )
        .mark_text(dx=6, dy=-8, fontSize=9, color='firebrick')
        .encode(x=x_enc, y=y_enc, text='label:N')
    )

    foreground_layers = [
        frontier_line,
        frontier_pts,
        heuristic_layer,
        whole_frame_layer,
        whole_frame_text,
    ]
    # Pareto-only variant drops the exhaustive background scatter; the remaining
    # foreground layers (Pareto line/points, heuristic, whole-frame) are unchanged.
    if not include_background:
        return foreground_layers
    return [bg_layer, *foreground_layers]


def _prune_for_pareto_only(plot_df: pd.DataFrame, pareto_flag_col: str) -> pd.DataFrame:
    """Drop exhaustive rows that are not on the Pareto frontier.

    The pareto-only chart variant references exhaustive rows only through
    the is_pareto filter, so non-pareto exhaustive points just inflate the
    Vega-Lite payload and freeze the browser. Heuristic and whole-frame
    rows are kept in full.
    """
    keep_pareto_exhaustive = (plot_df['method'] == 'exhaustive') & plot_df[pareto_flag_col]
    keep_non_exhaustive = plot_df['method'] != 'exhaustive'
    return plot_df[keep_pareto_exhaustive | keep_non_exhaustive].copy()


def _make_single_chart(
    results_df: pd.DataFrame,
    pair: PairDefinition,
    title: str,
    include_background: bool = True,
) -> alt.LayerChart:
    """Build a single-dataset Altair chart for one metric pair.

    Passing include_background=False produces the "Pareto + heuristic +
    whole-frame" variant (no gray exhaustive scatter).
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'
    plot_df = results_df.dropna(subset=[pair.x_col, pair.y_col]).copy()
    plot_df[pareto_flag_col] = (
        plot_df[pareto_flag_col].fillna(False).astype(bool)
        if pareto_flag_col in plot_df.columns
        else False
    )
    # Drop non-pareto exhaustive rows for the pareto-only variant — they are
    # filtered away by transform_filter anyway but still slow Vega-Lite down.
    if not include_background:
        plot_df = _prune_for_pareto_only(plot_df, pareto_flag_col)
    layers = _build_chart_layers(alt.Chart(plot_df), pair, include_background=include_background)
    # Pareto-only variant gets 3× width and 2× height for readability.
    width, height = (300, 220) if not include_background else (100, 110)
    return alt.layer(*layers).properties(title=title, width=width, height=height)


def plot_results(dataset: str, tracker_name: str) -> list[Path]:
    results_path = results_csv_path(dataset, tracker_name)
    if not results_path.exists():
        raise FileNotFoundError(f'Results CSV not found: {results_path}')

    # Re-annotate pareto flags against the current PAIR_DEFINITIONS so stale
    # cached CSVs (from earlier runs that used different slug names) still
    # produce correct Pareto layers. annotate_pareto_flags is idempotent and
    # internally calls _prepare_df, so we don't need to prepare first.
    results_df = annotate_pareto_flags(pd.read_csv(results_path))
    output_dir = ensure_dir(plots_dir(dataset, tracker_name))
    written_paths: list[Path] = []

    # Two chart variants per pair: full (with exhaustive background) and
    # pareto-only (frontier + heuristic + whole-frame markers only).
    variants = (
        ('', True),
        ('_pareto_only', False),
    )

    for pair in PAIR_DEFINITIONS:
        title = f'{dataset} {tracker_name}: {pair.x_label} vs {pair.y_label}'
        for slug_suffix, include_background in variants:
            chart = _make_single_chart(
                results_df,
                pair,
                title,
                include_background=include_background,
            )
            for suffix in ('.png', '.html'):
                out_path = output_dir / f'{pair.slug}{slug_suffix}{suffix}'
                # Render PNGs at 4× resolution for print-quality output; HTML
                # is vector-based and does not need a scale factor.
                save_kwargs = {'scale_factor': 4.0} if suffix == '.png' else {}
                chart.save(str(out_path), **save_kwargs)
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

    # Re-annotate each CSV against the current PAIR_DEFINITIONS so stale cached
    # flag columns (e.g. `is_pareto_retention_vs_hota` from older runs) don't
    # cause missing-flag → empty-Pareto-layer rendering. Annotation runs
    # per-dataset, preserving each dataset's independent Pareto frontier.
    frames = [annotate_pareto_flags(pd.read_csv(p)) for p in csv_paths]
    return _prepare_df(pd.concat(frames, ignore_index=True))


def _make_altair_chart(
    combined_df: pd.DataFrame,
    pair: PairDefinition,
    tracker_name: str,
    include_background: bool = True,
) -> alt.FacetChart:
    """Build a faceted Altair chart for one metric pair; one subplot per dataset.

    All layers share a single DataFrame so Altair's facet requirement (uniform
    data source across layers) is satisfied. Row filtering is done via
    Vega-Lite transform_filter expressions inside _build_chart_layers.

    Passing include_background=False omits the gray exhaustive scatter to
    produce the "Pareto + heuristic + whole-frame" variant.
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
    # Drop non-pareto exhaustive rows for the pareto-only variant — they are
    # filtered away by transform_filter anyway but still slow Vega-Lite down.
    if not include_background:
        plot_df = _prune_for_pareto_only(plot_df, pareto_flag_col)
    layers = _build_chart_layers(alt.Chart(plot_df), pair, include_background=include_background)
    # Pareto-only variant gets 3× width and 2× height for readability.
    width, height = (300, 220) if not include_background else (100, 110)
    return (
        alt.layer(*layers)
        .properties(width=width, height=height)
        .facet(
            # Drop the shared 'Dataset' facet title and instead prefix each
            # per-facet header label with 'Dataset: ' via a Vega labelExpr.
            # labelPadding is negative so the 'Dataset: ...' label sits closer
            # to (just above) the subplot rather than floating well above it.
            facet=alt.Facet(
                'dataset:N',
                title=None,
                header=alt.Header(
                    labelExpr="'Dataset: ' + datum.value",
                    labelPadding=2,
                ),
            ),
            # columns=3,
        )
        .resolve_scale(
            x='independent',
            y='independent' if pair.resolve_y_independent else 'shared',
        )
        .properties(title=f'{tracker_name}: {pair.x_label} vs {pair.y_label}')
        # Collapse inter-facet spacing so subplots sit flush against each
        # other rather than being separated by Altair's default gap.
        .configure_facet(spacing=0)
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

    # Two chart variants per pair: full (with exhaustive background) and
    # pareto-only (frontier + heuristic + whole-frame markers only).
    variants = (
        ('', True),
        ('_pareto_only', False),
    )

    for pair in PAIR_DEFINITIONS:
        for slug_suffix, include_background in variants:
            chart = _make_altair_chart(
                combined_df,
                pair,
                tracker_name,
                include_background=include_background,
            )

            # Write PNG (requires vl-convert-python) and self-contained HTML.
            for suffix in ('.png', '.html'):
                out_path = output_dir / f'{pair.slug}{slug_suffix}{suffix}'
                # Render PNGs at 4× resolution for print-quality output; HTML
                # is vector-based and does not need a scale factor.
                save_kwargs = {'scale_factor': 4.0} if suffix == '.png' else {}
                chart.save(str(out_path), **save_kwargs)
                written_paths.append(out_path)

    return written_paths

