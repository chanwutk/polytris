from __future__ import annotations

import glob as _glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

from polyis.pareto import compute_pareto_front

from .common import GRID_COLS, GRID_ROWS, ensure_dir, plots_dir, results_csv_path

# Raw cache / folder names -> human-readable labels (tooltips, single-dataset titles).
_DATASET_SEQUENCE_LABELS: dict[str, str] = {
    'ams-y05': 'Amsterdam',
    'ams-y-05': 'Amsterdam',
    'caldot1-y05': 'CalDoT 1',
    'caldot2-y05': 'CalDoT 2',
    'jnc0': 'B3D1',
    'jnc2': 'B3D2',
    'jnc6': 'B3D3',
    'jnc7': 'B3D4',
}

# Raw dataset folder names included in cross-dataset summary charts only.
_SUMMARY_DATASET_RAW: frozenset[str] = frozenset({
    'ams-y05',
    'ams-y-05',
    'caldot1-y05',
    'caldot2-y05',
    'jnc0',
})

# Default facet column order for combined charts (unknown names append after).
_FACET_COLUMN_ORDER: tuple[str, ...] = (
    'Amsterdam',
    'CalDoT 1',
    'CalDoT 2',
    'B3D1',
)

# Internal tracker keys -> chart title labels (paths still use the raw key).
_TRACKER_DISPLAY_LABELS: dict[str, str] = {
    'bytetrackcython': 'BYTETrack',
    'sortcython': 'SORT',
    'ocsortcython': 'OCSORT',
}


def _tracker_display_label(raw: str) -> str:
    return _TRACKER_DISPLAY_LABELS.get(raw, raw)


def _sequence_label(raw: str) -> str:
    return _DATASET_SEQUENCE_LABELS.get(raw, raw)


def _facet_sort_order(values: pd.Series) -> list[str]:
    present = [str(v) for v in pd.unique(values.dropna())]
    ordered = [x for x in _FACET_COLUMN_ORDER if x in present]
    tail = sorted(x for x in present if x not in ordered)
    return ordered + tail


def _ensure_sequence_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence_label for chart tooltips (human-readable dataset name)."""
    out = df.copy()
    if 'sequence_label' not in out.columns:
        out['sequence_label'] = out['dataset'].map(_sequence_label)
    return out


def _prepare_combined_chart_df(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence_label and facet_dataset (display names) for summary charts."""
    df = _ensure_sequence_label(combined_df.copy())
    df['facet_dataset'] = df['sequence_label']
    order = _facet_sort_order(df['facet_dataset'])
    df['facet_dataset'] = pd.Categorical(df['facet_dataset'], categories=order, ordered=True)
    return df


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

_PRUNING_HOTA_FLAG_COL: str = 'is_pareto_pruning_vs_hota'


def _paper_figures_generated_dir() -> Path:
    """Repo-relative ``paper/figures/generated`` (same layout as ``run.py`` in Docker)."""
    return Path(__file__).resolve().parents[3] / 'paper' / 'figures' / 'generated'


def _reference_hota_at_matched_pruning(
    pareto_df: pd.DataFrame,
    p_heuristic: float,
) -> float | None:
    """Pareto exhaustive anchor: max pruning among points with pruning <= heuristic."""
    if pareto_df.empty:
        return None
    tol = 1e-9
    eligible = pareto_df[pareto_df['pruning_ratio'] <= p_heuristic + tol]
    if eligible.empty:
        eligible = pareto_df.sort_values('pruning_ratio').head(1)
    p_cap = float(eligible['pruning_ratio'].max())
    at_cap = eligible[eligible['pruning_ratio'] >= p_cap - tol]
    best = at_cap.loc[at_cap['HOTA_HOTA'].idxmax()]
    return float(best['HOTA_HOTA'])


def _hota_loss_percent(ref_hota: float, heu_hota: float) -> float | None:
    if ref_hota is None or (isinstance(ref_hota, float) and (math.isnan(ref_hota) or ref_hota <= 0)):
        return None
    return 100.0 * (ref_hota - heu_hota) / ref_hota


@dataclass(frozen=True)
class HeuristicHotaLossSummary:
    """Aggregate HOTA loss (%%) of heuristic vs matched-pruning Pareto anchor."""

    tracker_name: str
    max_loss_percent: float
    mean_loss_percent: float
    detail_rows: pd.DataFrame


def compute_heuristic_hota_loss_summary(combined_df: pd.DataFrame, tracker_name: str) -> HeuristicHotaLossSummary | None:
    """Compare each heuristic row to the pruning-vs-HOTA Pareto anchor at matched pruning."""
    df = annotate_pareto_flags(_prepare_df(combined_df.copy()))
    flag = _PRUNING_HOTA_FLAG_COL
    if flag not in df.columns:
        return None

    rows: list[dict[str, Any]] = []
    for dataset, g in df.groupby('dataset', sort=False):
        pareto = g[(g['method'] == 'exhaustive') & (g[flag])].dropna(
            subset=['pruning_ratio', 'HOTA_HOTA'],
        )
        heur = g[g['method'] == 'heuristic'].dropna(subset=['pruning_ratio', 'HOTA_HOTA'])
        for _, r in heur.iterrows():
            p_h = float(r['pruning_ratio'])
            h_h = float(r['HOTA_HOTA'])
            ref = _reference_hota_at_matched_pruning(pareto, p_h)
            loss = _hota_loss_percent(ref, h_h) if ref is not None else None
            rows.append(
                {
                    'dataset': dataset,
                    'pruning_ratio': p_h,
                    'heuristic_hota': h_h,
                    'ref_hota': ref,
                    'hota_loss_percent': loss,
                },
            )

    if not rows:
        return None

    detail = pd.DataFrame(rows)
    losses = detail['hota_loss_percent'].dropna()
    if losses.empty:
        return HeuristicHotaLossSummary(
            tracker_name=tracker_name,
            max_loss_percent=float('nan'),
            mean_loss_percent=float('nan'),
            detail_rows=detail,
        )
    return HeuristicHotaLossSummary(
        tracker_name=tracker_name,
        max_loss_percent=float(losses.max()),
        mean_loss_percent=float(losses.mean()),
        detail_rows=detail,
    )


def write_heuristic_hota_loss_macros(
    summary: HeuristicHotaLossSummary,
    dest_dir: Path | None = None,
) -> Path:
    """Write LaTeX ``\\providecommand`` macros for max / mean HOTA loss (percent)."""
    out_dir = dest_dir if dest_dir is not None else _paper_figures_generated_dir()
    ensure_dir(out_dir)
    safe = summary.tracker_name.replace('/', '_')
    path = out_dir / f'mistrack_rate_heuristic_hota_loss_macros_{safe}.tex'
    losses = summary.detail_rows['hota_loss_percent'].dropna()
    if losses.empty:
        mx_s, mn_s = '0.00', '0.00'
        warn = '% WARNING: no valid HOTA-loss samples (check heuristic and Pareto rows).\n'
    else:
        mx_s = f'{float(losses.max()):.2f}'
        mn_s = f'{float(losses.mean()):.2f}'
        warn = ''
    label = _tracker_display_label(summary.tracker_name)
    lines: list[str] = []
    if warn:
        lines.append(warn.rstrip('\n'))
    lines.extend(
        [
            '% Auto-generated by mistrack_rate.analysis.combine_visualize.',
            f'% Tracker key: {summary.tracker_name} ({label}).',
            '% Pareto anchor (pruning vs HOTA): exhaustive point with highest pruning',
            '% among those with pruning <= the heuristic point.',
            r'\providecommand{\MistrackHeuristicMaxHotaLossPercent}{%s}' % mx_s,
            r'\providecommand{\MistrackHeuristicAvgHotaLossPercent}{%s}' % mn_s,
        ],
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _print_heuristic_hota_loss_summary(summary: HeuristicHotaLossSummary, macro_path: Path) -> None:
    print('\n--- Heuristic vs Pareto (matched pruning), HOTA loss % ---')
    print(f'tracker: {summary.tracker_name} ({_tracker_display_label(summary.tracker_name)})')
    for _, r in summary.detail_rows.iterrows():
        loss = r['hota_loss_percent']
        loss_s = f'{loss:.4f}' if loss == loss else 'nan'
        ref_s = f"{r['ref_hota']:.6f}" if r['ref_hota'] == r['ref_hota'] else 'nan'
        print(
            f"  {r['dataset']}: pruning={r['pruning_ratio']:.4f}  heuristic_HOTA={r['heuristic_hota']:.6f}  "
            f'ref_HOTA={ref_s}  loss%={loss_s}',
        )
    mx = summary.max_loss_percent
    mn = summary.mean_loss_percent
    print(f"  MAX loss %: {mx:.4f}" if mx == mx else '  MAX loss %: nan')
    print(f"  MEAN loss %: {mn:.4f}" if mn == mn else '  MEAN loss %: nan')
    print(f'  macros: {macro_path}')
    print('---\n')


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


def _save_chart_export(chart: alt.Chart, output_dir: Path, file_stem: str) -> list[Path]:
    """Write PNG (scaled for print), vector PDF, and self-contained HTML."""
    written: list[Path] = []
    for ext, kwargs in (
        ('.png', {'scale_factor': 4.0}),
        ('.pdf', {}),
        ('.html', {}),
    ):
        out_path = output_dir / f'{file_stem}{ext}'
        chart.save(str(out_path), **kwargs)
        written.append(out_path)
    return written


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
      5–6. Whole-frame baseline (firebrick diamonds + rate labels): currently
         commented out in code; search for whole_frame_layer to restore.
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'
    x_enc = alt.X(f'{pair.x_col}:Q', title=pair.x_label)
    y_enc = alt.Y(
        f'{pair.y_col}:Q',
        title=pair.y_label,
        scale=alt.Scale(zero=pair.zero_y_axis),
    )

    exhaustive_tooltip = [
        alt.Tooltip('sequence_label:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('grid_key:N', title='Grid'),
    ]
    heuristic_tooltip = [
        alt.Tooltip('sequence_label:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('heuristic_threshold:Q', title='Threshold'),
        alt.Tooltip('grid_key:N', title='Grid'),
    ]
    # whole_frame_tooltip = [
    #     alt.Tooltip('dataset:N', title='Dataset'),
    #     alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
    #     alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
    #     alt.Tooltip('grid_key:N', title='Frame rate'),
    # ]

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

    # Layer 5: whole-frame baseline diamonds (firebrick).
    # whole_frame_layer = (
    #     base
    #     .transform_filter("datum.method === 'whole_frame'")
    #     .mark_point(size=80, shape='diamond', filled=True, color='firebrick', stroke='white', strokeWidth=0.5)
    #     .encode(x=x_enc, y=y_enc, tooltip=whole_frame_tooltip)
    # )

    # Layer 6: whole-frame rate labels via Vega expression.
    # split(grid_key, '_')[2] extracts the numeric rate from 'whole_frame_{rate}'.
    # whole_frame_text = (
    #     base
    #     .transform_filter("datum.method === 'whole_frame'")
    #     .transform_calculate(
    #         label="datum.grid_key === 'whole_frame_1' ? 'no skip' : '1/' + split(datum.grid_key, '_')[2]",
    #     )
    #     .mark_text(dx=6, dy=-8, fontSize=9, color='firebrick')
    #     .encode(x=x_enc, y=y_enc, text='label:N')
    # )

    foreground_layers = [
        frontier_line,
        frontier_pts,
        heuristic_layer,
        # whole_frame_layer,
        # whole_frame_text,
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
    plot_df = _ensure_sequence_label(results_df.dropna(subset=[pair.x_col, pair.y_col]).copy())
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
        title = (
            f'{_sequence_label(dataset)} {_tracker_display_label(tracker_name)}: '
            f'{pair.x_label} vs {pair.y_label}'
        )
        for slug_suffix, include_background in variants:
            chart = _make_single_chart(
                results_df,
                pair,
                title,
                include_background=include_background,
            )
            stem = f'{pair.slug}{slug_suffix}'
            written_paths.extend(_save_chart_export(chart, output_dir, stem))

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

    Facet column ``facet_dataset`` mirrors human-readable ``sequence_label``
    (Amsterdam, CalDoT 1/2, B3D1). Row filtering uses Vega-Lite transform_filter
    inside _build_chart_layers.

    Passing include_background=False omits the gray exhaustive scatter to
    produce the "Pareto + heuristic + whole-frame" variant.
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'
    prepared = _prepare_combined_chart_df(combined_df)
    # Sort by (facet, x) so the Pareto frontier line renders left-to-right.
    plot_df = (
        prepared
        .dropna(subset=[pair.x_col, pair.y_col])
        .sort_values(['facet_dataset', pair.x_col])
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
                'facet_dataset:N',
                title=None,
                sort=list(prepared['facet_dataset'].cat.categories),
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
        .properties(
            title=f'{_tracker_display_label(tracker_name)}: '
            f'{pair.x_label} vs {pair.y_label}',
        )
        # Collapse inter-facet spacing so subplots sit flush against each
        # other rather than being separated by Altair's default gap.
        .configure_facet(spacing=0)
    )


def combine_visualize(
    tracker_name: str,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Combine per-dataset results and write Altair summary charts to SUMMARY.

    Charts include only ``ams-y05``/``ams-y-05``, ``caldot1-y05``,
    ``caldot2-y05``, and ``jnc0``; other cached datasets are ignored here.

    For each of the three metric pairs (mistrack/HOTA, mistrack/retention,
    retention/HOTA) the function writes .png, .pdf, and .html under
    {cache_dir}/SUMMARY/ablation/mistrack-rate/{tracker_name}/.

    Returns the list of written paths, or an empty list when no cached
    results exist for *tracker_name*.
    """
    from polyis.io import cache as _polyis_cache

    # Load all available per-dataset results for this tracker; summary charts
    # only include ams-y05, caldot1-y05, caldot2-y05, and jnc0.
    combined_df = _collect_results(tracker_name, cache_dir=cache_dir)
    if combined_df.empty:
        return []
    combined_df = combined_df[combined_df['dataset'].isin(_SUMMARY_DATASET_RAW)].copy()
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
            stem = f'{pair.slug}{slug_suffix}'
            written_paths.extend(_save_chart_export(chart, output_dir, stem))

    loss_summary = compute_heuristic_hota_loss_summary(combined_df, tracker_name)
    if loss_summary is not None:
        macro_path = write_heuristic_hota_loss_macros(loss_summary)
        _print_heuristic_hota_loss_summary(loss_summary, macro_path)
    else:
        print(
            'mistrack-rate combine_visualize: skipped heuristic HOTA-loss stats '
            '(no heuristic rows in combined results).',
            flush=True,
        )

    return written_paths

