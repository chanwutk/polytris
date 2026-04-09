from __future__ import annotations

import glob as _glob
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


PAIR_DEFINITIONS = (
    PairDefinition(
        slug='mistrack_vs_hota',
        x_col='mistrack_rate',
        y_col='HOTA_HOTA',
        x_label='Mistrack Rate',
        y_label='HOTA',
        x_minimize=True,
        y_maximize=True,
    ),
    PairDefinition(
        slug='mistrack_vs_retention',
        x_col='mistrack_rate',
        y_col='retention_rate',
        x_label='Mistrack Rate',
        y_label='Retention Rate',
        x_minimize=True,
        y_maximize=False,
    ),
    PairDefinition(
        slug='retention_vs_hota',
        x_col='retention_rate',
        y_col='HOTA_HOTA',
        x_label='Retention Rate',
        y_label='HOTA',
        x_minimize=True,
        y_maximize=True,
    ),
)


def annotate_pareto_flags(results_df: pd.DataFrame) -> pd.DataFrame:
    flagged_df = results_df.copy()
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


def _plot_pair(results_df: pd.DataFrame, pair: PairDefinition, output_path: Path, title: str) -> None:
    exhaustive_df = results_df[
        (results_df['method'] == 'exhaustive')
        & results_df[pair.x_col].notna()
        & results_df[pair.y_col].notna()
    ].copy()
    heuristic_df = results_df[
        (results_df['method'] == 'heuristic')
        & results_df[pair.x_col].notna()
        & results_df[pair.y_col].notna()
    ].copy()
    frontier_df = exhaustive_df[exhaustive_df[f'is_pareto_{pair.slug}']].copy()

    figure, axis = plt.subplots(figsize=(8, 6))

    if not exhaustive_df.empty:
        axis.scatter(
            exhaustive_df[pair.x_col],
            exhaustive_df[pair.y_col],
            s=14,
            alpha=0.45,
            color='lightgray',
            label='Exhaustive points',
        )

    if not frontier_df.empty:
        frontier_df = frontier_df.sort_values(pair.x_col)
        axis.plot(
            frontier_df[pair.x_col],
            frontier_df[pair.y_col],
            linewidth=2.0,
            color='black',
            label='Exhaustive Pareto frontier',
        )
        axis.scatter(
            frontier_df[pair.x_col],
            frontier_df[pair.y_col],
            s=24,
            color='black',
        )

    if not heuristic_df.empty:
        axis.scatter(
            heuristic_df[pair.x_col],
            heuristic_df[pair.y_col],
            s=40,
            color='tab:blue',
            edgecolors='white',
            linewidths=0.4,
            label='Heuristic points',
        )

    # Overlay whole-frame sampling baselines (entire frames kept or dropped at rate 1/2/4).
    whole_frame_df = results_df[
        (results_df['method'] == 'whole_frame')
        & results_df[pair.x_col].notna()
        & results_df[pair.y_col].notna()
    ].copy()
    if not whole_frame_df.empty:
        # Extract numeric rate from grid_key 'whole_frame_{rate}'.
        whole_frame_df['_rate'] = (
            whole_frame_df['grid_key'].str.removeprefix('whole_frame_').astype(int)
        )
        axis.scatter(
            whole_frame_df[pair.x_col],
            whole_frame_df[pair.y_col],
            s=80,
            color='tab:red',
            marker='D',
            edgecolors='white',
            linewidths=0.5,
            zorder=5,
            label='Whole-frame baseline',
        )
        for _, row in whole_frame_df.iterrows():
            label = 'no skip' if row['_rate'] == 1 else f'1/{row["_rate"]} frames'
            axis.annotate(
                label,
                xy=(row[pair.x_col], row[pair.y_col]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=7,
                color='tab:red',
            )

    axis.set_xlabel(pair.x_label)
    axis.set_ylabel(pair.y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_results(dataset: str, tracker_name: str) -> list[Path]:
    results_path = results_csv_path(dataset, tracker_name)
    if not results_path.exists():
        raise FileNotFoundError(f'Results CSV not found: {results_path}')

    results_df = pd.read_csv(results_path)
    output_dir = ensure_dir(plots_dir(dataset, tracker_name))
    written_paths: list[Path] = []

    for pair in PAIR_DEFINITIONS:
        output_path = output_dir / f'{pair.slug}.png'
        title = f'{dataset} {tracker_name}: {pair.x_label} vs {pair.y_label}'
        _plot_pair(results_df, pair, output_path, title)
        written_paths.append(output_path)

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

    # Load each CSV and concatenate into a single flat DataFrame.
    frames = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def _make_altair_chart(
    combined_df: pd.DataFrame,
    pair: PairDefinition,
    tracker_name: str,
) -> alt.FacetChart:
    """Build a faceted Altair scatter chart for one metric pair.

    One subplot per dataset. Layers within each facet (back to front):
      1. All exhaustive points — small, semi-transparent, light gray.
      2. Pareto frontier connecting line — black.
      3. Pareto frontier points — black, slightly larger.
      4. Heuristic points — blue diamond markers.

    All layers share a single DataFrame so that Altair's facet requirement
    (uniform data source across layers) is satisfied. Row filtering is done
    via Vega-Lite transform_filter expressions.
    """
    pareto_flag_col = f'is_pareto_{pair.slug}'

    # Build a single plot DataFrame: drop rows with missing metrics, sort by
    # (dataset, x) so the Pareto frontier line is drawn left-to-right, and
    # normalise the Pareto flag column to a bool (False when absent).
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

    # All layers reference the same base chart so faceting is valid.
    base = alt.Chart(plot_df)

    x_enc = alt.X(f'{pair.x_col}:Q', title=pair.x_label)
    y_enc = alt.Y(f'{pair.y_col}:Q', title=pair.y_label)

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

    # Layer 1: exhaustive background scatter (light gray, semi-transparent).
    bg_layer = base.transform_filter(
        "datum.method === 'exhaustive'",
    ).mark_point(
        size=14, opacity=0.35, filled=True, color='lightgray',
    ).encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)

    # Layers 2 & 3 share the Pareto frontier filter expression.
    pareto_filter = f"datum.method === 'exhaustive' && datum.{pareto_flag_col}"

    # Layer 2: Pareto frontier connecting line.
    frontier_line = base.transform_filter(pareto_filter).mark_line(
        strokeWidth=2.0, color='black',
    ).encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)

    # Layer 3: Pareto frontier point markers.
    frontier_pts = base.transform_filter(pareto_filter).mark_point(
        size=32, filled=True, color='black',
    ).encode(x=x_enc, y=y_enc, tooltip=exhaustive_tooltip)

    # Layer 4: heuristic points as blue diamond markers.
    heuristic_layer = base.transform_filter(
        "datum.method === 'heuristic'",
    ).mark_point(
        size=80, shape='diamond', filled=True, color='steelblue',
        stroke='white', strokeWidth=0.5,
    ).encode(x=x_enc, y=y_enc, tooltip=heuristic_tooltip)

    # Layer 5: whole-frame baseline points as firebrick diamonds.
    whole_frame_tooltip = [
        alt.Tooltip('dataset:N', title='Dataset'),
        alt.Tooltip(f'{pair.x_col}:Q', title=pair.x_label, format='.4f'),
        alt.Tooltip(f'{pair.y_col}:Q', title=pair.y_label, format='.4f'),
        alt.Tooltip('grid_key:N', title='Frame rate'),
    ]
    whole_frame_layer = base.transform_filter(
        "datum.method === 'whole_frame'",
    ).mark_point(
        size=80, shape='diamond', filled=True, color='firebrick',
        stroke='white', strokeWidth=0.5,
    ).encode(x=x_enc, y=y_enc, tooltip=whole_frame_tooltip)

    # Facet by dataset — one subplot per dataset, up to 3 columns per row.
    # Independent x-axes let each panel use its own domain rather than the
    # global range, which avoids compressing sparse datasets.
    chart = (
        alt.layer(bg_layer, frontier_line, frontier_pts, heuristic_layer, whole_frame_layer)
        .properties(width=300, height=250)
        .facet(
            facet=alt.Facet('dataset:N', title='Dataset'),
            columns=3,
        )
        .resolve_scale(x='independent')
        .properties(title=f'{tracker_name}: {pair.x_label} vs {pair.y_label}')
    )

    return chart


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

