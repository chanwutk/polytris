from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from polyis.pareto import compute_pareto_front

from .common import ensure_dir, plots_dir, results_csv_path


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
            minimize_x=pair.x_minimize,
            maximize_y=True,
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

