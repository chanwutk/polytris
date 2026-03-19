#!/usr/local/bin/python

import argparse
import itertools
import json
import multiprocessing as mp
import os

import altair as alt
import numpy as np
import pandas as pd

from polyis.io import cache, store
from polyis.utilities import build_param_str, get_config


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
CLASSIFIERS: list[str] = config['EXEC']['CLASSIFIERS']
DATASETS: list[str] = config['EXEC']['DATASETS']
SAMPLE_RATES: list[int] = config['EXEC']['SAMPLE_RATES']
TRACKERS: list[str] = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS: list[float] = [
    t for t in config['EXEC']['TRACKING_ACCURACY_THRESHOLDS'] if t is not None
]

# ILP solver time limits used by p022t_exec_prune_timelimit.sh.
TIME_LIMITS: list[float] = [0.01, 0.05, 0.1, 0.5, 1.0]

# Operations tracked by p022_exec_prune_polyominoes.py.
OPERATIONS = ['prepare_bitmaps', 'group_tiles', 'build_ilp', 'solve_ilp', 'extract_grids']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize p022 results across ILP solver time limits'
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    return parser.parse_args()


def load_runtime(path: str) -> tuple[dict[str, float], float | None]:
    # Parse runtime.jsonl: {"runtime": [...], "time_limit": <float>}
    with open(path) as f:
        data = json.loads(f.readline())
    ops = {item['op']: item['time'] for item in data['runtime']}
    # time_limit is stored alongside the runtime list.
    tl = data.get('time_limit')
    return ops, tl


def count_active_tiles(score_path: str, threshold: int) -> int:
    # Count tiles with value >= threshold across all frames in a score.jsonl file.
    total = 0
    with open(score_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            flat = np.frombuffer(bytes.fromhex(entry['classification_hex']), dtype=np.uint8)
            total += int((flat >= threshold).sum())
    return total


def collect_row(args: tuple) -> dict | None:
    # Collect runtime and retention data for one (param_combo, time_limit) pair.
    # Returns None if the output files are missing (run not yet complete).
    dataset, videoset, video, classifier, tile_size, sample_rate, tracker, threshold, time_limit = args

    param_str = build_param_str(
        classifier=classifier, tilesize=tile_size,
        sample_rate=sample_rate, tracker=tracker,
        tracking_accuracy_threshold=threshold,
    )

    # Results live under the dedicated timelimit stage, keyed by tl{value}.
    score_dir = cache.exec(
        dataset, 'pruned-polyominoes-tl', video,
        param_str, f'tl{time_limit}', 'score',
    )
    runtime_path = os.path.join(score_dir, 'runtime.jsonl')
    pruned_score_path = os.path.join(score_dir, 'score.jsonl')
    orig_score_path = str(cache.exec(
        dataset, 'relevancy', video,
        f'{classifier}_{tile_size}_{sample_rate}', 'score', 'score.jsonl',
    ))

    if not (os.path.exists(runtime_path) and os.path.exists(pruned_score_path)
            and os.path.exists(orig_score_path)):
        return None

    runtimes, stored_tl = load_runtime(str(runtime_path))

    # Prefer the time_limit stored in the file; fall back to the directory-encoded value.
    effective_tl = stored_tl if stored_tl is not None else time_limit

    # Pruned output has binary values (0 or 255); count tiles == 255.
    pruned_tiles = count_active_tiles(str(pruned_score_path), threshold=255)
    # Original relevancy output may have 0-255 values; apply same threshold as p022 (>=128).
    orig_tiles = count_active_tiles(str(orig_score_path), threshold=128)
    retention_rate = pruned_tiles / orig_tiles if orig_tiles > 0 else 0.0

    row: dict = {
        'dataset': dataset,
        'videoset': videoset,
        'video': video,
        'classifier': classifier,
        'tile_size': tile_size,
        'sample_rate': sample_rate,
        'tracker': tracker,
        'threshold': threshold,
        'time_limit': effective_tl,
        'retention_rate': retention_rate,
    }
    for op in OPERATIONS:
        row[op] = runtimes.get(op, float('nan'))
    return row


def collect_data(videosets: list[str]) -> pd.DataFrame:
    # Build task list then collect all rows in parallel.
    tasks: list[tuple] = []
    for dataset in DATASETS:
        for videoset in videosets:
            videoset_dir = store.dataset(dataset, videoset)
            if not os.path.exists(videoset_dir):
                continue
            videos = sorted(
                f for f in os.listdir(videoset_dir)
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            )
            for combo in itertools.product(
                videos, CLASSIFIERS, TILE_SIZES, SAMPLE_RATES,
                TRACKERS, TRACKING_ACCURACY_THRESHOLDS, TIME_LIMITS,
            ):
                video, classifier, tile_size, sample_rate, tracker, threshold, tl = combo
                tasks.append((dataset, videoset, video, classifier,
                               tile_size, sample_rate, tracker, threshold, tl))

    print(f"Scanning {len(tasks)} (param_combo × time_limit) combinations...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(collect_row, tasks)

    rows = [r for r in results if r is not None]
    print(f"Collected {len(rows)} rows ({len(tasks) - len(rows)} missing/skipped)")
    return pd.DataFrame(rows)


def make_config_label(row: pd.Series) -> str:
    # Abbreviated parameter set label used on chart axes.
    return (
        f"{row['classifier'][:5]} ts{row['tile_size']} "
        f"sr{row['sample_rate']} {row['tracker'][:5]} {row['threshold']:.0%}"
    )


def create_line_chart_runtime(df: pd.DataFrame) -> alt.Chart:
    """
    Line chart: X = total runtime (sum of all operations), Y = mean retention rate.
    One line per parameter configuration, faceted by dataset.
    Points are overlaid on each line to show individual time-limit values.
    """
    plot_df = df.dropna(subset=['retention_rate']).copy()
    plot_df['config'] = plot_df.apply(make_config_label, axis=1)

    # Compute per-row total runtime as sum of all pipeline operations.
    plot_df['total_runtime'] = plot_df[OPERATIONS].sum(axis=1)

    # Aggregate across videos: mean retention and total runtime per (dataset, config, time_limit).
    agg = plot_df.groupby(['dataset', 'config', 'time_limit'], as_index=False).agg(
        retention_rate=('retention_rate', 'mean'),
        total_runtime=('total_runtime', 'sum'),
    )

    base = alt.Chart(agg)

    # Line connecting time-limit values for each configuration.
    lines = base.mark_line(opacity=0.7).encode(
        x=alt.X('total_runtime:Q', title='Total Runtime (ms)'),
        y=alt.Y('retention_rate:Q',
                title='Mean Retention Rate',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('config:N', title='Parameter Config'),
        tooltip=[
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('time_limit:Q', title='Time Limit (s)'),
            alt.Tooltip('total_runtime:Q', format='.1f', title='Total Runtime (ms)'),
            alt.Tooltip('retention_rate:Q', format='.4f', title='Mean Retention Rate'),
        ],
    )

    # Points overlaid on each line to show actual measured values.
    points = base.mark_circle(size=60, opacity=0.9).encode(
        x=alt.X('total_runtime:Q'),
        y=alt.Y('retention_rate:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('config:N'),
        tooltip=[
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('time_limit:Q', title='Time Limit (s)'),
            alt.Tooltip('total_runtime:Q', format='.1f', title='Total Runtime (ms)'),
            alt.Tooltip('retention_rate:Q', format='.4f', title='Mean Retention Rate'),
        ],
    )

    chart = (lines + points).facet(
        row=alt.Row('dataset:N', title='Dataset'),
    ).resolve_scale(
        y='independent',
    ).properties(
        title='Retention Rate vs Total Runtime',
    )
    return chart


def create_line_chart_retention(df: pd.DataFrame) -> alt.Chart:
    """
    Line chart: X = total runtime (sum of all operations), Y = mean retention rate.
    One line per parameter configuration, faceted by dataset.
    Points are overlaid on each line to show individual time-limit values.
    """
    plot_df = df.dropna(subset=['retention_rate']).copy()
    plot_df['config'] = plot_df.apply(make_config_label, axis=1)

    # Compute per-row total runtime as sum of all pipeline operations.
    plot_df['total_runtime'] = plot_df[OPERATIONS].sum(axis=1)

    # Aggregate across videos: mean retention and total runtime per (dataset, config, time_limit).
    agg = plot_df.groupby(['dataset', 'config', 'time_limit'], as_index=False).agg(
        retention_rate=('retention_rate', 'mean'),
        total_runtime=('total_runtime', 'sum'),
    )

    base = alt.Chart(agg)

    lines = base.mark_line(opacity=0.7).encode(
        x=alt.X('total_runtime:Q', title='Total Runtime (ms)'),
        y=alt.Y('retention_rate:Q',
                title='Mean Retention Rate',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('config:N', title='Parameter Config'),
        tooltip=[
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('time_limit:Q', title='Time Limit (s)'),
            alt.Tooltip('total_runtime:Q', format='.1f', title='Total Runtime (ms)'),
            alt.Tooltip('retention_rate:Q', format='.4f', title='Mean Retention Rate'),
        ],
    )

    points = base.mark_circle(size=60, opacity=0.9).encode(
        x=alt.X('total_runtime:Q'),
        y=alt.Y('retention_rate:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('config:N'),
        tooltip=[
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('time_limit:Q', title='Time Limit (s)'),
            alt.Tooltip('total_runtime:Q', format='.1f', title='Total Runtime (ms)'),
            alt.Tooltip('retention_rate:Q', format='.4f', title='Mean Retention Rate'),
        ],
    )

    chart = (lines + points).facet(
        row=alt.Row('dataset:N', title='Dataset'),
    ).resolve_scale(
        y='independent',
        x='independent',
    ).properties(
        title='Retention Rate vs Total Runtime',
    )
    return chart


def create_bar_breakdown_chart(df: pd.DataFrame, dataset: str) -> alt.Chart:
    """
    Stacked bar chart: one bar per time_limit, stacked by operation.
    Shows how total runtime composition changes as the solver budget grows.
    Bars are summed across all videos and parameter configurations for the dataset.
    """
    subset = df[df['dataset'] == dataset].copy()

    # Reshape to long format for stacking.
    melted = subset.melt(
        id_vars=['time_limit'],
        value_vars=OPERATIONS,
        var_name='operation',
        value_name='runtime_ms',
    )

    # Sum across all videos and configs for each (time_limit, operation).
    agg = melted.groupby(['time_limit', 'operation'], as_index=False)['runtime_ms'].sum()

    # Sort bars by ascending time limit.
    tl_order = sorted(agg['time_limit'].unique())

    return alt.Chart(agg).mark_bar().encode(
        x=alt.X('time_limit:O',
                sort=[str(t) for t in tl_order],
                title='Solver Time Limit (s)',
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('runtime_ms:Q', title='Total Runtime (ms)'),
        color=alt.Color('operation:N', title='Operation',
                        legend=alt.Legend(orient='bottom')),
        tooltip=[
            alt.Tooltip('time_limit:O', title='Time Limit (s)'),
            alt.Tooltip('operation:N', title='Operation'),
            alt.Tooltip('runtime_ms:Q', format='.1f', title='Total Runtime (ms)'),
        ],
    ).properties(
        title=f'Runtime Breakdown by Time Limit — {dataset}',
        width=max(80 * len(tl_order), 400),
        height=400,
    )


def main():
    args = parse_args()

    videosets: list[str] = []
    if args.test:
        videosets.append('test')
    if args.valid:
        videosets.append('valid')
    if not videosets:
        videosets = ['valid']

    df = collect_data(videosets)
    if df.empty:
        print("No data found. Run scripts/p022t_exec_prune_timelimit.sh first.")
        return

    # Output directory for all charts from this evaluation.
    out_dir = str(cache.summary('027_prune_timelimit'))
    os.makedirs(out_dir, exist_ok=True)

    # Line chart: retention rate vs total runtime.
    chart = create_line_chart_runtime(df)
    out_path = os.path.join(out_dir, 'line_runtime.html')
    chart.save(out_path)
    print(f"Saved: {out_path}")

    # Line chart: retention rate vs time_limit.
    chart = create_line_chart_retention(df)
    out_path = os.path.join(out_dir, 'line_retention_rate.html')
    chart.save(out_path)
    print(f"Saved: {out_path}")

    # Stacked bar breakdown per dataset.
    for dataset in df['dataset'].unique():
        chart = create_bar_breakdown_chart(df, dataset)
        dataset_dir = str(cache.eval(dataset, 'prune-breakdown'))
        os.makedirs(dataset_dir, exist_ok=True)
        out_path = os.path.join(dataset_dir, 'runtime_breakdown_by_timelimit.html')
        chart.save(out_path)
        print(f"Saved: {out_path}")

    print("All visualizations complete!")


if __name__ == '__main__':
    main()
