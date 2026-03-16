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
from polyis.pareto import build_pareto_combo_filter
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

# Operations tracked by p022_exec_prune_polyominoes.py
OPERATIONS = ['prepare_bitmaps', 'group_tiles', 'build_ilp', 'solve_ilp', 'extract_grids']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    return parser.parse_args()


def load_runtime(path: str) -> dict[str, float]:
    # Parse runtime.jsonl: {"runtime": [{"op": "...", "time": ms}, ...]}
    with open(path) as f:
        data = json.loads(f.readline())
    return {item['op']: item['time'] for item in data['runtime']}


def count_active_tiles(score_path: str, threshold: int) -> int:
    # Count tiles with value >= threshold across all frames in a score.jsonl file
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
    # Collect runtime and retention data for one parameter combination; returns None if files missing
    dataset, videoset, video, classifier, tile_size, sample_rate, tracker, threshold = args

    param_str = build_param_str(
        classifier=classifier, tilesize=tile_size,
        sample_rate=sample_rate, tracker=tracker,
        tracking_accuracy_threshold=threshold,
    )

    runtime_path = cache.exec(dataset, 'pruned-polyominoes', video, param_str, 'score', 'runtime.jsonl')
    pruned_score_path = cache.exec(dataset, 'pruned-polyominoes', video, param_str, 'score', 'score.jsonl')
    orig_score_path = cache.exec(
        dataset, 'relevancy', video,
        f'{classifier}_{tile_size}_{sample_rate}', 'score', 'score.jsonl'
    )

    if not (os.path.exists(runtime_path) and os.path.exists(pruned_score_path)
            and os.path.exists(orig_score_path)):
        return None

    runtimes = load_runtime(str(runtime_path))

    # Pruned output has binary values (0 or 255); count tiles == 255
    pruned_tiles = count_active_tiles(str(pruned_score_path), threshold=255)
    # Original relevancy output may have 0-255 values; apply same threshold as p022 (>=128)
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
        'retention_rate': retention_rate,
    }
    for op in OPERATIONS:
        row[op] = runtimes.get(op, float('nan'))
    return row


def collect_data(videosets: list[str]) -> pd.DataFrame:
    # Build task list then collect all rows in parallel
    # For the test videoset, restrict to Pareto-optimal parameter combinations only.
    pareto_filter = build_pareto_combo_filter(
        DATASETS, videosets,
        ['classifier', 'tilesize', 'sample_rate', 'tracker', 'tracking_accuracy_threshold'],
    )
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
                videos, CLASSIFIERS, TILE_SIZES, SAMPLE_RATES, TRACKERS, TRACKING_ACCURACY_THRESHOLDS
            ):
                video, classifier, tile_size, sample_rate, tracker, threshold = combo
                # Skip non-Pareto combinations when processing the test split.
                if videoset == 'test' and pareto_filter is not None:
                    if (classifier, tile_size, sample_rate, tracker, threshold) not in pareto_filter[dataset]:
                        continue
                tasks.append((dataset, videoset, video, classifier, tile_size, sample_rate, tracker, threshold))

    print(f"Scanning {len(tasks)} parameter combinations...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(collect_row, tasks)

    rows = [r for r in results if r is not None]
    print(f"Collected {len(rows)} rows ({len(tasks) - len(rows)} missing/skipped)")
    return pd.DataFrame(rows)


def make_config_label(row: pd.Series) -> str:
    # Abbreviated parameter set label used on chart axes
    return (
        f"{row['classifier'][:5]} {row['tile_size']} "
        f"SR{row['sample_rate']} {row['tracker'][:5]} {row['threshold']:.0%}"
    )


def create_runtime_breakdown_chart(df: pd.DataFrame, dataset: str) -> alt.Chart:
    # Stacked bar chart of mean operation runtimes per parameter set for one dataset
    subset = df[df['dataset'] == dataset].copy()
    subset['config'] = subset.apply(make_config_label, axis=1)

    # Reshape to long format: one row per (video, config, operation)
    melted = subset.melt(
        id_vars=['config'],
        value_vars=OPERATIONS,
        var_name='operation',
        value_name='runtime_ms',
    )

    # Sum across videos for each (config, operation) pair
    agg = melted.groupby(['config', 'operation'], as_index=False)['runtime_ms'].sum()

    # Sort configs by descending total mean runtime
    totals = agg.groupby('config')['runtime_ms'].sum().reset_index()
    config_order = totals.sort_values('runtime_ms', ascending=False)['config'].tolist()

    return alt.Chart(agg).mark_bar().encode(
        x=alt.X('config:N', sort=config_order, title='Parameter Set',
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('runtime_ms:Q', title='Mean Runtime (ms)'),
        color=alt.Color('operation:N', title='Operation',
                        legend=alt.Legend(orient='bottom')),
        tooltip=[
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('operation:N', title='Operation'),
            alt.Tooltip('runtime_ms:Q', format='.1f', title='Mean Runtime (ms)'),
        ],
    ).properties(
        title=f'Runtime Breakdown — {dataset}',
        width=max(60 * len(config_order), 400),
        height=400,
    )


def create_scatter_chart(df: pd.DataFrame, operation: str) -> alt.Chart:
    # Scatter plot of operation runtime vs retention rate, one panel per dataset.
    # Aggregate across videos: sum runtime, average retention rate.
    plot_df = df.dropna(subset=[operation, 'retention_rate']).copy()
    plot_df['config'] = plot_df.apply(make_config_label, axis=1)

    agg = plot_df.groupby(['dataset', 'config', 'threshold'], as_index=False).agg(
        **{operation: (operation, 'sum'), 'retention_rate': ('retention_rate', 'mean')}
    )

    base = alt.Chart(agg).mark_circle(opacity=0.6, size=60).encode(
        x=alt.X(f'{operation}:Q', title=f'{operation} total runtime (ms)'),
        y=alt.Y('retention_rate:Q', title='Mean Retention Rate',
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('threshold:Q', title='Accuracy Threshold',
                        scale=alt.Scale(scheme='viridis')),
        tooltip=[
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('config:N', title='Config'),
            alt.Tooltip('threshold:Q', format='.0%', title='Threshold'),
            alt.Tooltip('retention_rate:Q', format='.3f', title='Mean Retention Rate'),
            alt.Tooltip(f'{operation}:Q', format='.1f', title='Total Runtime (ms)'),
        ],
    ).properties(width=400, height=300)

    # Facet by dataset with independent x scale per panel
    return base.facet(
        row=alt.Row('dataset:N', title='Dataset'),
    ).resolve_scale(
        x='independent',
    ).properties(
        title=f'Runtime vs Retention Rate: {operation}',
    )


def main():
    args = parse_args()

    videosets: list[str] = []
    if args.test:
        videosets.append('test')
    if args.train:
        videosets.append('train')
    if args.valid:
        videosets.append('valid')
    if not videosets:
        videosets = ['valid']

    df = collect_data(videosets)
    if df.empty:
        print("No data found. Ensure p022 has been run first.")
        return

    # Bar chart per dataset: stacked by operation, sorted by total runtime
    for dataset in df['dataset'].unique():
        output_dir = str(cache.eval(dataset, 'prune-breakdown'))
        os.makedirs(output_dir, exist_ok=True)
        chart = create_runtime_breakdown_chart(df, dataset)
        out_path = os.path.join(output_dir, 'runtime_breakdown.html')
        chart.save(out_path)
        print(f"Saved bar chart: {out_path}")

    # Scatter plot per operation: X=operation runtime, Y=retention rate, faceted by dataset
    scatter_dir = str(cache.summary('026_prune_breakdown'))
    os.makedirs(scatter_dir, exist_ok=True)
    for op in OPERATIONS:
        chart = create_scatter_chart(df, op)
        out_path = os.path.join(scatter_dir, f'scatter_{op}.html')
        chart.save(out_path)
        print(f"Saved scatter plot: {out_path}")

    print("All visualizations complete!")


if __name__ == '__main__':
    main()
