#!/usr/local/bin/python

import argparse
import os

import altair as alt
from rich.progress import track

from polyis.io import cache
from polyis.utilities import (
    METRICS,
    get_config,
    load_tradeoff_data,
    split_tradeoff_variants,
    tradeoff_scatter_and_naive_baseline,
)


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def build_metric_columns(metric: str) -> tuple[str, str] | None:
    # Map each supported metric family to the canonical tradeoff column name.
    if metric == 'HOTA':
        return 'HOTA_HOTA', 'HOTA'
    # Map the CLEAR family to the canonical MOTA column when present.
    if metric == 'CLEAR':
        return 'MOTA_MOTA', 'MOTA'
    # Map the Count family to the tracking MAPE column used in existing plots.
    if metric == 'Count':
        return 'Count_TracksMAPE', 'Count'
    return None


def prepare_plot_df(tradeoff_df, x_column: str):
    # Split the canonical tradeoff table into Polytris and naive subsets.
    polytris_df, naive_df = split_tradeoff_variants(tradeoff_df)
    # Fail fast when the canonical tradeoff table does not contain Polytris rows.
    assert not polytris_df.empty, "No Polytris tradeoff rows found"
    # Fail fast when the canonical tradeoff table does not contain naive rows.
    assert not naive_df.empty, "No naive baseline rows found"

    # Keep only the split-aware naive baseline columns needed for the merge.
    naive_cols = ['dataset', 'videoset', x_column]
    naive_merge_df = naive_df[naive_cols].rename(columns={x_column: f'{x_column}_naive'})

    # Attach the split-specific naive baseline to each Polytris row.
    plot_df = polytris_df.merge(naive_merge_df, on=['dataset', 'videoset'], how='left')
    # Reuse the existing scatter utility by exposing the split name through the legacy video field.
    plot_df['video'] = plot_df['videoset']

    return plot_df


def visualize_tradeoff(tradeoff_df, output_dir: str, x_column: str, x_title: str, plot_suffix: str):
    # Prepare the merged split-aware plotting DataFrame for this x-axis.
    plot_df = prepare_plot_df(tradeoff_df, x_column)
    # Build the base Altair chart from the split-aware tradeoff rows.
    base_chart = alt.Chart(plot_df)

    # Render one chart per supported metric family.
    for metric in METRICS:
        metric_info = build_metric_columns(metric)
        if metric_info is None:
            continue
        accuracy_col, metric_name = metric_info

        # Skip metrics that are absent in the current dataset.
        if accuracy_col not in plot_df.columns or plot_df[accuracy_col].isna().all():
            continue

        # Reuse the shared tradeoff scatter helper for Polytris rows and naive baselines.
        scatter, baseline = tradeoff_scatter_and_naive_baseline(
            base_chart,
            x_column,
            x_title,
            accuracy_col,
            metric_name,
        )

        # Facet the split-level view by videoset instead of per-video.
        chart = (scatter + baseline).facet(
            column=alt.Column('videoset:N', title='Videoset'),
        ).resolve_scale(
            x='independent',
        )

        # Save the rendered plot for the current metric/x-axis pair.
        chart.save(os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_tradeoff.png'), scale_factor=2)


def visualize_tradeoffs(dataset: str):
    # Resolve the dataset-local tradeoff visualization directory.
    output_dir = cache.eval(dataset, 'tradeoff-vis')
    # Recreate the dataset-local output directory before rendering charts.
    os.makedirs(output_dir, exist_ok=True)

    # Load the canonical split-level tradeoff table.
    tradeoff_df = load_tradeoff_data(dataset)
    # Render both runtime and throughput visualizations from the same table.
    visualize_tradeoff(tradeoff_df, str(output_dir), 'time', 'Query Execution Runtime (seconds)', 'runtime')
    visualize_tradeoff(tradeoff_df, str(output_dir), 'throughput_fps', 'Throughput (frames/second)', 'throughput')


def main(args):
    # Log the configured datasets before visualization starts.
    print(f"Processing datasets: {DATASETS}")

    # Render split-level tradeoff charts for each configured dataset.
    for dataset in track(DATASETS, description="Processing datasets"):
        visualize_tradeoffs(dataset)


if __name__ == '__main__':
    main(parse_args())
