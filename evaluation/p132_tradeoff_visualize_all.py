#!/usr/local/bin/python

import os

import altair as alt

from polyis.io import cache
from polyis.utilities import (
    METRICS,
    get_config,
    load_all_datasets_tradeoff_data,
    print_best_data_points,
    split_tradeoff_variants,
    tradeoff_scatter_and_naive_baseline,
)


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


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


def prepare_plot_df(datasets: list[str]):
    # Load the canonical split-level tradeoff rows for all configured datasets.
    tradeoff_df = load_all_datasets_tradeoff_data(datasets, system_name=None)
    # Split the canonical table into Polytris and naive subsets.
    polytris_df, naive_df = split_tradeoff_variants(tradeoff_df)
    # Fail fast when any expected subset is missing.
    assert not polytris_df.empty, "No Polytris tradeoff rows found"
    assert not naive_df.empty, "No naive baseline rows found"

    return polytris_df, naive_df


def visualize_all_datasets_tradeoff(polytris_df, naive_df, x_column: str, x_title: str, plot_suffix: str, output_dir: str):
    # Keep only the split-aware naive columns required for the baseline merge.
    naive_cols = ['dataset', 'videoset', x_column]
    naive_merge_df = naive_df[naive_cols].rename(columns={x_column: f'{x_column}_naive'})

    # Attach the split-specific naive baseline to every Polytris row.
    plot_df = polytris_df.merge(naive_merge_df, on=['dataset', 'videoset'], how='left')
    # Reuse the shared scatter helper by exposing the split label through the legacy video field.
    plot_df['video'] = plot_df['videoset']
    # Create a combined facet label so valid and test stay separate across datasets.
    plot_df['dataset_split'] = plot_df['dataset'].astype(str) + ' / ' + plot_df['videoset'].astype(str)

    # Print the split-aware “best point” summary before plotting.
    print_best_data_points(plot_df.assign(dataset=plot_df['dataset_split']), METRICS, x_column, plot_suffix)

    # Build the base Altair chart from the merged split-aware tradeoff rows.
    base_chart = alt.Chart(plot_df)

    # Render one chart per supported metric family.
    for metric in METRICS:
        metric_info = build_metric_columns(metric)
        if metric_info is None:
            continue
        accuracy_col, metric_name = metric_info

        # Skip metrics that are absent across the loaded datasets.
        if accuracy_col not in plot_df.columns or plot_df[accuracy_col].isna().all():
            continue

        # Reuse the shared tradeoff scatter helper for Polytris rows and naive baselines.
        scatter, baseline = tradeoff_scatter_and_naive_baseline(
            base_chart,
            x_column,
            x_title,
            accuracy_col,
            metric_name,
            size=20,
            shape_field='videoset',
        )

        # Facet by the combined dataset/split label so both splits remain visible.
        chart = (scatter + baseline).facet(
            facet=alt.Facet('dataset_split:N', title=None),
            columns=3,
        ).resolve_scale(
            x='independent',
        )

        # Save the rendered plot for the current metric/x-axis pair.
        chart.save(os.path.join(output_dir, f'{metric.lower()}_{plot_suffix}_tradeoff.png'), scale_factor=4)


def visualize_all_datasets_tradeoffs(datasets: list[str]):
    # Resolve the shared all-datasets tradeoff visualization directory.
    output_dir = cache.summary('092_tradeoff_all')
    # Recreate the output directory before writing fresh charts.
    os.makedirs(output_dir, exist_ok=True)

    # Load the canonical tradeoff rows and split them into Polytris/naive subsets.
    polytris_df, naive_df = prepare_plot_df(datasets)

    # Render both runtime and throughput tradeoff charts from the same canonical table.
    visualize_all_datasets_tradeoff(polytris_df, naive_df, 'time', 'Query Execution Runtime (seconds)', 'runtime', str(output_dir))
    visualize_all_datasets_tradeoff(polytris_df, naive_df, 'throughput_fps', 'Throughput (frames/second)', 'throughput', str(output_dir))


def main():
    # Log the configured datasets before the all-datasets visualization stage starts.
    print(f"Processing datasets: {DATASETS}")

    # Render the split-aware all-datasets tradeoff visualizations.
    visualize_all_datasets_tradeoffs(DATASETS)


if __name__ == '__main__':
    main()
