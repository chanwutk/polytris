#!/usr/local/bin/python

import argparse
import os
from pathlib import Path
import shutil

import altair as alt
import pandas as pd

from polyis.io import cache
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true')
    group.add_argument('--test', action='store_true')
    return parser.parse_args()


def build_threshold_label(value: object) -> str:
    # Render missing thresholds as the explicit no-pruning label.
    if pd.isna(value):
        return 'TA-none'
    # Convert numeric thresholds into the compact percentage label used elsewhere.
    return f"TA{int(round(float(value) * 100)):03d}"


def build_config_labels(results: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the caller keeps the original DataFrame unchanged.
    df = results.copy()

    # Normalize missing sample rates to an explicit NA-friendly string representation.
    df['sample_rate_label'] = df['sample_rate'].apply(lambda value: 'SR-' if pd.isna(value) else f"SR{int(value)}")
    # Normalize missing thresholds into a consistent label.
    df['threshold_label'] = df['tracking_accuracy_threshold'].apply(build_threshold_label)
    # Normalize missing tile padding for the naive baseline.
    df['tilepadding_label'] = df['tilepadding'].fillna('naive')
    # Normalize missing trackers for the naive baseline.
    df['tracker_label'] = df['tracker'].fillna('naive')

    # Use a dedicated label for naive rows instead of fake Polytris params.
    naive_mask = df['variant'] == 'naive'
    df['config_label'] = 'Naive'

    # Build the Polytris configuration label from the real parameter columns.
    df.loc[~naive_mask, 'config_label'] = (
        df.loc[~naive_mask, 'classifier'].fillna('NA').astype(str).str.slice(0, 6)
        + ' '
        + df.loc[~naive_mask, 'sample_rate_label']
        + ' '
        + df.loc[~naive_mask, 'threshold_label']
        + ' '
        + df.loc[~naive_mask, 'tilepadding_label'].astype(str).str.slice(0, 6)
        + ' '
        + df.loc[~naive_mask, 'tracker_label'].astype(str).str.slice(0, 6)
    )

    return df


def visualize_metric(results: pd.DataFrame, metric_name: str, metric_label: str, output_path: Path):
    # Add the human-readable config labels used on the chart axes.
    df = build_config_labels(results)

    # Skip empty metrics early so the visualization stage stays robust.
    if metric_name not in df.columns or df[metric_name].isna().all():
        return

    # Build the horizontal split-level accuracy comparison chart.
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            f'{metric_name}:Q',
            title=metric_label,
            scale=alt.Scale(domain=[0, 1]) if not metric_name.startswith('Count_') else alt.Undefined,
        ),
        y=alt.Y('config_label:N', sort='-x', title='Variant'),
        color=alt.Color('variant:N', title='Variant'),
        tooltip=[
            'dataset',
            'videoset',
            'variant',
            'classifier',
            'sample_rate',
            'tracking_accuracy_threshold',
            'tilepadding',
            'canvas_scale',
            'tracker',
            alt.Tooltip(f'{metric_name}:Q', format='.4f'),
        ],
    ).properties(
        width=320,
        height=180,
    ).facet(
        column=alt.Column('videoset:N', title='Videoset'),
    ).resolve_scale(
        x='independent',
        y='independent',
    )

    # Save the rendered chart for the current metric.
    chart.save(output_path, scale_factor=2)


def visualize_tracking_accuracy(results: pd.DataFrame, output_dir: Path):
    # Define the metric columns rendered by the accuracy visualization stage.
    metrics = [
        ('HOTA_HOTA', 'HOTA Score'),
        ('HOTA_AssA', 'AssA Score'),
        ('HOTA_DetA', 'DetA Score'),
        ('Count_DetsMAPE', 'Dets MAPE (%)'),
        ('Count_TracksMAPE', 'Tracks MAPE (%)'),
    ]

    # Render one chart per metric into the dataset-local output directory.
    for metric_name, metric_label in metrics:
        visualize_metric(results, metric_name, metric_label, output_dir / f'{metric_name}.png')


def main(args):
    # Log the configured datasets before visualization starts.
    print(f"Starting tracking accuracy visualization for datasets: {DATASETS}")

    # Render each dataset independently so the output directories stay isolated.
    for dataset in DATASETS:
        # Resolve the dataset-local accuracy results directory.
        results_dir = cache.eval(dataset, 'acc')
        # Resolve the dataset-local visualization output directory.
        output_dir = cache.eval(dataset, 'acc_vis')

        # Load the canonical split-level accuracy table.
        results = pd.read_csv(results_dir / 'accuracy.csv')
        # Fail fast when the expected split-level CSV is empty.
        assert len(results) > 0, f"No accuracy results found for dataset {dataset}"

        # Remove stale visualization outputs before saving fresh charts.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        # Recreate the visualization output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Render all split-level accuracy charts for the current dataset.
        visualize_tracking_accuracy(results, output_dir)


if __name__ == '__main__':
    main(parse_args())
