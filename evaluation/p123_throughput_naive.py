#!/usr/local/bin/python

import json
import os

import altair as alt
import pandas as pd

from polyis.io import cache
from polyis.utilities import get_config
from evaluation.utilities import ColorScheme


config = get_config()
DATASETS = config['EXEC']['DATASETS']

# Map raw detection operation names to display categories
DETECTION_OP_MAP = {'read': 'Decode', 'detect': 'Detect'}

# Display name for all tracking operations (grouped into a single category)
TRACKING_OP_NAME = 'Track'

# Stacking order for the bar chart (left to right)
OP_ORDER = ['Detect', 'Decode', 'Track']
OP_ORDER_MAP = {op: i for i, op in enumerate(OP_ORDER)}


def load_naive_runtimes(dataset: str) -> pd.DataFrame:
    """Load naive detection and tracking runtimes for all videos in a dataset."""
    execution_dir = cache.execution(dataset)
    records: list[dict] = []

    # Skip datasets without an execution directory
    if not os.path.exists(execution_dir):
        return pd.DataFrame(columns=['dataset', 'video', 'op', 'time'])

    # Iterate over video directories that contain naive runtime files
    for video_name in sorted(os.listdir(execution_dir)):
        det_runtime_path = cache.exec(dataset, 'naive', video_name, 'detection_runtime.jsonl')
        track_runtime_path = cache.exec(dataset, 'naive', video_name, 'tracking_runtime.jsonl')

        # Skip videos missing either detection or tracking runtimes
        if not os.path.exists(det_runtime_path) or not os.path.exists(track_runtime_path):
            continue

        # Parse detection runtimes (each JSONL line is a list of {op, time} dicts in ms)
        with open(det_runtime_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line is a JSON array of operation timing entries
                for entry in json.loads(line):
                    display_op = DETECTION_OP_MAP.get(entry['op'], entry['op'])
                    records.append({
                        'dataset': dataset,
                        'video': video_name,
                        'op': display_op,
                        'time': entry['time'] / 1000.0,  # Convert ms to seconds
                    })

        # Parse tracking runtimes (each JSONL line has a 'runtime' key with timing entries in ms)
        with open(track_runtime_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line is a JSON object with 'runtime' containing operation timings
                frame_data = json.loads(line)
                for entry in frame_data['runtime']:
                    records.append({
                        'dataset': dataset,
                        'video': video_name,
                        'op': TRACKING_OP_NAME,
                        'time': entry['time'] / 1000.0,  # Convert ms to seconds
                    })

    return pd.DataFrame.from_records(records)


def main():
    """Visualize normalized runtime breakdown of naive detection and tracking."""

    # Load and concatenate naive runtimes across all configured datasets
    all_runtimes: list[pd.DataFrame] = []
    for dataset in DATASETS:
        df = load_naive_runtimes(dataset)
        if len(df) > 0:
            all_runtimes.append(df)
    assert len(all_runtimes) > 0, "No naive runtime data found for any dataset"
    runtimes = pd.concat(all_runtimes, ignore_index=True)

    # Aggregate total time per (dataset, op) across all videos
    agg = (
        runtimes
        .groupby(['dataset', 'op'])
        .agg(time=pd.NamedAgg(column='time', aggfunc='sum'))
        .reset_index()
    )

    # Shorten dataset names by removing anything after the first '-'
    agg['dataset'] = agg['dataset'].str.split('-').str[0]

    # Rename 'ams' to 'amsterdam' for clarity
    agg['dataset'] = agg['dataset'].replace({'ams': 'amsterdam'})

    # Compute and print detection proportion per dataset
    totals = agg.groupby('dataset')['time'].sum()
    detect_times = agg[agg['op'] == 'Detect'].set_index('dataset')['time']
    detect_pct = (detect_times / totals * 100).dropna()
    for ds in detect_pct.index:
        print(f"{ds}: Detect = {detect_pct[ds]:.1f}%")

    # Assign stacking order so bars render in consistent Read -> Detect -> Track order
    agg['op_order'] = agg['op'].map(OP_ORDER_MAP)

    # Build normalized horizontal stacked bar chart with thin bars
    chart = alt.Chart(agg).mark_bar(size=12).encode(
        x=alt.X('time:Q', title='Fraction of Total Runtime', stack='normalize',
                 axis=alt.Axis(format='%')),
        y=alt.Y('dataset:N', title='Dataset'),
        color=alt.Color('op:N', title='Operation', sort=OP_ORDER,
                        scale=alt.Scale(domain=OP_ORDER, range=ColorScheme.CarbonDark[:len(OP_ORDER)])),
        order=alt.Order('op_order:Q'),
        tooltip=[
            'dataset',
            alt.Tooltip('op:N', title='Operation'),
            alt.Tooltip('time:Q', format='.2f', title='Runtime (s)'),
        ],
    ).properties(
        # title='Naive Execution Runtime Breakdown',
        width=400,
        height=90,
    )

    # Save to SUMMARY folder in both PNG and PDF formats
    output_dir = str(cache.summary('083_naive_throughput'))
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'naive_runtime_breakdown.png')
    pdf_path = os.path.join(output_dir, 'naive_runtime_breakdown.pdf')
    chart.save(png_path, scale_factor=2)
    chart.save(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
