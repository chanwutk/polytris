#!/usr/local/bin/python

import json
import os
import shutil
from pathlib import Path

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

# Stacking order for the full breakdown bar chart (left to right)
OP_ORDER = ['Detect', 'Decode', 'Track']
OP_ORDER_MAP = {op: i for i, op in enumerate(OP_ORDER)}

# Stacking order for the detection-vs-tracking-only chart (Decode dropped)
DET_TRACK_OP_ORDER = ['Detect', 'Track']
DET_TRACK_OP_ORDER_MAP = {op: i for i, op in enumerate(DET_TRACK_OP_ORDER)}

# Canonical paper-facing labels for the truncated dataset ids produced after
# the ``split('-')`` mangling done in ``main``. Keep in sync with
# ``evaluation/p201_compare_pareto.DATASET_NAME_MAP`` so all paper figures use
# the same dataset spellings.
DATASET_DISPLAY_MAP = {
    'caldot1': 'CalDoT 1',
    'caldot2': 'CalDoT 2',
    'amsterdam': 'Amsterdam',
    'jnc0': 'B3D 1',
    'jnc2': 'B3D 2',
    'jnc6': 'B3D 3',
    'jnc7': 'B3D 4',
}

# Repo root used to locate the paper-figures directory for the copy step.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_FIGURES_GENERATED_DIR = os.path.join(REPO_ROOT, 'paper', 'figures', 'generated')

# Basename for the Detect+Track-only chart and its accompanying tex table.
DET_TRACK_BASENAME = 'naive_runtime_breakdown_det_track'


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


def _short_dataset_order() -> list[str]:
    """Return configured DATASETS in canonical order after short-name mangling.

    Mirrors the transformations done in ``main`` (``split('-')[0]`` and the
    ``ams`` -> ``amsterdam`` rename) so callers can align columns/rows with the
    aggregated DataFrame without recomputing the mapping.
    """
    ordered: list[str] = []
    for raw in DATASETS:
        # Truncate at the first '-' to match the aggregation key.
        short = raw.split('-')[0]
        # Apply the same explicit rename used downstream in ``main``.
        short = 'amsterdam' if short == 'ams' else short
        ordered.append(short)
    return ordered


def write_det_track_chart(agg: pd.DataFrame, png_path: str, pdf_path: str) -> None:
    """Render a normalized stacked bar chart restricted to Detect and Track.

    Drops the Decode bar entirely so the reader sees how much of the
    model-bound work in the naive pipeline is detection versus tracking.

    Args:
        agg: aggregated runtime DataFrame with columns ['dataset', 'op', 'time'].
        png_path: destination PNG path.
        pdf_path: destination PDF path.
    """
    # Keep only the two operations we care about for this view.
    det_track = agg[agg['op'].isin(DET_TRACK_OP_ORDER)].copy()

    # Translate short dataset ids to the paper's canonical display labels.
    det_track['dataset_display'] = (
        det_track['dataset'].map(DATASET_DISPLAY_MAP).fillna(det_track['dataset'])
    )

    # Stable y-axis ordering: follow the configured DATASETS order, mapped to
    # the display labels actually present in the aggregation.
    present_short = set(det_track['dataset'])
    dataset_sort = [
        DATASET_DISPLAY_MAP.get(short, short)
        for short in _short_dataset_order()
        if short in present_short
    ]

    # Stacking position for each operation (left -> right within each bar).
    det_track['op_order'] = det_track['op'].map(DET_TRACK_OP_ORDER_MAP)

    # Reuse the colors assigned to Detect/Track in the full breakdown chart so
    # the two figures remain visually consistent side-by-side.
    full_palette = ColorScheme.CarbonDark[:len(OP_ORDER)]
    det_track_palette = [full_palette[OP_ORDER.index(op)] for op in DET_TRACK_OP_ORDER]

    # Normalized stacked bar: x-axis is the fraction of (Detect + Track) only.
    chart = alt.Chart(det_track).mark_bar(size=12).encode(
        x=alt.X('time:Q', title='Fraction of Detect + Track Runtime',
                 stack='normalize', axis=alt.Axis(format='%')),
        y=alt.Y('dataset_display:N', title='Dataset', sort=dataset_sort),
        color=alt.Color('op:N', title='Operation', sort=DET_TRACK_OP_ORDER,
                        scale=alt.Scale(domain=DET_TRACK_OP_ORDER, range=det_track_palette)),
        order=alt.Order('op_order:Q'),
        tooltip=[
            alt.Tooltip('dataset_display:N', title='Dataset'),
            alt.Tooltip('op:N', title='Operation'),
            alt.Tooltip('time:Q', format='.2f', title='Runtime (s)'),
        ],
    ).properties(
        width=400,
        height=90,
    )

    # Save both raster and vector forms (PNG for previews, PDF for the paper).
    chart.save(png_path, scale_factor=2)
    chart.save(pdf_path)


def write_det_track_table(detect_share: pd.Series, tex_path: str) -> None:
    """Write a 2-row LaTeX tabular with one column per dataset.

    Row 1: empty corner cell followed by the dataset display names.
    Row 2: ``Detection time proportion (%)`` followed by the per-dataset values.

    Args:
        detect_share: per-dataset Detect / (Detect + Track) percentage, indexed
            by the short dataset id (e.g., ``caldot1``, ``jnc0``, ``amsterdam``).
        tex_path: destination .tex file path.
    """
    # Stable column order: follow the configured DATASETS order, dropping any
    # datasets missing from the share series (no data this run).
    ordered_short = [s for s in _short_dataset_order() if s in detect_share.index]

    # Map short ids to the paper's canonical display labels for the header row.
    display_names = [DATASET_DISPLAY_MAP.get(short, short) for short in ordered_short]

    # Build the two content rows: header (``Dataset`` corner + names) and
    # values (row label + percentages formatted to one decimal place with a
    # ``\%`` suffix so each cell is self-describing).
    header_cells = ['Dataset'] + display_names
    value_cells = ['Detection time proportion'] + [
        f'{detect_share[short]:.1f}\\%' for short in ordered_short
    ]

    # Column spec: a left-aligned label column followed by one centered column
    # per dataset; the vertical rule separates the row label from the values.
    col_spec = 'l|' + 'c' * len(ordered_short)

    # Emit a standalone tabular suitable for \input{...} inside a table float.
    lines = [
        '% Auto-generated by evaluation/p123_throughput_naive.py',
        '% Row label = description; each column is one dataset.',
        '% Value = Detect / (Detect + Track) * 100 (naive pipeline, Decode excluded).',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        ' & '.join(header_cells) + ' \\\\',
        '\\midrule',
        ' & '.join(value_cells) + ' \\\\',
        '\\bottomrule',
        '\\end{tabular}',
    ]
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _copy_paper_figure_artifacts(source_dir: str, base_name: str, destination_dir: str) -> None:
    """Mirror the chart PDF/PNG and the table tex into ``paper/figures/generated``.

    Mirrors the convention used in ``p038_packing_efficacy._copy_paper_figure_artifacts``:
    silently skip artifacts that are missing on disk (e.g., a chart save that
    failed), and copy with ``shutil.copy2`` to preserve mtimes.
    """
    dst_root = Path(destination_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Chart artifacts share the same base name and differ only by extension.
    for ext in ('.pdf', '.png'):
        src = Path(source_dir) / f'{base_name}{ext}'
        if not src.is_file():
            continue
        dst = dst_root / f'{base_name}{ext}'
        shutil.copy2(src, dst)
        print(f"  Copied to paper figures: {dst}")

    # The LaTeX table lives under ``{base_name}_table.tex`` next to the chart.
    tex_src = Path(source_dir) / f'{base_name}_table.tex'
    if tex_src.is_file():
        tex_dst = dst_root / f'{base_name}_table.tex'
        shutil.copy2(tex_src, tex_dst)
        print(f"  Copied to paper figures: {tex_dst}")


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

    # ---- Alternative view: Detect + Track only (Decode excluded) ----

    # Render the alternative chart with the same aggregation but only the
    # Detect and Track bars.
    det_track_png = os.path.join(output_dir, f'{DET_TRACK_BASENAME}.png')
    det_track_pdf = os.path.join(output_dir, f'{DET_TRACK_BASENAME}.pdf')
    write_det_track_chart(agg, det_track_png, det_track_pdf)
    print(f"Saved: {det_track_png}")
    print(f"Saved: {det_track_pdf}")

    # Detection share among Detect+Track only (Decode dropped from denominator).
    det_only = agg[agg['op'] == 'Detect'].set_index('dataset')['time']
    track_only = agg[agg['op'] == 'Track'].set_index('dataset')['time']
    det_track_share = (det_only / (det_only + track_only) * 100).dropna()

    # Print the alt-view share so the values are visible in the run log.
    for ds in det_track_share.index:
        print(f"{ds}: Detect / (Detect+Track) = {det_track_share[ds]:.1f}%")

    # Write the 2-row LaTeX tabular for inclusion in the paper.
    det_track_tex = os.path.join(output_dir, f'{DET_TRACK_BASENAME}_table.tex')
    write_det_track_table(det_track_share, det_track_tex)
    print(f"Saved: {det_track_tex}")

    # Mirror the Detect+Track artifacts into the paper repo so the LaTeX build
    # can pick them up via \includegraphics / \input. The original
    # ``naive_runtime_breakdown.{png,pdf}`` files are intentionally NOT mirrored
    # here; they live at ``figures/`` in the paper, not ``figures/generated/``.
    _copy_paper_figure_artifacts(output_dir, DET_TRACK_BASENAME, PAPER_FIGURES_GENERATED_DIR)


if __name__ == '__main__':
    main()
