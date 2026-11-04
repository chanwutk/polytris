#!/usr/local/bin/python

"""Combined relevance-classifier, packing-accuracy, and packing-efficacy figure for the paper.

This script does **no** recomputation. It reads the p205 classifier comparison
(``evaluation/p205_compare_classifiers.py``) and the p038 packing efficacy
(``evaluation/p038_packing_efficacy.py``) from the cache, and the
ground-truth-tile packing experiment (measurements-gpu/r1o4_oracle_packing.py)
from the paper's evidence snapshot (``packing.json``, with each dataset's 5%
HOTA-loss configuration from ``matched_points.json``), and ``hconcat``s:

  * Left   : Classifier comparison -- F1 against runtime, Modified vs.
             Baseline by shape (legend inside the panel).
  * Middle : Detector precision and recall on original frames (hollow) and
             packed canvases (filled) per dataset, at the tile padding of each
             dataset's 5% HOTA-loss configuration.
  * Right  : Packing efficacy per dataset (p038, ``none`` tile padding) as
             horizontal bars whose dataset labels double as the color legend.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

# Repo-root anchored constants so the paper-figures copy works regardless of
# the directory the script is invoked from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_FIGURES_GENERATED_DIR = os.path.join(REPO_ROOT, 'paper', 'figures', 'generated')
SCRIPT_ARTIFACT_BASENAME = 'p206_classifier_and_packing'

import altair as alt
import duckdb
import numpy as np
import pandas as pd

from polyis.io import cache
from evaluation.p201_compare_pareto import (
    _add_dataset_display_names,
    _get_dataset_display_sort,
)
from evaluation.colors import categorical_color_scale

# Subplot geometry: matches p205's per-metric subplot footprint so the
# stitched figure reads at a familiar density.
SUBPLOT_WIDTH: int = 110
SUBPLOT_HEIGHT: int = 130
PACKING_PANEL_WIDTH: int = 100
# Pixels added inside both ends of every axis so points on the data extremes draw whole.
SCALE_PADDING: int = 4

# Upstream summary directories (must match p205 / p038 ``cache.summary`` keys).
P205_SUMMARY_STAGE = '205_compare_classifiers'
P038_SUMMARY_STAGE = '038_packing_efficacy'
# The packing-efficacy bars use p038's unpadded slice.
PACKING_TILEPADDING: str = 'none'
# Width of the packing-efficacy bar chart.
EFFICACY_PANEL_WIDTH: int = 80

# The paper's evidence snapshot holding the packing experiment and the 5% operating points.
EVIDENCE_DIR = Path(REPO_ROOT) / 'paper' / 'scripts' / 'data' / 'revision_evidence'
# Packing metrics shown, as (summary field, axis title).
PACKING_METRICS: list[tuple[str, str]] = [('precision', 'Precision (%)'), ('recall', 'Recall (%)')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine p205 classifier metrics and packing accuracy into one figure'
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--valid', action='store_true')
    split_group.add_argument('--test', action='store_true')
    return parser.parse_args()


def _resolve_videoset(args: argparse.Namespace) -> str:
    """Pick the split the run aggregates over -- defaults to ``valid``.

    The split selects p038's packing-efficacy file and names the output; the paper figure uses ``--test``.
    """
    if args.test:
        return 'test'
    return 'valid'


def _p205_jsonl_path(videoset: str) -> Path:
    """Return the cached p205 JSONL path for the given split."""
    # p205 always suffixes its artifacts with the videoset name.
    return cache.summary(P205_SUMMARY_STAGE, f'p205_compare_classifiers_{videoset}.jsonl')


def _p038_jsonl_path(videoset: str) -> Path:
    """Return the cached p038 JSONL path for the given split.

    p038 leaves the validation split unsuffixed (canonical paper artifact)
    and suffixes the test split, so we mirror that naming here.
    """
    if videoset == 'valid':
        return cache.summary(P038_SUMMARY_STAGE, 'p038_packing_efficacy.jsonl')
    return cache.summary(P038_SUMMARY_STAGE, f'p038_packing_efficacy_{videoset}.jsonl')


def _load_jsonl(path: Path, source_label: str) -> pd.DataFrame:
    """Load a JSONL file produced by an upstream evaluation script.

    Fails fast with an actionable message so the user knows which upstream
    script to run when the cache is empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {source_label} cache at {path}. "
            f"Run the upstream script first."
        )
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{source_label} cache at {path} contains no rows")
    return pd.DataFrame(rows)


def _build_color_scale(domain: list[str]) -> alt.Scale:
    """Build a categorical color scale shared across both panels.

    Uses the shared Tetris palette from p201/p205/p038 so this
    figure visually composes with the rest of the paper.
    """
    return categorical_color_scale(domain or None)


def _y_domain_for_metric(df: pd.DataFrame, field: str) -> tuple[float, float]:
    """Padded ``[min, max]`` for a metric column, clamped to ``[0, 1]``.

    Mirrors the helper in p205 so the per-metric y-axis ranges line up
    with the original chart -- we do not import directly to keep p206
    self-contained against future p205 refactors.
    """
    s = df[field].astype(float)
    lo, hi = float(s.min()), float(s.max())
    # Guard against all-NaN columns, which would otherwise blow up the axis.
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, 1.0
    # Constant columns expand to a small symmetric window so the line still
    # has visible vertical extent.
    if hi - lo < 1e-9:
        mid = lo
        lo = max(0.0, mid - 0.02)
        hi = min(1.0, mid + 0.02)
    span = max(hi - lo, 1e-6)
    pad = max(0.1 * span, 0.004)
    d_lo = max(0.0, lo - pad)
    d_hi = min(1.0, hi + pad)
    if d_hi <= d_lo:
        d_hi = min(1.0, d_lo + 0.05)
    return d_lo, d_hi


def _inside_legend(items: list[tuple[str, dict]]) -> alt.LayerChart:
    """A small legend drawn inside a panel's bottom-left corner: one marker and label per (label, mark style).

    Vega-Lite places legends of concatenated views outside the plots, so in-panel legends are drawn as marks at
    fixed pixel positions, independent of the panel's scales.
    """
    layers = []
    for row, (label, style) in enumerate(reversed(items)):
        y = SUBPLOT_HEIGHT - 12 - 16 * row
        layers.append(alt.Chart(alt.Data(values=[{}])).mark_point(size=60, color='#555', **style)
                      .encode(x=alt.value(12), y=alt.value(y)))
        layers.append(alt.Chart(alt.Data(values=[{}])).mark_text(align='left', baseline='middle', fontSize=11)
                      .encode(x=alt.value(22), y=alt.value(y), text=alt.value(label)))
    return alt.layer(*layers)


def _build_classifier_panels(
    p205_df: pd.DataFrame,
    color_scale: alt.Scale,
) -> list[alt.Chart]:
    """Build the classifier subcharts (F1) from the p205 frame.

    Returns a list of subcharts (left-to-right order) ready to be
    horizontally concatenated. We deliberately drop accuracy and runtime to
    keep the combined figure focused on the metrics the paper cites.
    """
    # Mirror p205: tag rows so the line connects Modified -> Baseline per
    # (dataset, tile_size); ``modified`` is the boolean discriminator written
    # by p205, and we surface it as a categorical for the shape encoding.
    df = _add_dataset_display_names(p205_df.copy())
    df['classifier_kind'] = np.where(df['modified'], 'Modified', 'Baseline')

    # Stable sort so mark_line traces Modified -> Baseline within each
    # (dataset, tile_size) group; matches p205's chart construction.
    df_line = df.sort_values(
        ['dataset', 'tile_size', 'modified'],
        ascending=[True, True, False],
        kind='stable',
    ).reset_index(drop=True)

    # F1 only: the paper's classifier comparison reports F1.
    metrics: list[tuple[str, str]] = [
        ('f1_score', 'F1'),
    ]

    # Shared x-axis (runtime ms/frame) so the subcharts line up.
    x_vals = df_line['ms_per_frame'].astype(float)
    x_max = float(x_vals.max()) if len(x_vals) else 1.0
    if not (np.isfinite(x_max) and x_max > 0):
        x_max = 1.0
    x_scale = alt.Scale(domain=[0.0, x_max], nice=False, padding=SCALE_PADDING)

    subcharts: list[alt.Chart] = []
    for index, (field, title) in enumerate(metrics):
        # Per-metric y-domain so subtle gaps between metrics stay readable.
        y0, y1 = _y_domain_for_metric(df, field)
        y_scale = alt.Scale(domain=[y0, y1], nice=False, zero=False, padding=SCALE_PADDING)
        # Runtime ticks every 2 ms/frame.
        x_enc = alt.X('ms_per_frame:Q', title='runtime (ms/frame)', scale=x_scale,
                      axis=alt.Axis(values=list(range(0, int(np.ceil(x_max)) + 1, 2))))
        y_enc = alt.Y(f'{field}:Q', title=title, scale=y_scale)

        # Connecting line links the Modified/Baseline pair per dataset.
        lines = (
            alt.Chart(df_line)
            .mark_line(opacity=0.55, strokeWidth=2.5, interpolate='linear')
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color(
                    'dataset_display:N',
                    legend=None,
                    scale=color_scale,
                ),
                detail=['dataset', 'tile_size'],
            )
        )
        # Points encode the Modified vs Baseline distinction via shape, so
        # the legend stays small when the figure is rendered narrow.
        points = (
            alt.Chart(df_line)
            .mark_point(filled=True, size=90, opacity=0.9, stroke='white', strokeWidth=0.6)
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color(
                    'dataset_display:N',
                    legend=None,
                    scale=color_scale,
                ),
                # Shape distinguishes Modified/Baseline; its legend is drawn inside the F1 panel.
                shape=alt.Shape(
                    'classifier_kind:N',
                    legend=None,
                    scale=alt.Scale(domain=['Modified', 'Baseline'], range=['circle', 'square']),
                ),
                tooltip=[
                    'dataset_display',
                    'classifier_kind',
                    'tile_size',
                    alt.Tooltip(f'{field}:Q', format='.4f'),
                    alt.Tooltip('ms_per_frame:Q', format='.3f'),
                ],
            )
        )
        layers = [lines, points]
        # The Modified/Baseline legend, inside the first (F1) panel only.
        if index == 0:
            layers.append(_inside_legend([('Modified', dict(shape='circle', filled=True)),
                                          ('Baseline', dict(shape='square', filled=True))]))
        sub = alt.layer(*layers).properties(width=SUBPLOT_WIDTH, height=SUBPLOT_HEIGHT)
        subcharts.append(sub)

    return subcharts


def _load_packing_rows(evidence_dir: Path) -> pd.DataFrame:
    """Original and packed precision/recall per dataset at its 5% configuration's tile padding.

    Rows: dataset, input ('Original' or 'Packed'), metric field, efficacy_pct, accuracy_pct.
    """
    # The tile padding of each dataset's 5% HOTA-loss Tetris configuration (the token after r<T_r>).
    points = json.loads((evidence_dir / 'matched_points.json').read_text())['operating_points']['0.05']
    picks = {label: entry['systems']['tetris']['config'] for label, entry in points.items()}
    con = duckdb.connect()
    con.execute("CREATE TABLE s AS SELECT unnest(rows, recursive := true) FROM read_json(?)",
                [str(evidence_dir / 'packing.json')])
    con.execute('CREATE TABLE picks(label VARCHAR, config VARCHAR)')
    con.executemany('INSERT INTO picks VALUES (?, ?)', list(picks.items()))
    con.execute("CREATE TABLE pick AS SELECT label, regexp_extract(config, '_r[0-9]{3}_([a-z]+)_s[0-9]+', 1) AS padding FROM picks")
    # Every operating point names a padding that the packing experiment ran.
    missing = con.execute("SELECT label FROM pick LEFT JOIN s USING (label, padding) WHERE s.dataset IS NULL").fetchall()
    assert not missing, f'no packing result for the 5% padding of {missing}'
    # Unpivot the metrics and the two inputs.
    query = ' UNION ALL '.join(
        f"""SELECT s.dataset, '{side}' AS input, '{field}' AS metric, 100 * s.packing_efficacy AS efficacy_pct,
                   s.retained_{kind}_{field}_pct AS accuracy_pct
            FROM s JOIN pick USING (label, padding)"""
        for field, _ in PACKING_METRICS for kind, side in (('original', 'Original'), ('packed', 'Packed')))
    df = con.execute(query).df()
    assert len(df) == len(picks) * len(PACKING_METRICS) * 2, len(df)
    return df


def _build_packing_panels(packing_df: pd.DataFrame, color_scale: alt.Scale,
                          dataset_domain: list[str]) -> list[alt.LayerChart]:
    """One panel per packing metric: per dataset, hollow original and filled packed points joined by a line."""
    df = _add_dataset_display_names(packing_df.copy())
    panels: list[alt.LayerChart] = []
    for index, (field, title) in enumerate(PACKING_METRICS):
        base = alt.Chart(df[df['metric'] == field]).encode(
            # Datasets in the shared order; colors (labeled by the efficacy bars) identify them.
            x=alt.X('dataset_display:N', title='Dataset', sort=dataset_domain or None,
                    axis=alt.Axis(labels=False, ticks=False)),
            # labelFlush keeps an edge tick label (e.g., 100) from overhanging the plot and lifting the group title.
            y=alt.Y('accuracy_pct:Q', title=title, scale=alt.Scale(zero=False, padding=SCALE_PADDING),
                    axis=alt.Axis(labelFlush=True)),
            color=alt.Color('dataset_display:N', legend=None, scale=color_scale),
        )
        line = base.mark_line(opacity=0.55, strokeWidth=2.5).encode(detail='dataset_display:N')
        # Hollow original and filled packed points.
        original = base.transform_filter(alt.datum.input == 'Original').mark_point(filled=False, size=90, strokeWidth=2)
        packed = base.transform_filter(alt.datum.input == 'Packed').mark_point(filled=True, size=90, opacity=0.9)
        layers = [line, original, packed]
        # The Original/Packed legend, inside the first (precision) panel only.
        if index == 0:
            layers.append(_inside_legend([('Original', dict(filled=False, strokeWidth=2)),
                                          ('Packed', dict(filled=True))]))
        panels.append(alt.layer(*layers).properties(width=PACKING_PANEL_WIDTH, height=SUBPLOT_HEIGHT))
    return panels


def _build_efficacy_chart(p038_df: pd.DataFrame, color_scale: alt.Scale, dataset_domain: list[str]) -> alt.Chart:
    """Horizontal packing-efficacy bars per dataset; the dataset labels on y double as the figure's color legend."""
    # The unpadded slice, as in the paper's packing-efficacy text.
    df = p038_df[p038_df['tilepadding'] == PACKING_TILEPADDING].copy()
    if df.empty:
        raise ValueError(f"No p038 rows with tilepadding='{PACKING_TILEPADDING}'. Re-run p038 with this padding mode.")
    df = _add_dataset_display_names(df)
    # One bar per dataset: the mean over p038's pruning slices.
    return alt.Chart(df).mark_bar(opacity=0.9).encode(
        y=alt.Y('dataset_display:N', title=None, sort=dataset_domain or None, axis=alt.Axis(ticks=False, domain=False)),
        x=alt.X('mean(packing_efficacy_pct):Q', title='Packing efficacy (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('dataset_display:N', legend=None, scale=color_scale),
        tooltip=[alt.Tooltip('dataset_display:N', title='Dataset'),
                 alt.Tooltip('mean(packing_efficacy_pct):Q', title='Efficacy (%)', format='.2f')],
    ).properties(width=EFFICACY_PANEL_WIDTH, height=SUBPLOT_HEIGHT)


def _shared_dataset_domain(frames: list[pd.DataFrame]) -> list[str]:
    """Build a stable display-name domain spanning every dataset in either source.

    Concatenates the datasets from each frame so ``_get_dataset_display_sort``
    sees the union, which keeps configured ordering and avoids dropping
    datasets that only appear in one panel.
    """
    union_pieces: list[pd.DataFrame] = []
    for df in frames:
        if df.empty or 'dataset' not in df.columns:
            continue
        union_pieces.append(df[['dataset']].drop_duplicates())
    if not union_pieces:
        return []
    union_df = pd.concat(union_pieces, ignore_index=True).drop_duplicates()
    union_df = _add_dataset_display_names(union_df)
    return _get_dataset_display_sort(union_df) or []


def _copy_paper_figure_artifacts(
    source_dir: str | Path,
    base_name: str,
    destination_dir: str | Path,
) -> None:
    """Copy the rendered PDF into ``paper/figures/generated``.

    Mirrors the helper in p205; we only ship the PDF to the paper because
    the LaTeX figure environment expects a vector format. The PNG companion
    stays in the cache for quick eyeballing.
    """
    src_root = Path(source_dir)
    dst_root = Path(destination_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    src = src_root / f'{base_name}.pdf'
    if not src.is_file():
        print(f"  PDF missing at {src}; skipping paper-figures copy")
        return
    dst = dst_root / f'{base_name}.pdf'
    shutil.copy2(src, dst)
    print(f"  Copied to paper figures: {dst}")


def main() -> None:
    args = parse_args()
    videoset = _resolve_videoset(args)

    # Locate the upstream inputs up front so we can fail before doing chart work.
    p205_path = _p205_jsonl_path('test')
    print(f"  Loading p205 cache from {p205_path}")
    p205_df = _load_jsonl(p205_path, 'p205_compare_classifiers')
    p038_path = _p038_jsonl_path(videoset)
    print(f"  Loading p038 cache from {p038_path}")
    p038_df = _load_jsonl(p038_path, 'p038_packing_efficacy')
    print(f"  Loading packing evidence from {EVIDENCE_DIR}")
    packing_df = _load_packing_rows(EVIDENCE_DIR)

    # Shared color encoding: union of datasets across all sources, mapped to
    # display names and ordered by the configured DATASETS list.
    dataset_domain = _shared_dataset_domain([p205_df, p038_df, packing_df])
    color_scale = _build_color_scale(dataset_domain)

    # The classifier subchart under its title.
    classifier_group = alt.hconcat(*_build_classifier_panels(p205_df, color_scale), spacing=0).properties(
        title=alt.TitleParams(text='Classification Accuracy', anchor='start', fontSize=12)
    )
    # The two detection panels (independent y-axes) under their own title.
    packing_group = alt.hconcat(*_build_packing_panels(packing_df, color_scale, dataset_domain), spacing=8).resolve_scale(
        y='independent'
    ).properties(title=alt.TitleParams(text='Detection after Packing', anchor='start', fontSize=12))
    # The packing-efficacy bars, which also label the dataset colors.
    efficacy_chart = _build_efficacy_chart(p038_df, color_scale, dataset_domain).properties(
        title=alt.TitleParams(text='Packing Efficacy', anchor='start', fontSize=12)
    )

    # One color scale across the figure; the classifier shape legend stays inside its panel.
    combined = (
        alt.hconcat(classifier_group, packing_group, efficacy_chart, spacing=16)
        .resolve_scale(color='shared', shape='independent')
        .properties(padding=0)
        .configure_view(stroke=None)
    )

    # Write into the dedicated p206 summary cache so the artifact does not
    # collide with the upstream caches we just consumed.
    summary_dir = cache.summary('206_classifier_and_packing')
    os.makedirs(summary_dir, exist_ok=True)

    # Validation split is the canonical paper artifact; suffix non-default
    # splits so both can coexist in the cache.
    base_name = (
        SCRIPT_ARTIFACT_BASENAME
        if videoset == 'valid'
        else f'{SCRIPT_ARTIFACT_BASENAME}_{videoset}'
    )

    chart_pdf = os.path.join(summary_dir, f'{base_name}.pdf')
    chart_png = os.path.join(summary_dir, f'{base_name}.png')
    # ``scale_factor`` mirrors p205's PDF and PNG defaults so the
    # rendered density matches the surrounding paper figures.
    combined.save(chart_pdf, scale_factor=2)
    combined.save(chart_png, scale_factor=4)
    print(f"  Wrote {chart_pdf}")
    print(f"  Wrote {chart_png}")

    # Propagate either split to paper/figures/generated under its split-suffixed name; the paper
    # shows the test split, matching the p205 test macros its text quotes.
    _copy_paper_figure_artifacts(summary_dir, base_name, PAPER_FIGURES_GENERATED_DIR)


if __name__ == '__main__':
    main()
