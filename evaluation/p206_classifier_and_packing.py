#!/usr/local/bin/python

"""Combined classifier-vs-baseline + packing-efficacy figure for the paper.

This script does **no** recomputation. It reads the JSONL artifacts already
produced by ``evaluation/p205_compare_classifiers.py`` and
``evaluation/p038_packing_efficacy.py``, rebuilds two trimmed-down panels,
and ``hconcat``s them into a single figure.

Panels (left -> right):
  * Left  : Classifier comparison from ``p205`` -- F1, Recall, Precision only
            (drops accuracy; runtime is intentionally omitted because the
            packing panel does not have a comparable axis).
  * Right : Packing efficacy from ``p038`` -- only the ``none`` tile-padding
            misrate / pruning slice; horizontal x-axis labels (no rotation).

Both panels share a single dataset color encoding derived from the union of
datasets present in the two source frames, so the legend is consistent and
identical colors map to identical datasets across the figure.

Run order: ``p205`` and ``p038`` must have been run first for the chosen
``--valid`` / ``--test`` split (default ``valid``). The script fails fast
with a clear error message when an upstream JSONL is missing.
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
import numpy as np
import pandas as pd

from polyis.io import cache
from evaluation.p201_compare_pareto import (
    SYSTEM_COLOR_SCHEME,
    _add_dataset_display_names,
    _get_dataset_display_sort,
)

# Subplot geometry: matches p205's per-metric subplot footprint so the
# stitched figure reads at a familiar density. Packing panel reuses the
# same height and is sized wider to accommodate the categorical x-axis.
SUBPLOT_WIDTH: int = 150
SUBPLOT_HEIGHT: int = 200
PACKING_PANEL_WIDTH: int = 100

# Upstream summary directories (must match p205 / p038 ``cache.summary`` keys).
P205_SUMMARY_STAGE = '205_compare_classifiers'
P038_SUMMARY_STAGE = '038_packing_efficacy'

# Restrict the packing panel to the ``none`` tile-padding misrate slice as
# requested. We still expect multiple datasets at this padding mode.
PACKING_TILEPADDING: str = 'none'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine p205 classifier metrics and p038 packing efficacy into one figure'
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--valid', action='store_true')
    split_group.add_argument('--test', action='store_true')
    return parser.parse_args()


def _resolve_videoset(args: argparse.Namespace) -> str:
    """Pick the split the run aggregates over -- defaults to ``valid``.

    Mirrors p038's behaviour so the paper figure (which lives under the
    canonical name) defaults to the validation split.
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

    Uses the same Vega ``observable10`` scheme as p201/p205/p038 so this
    figure visually composes with the rest of the paper.
    """
    if not domain:
        return alt.Scale(scheme=SYSTEM_COLOR_SCHEME)
    return alt.Scale(domain=domain, scheme=SYSTEM_COLOR_SCHEME)


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


def _build_classifier_panels(
    p205_df: pd.DataFrame,
    color_scale: alt.Scale,
) -> list[alt.Chart]:
    """Build the F1 / Recall / Precision subcharts from the p205 frame.

    Returns a list of three subcharts (left-to-right order) ready to be
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

    # Requested subset and order: F1, Recall, Precision (no accuracy).
    metrics: list[tuple[str, str]] = [
        ('f1_score', 'F1'),
        ('recall', 'Recall'),
        ('precision', 'Precision'),
    ]

    # Shared x-axis (runtime ms/frame) so the three subcharts line up.
    x_vals = df_line['ms_per_frame'].astype(float)
    x_max = float(x_vals.max()) if len(x_vals) else 1.0
    if not (np.isfinite(x_max) and x_max > 0):
        x_max = 1.0
    x_scale = alt.Scale(domain=[0.0, x_max], nice=False)

    subcharts: list[alt.Chart] = []
    for field, title in metrics:
        # Per-metric y-domain so subtle gaps between metrics stay readable.
        y0, y1 = _y_domain_for_metric(df, field)
        y_scale = alt.Scale(domain=[y0, y1], nice=False, zero=False)
        x_enc = alt.X('ms_per_frame:Q', title='runtime (ms/frame)', scale=x_scale)
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
                    legend=alt.Legend(title=None),
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
                    legend=alt.Legend(title=None),
                    scale=color_scale,
                ),
                shape=alt.Shape(
                    'classifier_kind:N',
                    legend=alt.Legend(title=None),
                    scale=alt.Scale(domain=['Modified', 'Baseline']),
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
        sub = (lines + points).properties(width=SUBPLOT_WIDTH, height=SUBPLOT_HEIGHT)
        subcharts.append(sub)

    return subcharts


def _build_packing_panel(
    p038_df: pd.DataFrame,
    color_scale: alt.Scale,
    dataset_domain: list[str],
) -> alt.Chart:
    """Build the packing-efficacy panel restricted to ``tilepadding='none'``.

    Drops the row facet over pruning thresholds and keeps a single
    grouped-bar chart with a horizontal x-axis (label angle 0).
    """
    # Restrict to the requested misrate threshold (tilepadding=='none').
    panel_df = p038_df[p038_df['tilepadding'] == PACKING_TILEPADDING].copy()
    if panel_df.empty:
        raise ValueError(
            f"No p038 rows with tilepadding='{PACKING_TILEPADDING}'. "
            'Re-run p038 with a config that includes this padding mode.'
        )

    # Attach display names so the bars share the legend with the classifier
    # panels via the shared color scale.
    panel_df = _add_dataset_display_names(panel_df)

    # Single bar per dataset within this padding slice -- aggregating with
    # ``mean`` is a no-op when each (dataset, tilepadding) bucket has one row,
    # but acts as a guardrail if multiple pruning thresholds slip through.
    chart = (
        alt.Chart(panel_df)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X(
                'dataset_display:N',
                title='Dataset',
                sort=dataset_domain or None,
                # Datasets are already encoded by color (shared legend), so
                # the categorical x-axis would just duplicate that channel;
                # strip ticks and labels to keep the thin panel uncluttered.
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            y=alt.Y(
                'mean(packing_efficacy_pct):Q',
                title='Packing Efficacy (%)',
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                'dataset_display:N',
                legend=alt.Legend(title=None),
                scale=color_scale,
            ),
            tooltip=[
                alt.Tooltip('dataset_display:N', title='Dataset'),
                alt.Tooltip('tilepadding:N', title='Tile Padding'),
                alt.Tooltip(
                    'mean(packing_efficacy_pct):Q',
                    title='Efficacy (%)',
                    format='.2f',
                ),
            ],
        )
        .properties(width=PACKING_PANEL_WIDTH, height=SUBPLOT_HEIGHT)
    )
    return chart


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

    # Locate the upstream caches up front so we can fail before doing chart work.
    p205_path = _p205_jsonl_path('test')
    p038_path = _p038_jsonl_path(videoset)

    print(f"  Loading p205 cache from {p205_path}")
    p205_df = _load_jsonl(p205_path, 'p205_compare_classifiers')
    print(f"  Loading p038 cache from {p038_path}")
    p038_df = _load_jsonl(p038_path, 'p038_packing_efficacy')

    # Shared color encoding: union of datasets across both frames, mapped to
    # display names and ordered by the configured DATASETS list.
    dataset_domain = _shared_dataset_domain([p205_df, p038_df])
    color_scale = _build_color_scale(dataset_domain)

    # Build the panels: classifier metrics on the left, packing on the right.
    classifier_panels = _build_classifier_panels(p205_df, color_scale)
    packing_panel = _build_packing_panel(p038_df, color_scale, dataset_domain)

    # Group the three classifier subcharts under one shared title so it spans
    # F1/Recall/Precision rather than sitting above just the leftmost panel.
    classifier_group = alt.hconcat(*classifier_panels, spacing=0).properties(
        title=alt.TitleParams(
            text='Classification Accuracy',
            anchor='start',
            fontSize=12,
        )
    )
    # Packing panel is a single chart so its title attaches directly.
    packing_panel = packing_panel.properties(
        title=alt.TitleParams(
            text='Packing Efficacy',
            anchor='start',
            fontSize=12,
        )
    )

    # Stitch into a single horizontal layout. ``resolve_scale`` keeps color
    # and shape shared so one legend covers both halves.
    combined = (
        alt.hconcat(classifier_group, packing_panel, spacing=12)
        .resolve_scale(color='shared', shape='shared')
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
    # ``scale_factor`` mirrors p205 (PDF) and p038 (PNG) defaults so the
    # rendered density matches the surrounding paper figures.
    combined.save(chart_pdf, scale_factor=2)
    combined.save(chart_png, scale_factor=4)
    print(f"  Wrote {chart_pdf}")
    print(f"  Wrote {chart_png}")

    # Only the validation split (canonical paper artifact) is propagated to
    # paper/figures/generated; test runs stay cache-only.
    if videoset == 'valid':
        _copy_paper_figure_artifacts(summary_dir, base_name, PAPER_FIGURES_GENERATED_DIR)
    else:
        print(f"  Skipping paper-figures copy for videoset={videoset}")


if __name__ == '__main__':
    main()
