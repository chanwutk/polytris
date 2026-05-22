#!/usr/local/bin/python

"""Packing efficacy of polyomino canvases at ``sample_rate=1``.

Walks each dataset's ``033_compressed_frames`` output and measures, for
every (dataset, tile-padding) combination, how densely the packing
algorithm fills its canvases.

Packing efficacy here measures the packing algorithm itself and is
deliberately *independent of tile-padding*::

    packing_efficacy = occupied_tiles / (num_canvases * canvas_w_tiles * canvas_h_tiles)
                     = occupied_tiles / (occupied_tiles + empty_tiles)

where ``occupied_tiles`` is every non-zero cell in the index map (which
includes padding tiles -- they still occupy space on the canvas) and the
denominator is the total number of tile-cells across all canvases. This
answers "how much of the canvas does the packer actually use?" rather
than "how relevant are the tiles it packed?".

Outputs a grouped bar chart faceted by pruning accuracy threshold (row
facets): within each row, x = tile-padding, color/x-offset = dataset,
y = packing efficacy in %. Exports PDF and PNG, plus a TeX macro file
summarizing the best/average/worst values so the paper prose can cite them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, cast
import multiprocessing
import os
import shutil
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_FIGURES_GENERATED_DIR = os.path.join(REPO_ROOT, 'paper', 'figures', 'generated')
SCRIPT_ARTIFACT_BASENAME = 'p038_packing_efficacy'

import altair as alt
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn

from polyis.io import cache
from polyis.utilities import get_config, parse_execution_param_str
from evaluation.p036_compress_effectiveness_compute import count_tiles_for_path
from evaluation.p201_compare_pareto import (
    SYSTEM_COLOR_SCHEME,
    _add_dataset_display_names,
    _get_dataset_display_sort,
)

# Fixed experimental knobs -- this chart only makes sense at every-frame
# compression with the canonical canvas size.
SAMPLE_RATE: int = 1
CANVAS_SCALE: float = 1.0
# Restrict to runs at the canonical relevance threshold T_r = 0.5. Other T_r
# values would mix incompatible relevancy bitmaps into the same aggregation
# (the padding-tile counter uses the same T_r to rebuild relevancy, so mixing
# T_r values would also corrupt any padding-based stats that consumers of the
# JSONL might compute downstream).
RELEVANCE_THRESHOLD: float = 0.5

# Only the Best-Fit (033) compressed-frames layout is consumed here; other
# pack modes live under different stage directories.
COMPRESSION_STAGE: str = '033_compressed_frames'

# Canonical slice used when emitting per-dataset polyomino-size statistics
# (paper prose in ``Packing Efficacy``). Polyomino shapes are a property of
# the dataset, but the on-disk polyominoes we can inspect depend on the
# pruning config that wrote them; we pin to zero-padding + the lowest pruning
# threshold so the distribution reflects the full foreground as closely as
# possible.
POLY_STATS_TILEPADDING: str = 'none'

# Map from ``--valid`` / ``--test`` flags to (videoset, video filename prefix)
# matching ``polyis.utilities`` conventions: ``te*`` -> test, ``va*`` -> valid.
_VIDEOSET_TO_PREFIX: dict[str, str] = {'test': 'te', 'valid': 'va'}

# Chart geometry -- kept small so the figure fits in a single paper column.
CHART_WIDTH: int = 360
# Height of each row facet (pruning threshold); overall figure scales with
# the number of thresholds in ``TRACKING_ACCURACY_THRESHOLDS``.
CHART_ROW_HEIGHT: int = 160

config = get_config()
DATASETS: list[str] = config['EXEC']['DATASETS']
CLASSIFIERS: list[str] = config['EXEC']['CLASSIFIERS']
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
TILEPADDING_MODES: list[str] = config['EXEC']['TILEPADDING_MODES']
TRACKING_ACCURACY_THRESHOLDS: list[float | None] | None = config['EXEC'].get(
    'TRACKING_ACCURACY_THRESHOLDS'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize packing efficacy at sample_rate=1 across datasets and tile-padding modes'
    )
    parser.add_argument('--verbose', action='store_true', help='Print verbose progress output')
    # Mutually exclusive split selector matching the rest of the pipeline
    # (``--valid`` / ``--test``). Default is the validation split: paper
    # figures and ``\\packingEfficacy*`` macros are generated from ``va*``.
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--valid', action='store_true', help='Aggregate over the valid split (va*) [default]')
    split_group.add_argument('--test', action='store_true', help='Aggregate over the test split (te*)')
    return parser.parse_args()


def _resolve_videoset(args: argparse.Namespace) -> str:
    """Pick the split the run aggregates over -- defaults to ``valid``."""
    if args.test:
        return 'test'
    return 'valid'


def _iter_config_dirs(dataset: str, video_prefix: str) -> list[tuple[Path, str, Path]]:
    """Yield ``(video_dir, video_name, config_dir)`` triples for the chosen split.

    Scanning is split out so ``collect_rows`` can fan out over the work with
    a multiprocessing pool without reconstructing the iteration logic.
    """
    # Rebuild the on-disk root p030 writes to for this dataset.
    dataset_exec_dir = cache.execution(dataset)
    if not dataset_exec_dir.exists():
        return []

    triples: list[tuple[Path, str, Path]] = []
    # ``video_prefix`` selects the split (``te`` for test, ``va`` for valid).
    for video_dir in sorted(dataset_exec_dir.iterdir()):
        if not video_dir.is_dir() or not video_dir.name.startswith(video_prefix):
            continue
        stage_dir = video_dir / COMPRESSION_STAGE
        if not stage_dir.exists():
            continue
        for config_dir in sorted(stage_dir.iterdir()):
            if config_dir.is_dir():
                triples.append((video_dir, video_dir.name, config_dir))
    return triples


def _pruning_accuracy_label(th: object) -> str:
    """Human-readable row label for a tracking / pruning accuracy threshold."""
    try:
        if pd.isna(th):
            return 'No pruning'
    except (ValueError, TypeError):
        pass
    if th is None:
        return 'No pruning'
    if isinstance(th, float) and not np.isfinite(th):
        return 'No pruning'
    if isinstance(th, (int, float, np.integer, np.floating)):
        return f'{int(round(float(cast(Any, th)) * 100))}%'
    return 'No pruning'


def _facet_row_label_order() -> list[str]:
    """Facet row order from config when present; otherwise a sensible default."""
    if TRACKING_ACCURACY_THRESHOLDS:
        return [_pruning_accuracy_label(t) for t in TRACKING_ACCURACY_THRESHOLDS]
    return ['No pruning']


def _tracking_threshold_allowed(th: object) -> bool:
    """True when ``th`` is in the configured EXEC sweep (including unpruned)."""
    if TRACKING_ACCURACY_THRESHOLDS is None:
        return True
    allowed = TRACKING_ACCURACY_THRESHOLDS
    include_no_pruning = any(t is None for t in allowed)
    allowed_vals: list[float] = [float(cast(Any, t)) for t in allowed if t is not None]
    if th is None:
        return include_no_pruning
    if not isinstance(th, (int, float, np.integer, np.floating)):
        return False
    th_f = float(cast(Any, th))
    for a in allowed_vals:
        if np.isclose(th_f, a):
            return True
    return False


def _filter_config(parsed: dict[str, object]) -> bool:
    """Return True when a config matches the experimental slice we plot."""
    # Only every-frame compression + canonical canvas scale are reported.
    if parsed.get('sample_rate') != SAMPLE_RATE:
        return False
    canvas_scale = parsed.get('canvas_scale')
    if canvas_scale is None or not isinstance(canvas_scale, (int, float, np.integer, np.floating)):
        return False
    if not np.isclose(float(cast(Any, canvas_scale)), CANVAS_SCALE):
        return False
    # Pin to the canonical T_r so the padding metric (and any downstream
    # consumer of the JSONL) sees a single, consistent relevancy convention.
    relevance_threshold = parsed.get('relevance_threshold')
    if relevance_threshold is None or not isinstance(
        relevance_threshold, (int, float, np.integer, np.floating)
    ):
        return False
    if not np.isclose(float(cast(Any, relevance_threshold)), RELEVANCE_THRESHOLD):
        return False
    # Drop rows outside the configured search space (guards against stale dirs).
    if parsed.get('classifier') not in CLASSIFIERS:
        return False
    if parsed.get('tilesize') not in TILE_SIZES:
        return False
    if parsed.get('tilepadding') not in TILEPADDING_MODES:
        return False
    if not _tracking_threshold_allowed(parsed.get('tracking_accuracy_threshold')):
        return False
    return True


def _canonical_poly_tracking_threshold() -> float | None:
    """Pick the pruning threshold used for per-dataset polyomino statistics.

    Pinned to ``None`` (no pruning) so the polyomino-shape macros agree
    with the figure's canonical slice (``tilepadding=none``,
    ``tracking_accuracy_threshold=None``). The un-pruned slice also
    reflects the dataset's raw foreground rather than any post-pruning
    subset, so the prose statements about polyomino-shape distributions
    remain meaningful.
    """
    return None


def _is_canonical_poly_slice(tilepadding: str, tracking_th: float | None) -> bool:
    """True when this (tilepadding, tracking_th) pair is the canonical poly-stats slice."""
    if tilepadding != POLY_STATS_TILEPADDING:
        return False
    target = _canonical_poly_tracking_threshold()
    if target is None:
        return tracking_th is None
    if tracking_th is None:
        return False
    return bool(np.isclose(float(tracking_th), target))


def _collect_poly_sizes(
    config_dir: Path,
) -> tuple[int, dict[int, int], dict[int, int], dict[int, int]] | None:
    """Return per-config polyomino-shape histograms.

    The result is ``(num_canvases, size_hist, bbox_h_hist, bbox_w_hist)`` where
    each histogram maps a value to the count of polyominoes with that value
    summed across every canvas in this config. Each ``index_maps/*.npy`` file
    is one canvas with integer polyomino IDs (``0`` = empty tile, positive =
    polyomino ID).
    """
    index_maps_dir = config_dir / 'index_maps'
    if not index_maps_dir.exists():
        return None
    size_hist: Counter = Counter()
    bbox_h_hist: Counter = Counter()
    bbox_w_hist: Counter = Counter()
    n_canvases = 0
    for npy in index_maps_dir.glob('*.npy'):
        try:
            m = np.load(str(npy))
        except Exception:
            # Stale / corrupt file; skip rather than poison the pool.
            continue
        n_canvases += 1
        # ``np.unique`` + ``return_counts`` gives us tile counts per polyomino
        # ID in a single pass; we drop the 0 (empty) bin.
        ids, counts = np.unique(m, return_counts=True)
        for uid, cnt in zip(ids.tolist(), counts.tolist()):
            if uid == 0:
                continue
            size_hist[int(cnt)] += 1
        # Bounding-box extents per polyomino require per-id masking; iterate
        # over non-zero ids only so trivially-empty canvases stay free.
        nz_ids = [int(uid) for uid in ids.tolist() if uid != 0]
        if nz_ids:
            for uid in nz_ids:
                ys, xs = np.where(m == uid)
                if ys.size == 0:
                    continue
                bbox_h_hist[int(ys.max() - ys.min() + 1)] += 1
                bbox_w_hist[int(xs.max() - xs.min() + 1)] += 1
    if n_canvases == 0:
        return None
    # Return plain dicts so the result pickles cleanly back to the parent.
    return n_canvases, dict(size_hist), dict(bbox_h_hist), dict(bbox_w_hist)


def _count_one_config(
    task: tuple[str, str, str, str, str, int, float | None],
) -> dict[str, object] | None:
    """Worker: count tiles for a single ``(dataset, video, config_dir)`` triple.

    The worker signature accepts primitive types so multiprocessing can
    pickle the arguments without carrying ``Path`` instances.
    """
    dataset, video, config_dir_str, classifier, tilepadding, tilesize, tracking_th = task

    # Reconstruct the rich ``Path`` locally; the tile-counting helper needs it.
    config_dir = Path(config_dir_str)
    try:
        empty, occupied, padding = count_tiles_for_path(
            config_dir, dataset, video, classifier, tilesize,
            threshold=RELEVANCE_THRESHOLD,
        )
    except Exception as exc:  # pragma: no cover - defensive against stale dirs
        # One bad config shouldn't wipe the whole run; log and move on.
        print(f"  [{dataset}/{video}] Skip {config_dir.name}: {exc}")
        return None

    # Skip rows with no occupied tiles -- they contribute nothing to the ratio.
    if occupied <= 0:
        return None

    record: dict[str, object] = {
        'dataset': dataset,
        'video': video,
        'classifier': classifier,
        'tilesize': int(tilesize),
        'tilepadding': str(tilepadding),
        'tracking_accuracy_threshold': tracking_th,
        'empty_tiles': int(empty),
        'occupied_tiles': int(occupied),
        'padding_tiles': int(padding),
    }

    # Only compute polyomino-size statistics on the canonical slice; these
    # rows feed the per-dataset ``\packingEfficacy<Key>Poly*`` macros but are
    # stripped from the aggregated DataFrame before JSONL/chart writers run.
    if _is_canonical_poly_slice(str(tilepadding), tracking_th):
        poly = _collect_poly_sizes(config_dir)
        if poly is not None:
            n_canv, size_hist, bbox_h_hist, bbox_w_hist = poly
            record['_poly_n_canvases'] = int(n_canv)
            record['_poly_size_hist'] = size_hist
            record['_poly_bbox_h_hist'] = bbox_h_hist
            record['_poly_bbox_w_hist'] = bbox_w_hist

    return record


def collect_rows(verbose: bool, videoset: str) -> pd.DataFrame:
    """Walk every dataset, count tiles per config, and return aggregated rows."""
    # Build a single flat task list so the pool can load-balance across datasets.
    tasks: list[tuple[str, str, str, str, str, int, float | None]] = []
    dataset_task_counts: dict[str, int] = {}

    video_prefix = _VIDEOSET_TO_PREFIX[videoset]
    for dataset in DATASETS:
        triples = _iter_config_dirs(dataset, video_prefix)
        if not triples:
            print(
                f"  Skip {dataset}: no {COMPRESSION_STAGE} outputs found for "
                f"videoset={videoset} ({video_prefix}*)"
            )
            continue

        added = 0
        for _video_dir, video_name, config_dir in triples:
            try:
                parsed = parse_execution_param_str(config_dir.name)
            except (AssertionError, ValueError) as exc:
                if verbose:
                    print(f"  [{dataset}/{video_name}] Unparsable config {config_dir.name}: {exc}")
                continue
            if not _filter_config(parsed):
                continue

            th_raw = parsed.get('tracking_accuracy_threshold')
            th: float | None
            if th_raw is None:
                th = None
            else:
                th = float(th_raw)  # type: ignore[arg-type]
            tasks.append((
                dataset,
                video_name,
                str(config_dir),
                str(parsed['classifier']),
                str(parsed['tilepadding']),
                int(parsed['tilesize']),  # type: ignore[arg-type]
                th,
            ))
            added += 1

        dataset_task_counts[dataset] = added
        if verbose:
            print(f"  [{dataset}] Queued {added} configs at sample_rate={SAMPLE_RATE}, canvas_scale={CANVAS_SCALE:g}")

    if not tasks:
        print(f"  No configs matched sample_rate={SAMPLE_RATE} / canvas_scale={CANVAS_SCALE:g}; nothing to do")
        return pd.DataFrame()

    # Tile counting reads dozens of .npy / .jsonl files per config -- parallelize.
    records: list[dict[str, object]] = []
    num_workers = max(1, min(multiprocessing.cpu_count(), len(tasks)))
    with multiprocessing.Pool(processes=num_workers) as pool:
        with Progress(
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}% ({task.completed}/{task.total})"),
            transient=False,
        ) as progress:
            job = progress.add_task("Counting tiles", total=len(tasks))
            for result in pool.imap_unordered(_count_one_config, tasks):
                if result is not None:
                    records.append(result)
                progress.update(job, advance=1)

    if not records:
        return pd.DataFrame()

    raw_df = pd.DataFrame(records)

    # Aggregate per-dataset polyomino-size statistics from the canonical
    # slice before groupby drops the list/dict-valued helper columns.
    poly_stats = _aggregate_poly_stats(raw_df)

    # Sum over videos + (classifier, tilesize) within each
    # (dataset, tilepadding, pruning threshold) so the efficacy ratio is
    # tile-volume-weighted, not mean-of-means.
    # ``dropna=False`` is critical: the unpruned slice has
    # ``tracking_accuracy_threshold=None`` which becomes ``NaN`` after
    # DataFrame construction, and pandas' default ``dropna=True`` would
    # silently drop every unpruned row (the "no pruning" facet).
    grouped = (
        raw_df.groupby(
            ['dataset', 'tilepadding', 'tracking_accuracy_threshold'],
            as_index=False,
            dropna=False,
        )[['empty_tiles', 'occupied_tiles', 'padding_tiles']].sum()
    )

    # Total canvas tiles = empty + occupied. Each .npy index map covers one
    # canvas of (canvas_h_tiles * canvas_w_tiles) cells, so summing across
    # all canvases gives ``num_canvases * canvas_h_tiles * canvas_w_tiles``.
    grouped['total_canvas_tiles'] = (
        grouped['empty_tiles'].astype(float) + grouped['occupied_tiles'].astype(float)
    )
    total = grouped['total_canvas_tiles']
    # Packing efficacy: how much of the canvas the packer actually uses,
    # independent of which tiles are "relevant" vs padding.
    grouped['packing_efficacy_pct'] = np.where(
        total > 0,
        100.0 * grouped['occupied_tiles'].astype(float) / total,
        np.nan,
    )
    grouped['pruning_accuracy_label'] = grouped['tracking_accuracy_threshold'].map(
        _pruning_accuracy_label
    )
    # Carry poly stats through ``df.attrs`` so downstream writers (macros)
    # can consume them without leaking list/dict cells into JSONL/markdown.
    grouped.attrs['poly_stats'] = poly_stats
    return grouped


def _median_from_hist(hist: dict[int, int]) -> float | None:
    """Compute the (lower) median of a histogram ``{value: count}``.

    Using a histogram avoids materializing a potentially large list of
    integer tile counts per polyomino, which can easily run into the
    hundreds of thousands across all canvases in a dataset.
    """
    total = sum(hist.values())
    if total <= 0:
        return None
    target = (total - 1) // 2  # zero-indexed lower-median position
    cumulative = 0
    for value in sorted(hist.keys()):
        cumulative += hist[value]
        if cumulative > target:
            return float(value)
    return None


def _aggregate_poly_stats(raw_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate canonical-slice polyomino stats into per-dataset summaries.

    Returns a mapping ``dataset -> {polys_per_canvas, median_tiles,
    mean_tiles, mean_bbox_h, mean_bbox_w, total_polys, n_canvases}``.
    Datasets with no canonical-slice rows are omitted so the caller can
    detect missingness cleanly.
    """
    if '_poly_size_hist' not in raw_df.columns:
        return {}
    summary: dict[str, dict[str, float]] = {}
    # ``_poly_size_hist`` is object-dtype; ``notna`` treats the missing
    # entries (other slices) as NaN and filters them out in one step.
    canonical = raw_df[raw_df['_poly_size_hist'].notna()]
    if canonical.empty:
        return {}
    for dataset, group in canonical.groupby('dataset'):
        size_combined: Counter = Counter()
        bbox_h_combined: Counter = Counter()
        bbox_w_combined: Counter = Counter()
        n_canvases = 0
        for _, row in group.iterrows():
            sz = row.get('_poly_size_hist')
            if isinstance(sz, dict):
                for k, v in sz.items():
                    size_combined[int(k)] += int(v)
            bh = row.get('_poly_bbox_h_hist')
            if isinstance(bh, dict):
                for k, v in bh.items():
                    bbox_h_combined[int(k)] += int(v)
            bw = row.get('_poly_bbox_w_hist')
            if isinstance(bw, dict):
                for k, v in bw.items():
                    bbox_w_combined[int(k)] += int(v)
            n_canvases += int(row.get('_poly_n_canvases', 0) or 0)
        total_polys = sum(size_combined.values())
        if total_polys == 0 or n_canvases == 0:
            continue
        mean_tiles = sum(k * v for k, v in size_combined.items()) / total_polys
        median_tiles = _median_from_hist(size_combined) or 0.0
        bh_total = sum(bbox_h_combined.values())
        bw_total = sum(bbox_w_combined.values())
        mean_bbox_h = (
            sum(k * v for k, v in bbox_h_combined.items()) / bh_total
            if bh_total > 0 else 0.0
        )
        mean_bbox_w = (
            sum(k * v for k, v in bbox_w_combined.items()) / bw_total
            if bw_total > 0 else 0.0
        )
        summary[str(dataset)] = {
            'polys_per_canvas': total_polys / n_canvases,
            'median_tiles': float(median_tiles),
            'mean_tiles': float(mean_tiles),
            'mean_bbox_h': float(mean_bbox_h),
            'mean_bbox_w': float(mean_bbox_w),
            'total_polys': float(total_polys),
            'n_canvases': float(n_canvases),
        }
    return summary


def _dataset_macro_key(display: str) -> str:
    """Map a human-readable dataset display name to a TeX-safe camelCase key.

    Digits are spelled out (``"B3D 2"`` -> ``"BThreeDTwo"``) and whitespace
    is stripped so the result is a legal ``\\newcommand`` suffix.
    """
    digit_words = {
        '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
        '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
    }
    out: list[str] = []
    for ch in str(display):
        if ch.isspace():
            continue
        if ch.isdigit():
            out.append(digit_words[ch])
        elif ch.isalpha():
            out.append(ch)
        # Drop any other punctuation so the resulting macro name stays safe.
    return ''.join(out)


def _facet_row_sort_labels(plot_df: pd.DataFrame) -> list[str]:
    """Order row facets: config order first, then any extra labels lexically."""
    preferred = _facet_row_label_order()
    present = set(plot_df['pruning_accuracy_label'].dropna().unique())
    ordered = [lab for lab in preferred if lab in present]
    ordered.extend(sorted(lab for lab in present if lab not in ordered))
    return ordered


def write_chart(df: pd.DataFrame, path: str) -> None:
    """Grouped bar chart faceted by pruning threshold (rows).

    Within each row: X=tile-padding, color/x-offset=dataset, Y=efficacy %.
    """
    if df.empty:
        return

    plot_df = _add_dataset_display_names(df)

    # Keep the tile-padding axis in config order so paddings line up with the
    # order the runtime uses; fall back to whatever is present otherwise.
    present_paddings = [mode for mode in TILEPADDING_MODES if mode in set(plot_df['tilepadding'])]
    extra_paddings = [
        mode for mode in plot_df['tilepadding'].dropna().unique() if mode not in present_paddings
    ]
    padding_sort = present_paddings + extra_paddings

    row_sort = _facet_row_sort_labels(plot_df)

    # Match p201/p205 categorical coloring so datasets keep a stable palette.
    dataset_domain = _get_dataset_display_sort(plot_df) or []
    color_scale = (
        alt.Scale(domain=dataset_domain, scheme=SYSTEM_COLOR_SCHEME)
        if dataset_domain
        else alt.Scale(scheme=SYSTEM_COLOR_SCHEME)
    )

    chart = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X('tilepadding:N', title='Tile Padding', sort=padding_sort),
            xOffset=alt.XOffset('dataset_display:N', sort=dataset_domain or None),
            y=alt.Y(
                'packing_efficacy_pct:Q',
                title='Packing Efficacy (%)',
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                'dataset_display:N',
                legend=alt.Legend(title=None),
                scale=color_scale,
            ),
            tooltip=[
                alt.Tooltip('pruning_accuracy_label:N', title='Pruning accuracy'),
                alt.Tooltip('dataset_display:N', title='Dataset'),
                alt.Tooltip('tilepadding:N', title='Tile Padding'),
                alt.Tooltip('packing_efficacy_pct:Q', title='Efficacy (%)', format='.2f'),
                alt.Tooltip('occupied_tiles:Q', title='Occupied tiles', format=','),
                alt.Tooltip('total_canvas_tiles:Q', title='Canvas tiles', format=','),
            ],
        )
        .properties(width=CHART_WIDTH, height=CHART_ROW_HEIGHT)
        .facet(
            row=alt.Row(
                'pruning_accuracy_label:N',
                title='Pruning accuracy',
                sort=row_sort,
                header=alt.Header(labelFontSize=10),
            ),
        )
        .resolve_scale(y='shared')
        .configure_view(stroke=None)
    )
    chart.save(path, scale_factor=2)
    png_path = f'{os.path.splitext(path)[0]}.png'
    chart.save(png_path, scale_factor=4)


def _autogen(inner: str) -> str:
    """Wrap generated TeX replacement text for ``\\autogen`` tagging in the paper."""
    return r'\autogen{%s}' % inner


def _tex_escape(text: str) -> str:
    """Escape TeX special characters so dataset display names render safely."""
    return (
        str(text)
        .replace('\\', r'\textbackslash{}')
        .replace('{', r'\{')
        .replace('}', r'\}')
        .replace('#', r'\#')
        .replace('%', r'\%')
        .replace('&', r'\&')
        .replace('_', r'\_')
        .replace('~', r'\textasciitilde{}')
        .replace('^', r'\textasciicircum{}')
    )


def _fmt(value: float | None, spec: str) -> str:
    """Format a numeric; emit an ``n/a`` placeholder when unavailable."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return r'\mbox{n/a}'
    return format(value, spec)


def write_tex_stat_macros(
    df: pd.DataFrame,
    path: str,
    macro_prefix: str = 'packingEfficacy',
    videoset: str = 'valid',
) -> None:
    """Summarize best/average/worst packing efficacy into ``\\newcommand`` macros.

    ``macro_prefix`` lets the caller emit a parallel macro family (e.g.,
    ``packingEfficacyTest``) when running on a non-default split, so both
    splits' macros can coexist if the paper ever needs them.
    """
    lines = [
        f'% Auto-generated by evaluation/{SCRIPT_ARTIFACT_BASENAME}.py',
        f'% videoset={videoset} (va*=validation, te*=test).',
        r'% Packing efficacy = occupied_tiles / (occupied_tiles + empty_tiles),',
        r'% i.e. fraction of total canvas tile-cells that the packer uses',
        r'% (independent of tile-padding / relevance),',
        f'% measured at sample_rate={SAMPLE_RATE}, canvas_scale={CANVAS_SCALE:g}.',
        r'% Aggregate macros below are restricted to the figure''s canonical slice:',
        f'% tilepadding={POLY_STATS_TILEPADDING}, tracking_accuracy_threshold=None (no pruning),',
        r'% so every cited number agrees with the rendered figure.',
        '',
    ]

    # Helper so every macro is wrapped in \autogen and follows a consistent shape.
    def emit(name: str, replacement: str) -> None:
        lines.append(r'\newcommand{\%s}{%s}' % (macro_prefix + name, _autogen(replacement)))

    emit('SampleRate', str(int(SAMPLE_RATE)))
    emit('CanvasScale', f'{CANVAS_SCALE:g}')
    split_word = 'validation' if videoset == 'valid' else 'test'
    emit('EvalSplit', split_word)

    # Restrict the aggregate macros to the figure's canonical slice so every
    # cited number matches what the reader sees in the rendered figure.
    # ``tracking_accuracy_threshold`` is the per-row pruning threshold; ``None``
    # is the un-pruned slice (the figure's "no pruning" facet that we now show
    # exclusively).
    if df.empty:
        slice_df = df
    else:
        labeled_full = _add_dataset_display_names(df.copy())
        slice_df = labeled_full[
            (labeled_full['tilepadding'] == POLY_STATS_TILEPADDING)
            & (labeled_full['tracking_accuracy_threshold'].isna())
        ]

    # Without any rows in the slice we still emit placeholder macros so the
    # paper compiles even when the underlying sweep is incomplete.
    if slice_df.empty:
        for macro in (
            'BestValue', 'BestDatasetDisplay',
            'WorstValue', 'WorstDatasetDisplay',
            'MeanAcrossDatasets',
            'DatasetCount',
        ):
            emit(macro, r'\mbox{n/a}')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return

    labeled = slice_df
    values = labeled['packing_efficacy_pct'].astype(float)

    # Rank-one extremes within the slice drive the "best"/"worst" prose.
    best_idx = int(values.idxmax())
    worst_idx = int(values.idxmin())
    best_row = labeled.loc[best_idx]
    worst_row = labeled.loc[worst_idx]

    emit('BestValue', _fmt(float(best_row['packing_efficacy_pct']), '.2f'))
    emit('BestDatasetDisplay', _tex_escape(str(best_row['dataset_display'])))

    emit('WorstValue', _fmt(float(worst_row['packing_efficacy_pct']), '.2f'))
    emit('WorstDatasetDisplay', _tex_escape(str(worst_row['dataset_display'])))

    # Single headline number: mean efficacy across datasets at this slice.
    # With one row per dataset in the slice, this reduces to a plain mean.
    emit('MeanAcrossDatasets', _fmt(float(values.mean()), '.2f'))

    # Dataset count helps the paper describe how the numbers were computed.
    emit('DatasetCount', str(int(labeled['dataset'].nunique())))

    # Per-dataset polyomino-size statistics (paper prose in ``Packing Efficacy``).
    # Keyed by dataset display name so callers cite e.g. ``\packingEfficacyBThreeDTwoPolyPerCanvas``.
    poly_stats = df.attrs.get('poly_stats', {}) if hasattr(df, 'attrs') else {}
    if poly_stats:
        # Look up the display name for each dataset from the already-labelled
        # frame so we don't duplicate the mapping logic.
        display_map = (
            labeled[['dataset', 'dataset_display']].drop_duplicates().set_index('dataset')['dataset_display']
        )
        canonical_th = _canonical_poly_tracking_threshold()
        th_str = (
            _pruning_accuracy_label(canonical_th)
            if canonical_th is not None
            else 'No pruning'
        )
        lines.append('')
        lines.append(
            '% Per-dataset polyomino-size statistics, canonical slice: '
            f"tilepadding={POLY_STATS_TILEPADDING}, pruning accuracy={th_str}."
        )
        for dataset_name in sorted(poly_stats.keys()):
            stats = poly_stats[dataset_name]
            display = str(display_map.get(dataset_name, dataset_name))
            key = _dataset_macro_key(display)
            if not key:
                continue
            emit(f'{key}PolyPerCanvas', _fmt(stats['polys_per_canvas'], '.1f'))
            # Median is already integer-valued for a histogram over integer
            # tile counts, so format with no decimal places.
            emit(f'{key}PolyMedianTiles', _fmt(stats['median_tiles'], '.0f'))
            emit(f'{key}PolyMeanTiles', _fmt(stats['mean_tiles'], '.2f'))
            # Mean bbox extents (in tile units) describe polyomino aspect; the
            # paper cites them to contrast tall+square vs wide+thin shapes.
            if 'mean_bbox_h' in stats:
                emit(f'{key}PolyMeanBboxH', _fmt(stats['mean_bbox_h'], '.2f'))
            if 'mean_bbox_w' in stats:
                emit(f'{key}PolyMeanBboxW', _fmt(stats['mean_bbox_w'], '.2f'))

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _copy_paper_figure_artifacts(source_dir: str | Path, base_name: str, destination_dir: str | Path) -> None:
    """Copy publication PDF/PNG, stats jsonl, and TeX macros into ``paper/figures/generated``."""
    src_root = Path(source_dir)
    dst_root = Path(destination_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.png', '.jsonl'):
        src = src_root / f'{base_name}{ext}'
        if not src.is_file():
            continue
        dst = dst_root / f'{base_name}{ext}'
        shutil.copy2(src, dst)
        print(f"  Copied to paper figures: {dst}")
    tex_src = src_root / f'{base_name}_macros.tex'
    if tex_src.is_file():
        tex_dst = dst_root / f'{base_name}_macros.tex'
        shutil.copy2(tex_src, tex_dst)
        print(f"  Copied to paper figures: {tex_dst}")


def write_markdown(df: pd.DataFrame, path: str, videoset: str) -> None:
    """Emit a short Markdown table alongside the chart for quick inspection."""
    lines = [
        f'# Packing efficacy (videoset={videoset}, sample_rate={SAMPLE_RATE}, canvas_scale={CANVAS_SCALE:g})',
        '',
        '| dataset | pruning | tile-padding | efficacy (%) | occupied | canvas tiles |',
        '|---------|---------|--------------|--------------|----------|--------------|',
    ]
    if df.empty:
        lines.append('| (no data) | | | | | |')
        lines.append('')
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        return
    # Sort so the table reads best-first within each dataset -- easier to scan.
    ordered = df.sort_values(
        ['dataset', 'pruning_accuracy_label', 'packing_efficacy_pct'],
        ascending=[True, True, False],
    )
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['pruning_accuracy_label']} | {row['tilepadding']} | "
            f"{float(row['packing_efficacy_pct']):.2f} | {int(row['occupied_tiles'])} | "
            f"{int(row['total_canvas_tiles'])} |"
        )
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main() -> None:
    args = parse_args()
    videoset = _resolve_videoset(args)

    # Unsuffixed outputs and ``\\packingEfficacy*`` macros: validation (paper).
    # Test runs are suffixed so they do not clobber the canonical artifacts.
    base_name = (
        SCRIPT_ARTIFACT_BASENAME
        if videoset == 'valid'
        else f'{SCRIPT_ARTIFACT_BASENAME}_{videoset}'
    )
    macro_prefix = (
        'packingEfficacy' if videoset == 'valid' else f'packingEfficacy{videoset.capitalize()}'
    )

    # Destination for the chart PDF, macros .tex, and intermediate JSONL/MD.
    summary_dir = cache.summary('038_packing_efficacy')
    os.makedirs(summary_dir, exist_ok=True)

    print(f"  Aggregating videoset={videoset} ({_VIDEOSET_TO_PREFIX[videoset]}*)")

    # Single aggregated frame drives every downstream writer.
    df = collect_rows(verbose=args.verbose, videoset=videoset)

    # Persist the raw rows so downstream re-analysis never has to recompute.
    jsonl_path = os.path.join(summary_dir, f'{base_name}.jsonl')
    # ``to_json`` maps NaN to JSON null (``json.dumps`` chokes on float NaN).
    df.to_json(jsonl_path, orient='records', lines=True, double_precision=15)

    # Human-friendly Markdown companion; keeps parity with p205's md dump.
    md_path = os.path.join(summary_dir, f'{base_name}.md')
    write_markdown(df, md_path, videoset=videoset)

    # Main deliverable for the paper: a grouped bar chart (PDF + PNG).
    chart_pdf = os.path.join(summary_dir, f'{base_name}.pdf')
    chart_png = os.path.join(summary_dir, f'{base_name}.png')
    try:
        write_chart(df, chart_pdf)
    except Exception as e:
        print(f"Chart skipped: {e}")

    # TeX macros summarizing best/average/worst for the paper prose.
    tex_macros_path = os.path.join(summary_dir, f'{base_name}_macros.tex')
    write_tex_stat_macros(
        df, tex_macros_path, macro_prefix=macro_prefix, videoset=videoset
    )
    print(f"  Wrote TeX macros: {tex_macros_path}")

    # Canonical validation split updates ``paper/figures/generated``; test runs
    # stay a side-by-side cache-only artifact.
    if videoset == 'valid':
        _copy_paper_figure_artifacts(summary_dir, base_name, PAPER_FIGURES_GENERATED_DIR)
    else:
        print(f"  Skipping paper-figures copy for videoset={videoset}")

    print(f"Wrote {jsonl_path} ({len(df)} rows), {md_path}, {chart_pdf}, {chart_png}")


if __name__ == '__main__':
    main()
