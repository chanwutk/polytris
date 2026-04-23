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

# Restrict the sweep to the test split -- ``te*`` videos -- matching the
# convention used by p036 and other evaluation scripts.
TEST_VIDEO_PREFIX: str = 'te'

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
    return parser.parse_args()


def _iter_config_dirs(dataset: str) -> list[tuple[Path, str, Path]]:
    """Yield ``(video_dir, video_name, config_dir)`` triples for the test split.

    Scanning is split out so ``collect_rows`` can fan out over the work with
    a multiprocessing pool without reconstructing the iteration logic.
    """
    # Rebuild the on-disk root p030 writes to for this dataset.
    dataset_exec_dir = cache.execution(dataset)
    if not dataset_exec_dir.exists():
        return []

    triples: list[tuple[Path, str, Path]] = []
    # Only consider test videos -- valid-split outputs are not part of the
    # packing efficacy story we report in the paper.
    for video_dir in sorted(dataset_exec_dir.iterdir()):
        if not video_dir.is_dir() or not video_dir.name.startswith(TEST_VIDEO_PREFIX):
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

    return {
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


def collect_rows(verbose: bool) -> pd.DataFrame:
    """Walk every dataset, count tiles per config, and return aggregated rows."""
    # Build a single flat task list so the pool can load-balance across datasets.
    tasks: list[tuple[str, str, str, str, str, int, float | None]] = []
    dataset_task_counts: dict[str, int] = {}

    for dataset in DATASETS:
        triples = _iter_config_dirs(dataset)
        if not triples:
            print(f"  Skip {dataset}: no {COMPRESSION_STAGE} outputs found")
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

    # Sum over videos + (classifier, tilesize) within each
    # (dataset, tilepadding, pruning threshold) so the efficacy ratio is
    # tile-volume-weighted, not mean-of-means.
    grouped = (
        raw_df.groupby(['dataset', 'tilepadding', 'tracking_accuracy_threshold'], as_index=False)[
            ['empty_tiles', 'occupied_tiles', 'padding_tiles']
        ].sum()
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
    return grouped


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


def write_tex_stat_macros(df: pd.DataFrame, path: str) -> None:
    """Summarize best/average/worst packing efficacy into ``\\newcommand`` macros."""
    lines = [
        f'% Auto-generated by evaluation/{SCRIPT_ARTIFACT_BASENAME}.py',
        r'% Packing efficacy = occupied_tiles / (occupied_tiles + empty_tiles),',
        r'% i.e. fraction of total canvas tile-cells that the packer uses',
        r'% (independent of tile-padding / relevance),',
        f'% measured at sample_rate={SAMPLE_RATE}, canvas_scale={CANVAS_SCALE:g}.',
        '',
    ]

    # Helper so every macro is wrapped in \autogen and follows a consistent shape.
    def emit(name: str, replacement: str) -> None:
        lines.append(r'\newcommand{\%s}{%s}' % (name, _autogen(replacement)))

    emit('packingEfficacySampleRate', str(int(SAMPLE_RATE)))
    emit('packingEfficacyCanvasScale', f'{CANVAS_SCALE:g}')

    # Without any rows we still emit placeholder macros to avoid compile breaks.
    if df.empty:
        for macro in (
            'packingEfficacyBestValue', 'packingEfficacyBestDatasetDisplay',
            'packingEfficacyBestTilePadding', 'packingEfficacyWorstValue',
            'packingEfficacyWorstDatasetDisplay', 'packingEfficacyWorstTilePadding',
            'packingEfficacyMeanValue', 'packingEfficacyMinValue', 'packingEfficacyMaxValue',
            'packingEfficacyBestPerDatasetMean',
            'packingEfficacyDatasetCount', 'packingEfficacyTilePaddingCount',
            'packingEfficacyRowCount',
        ):
            emit(macro, r'\mbox{n/a}')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return

    # Reuse the shared display-name mapping so macros read like the paper prose.
    labeled = _add_dataset_display_names(df.copy())
    values = labeled['packing_efficacy_pct'].astype(float)

    # Rank-one extremes drive the "best"/"worst" prose in the paper.
    best_idx = int(values.idxmax())
    worst_idx = int(values.idxmin())
    best_row = labeled.loc[best_idx]
    worst_row = labeled.loc[worst_idx]

    emit('packingEfficacyBestValue', _fmt(float(best_row['packing_efficacy_pct']), '.2f'))
    emit('packingEfficacyBestDatasetDisplay', _tex_escape(str(best_row['dataset_display'])))
    emit('packingEfficacyBestTilePadding', _tex_escape(str(best_row['tilepadding'])))

    emit('packingEfficacyWorstValue', _fmt(float(worst_row['packing_efficacy_pct']), '.2f'))
    emit('packingEfficacyWorstDatasetDisplay', _tex_escape(str(worst_row['dataset_display'])))
    emit('packingEfficacyWorstTilePadding', _tex_escape(str(worst_row['tilepadding'])))

    # Aggregate statistics: macro-average across (dataset, padding) cells so
    # every cell weighs equally, matching how the bar chart reads visually.
    emit('packingEfficacyMeanValue', _fmt(float(values.mean()), '.2f'))
    emit('packingEfficacyMinValue', _fmt(float(values.min()), '.2f'))
    emit('packingEfficacyMaxValue', _fmt(float(values.max()), '.2f'))

    # Best-per-dataset average shows how well the best configuration packs
    # tiles on average -- useful when the paper wants a single headline number.
    best_per_dataset = (
        labeled.groupby('dataset_display', as_index=False)['packing_efficacy_pct']
        .max()
    )
    emit(
        'packingEfficacyBestPerDatasetMean',
        _fmt(float(best_per_dataset['packing_efficacy_pct'].mean()), '.2f'),
    )

    # Grain counts help the paper describe how the numbers were computed.
    emit('packingEfficacyDatasetCount', str(int(labeled['dataset'].nunique())))
    emit('packingEfficacyTilePaddingCount', str(int(labeled['tilepadding'].nunique())))
    emit('packingEfficacyRowCount', str(int(len(labeled))))

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


def write_markdown(df: pd.DataFrame, path: str) -> None:
    """Emit a short Markdown table alongside the chart for quick inspection."""
    lines = [
        f'# Packing efficacy (sample_rate={SAMPLE_RATE}, canvas_scale={CANVAS_SCALE:g})',
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

    # Destination for the chart PDF, macros .tex, and intermediate JSONL/MD.
    summary_dir = cache.summary('038_packing_efficacy')
    os.makedirs(summary_dir, exist_ok=True)

    # Single aggregated frame drives every downstream writer.
    df = collect_rows(verbose=args.verbose)

    # Persist the raw rows so downstream re-analysis never has to recompute.
    jsonl_path = os.path.join(summary_dir, f'{SCRIPT_ARTIFACT_BASENAME}.jsonl')
    # ``to_json`` maps NaN to JSON null (``json.dumps`` chokes on float NaN).
    df.to_json(jsonl_path, orient='records', lines=True, double_precision=15)

    # Human-friendly Markdown companion; keeps parity with p205's md dump.
    md_path = os.path.join(summary_dir, f'{SCRIPT_ARTIFACT_BASENAME}.md')
    write_markdown(df, md_path)

    # Main deliverable for the paper: a grouped bar chart (PDF + PNG).
    chart_pdf = os.path.join(summary_dir, f'{SCRIPT_ARTIFACT_BASENAME}.pdf')
    chart_png = os.path.join(summary_dir, f'{SCRIPT_ARTIFACT_BASENAME}.png')
    try:
        write_chart(df, chart_pdf)
    except Exception as e:
        print(f"Chart skipped: {e}")

    # TeX macros summarizing best/average/worst for the paper prose.
    tex_macros_path = os.path.join(summary_dir, f'{SCRIPT_ARTIFACT_BASENAME}_macros.tex')
    write_tex_stat_macros(df, tex_macros_path)
    print(f"  Wrote TeX macros: {tex_macros_path}")

    # Copy PDF + macros into paper/figures/generated so latex picks them up.
    _copy_paper_figure_artifacts(summary_dir, SCRIPT_ARTIFACT_BASENAME, PAPER_FIGURES_GENERATED_DIR)

    print(f"Wrote {jsonl_path} ({len(df)} rows), {md_path}, {chart_pdf}, {chart_png}")


if __name__ == '__main__':
    main()
