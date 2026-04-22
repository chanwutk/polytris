#!/usr/local/bin/python

"""Compare experimental vs image-only baseline tile classifiers against GT detections."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPER_FIGURES_GENERATED_DIR = os.path.join(REPO_ROOT, 'paper', 'figures', 'generated')
SCRIPT_ARTIFACT_BASENAME = 'p205_compare_classifiers'

import altair as alt
import numpy as np
import pandas as pd

from polyis.io import cache, store
from evaluation.p201_compare_pareto import (
    FACET_SUBPLOT_HEIGHT,
    FACET_SUBPLOT_WIDTH,
    SYSTEM_COLOR_SCHEME,
    _add_dataset_display_names,
    _get_dataset_display_sort,
)
from polyis.utilities import (
    get_accuracy,
    get_config,
    get_f1_score,
    get_precision,
    get_recall,
    load_classification_results,
    load_detection_results,
    mark_detections,
)

FACET_SUBPLOT_WIDTH = 150
FACET_SUBPLOT_HEIGHT = 200

config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
DATASETS: list[str] = config['EXEC']['DATASETS']
CLASSIFIERS: list[str] = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']

# Fixed sampling rate for this experiment: every frame.
SAMPLE_RATE: int = 1

# Experimental classifiers that have a baseline twin (must match p017 / p020b).
BASELINE_PAIRS: dict[str, str] = {'ShuffleNet05': 'ShuffleNet05Baseline'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare experimental vs baseline classifier metrics (F1, accuracy, precision, recall)'
    )
    parser.add_argument('--test', action='store_true', help='Evaluate test videoset')
    parser.add_argument('--valid', action='store_true', help='Evaluate valid videoset')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold for scores in [0,1]')
    return parser.parse_args()


def evaluate_classification_accuracy(
    classifications: np.ndarray,
    detections: list[list[float]],
    tile_size: int,
    threshold: float,
    eval_mask: np.ndarray,
) -> dict[str, Any]:
    """Per-frame tile metrics vs GT boxes (detections are [x1,y1,x2,y2,score]).

    Tiles where ``eval_mask`` is zero are considered always-discarded and are
    skipped entirely (they count toward neither TP/TN nor FP/FN).
    """
    grid_height = classifications.shape[0]
    grid_width = classifications.shape[1] if grid_height > 0 else 0
    assert eval_mask.shape == (grid_height, grid_width), (
        f"eval_mask shape {eval_mask.shape} does not match grid "
        f"({grid_height},{grid_width})"
    )

    metrics = np.zeros((2, 2), dtype=int)

    total_height = grid_height * tile_size
    total_width = grid_width * tile_size

    detection_bitmap = mark_detections(detections, total_width, total_height, tile_size, slice(0, 4))

    counted = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if eval_mask[i, j] == 0:
                continue
            score = classifications[i][j]
            predicted_positive = score >= threshold
            actual_positive = detection_bitmap[i, j] > 0
            metrics[int(predicted_positive), int(actual_positive)] += 1
            counted += 1

    tp, fp, fn, tn = map(int, metrics.flatten().tolist())
    assert tp + fp + fn + tn == counted

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def _decode_classification_hex(entry: dict) -> np.ndarray:
    classifications = entry['classification_hex']
    classification_size = entry['classification_size']
    return (
        np.frombuffer(bytes.fromhex(classifications), dtype=np.uint8)
        .reshape(classification_size)
        .astype(np.float32)
        / 255.0
    )


def _build_gt_frame_map(groundtruth_frames: list[dict]) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    for row in groundtruth_frames:
        if 'frame_idx' in row:
            idx = int(row['frame_idx'])
            dets = row.get('detections', [])
            out[idx] = dets
    return out


def load_runtime(dataset: str, video: str, classifier: str, tile_size: int,
                 sample_rate: int) -> tuple[float, int]:
    """Return (total_runtime_ms, num_classified_frames).

    Every ``time`` field from each ``runtime.jsonl`` line is summed (see
    ``format_time`` in classify scripts: per-batch ops plus one line for
    ``read``/``retrieve``). ``num_classified_frames`` is the line count of
    ``score.jsonl`` so ``total / frames`` is amortized ms/frame over the full
    scoring pipeline for that video.
    """
    runtime_path = Path(
        cache.exec(
            dataset, 'relevancy', video,
            f'{classifier}_{tile_size}_{sample_rate}', 'score', 'runtime.jsonl',
        )
    )
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime file not found: {runtime_path}")

    score_path = runtime_path.parent / 'score.jsonl'

    total_runtime_ms = 0.0
    with runtime_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ops = json.loads(line)
            if not isinstance(ops, list):
                continue
            for op in ops:
                if isinstance(op, dict) and 'time' in op:
                    total_runtime_ms += float(op['time'])

    frame_count = 0
    if score_path.exists():
        with score_path.open() as sf:
            for line in sf:
                if line.strip():
                    frame_count += 1

    if total_runtime_ms > 0 and frame_count == 0:
        print(
            f"Warning: runtime > 0 but no score.jsonl frames at {score_path}; "
            "ms/frame will be 0."
        )

    return total_runtime_ms, frame_count


def load_eval_mask(dataset: str, tile_size: int) -> np.ndarray:
    """Load the always-relevant bitmap used to skip always-discarded tiles."""
    mask_path = cache.index(dataset, 'never-relevant', f'{tile_size}_all.npy')
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Always-relevant mask not found: {mask_path}")
    mask = np.load(mask_path)
    return (mask > 0).astype(np.uint8)


def evaluate_video_pair(
    dataset: str,
    video: str,
    tile_size: int,
    sample_rate: int,
    experimental_name: str,
    baseline_name: str,
    threshold: float,
    eval_mask: np.ndarray,
) -> tuple[dict[str, int], dict[str, int]]:
    gt_path = cache.exec(dataset, 'groundtruth', video, 'detection.jsonl')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing groundtruth detections: {gt_path}")

    gt_frames = load_detection_results(dataset, video, groundtruth=True)
    gt_by_idx = _build_gt_frame_map(gt_frames)

    exp_results = load_classification_results(dataset, video, tile_size, experimental_name, sample_rate)
    base_results = load_classification_results(dataset, video, tile_size, baseline_name, sample_rate)

    exp_by_idx = {int(r['idx']): r for r in exp_results}
    base_by_idx = {int(r['idx']): r for r in base_results}
    common_idx = sorted(set(exp_by_idx.keys()) & set(base_by_idx.keys()))

    totals_exp = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    totals_base = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for idx in common_idx:
        if idx not in gt_by_idx:
            continue
        dets = gt_by_idx[idx]

        exp_grid = _decode_classification_hex(exp_by_idx[idx])
        base_grid = _decode_classification_hex(base_by_idx[idx])
        if exp_grid.shape != base_grid.shape:
            continue

        ev_e = evaluate_classification_accuracy(exp_grid, dets, tile_size, threshold, eval_mask)
        ev_b = evaluate_classification_accuracy(base_grid, dets, tile_size, threshold, eval_mask)

        for k in ('tp', 'tn', 'fp', 'fn'):
            totals_exp[k] += ev_e[k]
            totals_base[k] += ev_b[k]

    if sum(totals_exp.values()) == 0:
        raise ValueError('No overlapping frames with groundtruth for this video')

    return totals_exp, totals_base


def _finalize(totals: dict[str, int], extra: dict[str, Any]) -> dict[str, Any]:
    tp, tn, fp, fn = totals['tp'], totals['tn'], totals['fp'], totals['fn']
    frames = int(extra.get('frames', 0))
    total_runtime_ms = float(
        extra.get('total_runtime_ms', extra.get('total_inference_ms', 0.0))
    )
    ms_per_frame = total_runtime_ms / frames if frames > 0 else 0.0
    return {
        **extra,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': get_precision(tp, fp),
        'recall': get_recall(tp, fn),
        'accuracy': get_accuracy(tp, tn, fp, fn),
        'f1_score': get_f1_score(tp, fp, fn),
        'ms_per_frame': ms_per_frame,
    }


def run_for_videoset(videoset: str, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        videoset_dir = store.dataset(dataset, videoset)
        if not videoset_dir.exists():
            print(f"Skip {dataset}/{videoset}: directory missing")
            continue

        videos = [
            f
            for f in os.listdir(videoset_dir)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        for experimental_name, baseline_name in BASELINE_PAIRS.items():
            if experimental_name not in CLASSIFIERS:
                continue
            for tile_size in TILE_SIZES:
                try:
                    eval_mask = load_eval_mask(dataset, tile_size)
                except FileNotFoundError as e:
                    print(f"Skip {dataset} tile={tile_size}: {e}")
                    continue

                agg = {
                    experimental_name: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                    baseline_name: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                }
                runtime = {
                    experimental_name: {'total_ms': 0.0, 'frames': 0},
                    baseline_name: {'total_ms': 0.0, 'frames': 0},
                }
                n_ok = 0
                for video in sorted(videos):
                    try:
                        te, tb = evaluate_video_pair(
                            dataset, video, tile_size, SAMPLE_RATE,
                            experimental_name, baseline_name, threshold,
                            eval_mask,
                        )
                    except (FileNotFoundError, OSError, ValueError) as e:
                        print(f"Skip {dataset}/{video}: {e}")
                        continue

                    for k in ('tp', 'tn', 'fp', 'fn'):
                        agg[experimental_name][k] += int(te[k])
                        agg[baseline_name][k] += int(tb[k])

                    for name in (experimental_name, baseline_name):
                        try:
                            ms, frames = load_runtime(dataset, video, name, tile_size, SAMPLE_RATE)
                        except FileNotFoundError as e:
                            print(f"Runtime missing for {dataset}/{video} [{name}]: {e}")
                            continue
                        runtime[name]['total_ms'] += ms
                        runtime[name]['frames'] += frames
                    n_ok += 1

                if n_ok == 0:
                    continue

                for name in (experimental_name, baseline_name):
                    rows.append(_finalize(
                        agg[name],
                        {
                            'videoset': videoset,
                            'dataset': dataset,
                            'classifier': name,
                            'modified': name == experimental_name,
                            'tile_size': tile_size,
                            'sample_rate': SAMPLE_RATE,
                            'threshold': threshold,
                            'videos_evaluated': n_ok,
                            'total_runtime_ms': runtime[name]['total_ms'],
                            'frames': runtime[name]['frames'],
                        },
                    ))

    return rows


def write_markdown(rows: list[dict[str, Any]], path: str, videoset: str):
    lines = [
        f'# Classifier comparison ({videoset}, sample_rate={SAMPLE_RATE})',
        '',
        '| dataset | classifier | tile | F1 | Acc | P | R | total ms/frame | videos |',
        '|---------|------------|------|----|-----|---|---|----------------|--------|',
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['classifier']} | {r['tile_size']} | "
            f"{r['f1_score']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['ms_per_frame']:.3f} | {r['videos_evaluated']} |"
        )
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _y_domain_for_metric(df: pd.DataFrame, field: str) -> tuple[float, float]:
    """Padded [min, max] for the metric column, clamped to [0, 1] (no fixed 0–1 axis)."""
    s = df[field].astype(float)
    lo, hi = float(s.min()), float(s.max())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, 1.0
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


def write_chart(rows: list[dict[str, Any]], path: str, _videoset: str):
    if not rows:
        return
    df = _add_dataset_display_names(pd.DataFrame(rows))
    df['classifier_kind'] = np.where(df['modified'], 'Modified', 'Baseline')

    metrics = [
        ('f1_score', 'F1'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('accuracy', 'Accuracy'),
    ]

    # Stable order so mark_line connects Modified -> Baseline per (dataset, tile_size).
    df_line = df.sort_values(
        ['dataset', 'tile_size', 'modified'],
        ascending=[True, True, False],
        kind='stable',
    ).reset_index(drop=True)

    dataset_domain = _get_dataset_display_sort(df_line)
    # Match p201_compare_pareto nominal colors (Vega ``observable10``).
    color_scale = (
        alt.Scale(domain=dataset_domain, scheme=SYSTEM_COLOR_SCHEME)
        if dataset_domain
        else alt.Scale(scheme=SYSTEM_COLOR_SCHEME)
    )
    subplot_w = max(1, int(round(FACET_SUBPLOT_WIDTH)))
    subplot_h = max(1, int(round(FACET_SUBPLOT_HEIGHT)))

    x_vals = df_line['ms_per_frame'].astype(float)
    x_max = float(x_vals.max()) if len(x_vals) else 1.0
    if not (np.isfinite(x_max) and x_max > 0):
        x_max = 1.0
    x_scale = alt.Scale(domain=[0.0, x_max], nice=False)

    subcharts = []
    for field, title in metrics:
        y0, y1 = _y_domain_for_metric(df, field)
        y_scale = alt.Scale(domain=[y0, y1], nice=False, zero=False)
        x_enc = alt.X('ms_per_frame:Q', title='total runtime (ms/frame)', scale=x_scale)
        y_enc = alt.Y(f'{field}:Q', title=title, scale=y_scale)

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
        sub = (lines + points).properties(width=subplot_w, height=subplot_h)
        subcharts.append(sub)

    grid = (
        alt.hconcat(*subcharts, spacing=0)
        .properties(padding=0)
        .configure_view(stroke=None)
    )
    grid.save(path, scale_factor=2)


def _copy_paper_figure_artifacts(source_dir: str | Path, base_name: str, destination_dir: str | Path) -> None:
    """Copy publication PDF, stats jsonl, and TeX macros into ``paper/figures/generated``."""
    src_root = Path(source_dir)
    dst_root = Path(destination_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.jsonl'):
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


def _autogen(inner: str) -> str:
    """Wrap generated TeX replacement text for ``\\autogen`` tagging in the paper."""
    return r'\autogen{%s}' % inner


def _tex_escape(text: str) -> str:
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


def _summarize_best_f1_gain(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Largest (modified − baseline) F1 on matching (dataset, tile_size, videoset)."""
    if not rows:
        return None
    df = _add_dataset_display_names(pd.DataFrame(rows))
    mod = df[df['modified']].copy()
    base = df[~df['modified']].copy()
    if mod.empty or base.empty:
        return None
    merged = mod.merge(
        base,
        on=['dataset', 'tile_size', 'videoset'],
        suffixes=('_mod', '_base'),
        how='inner',
    )
    if merged.empty:
        return None
    merged['f1_delta'] = merged['f1_score_mod'].astype(float) - merged['f1_score_base'].astype(float)
    merged['ms_ratio'] = np.where(
        merged['ms_per_frame_base'].astype(float) > 0,
        merged['ms_per_frame_mod'].astype(float) / merged['ms_per_frame_base'].astype(float),
        np.nan,
    )
    j = int(merged['f1_delta'].values.argmax())
    r = merged.iloc[j]
    return {
        'videoset': str(r['videoset']),
        'dataset': str(r['dataset']),
        'dataset_display': str(r['dataset_display_mod']),
        'tile_size': int(r['tile_size']),
        'f1_delta': float(r['f1_delta']),
        'f1_mod': float(r['f1_score_mod']),
        'f1_base': float(r['f1_score_base']),
        'ms_per_frame_mod': float(r['ms_per_frame_mod']),
        'ms_per_frame_base': float(r['ms_per_frame_base']),
        'ms_ratio': float(r['ms_ratio']) if np.isfinite(r['ms_ratio']) else None,
    }


def _per_dataset_merged(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Collapse tile sizes to one modified/baseline pair per dataset.

    Sums confusion-matrix counts and runtime totals across tile sizes within
    each ``(dataset, classifier)`` bucket so every aggregate macro has a
    consistent per-dataset definition. Returned frame has one row per dataset
    with ``_mod`` and ``_base`` columns.
    """
    if not rows:
        return pd.DataFrame()

    df = _add_dataset_display_names(pd.DataFrame(rows))
    if df.empty or 'modified' not in df.columns:
        return pd.DataFrame()

    sum_cols = ['tp', 'tn', 'fp', 'fn', 'total_runtime_ms', 'frames']
    sum_cols = [c for c in sum_cols if c in df.columns]
    grouped = (
        df.groupby(['dataset', 'dataset_display', 'modified'], as_index=False)[sum_cols]
        .sum(numeric_only=True)
    )

    def _derive(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        counts = (out['tp'] + out['tn'] + out['fp'] + out['fn']).astype(float)
        out['accuracy'] = np.where(counts > 0, (out['tp'] + out['tn']).astype(float) / counts, np.nan)
        f1_denom = (2 * out['tp'] + out['fp'] + out['fn']).astype(float)
        out['f1_score'] = np.where(f1_denom > 0, 2 * out['tp'].astype(float) / f1_denom, np.nan)
        frames = out['frames'].astype(float) if 'frames' in out.columns else 0.0
        runtime = out['total_runtime_ms'].astype(float) if 'total_runtime_ms' in out.columns else 0.0
        out['ms_per_frame'] = np.where(frames > 0, runtime / frames, np.nan)
        return out

    grouped = _derive(grouped)
    mod = grouped[grouped['modified']].drop(columns=['modified'])
    base = grouped[~grouped['modified']].drop(columns=['modified'])
    if mod.empty or base.empty:
        return pd.DataFrame()
    merged = mod.merge(base, on=['dataset', 'dataset_display'], suffixes=('_mod', '_base'), how='inner')
    if merged.empty:
        return merged

    merged['accuracy_delta_pp'] = (
        merged['accuracy_mod'].astype(float) - merged['accuracy_base'].astype(float)
    ) * 100.0
    merged['f1_delta'] = (
        merged['f1_score_mod'].astype(float) - merged['f1_score_base'].astype(float)
    )
    merged['runtime_ratio'] = np.where(
        merged['ms_per_frame_base'].astype(float) > 0,
        merged['ms_per_frame_mod'].astype(float) / merged['ms_per_frame_base'].astype(float),
        np.nan,
    )
    merged['runtime_slower_pct'] = (merged['runtime_ratio'] - 1.0) * 100.0
    return merged


def _micro_accuracy_delta_pp(rows: list[dict[str, Any]]) -> float | None:
    """Global (pooled) accuracy gain in percentage points, across every tile row."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty or 'modified' not in df.columns:
        return None
    mod = df[df['modified']]
    base = df[~df['modified']]
    if mod.empty or base.empty:
        return None
    counts = {}
    for name, sub in (('mod', mod), ('base', base)):
        tp = float(sub['tp'].sum())
        tn = float(sub['tn'].sum())
        fp = float(sub['fp'].sum())
        fn = float(sub['fn'].sum())
        total = tp + tn + fp + fn
        if total <= 0:
            return None
        counts[name] = (tp + tn) / total
    return (counts['mod'] - counts['base']) * 100.0


def _fmt(value: float | None, spec: str) -> str:
    """Format a numeric or return an ``n/a`` placeholder for the TeX macros."""
    if value is None or not np.isfinite(value):
        return r'\mbox{n/a}'
    return format(value, spec)


def write_tex_stat_macros(
    rows: list[dict[str, Any]],
    path: str,
    videoset: str,
    threshold: float,
    sample_rate: int,
) -> None:
    """Write ``\\newcommand`` macros summarizing the classifier comparison."""
    best = _summarize_best_f1_gain(rows)
    per_dataset = _per_dataset_merged(rows)
    micro_acc_pp = _micro_accuracy_delta_pp(rows)

    lines = [
        f'% Auto-generated by evaluation/{SCRIPT_ARTIFACT_BASENAME}.py ({videoset} split).',
        r'% Macros use a numeric prefix spelling (p205) because command names cannot',
        r'% start with digits in LaTeX.',
        '',
    ]

    def emit(name: str, replacement: str) -> None:
        lines.append(r'\newcommand{\%s}{%s}' % (name, _autogen(replacement)))

    emit('PtwozerofiveCompareClassifiersVideoset', _tex_escape(videoset))
    emit('PtwozerofiveCompareClassifiersThreshold', f'{float(threshold):.2f}')
    emit('PtwozerofiveCompareClassifiersSampleRate', str(int(sample_rate)))

    if best is None:
        lines.append(r'% No paired modified/baseline rows; best-F1 macros omitted.')
    else:
        ratio = best['ms_ratio']
        emit('PtwozerofiveCompareClassifiersBestDatasetDisplay', _tex_escape(best['dataset_display']))
        emit('PtwozerofiveCompareClassifiersBestTileSize', str(best['tile_size']))
        emit('PtwozerofiveCompareClassifiersBestFOneDelta', f'{best["f1_delta"]:.4f}')
        emit('PtwozerofiveCompareClassifiersBestFOneModified', f'{best["f1_mod"]:.4f}')
        emit('PtwozerofiveCompareClassifiersBestFOneBaseline', f'{best["f1_base"]:.4f}')
        emit(
            'PtwozerofiveCompareClassifiersModifiedRuntimeSlowdown',
            _fmt(ratio, '.3f'),
        )
        slower_pct = (ratio - 1.0) * 100.0 if (ratio is not None and np.isfinite(ratio)) else None
        emit(
            'PtwozerofiveCompareClassifiersModifiedRuntimeSlowerPct',
            _fmt(slower_pct, '.1f'),
        )

    if per_dataset.empty:
        lines.append(r'% No per-dataset pairs; accuracy/runtime aggregate macros omitted.')
        emit('PtwozerofiveCompareClassifiersMicroAccuracyImprovementPp', _fmt(micro_acc_pp, '.2f'))
    else:
        acc_gains = per_dataset['accuracy_delta_pp'].astype(float)
        f1_deltas = per_dataset['f1_delta'].astype(float)
        slower_pct_series = per_dataset['runtime_slower_pct'].astype(float).replace(
            [np.inf, -np.inf], np.nan
        )

        emit(
            'PtwozerofiveCompareClassifiersMinAccuracyImprovementPp',
            _fmt(float(acc_gains.min()), '.2f'),
        )
        emit(
            'PtwozerofiveCompareClassifiersMaxAccuracyImprovementPp',
            _fmt(float(acc_gains.max()), '.2f'),
        )
        emit(
            'PtwozerofiveCompareClassifiersMacroAvgAccuracyImprovementPp',
            _fmt(float(acc_gains.mean()), '.2f'),
        )
        emit(
            'PtwozerofiveCompareClassifiersMicroAccuracyImprovementPp',
            _fmt(micro_acc_pp, '.2f'),
        )

        j = int(acc_gains.values.argmax())
        best_acc_row = per_dataset.iloc[j]
        emit(
            'PtwozerofiveCompareClassifiersBestAccuracyDatasetDisplay',
            _tex_escape(str(best_acc_row['dataset_display'])),
        )
        emit(
            'PtwozerofiveCompareClassifiersBestAccuracyImprovementPp',
            _fmt(float(best_acc_row['accuracy_delta_pp']), '.2f'),
        )

        slower_mean = float(slower_pct_series.mean(skipna=True)) if slower_pct_series.notna().any() else None
        emit(
            'PtwozerofiveCompareClassifiersMacroAvgRuntimeSlowerPct',
            _fmt(slower_mean, '.1f'),
        )

        improved_count = int((f1_deltas > 0).sum())
        dataset_count = int(len(per_dataset))
        emit('PtwozerofiveCompareClassifiersImprovedFOneDatasetCount', str(improved_count))
        emit('PtwozerofiveCompareClassifiersDatasetCount', str(dataset_count))
        emit('PtwozerofiveCompareClassifiersMinFOneDelta', _fmt(float(f1_deltas.min()), '.4f'))
        emit('PtwozerofiveCompareClassifiersMaxFOneDelta', _fmt(float(f1_deltas.max()), '.4f'))
        emit(
            'PtwozerofiveCompareClassifiersMacroAvgFOneDelta',
            _fmt(float(f1_deltas.mean()), '.4f'),
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError('threshold must be in [0, 1]')

    videosets: list[str] = []
    if args.test:
        videosets.append('test')
    if args.valid:
        videosets.append('valid')
    if not videosets:
        videosets = ['valid']

    summary_dir = cache.summary('205_compare_classifiers')
    os.makedirs(summary_dir, exist_ok=True)

    for vs in videosets:
        rows = run_for_videoset(vs, args.threshold)
        base_name = f'{SCRIPT_ARTIFACT_BASENAME}_{vs}'

        jsonl_path = os.path.join(summary_dir, f'{base_name}.jsonl')
        with open(jsonl_path, 'w') as jf:
            for row in rows:
                jf.write(json.dumps(row) + '\n')

        md_path = os.path.join(summary_dir, f'{base_name}.md')
        write_markdown(rows, md_path, vs)

        chart_pdf = os.path.join(summary_dir, f'{base_name}.pdf')
        try:
            write_chart(rows, chart_pdf, vs)
        except Exception as e:
            print(f"Chart skipped: {e}")

        tex_macros_path = os.path.join(summary_dir, f'{base_name}_macros.tex')
        write_tex_stat_macros(rows, tex_macros_path, vs, args.threshold, SAMPLE_RATE)
        print(f"  Wrote TeX macros: {tex_macros_path}")

        _copy_paper_figure_artifacts(summary_dir, base_name, PAPER_FIGURES_GENERATED_DIR)

        print(f"Wrote {jsonl_path} ({len(rows)} rows), {md_path}, {chart_pdf}")


if __name__ == '__main__':
    main()
