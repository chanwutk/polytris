#!/usr/local/bin/python

"""Compare experimental vs image-only baseline tile classifiers against GT detections."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from polyis.io import cache, store
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


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
DATASETS: list[str] = config['EXEC']['DATASETS']
CLASSIFIERS: list[str] = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']
SAMPLE_RATES: list[int] = config['EXEC']['SAMPLE_RATES']

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
) -> dict[str, Any]:
    """Per-frame tile metrics vs GT boxes (detections are [x1,y1,x2,y2,score])."""
    grid_height = classifications.shape[0]
    grid_width = classifications.shape[1] if grid_height > 0 else 0

    metrics = np.zeros((2, 2), dtype=int)

    total_height = grid_height * tile_size
    total_width = grid_width * tile_size

    detection_bitmap = mark_detections(detections, total_width, total_height, tile_size, slice(0, 4))

    for i in range(grid_height):
        for j in range(grid_width):
            score = classifications[i][j]
            predicted_positive = score >= threshold
            actual_positive = detection_bitmap[i, j] > 0
            metrics[int(predicted_positive), int(actual_positive)] += 1

    tp, fp, fn, tn = map(int, metrics.flatten().tolist())
    assert tp + fp + fn + tn == grid_height * grid_width

    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    accuracy = get_accuracy(tp, tn, fp, fn)
    f1_score = get_f1_score(tp, fp, fn)

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'total_tiles': grid_height * grid_width,
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
        else:
            continue
        out[idx] = dets
    return out


def evaluate_video_pair(
    dataset: str,
    video: str,
    tile_size: int,
    sample_rate: int,
    experimental_name: str,
    baseline_name: str,
    threshold: float,
) -> tuple[dict[str, int | float], dict[str, int | float]]:
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

        ev_e = evaluate_classification_accuracy(exp_grid, dets, tile_size, threshold)
        ev_b = evaluate_classification_accuracy(base_grid, dets, tile_size, threshold)

        for k in ('tp', 'tn', 'fp', 'fn'):
            totals_exp[k] += ev_e[k]
            totals_base[k] += ev_b[k]

    if sum(totals_exp.values()) == 0:
        raise ValueError('No overlapping frames with groundtruth for this video')

    def finalize(totals: dict[str, int]) -> dict[str, int | float]:
        tp, tn, fp, fn = totals['tp'], totals['tn'], totals['fp'], totals['fn']
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': get_precision(tp, fp),
            'recall': get_recall(tp, fn),
            'accuracy': get_accuracy(tp, tn, fp, fn),
            'f1_score': get_f1_score(tp, fp, fn),
        }

    return finalize(totals_exp), finalize(totals_base)


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
                for sample_rate in SAMPLE_RATES:
                    agg_exp = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
                    agg_base = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
                    n_ok = 0
                    for video in sorted(videos):
                        try:
                            te, tb = evaluate_video_pair(
                                dataset,
                                video,
                                tile_size,
                                sample_rate,
                                experimental_name,
                                baseline_name,
                                threshold,
                            )
                        except (FileNotFoundError, OSError, ValueError) as e:
                            print(f"Skip {dataset}/{video}: {e}")
                            continue
                        for k in ('tp', 'tn', 'fp', 'fn'):
                            agg_exp[k] += int(te[k])
                            agg_base[k] += int(tb[k])
                        n_ok += 1

                    if n_ok == 0:
                        continue

                    def row_for(classifier_label: str, totals: dict[str, int]) -> dict[str, Any]:
                        tp, tn, fp, fn = totals['tp'], totals['tn'], totals['fp'], totals['fn']
                        return {
                            'videoset': videoset,
                            'dataset': dataset,
                            'classifier': classifier_label,
                            'tile_size': tile_size,
                            'sample_rate': sample_rate,
                            'threshold': threshold,
                            'videos_evaluated': n_ok,
                            'tp': tp,
                            'tn': tn,
                            'fp': fp,
                            'fn': fn,
                            'precision': get_precision(tp, fp),
                            'recall': get_recall(tp, fn),
                            'accuracy': get_accuracy(tp, tn, fp, fn),
                            'f1_score': get_f1_score(tp, fp, fn),
                        }

                    rows.append(row_for(experimental_name, agg_exp))
                    rows.append(row_for(baseline_name, agg_base))

    return rows


def write_markdown(rows: list[dict[str, Any]], path: str, videoset: str):
    lines = [
        f'# Classifier comparison ({videoset})',
        '',
        '| dataset | classifier | tile | sr | F1 | Acc | P | R | videos |',
        '|---------|------------|------|----|----|-----|---|---|--------|',
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['classifier']} | {r['tile_size']} | {r['sample_rate']} | "
            f"{r['f1_score']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['videos_evaluated']} |"
        )
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def write_chart(rows: list[dict[str, Any]], path: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    title = 'F1 by dataset'
    if TILE_SIZES and SAMPLE_RATES:
        t0, s0 = TILE_SIZES[0], SAMPLE_RATES[0]
        df = df[(df['tile_size'] == t0) & (df['sample_rate'] == s0)]
        title = f'F1 (tile={t0}, sr={s0})'
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('dataset:N', title='dataset'),
            y=alt.Y('f1_score:Q', title='F1', scale=alt.Scale(domain=[0, 1])),
            color='classifier:N',
        )
        .properties(width=200, height=200, title=title)
    )
    chart.save(path, scale_factor=2)


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
        jsonl_path = os.path.join(summary_dir, f'{vs}.jsonl')
        with open(jsonl_path, 'w') as jf:
            for row in rows:
                jf.write(json.dumps(row) + '\n')

        md_path = os.path.join(summary_dir, f'{vs}.md')
        write_markdown(rows, md_path, vs)

        chart_path = os.path.join(summary_dir, f'{vs}_f1_chart.png')
        try:
            write_chart(rows, chart_path)
        except Exception as e:
            print(f"Chart skipped: {e}")

        print(f"Wrote {jsonl_path} ({len(rows)} rows) and {md_path}")


if __name__ == '__main__':
    main()
