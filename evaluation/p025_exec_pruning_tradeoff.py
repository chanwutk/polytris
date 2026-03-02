#!/usr/bin/env python3
"""
export-p022-pruning.py

Compare tile coverage and runtime before/after polyomino pruning across
accuracy thresholds.

X-axis: total p022 runtime (seconds, summed across all videos)
Y-axis: total active tiles (summed across all videos and frames)
Each point is labelled as "Unpruned" or its accuracy threshold in the legend.

Output:
  - Table to stdout
  - p022_pruning_results.png in the same directory as this script

Usage:
  python export-p022-pruning.py [--dataset caldot2-y05] [--videoset test]
                                [--classifier Perfect] [--tile-size 60]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Accuracy threshold labels (must match p022 ACCURACY_THRESHOLDS order) ──
ACCURACY_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
ACCURACY_LABELS     = ['60%', '70%', '80%', '90%', '95%', '100%']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config (same source as p022 via polyis config) ──────────────────────────
try:
    sys.path.insert(0, '/polyis')
    from polyis.utilities import get_config
    config       = get_config()
    CACHE_DIR    = config['DATA']['CACHE_DIR']
    DATASETS_DIR = config['DATA']['DATASETS_DIR']
except Exception:
    CACHE_DIR    = '/polyis-cache'
    DATASETS_DIR = '/polyis-data'


# ── Helpers ─────────────────────────────────────────────────────────────────

def count_active_tiles(score_path: str, binarize: bool = False) -> int:
    """Sum active (>0) tiles across every frame in a score.jsonl file."""
    total = 0
    with open(score_path) as f:
        for line in f:
            row = json.loads(line)
            flat = np.frombuffer(bytes.fromhex(row['classification_hex']), dtype=np.uint8)
            if binarize:
                flat = (flat >= 128).astype(np.uint8)
            total += int((flat > 0).sum())
    return total


def read_runtime_ms(runtime_path: str) -> float:
    """Return total processing time in milliseconds from a runtime.jsonl file.

    runtime.jsonl format: {"runtime": [{"op": "...", "time": <ms>}, ...]}
    """
    with open(runtime_path) as f:
        row = json.loads(f.readline())
    return sum(entry['time'] for entry in row['runtime'])


def collect_baseline(cache_dir: str, dataset: str, videos: list[str],
                     classifier: str, tile_size: int) -> int:
    """Sum active tiles from 020_relevancy (binarized) across all videos."""
    total = 0
    for video in videos:
        path = os.path.join(
            cache_dir, dataset, 'execution', video,
            '020_relevancy', f'{classifier}_{tile_size}', 'score', 'score.jsonl'
        )
        if not os.path.exists(path):
            print(f'  [baseline] missing: {path}', flush=True)
            continue
        total += count_active_tiles(path, binarize=True)
    return total


def collect_pruned(cache_dir: str, dataset: str, videos: list[str],
                   classifier: str, tile_size: int,
                   accuracy_idx: int) -> tuple[int, float] | None:
    """
    Sum active tiles and total runtime from 022_pruned_polyominoes for a
    given accuracy_idx.

    Returns (total_tiles, total_runtime_seconds), or None if no results found.
    """
    total_tiles = 0
    total_ms    = 0.0
    found_any   = False

    for video in videos:
        base = os.path.join(
            cache_dir, dataset, 'execution', video,
            '022_pruned_polyominoes', f'{classifier}_{tile_size}',
            str(accuracy_idx), 'score'
        )
        score_path   = os.path.join(base, 'score.jsonl')
        runtime_path = os.path.join(base, 'runtime.jsonl')

        if not os.path.exists(score_path):
            continue
        found_any    = True
        total_tiles += count_active_tiles(score_path, binarize=False)
        if os.path.exists(runtime_path):
            total_ms += read_runtime_ms(runtime_path)

    if not found_any:
        return None
    return total_tiles, total_ms / 1000.0   # ms → seconds


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset',    default='caldot2-y05')
    parser.add_argument('--videoset',   default='test')
    parser.add_argument('--classifier', default='Perfect')
    parser.add_argument('--tile-size',  type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset    = args.dataset
    videoset   = args.videoset
    classifier = args.classifier
    tile_size  = args.tile_size

    # ── Discover videos ─────────────────────────────────────────────────────
    videoset_dir = os.path.join(DATASETS_DIR, dataset, videoset)
    if not os.path.isdir(videoset_dir):
        exec_dir = os.path.join(CACHE_DIR, dataset, 'execution')
        videos = sorted([
            d for d in os.listdir(exec_dir)
            if os.path.isdir(os.path.join(exec_dir, d))
        ])
    else:
        videos = sorted([
            f for f in os.listdir(videoset_dir)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])

    print(f'Dataset : {dataset}')
    print(f'Videoset: {videoset}  ({len(videos)} videos)')
    print(f'Classifier: {classifier}  Tile size: {tile_size}')
    print()

    # ── Baseline (no pruning; runtime treated as 0) ──────────────────────────
    print('Computing baseline (020_relevancy, binarized)...')
    baseline = collect_baseline(CACHE_DIR, dataset, videos, classifier, tile_size)
    print(f'  Baseline total tiles: {baseline:,}')
    print()

    # ── Pruned results per accuracy threshold ────────────────────────────────
    # Each entry: (label, tile_count, runtime_seconds)
    points: list[tuple[str, int, float]] = []

    for idx, _ in enumerate(ACCURACY_THRESHOLDS):
        label  = ACCURACY_LABELS[idx]
        result = collect_pruned(CACHE_DIR, dataset, videos, classifier, tile_size, idx)
        if result is None:
            print(f'  acc_idx={idx} ({label}): no results found, skipping.')
        else:
            tiles, runtime_s = result
            reduction = (1 - tiles / baseline) * 100 if baseline > 0 else 0
            print(f'  acc_idx={idx} ({label}): {tiles:>10,} tiles  '
                  f'{runtime_s:>8.1f}s  ({reduction:.1f}% reduction)')
            points.append((label, tiles, runtime_s))

    print()

    # ── Table ────────────────────────────────────────────────────────────────
    print(f'{"Label":>12}  {"Tiles":>14}  {"Runtime (s)":>12}  {"Reduction":>10}')
    print('-' * 54)
    print(f'{"Unpruned":>12}  {baseline:>14,}  {"—":>12}  {"—":>10}')
    for label, tiles, runtime_s in points:
        reduction = (1 - tiles / baseline) * 100 if baseline > 0 else 0
        print(f'{label:>12}  {tiles:>14,}  {runtime_s:>12.1f}  {reduction:>9.1f}%')

    if not points:
        print('\nNo pruning results found. Run p022 with --accuracy-idx for at least one threshold.')
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # Unpruned baseline at runtime = 0
    ax.scatter([0], [baseline], marker='D', s=110, color='steelblue', zorder=5,
               label=f'Unpruned: {baseline:,} tiles')

    for i, (label, tiles, runtime_s) in enumerate(points):
        reduction = (1 - tiles / baseline) * 100 if baseline > 0 else 0
        ax.scatter([runtime_s], [tiles], marker='o', s=90,
                   color=colors[i % len(colors)], zorder=5,
                   label=f'{label} accuracy: {tiles:,} tiles  ({reduction:.1f}% fewer)')

    ax.set_xlabel('Total p022 runtime  (seconds, summed across all videos)', fontsize=11)
    ax.set_ylabel('Total active tiles', fontsize=11)
    ax.set_title(
        f'Tile coverage vs. pruning runtime\n'
        f'{dataset} / {videoset} / {classifier}_{tile_size}',
        fontsize=13
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}s'))
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, 'p022_pruning_results.png')
    fig.savefig(out_path, dpi=150)
    print(f'\nPlot saved: {out_path}')


if __name__ == '__main__':
    main()
