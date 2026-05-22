#!/usr/local/bin/python
"""
Scan all test videos across configured datasets and report 64-frame windows
that produce the most interesting polyomino visualizations.

Ranks windows by a composite signal that combines:
- total polyominoes across the 64 frames
- per-frame mean polyomino count (visible activity)
- spatial spread (unique tile positions used) so windows where everything
  overlaps a single road segment don't dominate

Prints the top-N candidates per dataset; lower the threshold or change the
sorting key to taste.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyis.io import store
from polyis.pack.adapters import group_tiles_all
from polyis.utilities import (
    TILEPADDING_MAPS,
    load_tracking_results,
    mark_detections,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+',
                        default=['caldot1-y05', 'caldot2-y05', 'jnc0', 'jnc2', 'jnc6', 'jnc7', 'ams-y05'])
    parser.add_argument('--videoset', default='test')
    parser.add_argument('--num-frames', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride between scanned windows (smaller = denser scan, slower)')
    parser.add_argument('--tile-size', type=int, default=60)
    parser.add_argument('--tilepadding', default='none')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Top windows per dataset')
    parser.add_argument('--min-polyominoes', type=int, default=30,
                        help='Skip windows with fewer polyominoes than this')
    parser.add_argument('--max-polyominoes', type=int, default=400,
                        help='Skip windows with more polyominoes than this (would clutter the viz)')
    return parser.parse_args()


def scan_video(dataset: str, videoset: str, video: str, args) -> list[dict]:
    """Return one record per 64-frame window for this video."""
    video_path = store.dataset(dataset, videoset, video)
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_h = (height // args.tile_size) * args.tile_size
    target_w = (width // args.tile_size) * args.tile_size
    grid_h = target_h // args.tile_size
    grid_w = target_w // args.tile_size

    try:
        frame_tracks = load_tracking_results(dataset, video)
    except FileNotFoundError:
        return []

    scale_x = target_w / width if width else 1.0
    scale_y = target_h / height if height else 1.0

    bitmaps_full = np.zeros((total_frames, grid_h, grid_w), dtype=np.uint8)
    for f in range(total_frames):
        dets = frame_tracks.get(f, [])
        if not dets:
            continue
        if scale_x != 1.0 or scale_y != 1.0:
            scaled = []
            for d in dets:
                d = list(d)
                d[-4] *= scale_x
                d[-3] *= scale_y
                d[-2] *= scale_x
                d[-1] *= scale_y
                scaled.append(d)
            dets = scaled
        bitmaps_full[f] = mark_detections(dets, target_w, target_h, args.tile_size)

    tilepadding_mode = TILEPADDING_MAPS[args.tilepadding]
    records: list[dict] = []
    last_start = total_frames - args.num_frames
    if last_start < 0:
        return []

    for start in range(0, last_start + 1, args.stride):
        window = bitmaps_full[start:start + args.num_frames]
        if window.shape[0] != args.num_frames:
            continue
        _, polyomino_lengths = group_tiles_all(window, tilepadding_mode)
        total_polys = sum(len(p) for p in polyomino_lengths)
        if total_polys < args.min_polyominoes or total_polys > args.max_polyominoes:
            continue
        mean_per_frame = total_polys / args.num_frames

        # Spatial spread: union of polyomino tiles across all frames.
        union_mask = (window > 0).any(axis=0)
        spread = int(union_mask.sum())
        spread_frac = spread / (grid_h * grid_w)

        # Composite score favoring busy-but-spread-out windows.
        score = total_polys * (0.5 + 0.5 * spread_frac)

        records.append({
            'dataset': dataset,
            'video': video,
            'start': start,
            'end': start + args.num_frames - 1,
            'total_polys': total_polys,
            'mean_per_frame': mean_per_frame,
            'spread_tiles': spread,
            'spread_frac': spread_frac,
            'grid': f'{grid_h}x{grid_w}',
            'score': score,
        })
    return records


def main():
    args = parse_args()
    all_records: list[dict] = []
    for dataset in args.datasets:
        videoset_dir = store.dataset(dataset, args.videoset)
        if not videoset_dir.exists():
            print(f"# skip {dataset}: no {args.videoset} dir at {videoset_dir}", flush=True)
            continue
        videos = sorted(p.name for p in videoset_dir.iterdir()
                        if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'})
        print(f"# scanning {dataset} ({len(videos)} videos)", flush=True)
        for video in videos:
            records = scan_video(dataset, args.videoset, video, args)
            all_records.extend(records)

    if not all_records:
        print("No qualifying windows found.")
        return

    all_records.sort(key=lambda r: -r['score'])

    print(f"\n=== Global top {args.top_k * 2} by score (busy + spread out) ===")
    header = f"{'dataset':18} {'video':14} {'frames':12} {'polys':>6} {'mean/f':>7} {'spread':>10} {'grid':>8} {'score':>8}"
    print(header)
    print('-' * len(header))
    for r in all_records[:args.top_k * 2]:
        print(f"{r['dataset']:18} {r['video']:14} {r['start']:>4}-{r['end']:<7} "
              f"{r['total_polys']:>6} {r['mean_per_frame']:>7.2f} "
              f"{r['spread_tiles']:>3}/{int(r['spread_tiles']/r['spread_frac']):<3} "
              f"({r['spread_frac']*100:>4.1f}%) "
              f"{r['grid']:>8} {r['score']:>8.1f}")

    by_dataset: dict[str, list[dict]] = {}
    for r in all_records:
        by_dataset.setdefault(r['dataset'], []).append(r)
    for dataset, records in by_dataset.items():
        print(f"\n=== Top {args.top_k} in {dataset} ===")
        print(header)
        print('-' * len(header))
        for r in records[:args.top_k]:
            print(f"{r['dataset']:18} {r['video']:14} {r['start']:>4}-{r['end']:<7} "
                  f"{r['total_polys']:>6} {r['mean_per_frame']:>7.2f} "
                  f"{r['spread_tiles']:>3}/{int(r['spread_tiles']/r['spread_frac']):<3} "
                  f"({r['spread_frac']*100:>4.1f}%) "
                  f"{r['grid']:>8} {r['score']:>8.1f}")


if __name__ == '__main__':
    main()
