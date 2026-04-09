#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mistrack_rate.analysis import plot_results
from mistrack_rate.evaluate import run_evaluation_stage
from mistrack_rate.heuristic import run_heuristic_stage
from polyis.utilities import get_config


CONFIG = get_config()
DEFAULT_TRACKER = CONFIG['EXEC']['TRACKERS'][0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the mistrack-rate ablation pipeline (heuristic -> evaluate -> plot)',
    )
    parser.add_argument('--dataset', default='jnc0')
    parser.add_argument('--tracker', default=DEFAULT_TRACKER)
    parser.add_argument('--iou-threshold', type=float, default=0.3)
    parser.add_argument('--time-limit', type=float, default=0.1)
    parser.add_argument('--limit-configs', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=75)
    parser.add_argument('--video-fraction-divisor', type=int, default=1)
    parser.add_argument('--no-hota', action='store_true')
    parser.add_argument('--keep-temp-tracks', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # Stage 1: build train-side heuristic tables.
    run_heuristic_stage(
        dataset=args.dataset,
        tracker_name=args.tracker,
        iou_threshold=args.iou_threshold,
        video_fraction_divisor=args.video_fraction_divisor,
        num_workers=args.num_workers,
        force=args.force,
    )

    # Stage 2: evaluate exhaustive and heuristic grids on test.
    run_evaluation_stage(
        dataset=args.dataset,
        tracker_name=args.tracker,
        iou_threshold=args.iou_threshold,
        time_limit_seconds=args.time_limit,
        limit_configs=args.limit_configs,
        num_workers=args.num_workers,
        video_fraction_divisor=args.video_fraction_divisor,
        compute_hota=not args.no_hota,
        keep_temp_tracks=args.keep_temp_tracks,
        force=args.force,
    )

    # Stage 3: plot cached ablation results.
    plot_results(dataset=args.dataset, tracker_name=args.tracker)


if __name__ == '__main__':
    main()
