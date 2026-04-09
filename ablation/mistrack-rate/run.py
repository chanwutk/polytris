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
    parser = argparse.ArgumentParser(description='Run the mistrack-rate ablation pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    heuristic_parser = subparsers.add_parser('heuristic', help='Build train-side heuristic tables')
    heuristic_parser.add_argument('--dataset', default='jnc0')
    heuristic_parser.add_argument('--tracker', default=DEFAULT_TRACKER)
    heuristic_parser.add_argument('--iou-threshold', type=float, default=0.3)
    heuristic_parser.add_argument('--video-fraction-divisor', type=int, default=1)
    heuristic_parser.add_argument('--force', action='store_true')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate exhaustive and heuristic grids on test')
    evaluate_parser.add_argument('--dataset', default='jnc0')
    evaluate_parser.add_argument('--tracker', default=DEFAULT_TRACKER)
    evaluate_parser.add_argument('--iou-threshold', type=float, default=0.3)
    evaluate_parser.add_argument('--time-limit', type=float, default=0.1)
    evaluate_parser.add_argument('--limit-configs', type=int, default=None)
    evaluate_parser.add_argument('--num-workers', type=int, default=75)
    evaluate_parser.add_argument('--video-fraction-divisor', type=int, default=1)
    evaluate_parser.add_argument('--no-hota', action='store_true')
    evaluate_parser.add_argument('--keep-temp-tracks', action='store_true')
    evaluate_parser.add_argument('--force', action='store_true')

    plot_parser = subparsers.add_parser('plot', help='Plot cached ablation results')
    plot_parser.add_argument('--dataset', default='jnc0')
    plot_parser.add_argument('--tracker', default=DEFAULT_TRACKER)

    all_parser = subparsers.add_parser('all', help='Run heuristic, evaluate, and plot')
    all_parser.add_argument('--dataset', default='jnc0')
    all_parser.add_argument('--tracker', default=DEFAULT_TRACKER)
    all_parser.add_argument('--iou-threshold', type=float, default=0.3)
    all_parser.add_argument('--time-limit', type=float, default=0.1)
    all_parser.add_argument('--limit-configs', type=int, default=None)
    all_parser.add_argument('--num-workers', type=int, default=75)
    all_parser.add_argument('--video-fraction-divisor', type=int, default=1)
    all_parser.add_argument('--no-hota', action='store_true')
    all_parser.add_argument('--keep-temp-tracks', action='store_true')
    all_parser.add_argument('--force', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'heuristic':
        run_heuristic_stage(
            dataset=args.dataset,
            tracker_name=args.tracker,
            iou_threshold=args.iou_threshold,
            video_fraction_divisor=args.video_fraction_divisor,
            force=args.force,
        )
        return

    if args.command == 'evaluate':
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
        return

    if args.command == 'plot':
        plot_results(dataset=args.dataset, tracker_name=args.tracker)
        return

    if args.command == 'all':
        run_heuristic_stage(
            dataset=args.dataset,
            tracker_name=args.tracker,
            iou_threshold=args.iou_threshold,
            video_fraction_divisor=args.video_fraction_divisor,
            force=args.force,
        )
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
        plot_results(dataset=args.dataset, tracker_name=args.tracker)
        return

    raise ValueError(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    main()
