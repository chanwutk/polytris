#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


# Re-exec inside the project container when running outside Docker.
# Must happen before any project imports that require in-container paths.
if not Path('/.dockerenv').exists():
    _args = [shlex.quote(a) for a in sys.argv[1:]]
    _cmd = 'cd /polyis && python /polyis/ablation/mistrack-rate/run.py ' + ' '.join(_args)
    os.execvp('docker', ['docker', 'exec', 'polyis', 'sh', '-lc', _cmd])


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mistrack_rate.analysis import combine_visualize, plot_results
from mistrack_rate.evaluate import run_evaluation_stage
from mistrack_rate.heuristic import run_heuristic_stage
from polyis.utilities import get_config


CONFIG = get_config()
CONFIGURED_DATASETS: list[str] = CONFIG['EXEC']['DATASETS']
CONFIGURED_TRACKERS: list[str] = CONFIG['EXEC']['TRACKERS']
DEFAULT_TRACKER: str = CONFIGURED_TRACKERS[0]

# Default video-fraction-divisor applied to non-jnc datasets.
_NON_JNC_VIDEO_FRACTION_DIVISOR = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the mistrack-rate ablation pipeline (heuristic -> evaluate -> plot)',
    )
    parser.add_argument(
        '--datasets',
        default=None,
        help='Comma-separated list of datasets to run (default: all configured datasets)',
    )
    parser.add_argument(
        '--tracker',
        default=None,
        help='Run a single named tracker (default: first configured tracker)',
    )
    parser.add_argument('--all-trackers', action='store_true',
                        help='Run every configured tracker (mutually exclusive with --tracker)')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for matching tracker detections to ground-truth boxes (default: 0.3)')
    parser.add_argument('--limit-configs', type=int, default=None,
                        help='Cap the number of exhaustive grid configs evaluated; heuristic configs are always included (default: unlimited)')
    parser.add_argument('--num-workers', type=int, default=75,
                        help='Number of parallel worker processes (default: 75)')
    parser.add_argument(
        '--video-fraction-divisor',
        type=int,
        default=None,
        help='Subsample denominator (default: 3 for non-jnc datasets, 1 otherwise)',
    )
    parser.add_argument('--no-hota', action='store_true',
                        help='Skip HOTA metric computation to speed up evaluation')
    parser.add_argument('--keep-temp-tracks', action='store_true',
                        help='Preserve temporary per-evaluation track files under the evaluation directory for debugging')
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite cached results even if they already exist')
    parser.add_argument('--combine-visualize', action='store_true',
                        help='Skip pipeline stages and produce the cross-dataset summary from existing cached results')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reject conflicting tracker-selection flags.
    if args.all_trackers and args.tracker is not None:
        print('Choose either --tracker or --all-trackers, not both.', file=sys.stderr)
        sys.exit(1)

    # Resolve the tracker list for this invocation.
    if args.all_trackers:
        trackers = CONFIGURED_TRACKERS
    elif args.tracker is not None:
        trackers = [args.tracker]
    else:
        trackers = [DEFAULT_TRACKER]

    # When --combine-visualize is set, skip all pipeline stages and read
    # existing cached results directly to produce the cross-dataset summary.
    if args.combine_visualize:
        for tracker in trackers:
            written = combine_visualize(tracker_name=tracker)
            if not written:
                print(f'No cached results found for tracker: {tracker}', file=sys.stderr)
            else:
                for p in written:
                    print(f'  wrote: {p}')
        return

    # Resolve the dataset list for this invocation.
    datasets = args.datasets.split(',') if args.datasets is not None else CONFIGURED_DATASETS

    # Run the ablation for every requested tracker and dataset.
    for tracker in trackers:
        for dataset in datasets:
            # Print a stable header before each long-running invocation.
            print(f'\n{"#" * 40}')
            print(f'>>> {tracker} | {dataset}')
            print(f'{"#" * 40}\n')

            # Non-jnc datasets default to a 1/3 video subsample unless the
            # caller explicitly provides --video-fraction-divisor.
            if args.video_fraction_divisor is not None:
                vfd = args.video_fraction_divisor
            elif not dataset.startswith('jnc'):
                vfd = _NON_JNC_VIDEO_FRACTION_DIVISOR
            else:
                vfd = 1

            # Stage 1: build train-side heuristic tables.
            run_heuristic_stage(
                dataset=dataset,
                tracker_name=tracker,
                iou_threshold=args.iou_threshold,
                video_fraction_divisor=vfd,
                num_workers=args.num_workers,
                force=args.force,
            )

            # Stage 2: evaluate exhaustive and heuristic grids on test.
            run_evaluation_stage(
                dataset=dataset,
                tracker_name=tracker,
                iou_threshold=args.iou_threshold,
                limit_configs=args.limit_configs,
                num_workers=args.num_workers,
                video_fraction_divisor=vfd,
                compute_hota=not args.no_hota,
                keep_temp_tracks=args.keep_temp_tracks,
                force=args.force,
            )

            # Stage 3: plot cached ablation results.
            plot_results(dataset=dataset, tracker_name=tracker)


if __name__ == '__main__':
    main()
