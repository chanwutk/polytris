#!/usr/local/bin/python
"""
Pipeline-parallel execution entrypoint.

Replaces Phase 4 (test pass) of scripts/run.sh.  Iterates over
Pareto-filtered parameter combinations, distributing one pipeline
per GPU via ProgressBar.

Usage:
    python execution/main.py --test
    python execution/main.py --test --preload
"""

import argparse
import itertools
import os
from functools import partial
from typing import Callable

import torch
import torch.multiprocessing as mp

from polyis.io import store
from polyis.pareto import build_pareto_combo_filter
from polyis.utilities import ProgressBar, TilePadding, get_config

from execution.pipeline import PipelineConfig, run_pipeline


config = get_config()
TILE_SIZES: list[int] = config['EXEC']['TILE_SIZES']
CLASSIFIERS: list[str] = config['EXEC']['CLASSIFIERS']
DATASETS: list[str] = config['EXEC']['DATASETS']
SAMPLE_RATES: list[int] = config['EXEC']['SAMPLE_RATES']
TILEPADDING_MODES: list[TilePadding] = config['EXEC']['TILEPADDING_MODES']
CANVAS_SCALES: list[float] = config['EXEC']['CANVAS_SCALE']
TRACKERS: list[str] = config['EXEC']['TRACKERS']
TRACKING_ACCURACY_THRESHOLDS: list[float] = config['EXEC']['TRACKING_ACCURACY_THRESHOLDS']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline-parallel execution of the video processing pipeline')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--preload', action='store_true',
                        help='Preload all video frames to GPU before starting the timer')
    parser.add_argument('--no-interpolate', action='store_true', dest='no_interpolate',
                        help='Disable trajectory interpolation in the tracker')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification probability threshold (default: 0.5)')
    return parser.parse_args()


def _pipeline_task(
    videos: list[str],
    pipeline_config: PipelineConfig,
    gpu_id: int,
    command_queue: mp.Queue,
):
    """Wrapper matching the ProgressBar worker signature (gpu_id, command_queue)."""
    run_pipeline(videos, pipeline_config, gpu_id, command_queue)


def main():
    args = parse_args()
    mp.set_start_method('spawn', force=True)

    # Determine which videosets to process.
    selected_videosets: list[str] = []
    if args.test:
        selected_videosets.append('test')
    if args.valid:
        selected_videosets.append('valid')
    if not selected_videosets:
        selected_videosets = ['test']

    # Build Pareto filter (returns None when not running the test pass).
    allowed_combos = build_pareto_combo_filter(
        DATASETS,
        selected_videosets,
        ['classifier', 'tilesize', 'sample_rate', 'tilepadding', 'canvas_scale',
         'tracker', 'tracking_accuracy_threshold'],
        collapse_tracker_when_no_threshold=True,
    )

    # Build one pipeline task per (dataset, videoset, parameter combo).
    funcs: list[Callable[[int, mp.Queue], None]] = []
    for dataset, videoset in itertools.product(DATASETS, selected_videosets):
        videoset_dir = store.dataset(dataset, videoset)
        assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"

        videos = sorted(
            f for f in os.listdir(videoset_dir)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        )
        assert len(videos) > 0, f"No videos found in {videoset_dir}"

        for (classifier, tile_size, sample_rate, tilepadding,
             canvas_scale, threshold) in itertools.product(
                CLASSIFIERS, TILE_SIZES, SAMPLE_RATES, TILEPADDING_MODES,
                CANVAS_SCALES, TRACKING_ACCURACY_THRESHOLDS):

            # Expand tracker dimension: None when no pruning, else iterate.
            trackers: list[str | None] = [None] if threshold is None else TRACKERS
            for tracker in trackers:
                combo = (classifier, tile_size, sample_rate, tilepadding,
                         canvas_scale, tracker, threshold)
                if allowed_combos is not None and combo not in allowed_combos[dataset]:
                    continue

                pipeline_config = PipelineConfig(
                    dataset=dataset,
                    videoset=videoset,
                    classifier=classifier,
                    tile_size=tile_size,
                    sample_rate=sample_rate,
                    tilepadding=tilepadding,
                    canvas_scale=canvas_scale,
                    tracker=tracker,
                    tracking_accuracy_threshold=threshold,
                    preload=args.preload,
                    compress_threshold=args.threshold,
                    no_interpolate=args.no_interpolate,
                )
                func = partial(_pipeline_task, videos, pipeline_config)
                funcs.append(func)

    print(f"Created {len(funcs)} pipeline tasks")

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"
    print(f"Using {num_gpus} GPUs for parallel pipelines")

    if len(funcs) > 0:
        ProgressBar(
            num_workers=num_gpus,
            num_tasks=len(funcs),
            refresh_per_second=5,
        ).run_all(funcs)

    print("All pipeline tasks completed!")


if __name__ == '__main__':
    main()
