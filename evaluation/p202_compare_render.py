#!/usr/local/bin/python

import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Queue
from pathlib import Path

from evaluation.manifests import build_split_video_manifest
from polyis.io import cache, store
from polyis.pareto import load_pareto_params, pareto_params_exist
from polyis.utilities import PREFIX_TO_VIDEOSET, ProgressBar, create_tracking_visualization, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']


def parse_args():
    parser = argparse.ArgumentParser(description='Render Polytris and naive tracking results on original videos')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--speed_up', type=int, default=4)
    parser.add_argument('--track_ids', type=int, nargs='*', default=None)
    parser.add_argument('--detection_only', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--variant_id', type=str, default=None)
    return parser.parse_args()


def load_tracking_jsonl(tracking_path: Path) -> dict[int, list[list[float]]]:
    """Load a tracking JSONL file into the frame-indexed dict expected by create_tracking_visualization."""
    # Fail fast when the tracking file is missing.
    assert tracking_path.exists(), f"Tracking file not found: {tracking_path}"

    # Parse the JSONL lines into a frame-indexed dict.
    frame_tracks: dict[int, list[list[float]]] = {}
    with open(tracking_path, 'r') as f:
        for line in f:
            if line.strip():
                frame_data = json.loads(line)
                frame_tracks[frame_data['frame_idx']] = frame_data['tracks']

    return frame_tracks


def visualize_video(dataset: str, video_file: str, stage: str, variant_id: str | None,
                    speed_up: int, track_ids: list[int] | None, detection_only: bool,
                    process_id: int, progress_queue: Queue):
    """Render a single tracking result on the original video."""
    # Resolve the tracking JSONL path for this stage/variant.
    if variant_id is not None:
        tracking_path = cache.exec(dataset, stage, video_file, variant_id, 'tracking.jsonl')
        output_path = cache.exec(dataset, stage, video_file, variant_id, f'annotated_{video_file}')
    else:
        tracking_path = cache.exec(dataset, stage, video_file, 'tracking.jsonl')
        output_path = cache.exec(dataset, stage, video_file, f'annotated_{video_file}')

    # Load the frame-indexed tracking dict.
    tracking_results = load_tracking_jsonl(tracking_path)

    # Resolve the path to the original video.
    video_path = store.dataset(dataset, PREFIX_TO_VIDEOSET[video_file[:2]], video_file)
    assert video_path.exists(), f"Original video not found: {video_path}"

    # Create the annotated visualization.
    # print(output_path)
    create_tracking_visualization(str(video_path), tracking_results, str(output_path), speed_up,
                                  process_id, progress_queue, track_ids, detection_only)


def build_render_tasks(datasets: list[str], variant_id_filter: str | None) -> list[partial]:
    """Discover and return render tasks for Pareto-optimal Polytris variants and naive baseline."""
    tasks: list[partial] = []

    for dataset in datasets:
        # Materialize the test videos for this dataset.
        videos_df = build_split_video_manifest(datasets=[dataset], videosets=['test'])
        videos = sorted(videos_df['video'].drop_duplicates().tolist())
        if not videos:
            print(f"  No test videos found for dataset {dataset}, skipping")
            continue

        # --- Naive baseline tasks ---
        # Include naive unless the caller filtered to a specific variant_id.
        if variant_id_filter is None or variant_id_filter == 'naive':
            for video in videos:
                tracking_path = cache.exec(dataset, 'naive', video, 'tracking.jsonl')
                assert os.path.exists(tracking_path), f"Missing naive tracking: {tracking_path}"

                tasks.append(partial(visualize_video, dataset, video, 'naive', None))

        # --- Pareto-optimal Polytris tasks ---
        # Skip datasets without Pareto params.
        if not pareto_params_exist(dataset):
            print(f"  No Pareto params found for dataset {dataset}, skipping Polytris variants")
            continue

        # Load the Pareto-optimal variant_ids for this dataset.
        pareto_df = load_pareto_params(dataset)
        variant_ids = sorted(pareto_df['variant_id'].dropna().unique().tolist())

        # Apply the optional variant_id filter.
        if variant_id_filter is not None and variant_id_filter != 'naive':
            variant_ids = [v for v in variant_ids if v == variant_id_filter]

        # Schedule one render task per variant_id × video combination.
        for variant_id in variant_ids:
            for video in videos:
                tracking_path = cache.exec(dataset, 'ucomp-tracks', video, variant_id, 'tracking.jsonl')
                assert os.path.exists(tracking_path), f"Missing Polytris tracking: {tracking_path}"

                tasks.append(partial(visualize_video, dataset, video, 'ucomp-tracks', variant_id))

    return tasks


def main(args):
    assert args.test, "This script only supports the test videoset (pass --test)"

    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")

    # Resolve the datasets to process.
    datasets = [args.dataset] if args.dataset else DATASETS

    # Discover all render tasks matching the filters.
    task_partials = build_render_tasks(datasets, args.variant_id)

    assert len(task_partials) > 0, "No tracking results found to render"
    print(f"Found {len(task_partials)} videos to render")

    # Complete each partial with the shared visualization arguments.
    funcs = [
        partial(task, args.speed_up, args.track_ids, args.detection_only)
        for task in task_partials
    ]

    # Determine the number of parallel workers.
    num_processes = min(mp.cpu_count(), len(funcs), 40)

    mp.set_start_method('spawn', force=True)
    ProgressBar(num_workers=num_processes, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
