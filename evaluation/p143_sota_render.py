#!/usr/local/bin/python

import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Queue
from pathlib import Path

from evaluation.manifests import build_split_video_manifest, load_sota_stat_manifest
from polyis.io import cache, store
from polyis.utilities import PREFIX_TO_VIDEOSET, ProgressBar, create_tracking_visualization, get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']

# SOTA systems supported by the transform stage.
SOTA_SYSTEMS = ['otif', 'leap']


def parse_args():
    parser = argparse.ArgumentParser(description='Render SOTA tracking results on original videos')
    parser.add_argument('--speed_up', type=int, default=4)
    parser.add_argument('--track_ids', type=int, nargs='*', default=None)
    parser.add_argument('--detection_only', action='store_true')
    parser.add_argument('--system', type=str, choices=SOTA_SYSTEMS, default=None)
    parser.add_argument('--param_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args()


def load_sota_tracking(tracking_path: Path) -> dict[int, list[list[float]]]:
    """Load a SOTA tracking JSONL file into the frame-indexed dict expected by create_tracking_visualization."""
    # Fail fast when the tracking file is missing.
    assert tracking_path.exists(), f"SOTA tracking file not found: {tracking_path}"

    # Parse the JSONL lines into a frame-indexed dict.
    frame_tracks: dict[int, list[list[float]]] = {}
    with open(tracking_path, 'r') as f:
        for line in f:
            if line.strip():
                frame_data = json.loads(line)
                frame_tracks[frame_data['frame_idx']] = frame_data['tracks']

    return frame_tracks


def visualize_sota_video(system: str, dataset: str, video_file: str, param_id: int,
                         speed_up: int, track_ids: list[int] | None, detection_only: bool,
                         process_id: int, progress_queue: Queue):
    """Render a single SOTA tracking result on the original video."""
    # Resolve the SOTA tracking JSONL path for this system/dataset/video/param_id.
    tracking_path = cache.sota(system, dataset, video_file, 'tracking_results', f'{param_id:03d}', 'tracking.jsonl')
    # Load the frame-indexed tracking dict.
    tracking_results = load_sota_tracking(tracking_path)

    # Resolve the path to the original video.
    video_path = store.dataset(dataset, PREFIX_TO_VIDEOSET[video_file[:2]], video_file)
    assert video_path.exists(), f"Original video not found: {video_path}"

    # Colocate the annotated output next to the tracking JSONL.
    output_path = cache.sota(system, dataset, video_file, 'tracking_results', f'{param_id:03d}', f'annotated_{video_file}')

    # Create the annotated visualization.
    create_tracking_visualization(str(video_path), tracking_results, str(output_path), speed_up,
                                  process_id, progress_queue, track_ids, detection_only)


def build_render_tasks(datasets: list[str], systems: list[str],
                       param_id_filter: int | None) -> list[partial]:
    """Discover and return render tasks for all matching SOTA system/dataset/param_id combos."""
    tasks: list[partial] = []

    for dataset in datasets:
        # Materialize the test videos for this dataset.
        videos_df = build_split_video_manifest(datasets=[dataset], videosets=['test'])
        videos = sorted(videos_df['video'].drop_duplicates().tolist())
        if not videos:
            print(f"  No test videos found for dataset {dataset}, skipping")
            continue

        for system in systems:
            # Skip systems that have not been transformed for this dataset.
            stat_path = cache.sota(system, dataset, 'stat.csv')
            if not os.path.exists(stat_path):
                continue

            # Load the stat manifest to enumerate available param_ids.
            stat_df = load_sota_stat_manifest(system, dataset)
            param_ids = sorted(stat_df['param_id'].drop_duplicates().tolist())

            # Apply the optional param_id filter.
            if param_id_filter is not None:
                param_ids = [p for p in param_ids if p == param_id_filter]

            # Validate that each tracking JSONL exists before scheduling the task.
            for param_id in param_ids:
                for video in videos:
                    tracking_path = cache.sota(system, dataset, video, 'tracking_results', f'{param_id:03d}', 'tracking.jsonl')
                    assert os.path.exists(tracking_path), f"Missing SOTA tracking: {tracking_path}"

                    tasks.append(partial(
                        visualize_sota_video, system, dataset, video, param_id,
                    ))

    return tasks


def main(args):
    print(f"Speed up factor: {args.speed_up} (processing every {args.speed_up}th frame)")

    # Resolve the datasets to process.
    datasets = [args.dataset] if args.dataset else DATASETS
    # Resolve the systems to process.
    systems = [args.system] if args.system else SOTA_SYSTEMS

    # Discover all render tasks matching the filters.
    task_partials = build_render_tasks(datasets, systems, args.param_id)

    assert len(task_partials) > 0, "No SOTA tracking results found to render"
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
