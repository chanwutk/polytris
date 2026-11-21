#!/usr/local/bin/python
"""
Run the execution2 pipeline over a directory of videos (PLAN.md "Interface", "Timing").

    python -m execution2.main --config <yaml> --videos <dir> --out <dir>

Writes one <out>/<video stem>.jsonl per video (p060's tracking format) and prints one JSON line
with the run's runtime.  Timing lives here, not in the pipeline: a copy of the last video is fed
first as a dummy (it also warms every model up), each result's arrival time is recorded, and the
runtime is the last arrival minus the first, so N real videos span N intervals.  The dummy's
result is discarded and its copy deleted.  Outputs are written only after the last arrival.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple

import cv2

from execution2 import frames
from execution2.config import Config, ConfigError, load_config
from execution2.pipeline import Stage, run
from execution2.stages import Tracked, build_stages, frame_selection
from polyis.utilities import save_tracking_results

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
SHM_DIR = '/dev/shm'


def parse_args():
    parser = argparse.ArgumentParser(description='Run the pipeline-parallel Tetris execution over a video directory.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


class VideoInfo(NamedTuple):
    path: str
    width: int
    height: int
    frame_count: int


class StartupError(ValueError):
    """The inputs cannot be run; raised before any worker starts."""


def list_videos(videos_dir: str) -> list[str]:
    """Sorted video files in videos_dir; other files are ignored, but outputs must not collide."""
    if not os.path.isdir(videos_dir):
        raise StartupError(f'--videos: {videos_dir} is not a directory')
    videos = sorted(os.path.join(videos_dir, f) for f in os.listdir(videos_dir)
                    if f.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(videos_dir, f)))
    if not videos:
        raise StartupError(f'--videos: no {"/".join(VIDEO_EXTENSIONS)} files in {videos_dir}')
    # Outputs are named by stem, so a.mp4 and a.avi would overwrite each other.
    clashes = [stem for stem, n in Counter(Path(v).stem for v in videos).items() if n > 1]
    if clashes:
        raise StartupError(f'--videos: several videos share a stem and would write the same output: {clashes}')
    return videos


def probe(path: str) -> VideoInfo:
    """Container metadata, checked enough that decode will not fail on it."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise StartupError(f'cannot open video {path}')
        info = VideoInfo(path, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if info.frame_count < 2:
        raise StartupError(f'{path}: has {info.frame_count} frame(s); the classifier diff needs at least 2')
    if info.width <= 0 or info.height <= 0:
        raise StartupError(f'{path}: invalid frame size {info.width}x{info.height}')
    return info


def max_videos_in_flight(cfg: Config) -> int:
    """Upper bound on videos holding shared blocks: every queue from decode to detect full, every worker busy."""
    holders = [n for n in ('decode', 'classify', 'prune', 'pack', 'detect') if n in cfg.system.stages]
    # Queues between consecutive holders (decode->...->detect), plus one in-flight video per worker.
    return cfg.system.queue_maxsize * (len(holders) - 1) + sum(cfg.system.stages[n].procs for n in holders)


def shm_bytes_per_video(info: VideoInfo, sample_rate: int, canvas_scale: float) -> int:
    """Estimated peak shared memory for one video: its frames block plus its canvases block."""
    _, needed, _ = frame_selection(info.frame_count, sample_rate)
    frames_bytes = len(needed) * info.height * info.width * 3
    # shortcut: assumes at most one collage per needed frame (packing usually yields far fewer), each
    # canvas_scale^2 times a frame; count collages if a run trips the /dev/shm check unexpectedly.
    canvas_bytes = int(frames_bytes * max(1.0, canvas_scale) ** 2)
    return frames_bytes + canvas_bytes


def check_shm_budget(cfg: Config, infos: list[VideoInfo], shm_dir: str = SHM_DIR) -> None:
    """Refuse to start when the worst case cannot fit in free shared memory (PLAN.md 15A)."""
    worst = max_videos_in_flight(cfg) * max(
        shm_bytes_per_video(i, cfg.params.sample_rate, cfg.params.canvas_scale) for i in infos)
    stat = os.statvfs(shm_dir)
    free = stat.f_bavail * stat.f_frsize
    if worst > free:
        raise StartupError(
            f'worst-case shared memory {worst / 2**30:.1f} GiB ({max_videos_in_flight(cfg)} videos in flight) exceeds '
            f'{free / 2**30:.1f} GiB free in {shm_dir}; lower system.queue_maxsize or stage procs, or raise shm_size')


def execute(cfg: Config, videos_dir: str, out_dir: str,
            build: Callable[[Config, str], list[Stage]] = build_stages) -> dict:
    """Validate, run timed, then write outputs.  Returns the runtime summary."""
    videos = list_videos(videos_dir)
    infos = [probe(v) for v in videos]
    check_shm_budget(cfg, infos)
    os.makedirs(out_dir, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix='execution2-')
    try:
        # The dummy is the last video under another path, so the pipeline treats it as any other video.
        dummy = os.path.join(tmp, 'timing-dummy' + Path(videos[-1]).suffix)
        shutil.copyfile(videos[-1], dummy)
        prefix = frames.new_prefix()
        arrivals: list[float] = []
        results: dict[str, Tracked] = {}
        for result in run(build(cfg, prefix), [dummy, *videos], cfg.system.queue_maxsize, shm_prefix=prefix):
            arrivals.append(time.perf_counter())
            if result.video != dummy:
                results[result.video] = result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Every video must come back exactly once (the engine guarantees it; check before writing anything).
    if sorted(results) != videos or len(arrivals) != len(videos) + 1:
        raise RuntimeError(f'expected {len(videos)} results plus the dummy, got {len(arrivals)} '
                           f'covering {len(results)} videos')
    # Writing happens after the last arrival so it cannot slow the timed run.
    for video, result in results.items():
        save_tracking_results(result.frame_tracks, os.path.join(out_dir, Path(video).stem + '.jsonl'))
    return {'videos': len(videos), 'runtime_s': arrivals[-1] - arrivals[0],
            'frames': sum(i.frame_count for i in infos)}


def main() -> int:
    args = parse_args()
    try:
        summary = execute(load_config(args.config), args.videos, args.out)
    except (StartupError, ConfigError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
