#!/usr/local/bin/python
"""
Run a profiling sweep: every configuration of a dataset's validation split in one pipeline
(experiments/fanout_profiling/PLAN.md "Shape", step 5).

    python -m experiments.fanout_profiling.main --config experiments/fanout_profiling/configs/<dataset>.yaml

Writes, under <output_root>/<dataset>/:
  execution/...                    every non-pixel result, in the scripts' formats and paths (the savers)
  profiling/records.jsonl          one line per (configuration, video): branch, stage timings, TrackEval result
  profiling/run.json               the run's summary (videos, configurations, wall clock, config path)

Refuses to start if <output_root>/<dataset> already exists, so an earlier sweep is never overwritten.
Main admits at most system.max_videos_in_flight videos; a video leaves when all its results have arrived.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter
from typing import Callable, Iterator

import numpy as np

from execution import frames
from execution.config import ConfigError
from execution.main import StartupError, VideoInfo, list_videos, probe
from execution.operators import frame_selection
from execution.pipeline import Stage, run
from experiments.fanout_profiling import stages as st
from experiments.fanout_profiling.completion import VideoCompletion
from experiments.fanout_profiling.config import SweepConfig, load_sweep_config
from experiments.fanout_profiling.messages import Evaluated
from tetris.io import store

SHM_DIR = '/dev/shm'


def parse_args():
    parser = argparse.ArgumentParser(description='Run a pipeline-parallel profiling sweep over a validation split.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def videos_dir(cfg: SweepConfig) -> str:
    """The split's videos (the repo's dataset layout)."""
    return str(store.dataset(cfg.dataset, cfg.videoset))


def check_references(cfg: SweepConfig, videos: list[str]) -> None:
    """Every video needs its reference tracks, or evaluation would fail mid-run."""
    missing = [v for v in videos if not os.path.exists(
        os.path.join(cfg.evaluation.reference_dir, os.path.basename(v), cfg.evaluation.reference_file))]
    if missing:
        raise StartupError(f'reference tracks missing for {len(missing)} videos, e.g. '
                           f'{os.path.join(cfg.evaluation.reference_dir, os.path.basename(missing[0]), cfg.evaluation.reference_file)}')


def worst_case_shm_bytes(cfg: SweepConfig, infos: list[VideoInfo]) -> int:
    """Frames blocks of every admitted video, plus every canvases block that can exist at once.

    A canvases block lives from pack until detect frees it.  Pack creates all of an input's outputs before
    sending them, so each pack worker can hold paddings x scales blocks, plus the queue into detect and one
    per detect worker.
    """
    s = cfg.system.stages
    frames_max = max(i.frame_count * i.height * i.width * 3 for i in infos)
    per_input = len(cfg.params.tile_paddings) * len(cfg.params.canvas_scales)
    canvas_holders = s['pack'].procs * per_input + cfg.system.queue_maxsize + s['detect'].procs
    # shortcut: at most one collage per sampled frame at the densest sample rate; count collages if the
    # check trips unexpectedly.
    ts, densest, scale = cfg.params.tile_size, min(cfg.params.sample_rates), max(cfg.params.canvas_scales)
    canvases_max = max(len(frame_selection(i.frame_count, densest)[0])
                       * max(1, int(round(i.height // ts * scale))) * ts * max(1, int(round(i.width // ts * scale))) * ts * 3
                       for i in infos)
    return cfg.system.max_videos_in_flight * frames_max + canvas_holders * canvases_max


def check_shm(cfg: SweepConfig, infos: list[VideoInfo], shm_dir: str = SHM_DIR) -> None:
    worst = worst_case_shm_bytes(cfg, infos)
    stat = os.statvfs(shm_dir)
    free = stat.f_bavail * stat.f_frsize
    if worst > free:
        raise StartupError(f'worst-case shared memory {worst / 2**30:.1f} GiB exceeds {free / 2**30:.1f} GiB free in '
                           f'{shm_dir}; lower system.max_videos_in_flight, queue_maxsize, or pack/detect procs')


def to_json(value):
    """TrackEval results hold numpy arrays and scalars; JSON gets lists and Python numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def record_of(result: Evaluated) -> dict:
    """One records.jsonl line."""
    return {'video': os.path.basename(result.video), 'branch': result.branch.as_dict(),
            'timings': [t._asdict() for t in result.timings], 'metrics': to_json(result.metrics)}


def admitted(videos: list[str], slots: threading.Semaphore, stopped: threading.Event) -> Iterator[str]:
    """Yield videos one at a time, each only once a slot is free (slots are returned as videos finish)."""
    for v in videos:
        while not slots.acquire(timeout=0.5):
            if stopped.is_set():
                return
        yield v


def execute(cfg: SweepConfig, config_path: str = '',
            build: Callable[[SweepConfig, str, VideoCompletion], list[Stage]] = st.build_stages,
            videos: list[str] | None = None) -> dict:
    """Validate, run the sweep, write records; returns the run summary."""
    # Wall clock covers the whole sweep: startup checks, video probing, workers, and every result.
    start = time.time()
    dataset_root = os.path.join(cfg.output_root, cfg.dataset)
    if os.path.exists(dataset_root):
        raise StartupError(f'{dataset_root} already exists; choose another output_root so an earlier sweep is kept')
    videos = videos if videos is not None else list_videos(videos_dir(cfg))
    infos = [probe(v) for v in videos]
    check_references(cfg, videos)
    check_shm(cfg, infos)
    out_dir = os.path.join(dataset_root, 'profiling')
    os.makedirs(out_dir)

    expected = st.expected_results(cfg)
    prefix = frames.new_prefix()
    slots, stopped = threading.Semaphore(cfg.system.max_videos_in_flight), threading.Event()
    received: Counter = Counter()
    manager = mp.get_context('spawn').Manager()
    try:
        completion = VideoCompletion(manager)
        with open(os.path.join(out_dir, 'records.jsonl'), 'w') as records:
            for result in run(build(cfg, prefix, completion), admitted(videos, slots, stopped),
                              cfg.system.queue_maxsize, shm_prefix=prefix,
                              save_queue_maxsize=cfg.system.save_queue_maxsize):
                records.write(json.dumps(record_of(result)) + '\n')
                received[result.video] += 1
                # A video whose every configuration has arrived frees its admission slot.
                if received[result.video] == expected:
                    slots.release()
        pending = completion.pending()
    finally:
        stopped.set()
        manager.shutdown()
    wall = time.time() - start
    short = {os.path.basename(v): received[v] for v in videos if received[v] != expected}
    if short or pending:
        raise RuntimeError(f'incomplete sweep: results per video {short} (expected {expected}); '
                           f'pack inputs still pending {pending}')
    summary = {'config': config_path, 'dataset': cfg.dataset, 'videoset': cfg.videoset, 'videos': len(videos),
               'configurations': expected, 'records': sum(received.values()), 'wall_s': wall,
               # The aggregator expands tolerance groups only if this run merged them.
               'reduce_search_space': cfg.params.reduce_search_space,
               'frames': sum(i.frame_count for i in infos),
               'video_frames': {os.path.basename(i.path): i.frame_count for i in infos}}
    with open(os.path.join(out_dir, 'run.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = execute(load_sweep_config(args.config), args.config)
    except (StartupError, ConfigError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
