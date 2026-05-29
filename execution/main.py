#!/usr/local/bin/python
"""One-command, pipeline-parallel runner for the polyis tracking pipeline.

Usage (inside the docker container):

  python execution/main.py \\
    --dataset caldot2-y05 --videoset test \\
    --classifier ShuffleNet05 --tile-size 60 --sample-rate 5 \\
    --tilepadding none --canvas-scale 0.5 \\
    --tracker bytetrackcython --tracking-accuracy-threshold 0.95 \\
    --relevance-threshold 0.5 \\
    --classify-gpu 0 --detect-gpu 1 \\
    --prune-workers 4 --compress-workers 8 \\
    --max-videos-in-flight 2

See ``--help`` for full flag descriptions.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback

import torch

from polyis.io import store
from polyis.utilities import TILEPADDING_MODES, TilePadding

from execution import shm as shm_mod
from execution.classify_stage import classify_stage
from execution.compress_stage import compress_worker
from execution.config import PipelineConfig
from execution.decode_stage import decode_stage
from execution.detect_stage import detect_stage
from execution.error_monitor import ErrorMonitor
from execution.messages import (
    StageTiming,
    TrackingResult,
    VideoCompressDone,
)
from execution.output import save_pipeline_runtime, save_tracking_result
from execution.pool import spawn_pool
from execution.prune_stage import prune_worker
from execution.track_stage import track_stage


WARMUP_MAX_FRAMES = 64


def _parse_threshold(s: str) -> float | None:
    """Parse a tracking-accuracy threshold; accepts ``null``/``none`` for None."""
    if s.lower() in ('null', 'none'):
        return None
    return float(s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pipeline-parallel runner for the polyis tracking pipeline'
    )

    # --- Algorithmic (required) ---
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--videoset', required=True, choices=['test', 'valid', 'train'])
    parser.add_argument('--classifier', required=True)
    parser.add_argument('--tile-size', dest='tile_size', type=int, required=True)
    parser.add_argument('--sample-rate', dest='sample_rate', type=int, required=True)
    parser.add_argument('--tilepadding', required=True, choices=list(TILEPADDING_MODES.keys()))
    parser.add_argument('--canvas-scale', dest='canvas_scale', type=float, required=True)
    parser.add_argument('--tracker', required=True)
    parser.add_argument(
        '--tracking-accuracy-threshold', dest='tracking_accuracy_threshold',
        type=_parse_threshold, required=True,
        help='Float in [0, 1] or "null"/"none" to disable the prune stage.',
    )
    parser.add_argument('--relevance-threshold', dest='relevance_threshold',
                        type=float, required=True)

    # --- Resource (defaults) ---
    cpu_count = max(1, os.cpu_count() or 1)
    parser.add_argument('--classify-gpu', dest='classify_gpu', type=int, default=0)
    parser.add_argument('--detect-gpu', dest='detect_gpu', type=int, default=0)
    parser.add_argument('--prune-workers', dest='prune_workers', type=int,
                        default=max(1, cpu_count // 4))
    parser.add_argument('--compress-workers', dest='compress_workers', type=int,
                        default=max(2, cpu_count // 2))
    parser.add_argument('--max-videos-in-flight', dest='max_videos_in_flight',
                        type=int, default=2)

    # --- Behavior ---
    parser.add_argument('--no-interpolate', dest='no_interpolate',
                        action='store_true', default=False)
    parser.add_argument('--no-warmup', dest='no_warmup',
                        action='store_true', default=False)
    parser.add_argument('--max-videos', dest='max_videos', type=int, default=None,
                        help='Debug: limit pipeline to the first N videos.')
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        dataset=args.dataset,
        videoset=args.videoset,
        classifier=args.classifier,
        tile_size=args.tile_size,
        sample_rate=args.sample_rate,
        tilepadding=args.tilepadding,
        canvas_scale=args.canvas_scale,
        tracker=args.tracker,
        tracking_accuracy_threshold=args.tracking_accuracy_threshold,
        relevance_threshold=args.relevance_threshold,
        classify_gpu=args.classify_gpu,
        detect_gpu=args.detect_gpu,
        prune_workers=args.prune_workers,
        compress_workers=args.compress_workers,
        max_videos_in_flight=args.max_videos_in_flight,
        no_interpolate=args.no_interpolate,
        warmup=not args.no_warmup,
    )


def _semaphore_release_loop(
    compress_done_q: mp.Queue,
    sem: threading.Semaphore,
):
    """Drain VideoCompressDone messages; unlink frame buffer + release semaphore."""
    while True:
        item = compress_done_q.get()
        if item is None:
            return
        assert isinstance(item, VideoCompressDone)
        # Unlink the frame buffer (idempotent).
        shm_mod.unlink(item.frame_shm_name)
        sem.release()


def _timings_collector_loop(
    timings_q: mp.Queue,
    collected: list,
):
    """Drain StageTiming messages into ``collected`` until None sentinel."""
    while True:
        item = timings_q.get()
        if item is None:
            return
        if isinstance(item, StageTiming):
            collected.append(item)


def _video_feeder(
    videos: list[str],
    video_q: queue.Queue,
    sem: threading.Semaphore,
    abort_event: threading.Event,
):
    """Feed videos to the decoder, gated by the videos-in-flight semaphore."""
    for video in videos:
        # Block until a slot is available; bail out fast on abort.
        while not sem.acquire(timeout=0.5):
            if abort_event.is_set():
                video_q.put(None)
                return
        if abort_event.is_set():
            video_q.put(None)
            return
        video_q.put(video)
    # All videos queued: terminate the decoder's loop.
    video_q.put(None)


def main() -> int:
    args = parse_args()
    config = build_config(args)

    # Required so worker processes inherit a clean state (no CUDA fork issues).
    mp.set_start_method('spawn', force=True)

    # Enumerate videos in the chosen videoset.
    videoset_dir = store.dataset(config.dataset, config.videoset)
    assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"
    videos = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )
    assert len(videos) > 0, f"No videos found in {videoset_dir}"
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    print(f"Found {len(videos)} videos in {config.dataset}/{config.videoset}")

    # --- Queues ---------------------------------------------------------------
    # Between threads in the main process: queue.Queue (reference passing).
    # To/from worker processes: mp.Queue.
    video_q: queue.Queue = queue.Queue()
    decode_q: queue.Queue = queue.Queue()
    classify_out_q: queue.Queue = queue.Queue()
    compress_to_detect_q: queue.Queue = queue.Queue()
    detect_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    error_q: mp.Queue = mp.Queue()
    timings_q: mp.Queue = mp.Queue()
    compress_done_q: mp.Queue = mp.Queue()

    # --- Helpers --------------------------------------------------------------
    sem = threading.Semaphore(config.max_videos_in_flight)
    abort_event = threading.Event()
    monitor = ErrorMonitor(error_q)
    monitor.start()

    # --- Threads + pools ------------------------------------------------------
    decode_t = threading.Thread(
        target=decode_stage,
        kwargs=dict(
            video_queue=video_q, out_queue=decode_q,
            config=config, error_q=error_q,
        ),
        daemon=True, name='decode',
    )
    classify_t = threading.Thread(
        target=classify_stage,
        kwargs=dict(
            in_queue=decode_q, out_queue=classify_out_q,
            config=config, error_q=error_q, timings_q=timings_q,
        ),
        daemon=True, name='classify',
    )

    # Optional Prune pool: wires classify_out_q -> prune_to_compress_q.
    prune_workers: list[mp.Process] = []
    prune_relays: list[threading.Thread] = []
    if config.use_prune:
        prune_to_compress_q: queue.Queue = queue.Queue()
        prune_workers, _, _, prune_relays = spawn_pool(
            name='prune',
            worker_target=prune_worker,
            worker_args=(config, error_q, timings_q),
            num_workers=config.prune_workers,
            upstream_q=classify_out_q,
            downstream_q=prune_to_compress_q,
        )
        compress_upstream = prune_to_compress_q
    else:
        compress_upstream = classify_out_q

    # Compress pool.
    compress_workers, _, _, compress_relays = spawn_pool(
        name='compress',
        worker_target=compress_worker,
        worker_args=(config, compress_done_q, error_q, timings_q),
        num_workers=config.compress_workers,
        upstream_q=compress_upstream,
        downstream_q=compress_to_detect_q,
    )

    detect_t = threading.Thread(
        target=detect_stage,
        kwargs=dict(
            in_queue=compress_to_detect_q, out_queue=detect_q,
            config=config, error_q=error_q, timings_q=timings_q,
        ),
        daemon=True, name='detect',
    )
    track_t = threading.Thread(
        target=track_stage,
        kwargs=dict(
            in_queue=detect_q, out_queue=result_q,
            config=config, error_q=error_q, timings_q=timings_q,
        ),
        daemon=True, name='track',
    )

    # Semaphore release thread (drains VideoCompressDone messages).
    sem_release_t = threading.Thread(
        target=_semaphore_release_loop,
        args=(compress_done_q, sem),
        daemon=True, name='sem-release',
    )

    # Timings collector thread.
    timings_collected: list[StageTiming] = []
    timings_collector_t = threading.Thread(
        target=_timings_collector_loop,
        args=(timings_q, timings_collected),
        daemon=True, name='timings',
    )

    # Start all threads.
    decode_t.start()
    classify_t.start()
    detect_t.start()
    track_t.start()
    sem_release_t.start()
    timings_collector_t.start()

    # --- Warmup (optional) ----------------------------------------------------
    if config.warmup:
        print(f"Warming up with {videos[0]} ({WARMUP_MAX_FRAMES} frames)...")
        # Acquire a permit; sent as a (video, max_frames) tuple for warmup mode.
        sem.acquire()
        video_q.put((videos[0], WARMUP_MAX_FRAMES))
        # Wait for the warmup result; discard.
        try:
            _ = result_q.get(timeout=300)
        except queue.Empty:
            _abort_pipeline(abort_event, monitor, "Warmup timed out (>300s)")
            return 2
        print("Warmup complete.")

    # --- Timer start ----------------------------------------------------------
    timer_start_ns = time.time_ns()

    # Start the feeder thread now that the timer is running.
    feeder_t = threading.Thread(
        target=_video_feeder,
        args=(videos, video_q, sem, abort_event),
        daemon=True, name='feeder',
    )
    feeder_t.start()

    # --- Collect TrackingResults ----------------------------------------------
    per_video_complete_ts: dict[str, float] = {}
    results: list[TrackingResult] = []
    try:
        for i in range(len(videos)):
            # Poll cooperatively so errors can abort within ~1s.
            result = None
            while result is None:
                if monitor.failed.is_set():
                    raise _ErrorFromWorker(monitor.first_error)
                try:
                    result = result_q.get(timeout=1.0)
                except queue.Empty:
                    continue
            assert isinstance(result, TrackingResult)
            elapsed_so_far_ms = (time.time_ns() - timer_start_ns) / 1e6
            per_video_complete_ts[result.video] = elapsed_so_far_ms
            results.append(result)
            # Save tracking output incrementally (off the timed path is fine —
            # save is small JSON).
            save_tracking_result(result, config)
            print(f"  [{i + 1}/{len(videos)}] {result.video} done at {elapsed_so_far_ms:.0f} ms")
    except _ErrorFromWorker as exc:
        print(f"\n=== Pipeline failure in stage {exc.err.stage} ===", file=sys.stderr)
        if exc.err.video:
            print(f"  while processing video: {exc.err.video}", file=sys.stderr)
        print(exc.err.traceback, file=sys.stderr)
        _abort_pipeline(abort_event, monitor, None)
        return 1

    # --- Timer end ------------------------------------------------------------
    timer_end_ns = time.time_ns()
    elapsed_ms = (timer_end_ns - timer_start_ns) / 1e6

    # --- Clean shutdown -------------------------------------------------------
    feeder_t.join(timeout=10)
    # Tail of the pipeline already drained; sentinels propagated naturally.
    # Help anything still alive exit cleanly.
    for t in [decode_t, classify_t, detect_t, track_t]:
        t.join(timeout=10)
    for p in prune_workers + compress_workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    for t in prune_relays + compress_relays:
        t.join(timeout=5)
    # Stop helper threads.
    compress_done_q.put(None)
    sem_release_t.join(timeout=5)
    timings_q.put(None)
    timings_collector_t.join(timeout=5)
    monitor.stop()

    # --- Save pipeline runtime summary ----------------------------------------
    summary_path = save_pipeline_runtime(
        config=config,
        elapsed_ms=elapsed_ms,
        num_videos=len(videos),
        timings=timings_collected,
        per_video_complete_ts=per_video_complete_ts,
    )

    print(f"\nPipeline complete: {len(results)} videos in {elapsed_ms:.0f} ms")
    print(f"Runtime summary: {summary_path}")
    return 0


class _ErrorFromWorker(Exception):
    """Internal control-flow exception carrying the first PipelineError."""
    def __init__(self, err):
        super().__init__(err.stage if err else 'unknown')
        self.err = err


def _abort_pipeline(abort_event: threading.Event,
                    monitor: ErrorMonitor,
                    timeout_reason: str | None) -> None:
    """Set the abort flag so the feeder exits; cascade shutdown sentinels."""
    abort_event.set()
    if timeout_reason:
        print(f"\n=== Pipeline aborted: {timeout_reason} ===", file=sys.stderr)
    # Stop the monitor; we leave the rest for the OS to clean up because the
    # process is about to exit non-zero anyway.
    try:
        monitor.stop()
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())
