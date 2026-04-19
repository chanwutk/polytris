"""
Pipeline-parallel execution engine.

One Pipeline instance processes all videos for a single parameter combination
on a single GPU.  Each stage runs in its own process, connected by
torch.multiprocessing queues.  Video frames live on GPU from decode through
compression; only small metadata crosses process boundaries via queues.

Pipeline topology:
  Decoder -> Classify -> [Prune] -> Compress -> Detect+Uncompress -> Track

The Prune stage is conditional (only when tracking_accuracy_threshold is set).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.multiprocessing as mp

from polyis.io import cache, store
from polyis.utilities import (
    TilePadding,
    build_param_str,
    save_tracking_results,
)


# ---------------------------------------------------------------------------
# Queue message types
# ---------------------------------------------------------------------------

class VideoStart(NamedTuple):
    """Sent by Decoder at the beginning of each video."""
    video: str
    frame_buffer: torch.Tensor   # GPU [num_sampled_frames, H, W, 3] uint8
    width: int
    height: int
    frame_count: int             # total frames in the original video
    sampled_indices: list[int]   # absolute frame indices that were decoded


class FrameBatch(NamedTuple):
    """Sent by Decoder for each batch of frames ready for classification."""
    video: str
    # Indices into the frame_buffer (0-based within the sampled set).
    batch_positions: list[int]
    # Corresponding previous-frame positions in the frame_buffer.
    prev_positions: list[int]


class VideoEnd(NamedTuple):
    """Sent by Decoder when all frames of a video have been dispatched."""
    video: str


class VideoClassifications(NamedTuple):
    """All classification results for one video, sent Classify -> Prune/Compress."""
    video: str
    # List of per-frame dicts: {classification_size, classification_hex, idx}
    classifications: list[dict]
    # Frame buffer reference, passed through to Compress
    frame_buffer: torch.Tensor
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]


class CollageReady(NamedTuple):
    """One packed collage ready for detection, sent Compress -> Detect."""
    video: str
    collage_idx: int
    total_collages: int
    canvas: torch.Tensor          # GPU [canvas_H, canvas_W, 3] uint8
    index_map: np.ndarray         # [dst_grid_H, dst_grid_W] uint16
    offset_lookup: list           # list of ((py,px),(oy,ox),frame_idx)
    num_frames: int               # total frames in original video
    tile_size: int                # tile size used for compression


class VideoDetections(NamedTuple):
    """Accumulated uncompressed detections for one video, sent Detect -> Track."""
    video: str
    # {frame_idx: [[x1, y1, x2, y2, score, ...], ...]}
    frame_detections: dict[int, list[list[float]]]
    num_frames: int


class TrackingResult(NamedTuple):
    """Final tracking output for one video, sent Track -> main."""
    video: str
    frame_tracks: dict[int, list[list[float]]]


# Sentinel: ``None`` on any queue signals the receiver to shut down.
# It propagates through the pipeline: Decoder -> Classify -> ... -> Track.


# ---------------------------------------------------------------------------
# Pipeline configuration container
# ---------------------------------------------------------------------------

class PipelineConfig(NamedTuple):
    dataset: str
    videoset: str
    classifier: str
    tile_size: int
    sample_rate: int
    tilepadding: TilePadding
    canvas_scale: float
    tracker: str | None
    tracking_accuracy_threshold: float | None
    preload: bool
    # Extra config options from the original scripts that we thread through.
    relevance_threshold: float    # T_r: binarize classifier scores (0–1) before prune/compress
    no_interpolate: bool          # disable trajectory interpolation in tracker


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    videos: list[str],
    config: PipelineConfig,
    gpu_id: int,
    command_queue: mp.Queue,
):
    """
    Execute the full pipeline for *videos* on *gpu_id*.

    This function is the entry-point called by ProgressBar workers.  It:
    1.  Creates inter-stage queues.
    2.  Spawns one process per stage.
    3.  Feeds videos to the Decoder.
    4.  Collects TrackingResults from the Track stage.
    5.  Saves results to disk (after timing stops).
    """
    device = f'cuda:{gpu_id}'
    use_prune = config.tracking_accuracy_threshold is not None

    # --- create inter-stage queues -------------------------------------------
    video_queue: mp.Queue = mp.Queue()          # main -> Decoder
    decode_queue: mp.Queue = mp.Queue()          # Decoder -> Classify
    classify_queue: mp.Queue = mp.Queue()        # Classify -> (Prune | Compress)
    prune_queue: mp.Queue = mp.Queue()           # Prune -> Compress (only if pruning)
    compress_queue: mp.Queue = mp.Queue()        # Compress -> Detect+Uncompress
    detect_queue: mp.Queue = mp.Queue()          # Detect+Uncompress -> Track
    result_queue: mp.Queue = mp.Queue()          # Track -> main

    # Compress reads from prune_queue when pruning, else from classify_queue.
    compress_in_queue = prune_queue if use_prune else classify_queue

    # --- import stage entry-points (deferred to avoid top-level side effects) -
    from execution.stage_decode import decoder_process
    from execution.stage_classify import classify_process
    from execution.stage_prune import prune_process
    from execution.stage_compress import compress_process
    from execution.stage_detect_uncompress import detect_uncompress_process
    from execution.stage_track import track_process

    # --- spawn stage processes -----------------------------------------------
    common_kwargs: dict[str, Any] = dict(
        gpu_id=gpu_id,
        config=config,
    )

    processes: list[mp.Process] = []

    # Stage 1: Decoder
    p_decode = mp.Process(
        target=decoder_process,
        kwargs=dict(
            video_queue=video_queue,
            out_queue=decode_queue,
            **common_kwargs,
        ),
        daemon=True,
    )
    processes.append(p_decode)

    # Stage 2: Classify
    p_classify = mp.Process(
        target=classify_process,
        kwargs=dict(
            in_queue=decode_queue,
            out_queue=classify_queue,
            **common_kwargs,
        ),
        daemon=True,
    )
    processes.append(p_classify)

    # Stage 3: Prune (optional)
    if use_prune:
        p_prune = mp.Process(
            target=prune_process,
            kwargs=dict(
                in_queue=classify_queue,
                out_queue=prune_queue,
                **common_kwargs,
            ),
            daemon=True,
        )
        processes.append(p_prune)

    # Stage 4: Compress
    p_compress = mp.Process(
        target=compress_process,
        kwargs=dict(
            in_queue=compress_in_queue,
            out_queue=compress_queue,
            **common_kwargs,
        ),
        daemon=True,
    )
    processes.append(p_compress)

    # Stage 5: Detect + Uncompress
    p_detect = mp.Process(
        target=detect_uncompress_process,
        kwargs=dict(
            in_queue=compress_queue,
            out_queue=detect_queue,
            **common_kwargs,
        ),
        daemon=True,
    )
    processes.append(p_detect)

    # Stage 6: Track
    p_track = mp.Process(
        target=track_process,
        kwargs=dict(
            in_queue=detect_queue,
            out_queue=result_queue,
            **common_kwargs,
        ),
        daemon=True,
    )
    processes.append(p_track)

    # --- start all stage processes -------------------------------------------
    for p in processes:
        p.start()

    # --- Preload mode: decode all videos upfront -----------------------------
    if config.preload:
        # In preload mode the Decoder fills all frame buffers before the timer
        # starts.  We feed all videos, wait for decode to finish each one
        # (by draining decode_queue of VideoStart/FrameBatch/VideoEnd), then
        # replay the buffered messages after the timer starts.
        #
        # Implementation: the preload path is handled inside stage_decode.py.
        # We just set a flag on the config; the decoder reads all frames into
        # pre-allocated buffers and only starts sending FrameBatches after all
        # videos are loaded.
        pass

    # --- warmup: feed one video through the full pipeline first ---------------
    # This warms up all CUDA kernels (classifier, detector, etc.) before
    # timing starts, so runtime measurements exclude JIT/warmup overhead.
    WARMUP_MAX_FRAMES = 64
    warmup_video = videos[0]
    video_queue.put((warmup_video, WARMUP_MAX_FRAMES))
    # Wait for the warmup result (discard it).
    _warmup_result: TrackingResult = result_queue.get()

    # --- start timer after warmup -------------------------------------------
    timer_start_ns = time.time_ns()

    # --- feed videos to the Decoder ------------------------------------------
    description = build_param_str(
        classifier=config.classifier,
        tilesize=config.tile_size,
        sample_rate=config.sample_rate,
        tilepadding=config.tilepadding,
        canvas_scale=config.canvas_scale,
        tracker=config.tracker,
        tracking_accuracy_threshold=config.tracking_accuracy_threshold,
        relevance_threshold=config.relevance_threshold,
    )
    command_queue.put((device, {
        'completed': 0,
        'total': len(videos),
        'description': f'{config.dataset} {description}',
    }))

    for video in videos:
        video_queue.put(video)

    # Signal the Decoder that no more videos will arrive.
    video_queue.put(None)

    # --- collect results from Track ------------------------------------------
    results: list[TrackingResult] = []
    for i in range(len(videos)):
        result: TrackingResult = result_queue.get()
        results.append(result)
        command_queue.put((device, {
            'completed': i + 1,
            'total': len(videos),
            'description': f'{config.dataset} {description}',
        }))

    # Timer stop: recorded as soon as the last result is collected.
    timer_end_ns = time.time_ns()

    # --- shut down stage processes -------------------------------------------
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    # --- compute elapsed time ------------------------------------------------
    elapsed_ms = (timer_end_ns - timer_start_ns) / 1e6

    # --- save results to disk (after timing) ---------------------------------
    _save_results(results, config, elapsed_ms)

    return results, elapsed_ms


# ---------------------------------------------------------------------------
# Disk save (happens after timing stops)
# ---------------------------------------------------------------------------

def _save_results(
    results: list[TrackingResult],
    config: PipelineConfig,
    elapsed_ms: float,
):
    """Persist tracking results and runtime summary to the cache directory."""
    for result in results:
        # Build output path matching the p060 convention.
        output_param_str = build_param_str(
            classifier=config.classifier,
            tilesize=config.tile_size,
            sample_rate=config.sample_rate,
            tilepadding=config.tilepadding,
            canvas_scale=config.canvas_scale,
            tracker=config.tracker,
            tracking_accuracy_threshold=config.tracking_accuracy_threshold,
            relevance_threshold=config.relevance_threshold,
        )
        output_path = cache.exec(
            config.dataset, 'ucomp-tracks', result.video,
            output_param_str, 'tracking.jsonl',
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_tracking_results(result.frame_tracks, output_path)

    # Save a pipeline runtime summary.
    summary_dir = cache.exec(config.dataset, 'pipeline-runtime')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'runtime.jsonl')
    with open(summary_path, 'a') as f:
        f.write(json.dumps({
            'config': config._asdict(),
            'elapsed_ms': elapsed_ms,
            'num_videos': len(results),
        }, default=str) + '\n')
