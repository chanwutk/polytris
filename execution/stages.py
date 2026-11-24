"""
The six execution2 stages (PLAN.md step 4).  Each matches its p0xx script's computation:

  decode    p020/p030 frame reading: every s-th frame plus the last, and each one's diff partner
            (idx-1, or idx+1 for frame 0), into one shared BGR block per video.
  classify  p020: classify_batch's computation over the sampled frames in batches of 16 (RGB, diff vs
            partner), each batch sent to the GPU in one transfer.
  prune     p022 (only when mistrack_tolerance is set): binarize with score >= floor(T_r*255),
            group tiles with padding mode 0, solve the ILP (0.1 s), keep selected polyominoes.
  pack      p030: binarize with score > T_r*255, group tiles with padding P, best-fit pack,
            draw collages on the CPU into one shared block per video, then free the frames.
            canvas_format 'jpeg' passes each canvas through JPEG as p030 writes and p040 reads it;
            'raw' leaves it as drawn.
  detect    p040 + p050: detector on the canvases in batches of 4, unpack boxes to per-frame
            lists for every frame, then free the canvases.
  track     p060: tracker over every frame in order, with trajectory interpolation.

Messages are NamedTuples keyed by the video's path; only small handles to shared blocks
travel between stages, never pixels.  The computations, and the Decoded and Packed messages they take and
return, live in execution2/operators.py (profiling shares them); this module adds the other messages and the
policy of who frees which block.
"""

from functools import partial
from typing import NamedTuple

import numpy as np

from execution import frames, operators as ops
from execution.config import MISTRACK_TOLERANCES, Config
from execution.pipeline import Stage, Worker
# Re-exported: tests, tools, and main import these names from stages.
from execution.operators import (  # noqa: F401
    CLASSIFY_BATCH, DETECT_BATCH, DEVICE, PACK_MODE_BEST_FIT, PRUNE_TILEPADDING_MODE, PRUNE_TIME_LIMIT_S,
    detect_batch, frame_selection, unpack_detections,
)
# The decoded-video and packed-canvas messages are the operators' own types.
from execution.operators import Decoded, Packed
from polyis.utilities import TILEPADDING_MAPS, is_tile_padding


class Classified(NamedTuple):
    decoded: Decoded
    scores: np.ndarray          # uint8 [len(sampled), grid_h, grid_w]: classifier scores, or 0/255 after prune


class Detected(NamedTuple):
    video: str
    width: int
    height: int
    frame_count: int
    detections: dict[int, list[list[float]]]   # every frame 0..frame_count-1, in original coordinates


class Tracked(NamedTuple):
    video: str
    frame_tracks: dict[int, list[list[float]]]


# ---------------------------------------------------------------- decode

def decode_setup(cfg: Config, prefix: str, worker: Worker) -> dict:
    return {'sample_rate': cfg.params.sample_rate, 'prefix': prefix}


def decode(state: dict, video: str) -> Decoded:
    """Read the video once, keeping only the needed frames, into a new shared block."""
    return ops.decode_video(video, state['prefix'], state['sample_rate'])


# ---------------------------------------------------------------- classify

def classify_setup(cfg: Config, worker: Worker) -> dict:
    """Once per worker: the optimized classifier and the GPU tensors that depend only on the dataset's grid."""
    c = cfg.artifacts.classifier
    return ops.load_classifier_state(c.weights, c.compilation_benchmark, cfg.artifacts.never_relevant_bitmap,
                                     cfg.params.tile_size)


def classify(state: dict, d: Decoded) -> Classified:
    """p020's classification of every sampled frame, in batches of CLASSIFY_BATCH."""
    try:
        scores, _ = ops.classify_frames(state, d)
    except ValueError as e:
        raise ValueError(f'{d.video}: {e}') from e
    return Classified(d, scores)


# ---------------------------------------------------------------- prune

def prune_setup(cfg: Config, worker: Worker) -> dict:
    """p022's per-tile max sampling distance for this tolerance, loaded once."""
    assert cfg.params.mistrack_tolerance is not None and cfg.artifacts.max_rate_table is not None
    distance = ops.max_sampling_distance(np.load(cfg.artifacts.max_rate_table),
                                         MISTRACK_TOLERANCES.index(cfg.params.mistrack_tolerance),
                                         cfg.params.sample_rate)
    return {'distance': distance, 'relevance_threshold': cfg.params.relevance_threshold}


def prune(state: dict, c: Classified) -> Classified:
    """p022: keep only the polyominoes the ILP selects; scores become 0/255 grids."""
    try:
        pruned, _ = ops.prune_scores(c.scores, state['distance'], state['relevance_threshold'])
    except ValueError as e:
        raise ValueError(f'{c.decoded.video}: {e}') from e
    return c._replace(scores=pruned)


# ---------------------------------------------------------------- pack

def pack_setup(cfg: Config, prefix: str, worker: Worker) -> dict:
    p = cfg.params
    padding = p.tile_padding
    # The config already rejects unknown modes; the guard narrows str to TilePadding for the lookup.
    if not is_tile_padding(padding):
        raise ValueError(f'unknown tile padding {padding!r}')
    return {'tile_size': p.tile_size, 'padding': TILEPADDING_MAPS[padding], 'canvas_scale': p.canvas_scale,
            'relevance_threshold': p.relevance_threshold, 'prefix': prefix, 'jpeg': p.canvas_format == 'jpeg'}


def pack_video(state: dict, c: Classified) -> Packed:
    """p030: group, best-fit pack, draw every collage into one shared block; then free the frames."""
    d = c.decoded
    packed = ops.pack_collages(c.scores, d, state['prefix'], state['tile_size'], state['padding'],
                               state['canvas_scale'], state['relevance_threshold'], state['jpeg'])
    # Pack is the frames' last user.
    frames.unlink(d.frames)
    return packed


# ---------------------------------------------------------------- detect

def detect_setup(cfg: Config, worker: Worker) -> dict:
    det = cfg.artifacts.detector
    return {'detector': ops.load_detector(det.type, det.options), 'tile_size': cfg.params.tile_size}


def detect(state: dict, p: Packed) -> Detected:
    """p040 + p050: detect on each canvas, map boxes back to frames; then free the canvases."""
    detections = ops.detect_canvases(state['detector'], p, state['tile_size'])
    # Detect is the canvases' last user.
    frames.unlink(p.canvases)
    return Detected(p.video, p.width, p.height, p.frame_count, detections)


# ---------------------------------------------------------------- track

def track_setup(cfg: Config, worker: Worker) -> dict:
    return {'tracker': cfg.params.tracker}


def track(state: dict, d: Detected) -> Tracked:
    """p060: tracker over every frame in order, interpolating trajectories."""
    return Tracked(d.video, ops.track_detections(state['tracker'], d.width, d.height, d.detections))


# ---------------------------------------------------------------- wiring

def build_stages(cfg: Config, prefix: str) -> list[Stage]:
    """The pipeline for cfg, with process counts and GPUs from cfg.system; prune only when configured."""
    s = cfg.system.stages
    stages = [
        Stage('decode', partial(decode_setup, cfg, prefix), decode, s['decode'].procs),
        Stage('classify', partial(classify_setup, cfg), classify, s['classify'].procs, s['classify'].gpu),
    ]
    if cfg.use_prune:
        stages.append(Stage('prune', partial(prune_setup, cfg), prune, s['prune'].procs))
    stages += [
        Stage('pack', partial(pack_setup, cfg, prefix), pack_video, s['pack'].procs),
        Stage('detect', partial(detect_setup, cfg), detect, s['detect'].procs, s['detect'].gpu),
        Stage('track', partial(track_setup, cfg), track, s['track'].procs),
    ]
    return stages
