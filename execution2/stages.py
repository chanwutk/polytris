"""
The six execution2 stages (PLAN.md step 4).  Each matches its p0xx script's computation:

  decode    p020/p030 frame reading: every s-th frame plus the last, and each one's diff partner
            (idx-1, or idx+1 for frame 0), into one shared BGR block per video.
  classify  p020: classify_batch over the sampled frames in batches of 16 (RGB, diff vs partner).
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
travel between stages, never pixels.
"""

import json
from functools import partial
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch

from execution2 import frames
from execution2.config import MISTRACK_TOLERANCES, Config
from execution2.pipeline import Stage, Worker
from polyis import dtypes
from polyis.classify import classify_batch, load_classifier
from polyis.models.detector import build_detector, detect_batch
from polyis.pack.adapters import group_tiles_all
from polyis.pack.collage import (
    _compute_polyomino_tile_boundaries,
    _copy_polyomino_tiles_to_canvas,
    _update_polyomino_metadata,
    unpack_detections,
)
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.sample.ilp.c.gurobi import solve_ilp
from polyis.train.select_model_optimization import select_model_optimization
from polyis.utilities import TILEPADDING_MAPS, create_tracker, register_tracked_detections

# p020's BATCH_SIZE, p040's default --batch_size, p022's TILEPADDING_MODE and solve time limit,
# and p030's PackMode.Best_Fit.  tests/execution2/test_stages.py checks each against its script.
CLASSIFY_BATCH = 16
DETECT_BATCH = 4
PRUNE_TILEPADDING_MODE = 0
PRUNE_TIME_LIMIT_S = 0.1
PACK_MODE_BEST_FIT = 2

# Workers see their pinned GPU as the only device (pipeline sets CUDA_VISIBLE_DEVICES).
DEVICE = 'cuda:0'


class Decoded(NamedTuple):
    video: str
    width: int
    height: int
    frame_count: int
    sampled: np.ndarray         # absolute indices of sampled frames, ascending
    partner: np.ndarray         # each sampled frame's diff partner (parallel to sampled)
    needed: np.ndarray          # absolute indices stored in `frames`, ascending and unique (sampled + partners)
    frames: frames.Block        # BGR uint8 [len(needed), height, width, 3]


class Classified(NamedTuple):
    decoded: Decoded
    scores: np.ndarray          # uint8 [len(sampled), grid_h, grid_w]: classifier scores, or 0/255 after prune


class Packed(NamedTuple):
    video: str
    width: int
    height: int
    frame_count: int
    canvases: frames.Block      # BGR uint8 [num_collages, canvas_h, canvas_w, 3]
    index_maps: np.ndarray      # uint16 [num_collages, dst_grid_h, dst_grid_w]
    offset_lookups: list        # per collage: list of ((py, px), (oy, ox), frame_idx)


class Detected(NamedTuple):
    video: str
    width: int
    height: int
    frame_count: int
    detections: dict[int, list[list[float]]]   # every frame 0..frame_count-1, in original coordinates


class Tracked(NamedTuple):
    video: str
    frame_tracks: dict[int, list[list[float]]]


def frame_selection(frame_count: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """p020's sampled frames, each one's diff partner (parallel array), and the frames to store, ascending and unique."""
    # p020 reads frames[idx + 1] for frame 0, which needs a second frame.
    if frame_count < 2:
        raise ValueError(f'video has {frame_count} frame(s); the classifier diff needs at least 2')
    # Every s-th frame, plus the last frame when it is not already sampled.
    sampled = np.arange(0, frame_count, sample_rate)
    if sampled[-1] != frame_count - 1:
        sampled = np.append(sampled, frame_count - 1)
    # Diff partner: the previous frame, or the next one for frame 0.
    partner = sampled - 1
    partner[0] = 1
    # Frames 0 and 1 (frame 0 and its partner) always come first.  The remaining pairs interleave to
    # [s1-1, s1, s2-1, s2, ...], non-decreasing because sampled is strictly increasing, so no sort is needed;
    # their values below 2 (only when s1 <= 2) are 0 or 1, already at the front.
    pairs = np.stack([partner[1:], sampled[1:]], axis=1).ravel()
    needed = np.concatenate([[0, 1], pairs[pairs >= 2]])
    # Drop adjacent repeats: a partner that is itself sampled (s <= 2, or a last frame right after a sampled one).
    needed = needed[np.r_[True, needed[1:] != needed[:-1]]]
    return sampled, partner, needed


# ---------------------------------------------------------------- decode

def decode_setup(cfg: Config, prefix: str, worker: Worker) -> dict:
    return {'sample_rate': cfg.params.sample_rate, 'prefix': prefix}


def decode(state: dict, video: str) -> Decoded:
    """Read the video once, keeping only the needed frames, into a new shared block."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise OSError(f'could not open video {video}')
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sampled, partner, needed = frame_selection(frame_count, state['sample_rate'])
        with frames.create(state['prefix'], (len(needed), height, width, 3), 'uint8') as m:
            # needed is ascending: walk it alongside the frames being read.
            read = k = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (height, width, 3):
                    raise ValueError(f'{video}: frame {read} has shape {frame.shape}, expected {(height, width, 3)}')
                if k < len(needed) and read == needed[k]:
                    m.array[k] = frame
                    k += 1
                read += 1
            # p020 asserts every reported frame decodes; a short read would leave zero frames in the block.
            if read != frame_count:
                raise ValueError(f'{video}: read {read} frames but the container reports {frame_count}')
    finally:
        cap.release()
    return Decoded(video, width, height, frame_count, sampled, partner, needed, m.block)


# ---------------------------------------------------------------- classify

def classify_setup(cfg: Config, worker: Worker) -> dict:
    """Load the classifier, never-relevant mask, and normalization tensors once per worker."""
    with open(cfg.artifacts.classifier.compilation_benchmark) as f:
        benchmark_results = [json.loads(line) for line in f]
    return {
        'tile_size': cfg.params.tile_size,
        'model': load_classifier(cfg.artifacts.classifier.weights).to(DEVICE),
        'benchmark_results': benchmark_results,
        # The optimization is picked from the first video's grid, as p020 picks it from its first video.
        'optimized': False,
        'normalize_mean': torch.tensor([0.485, 0.456, 0.406] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1),
        'normalize_std': torch.tensor([0.229, 0.224, 0.225] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1),
        'never_relevant': np.load(cfg.artifacts.never_relevant_bitmap),
    }


def classify(state: dict, d: Decoded) -> Classified:
    """p020's classification of every sampled frame, in batches of CLASSIFY_BATCH."""
    ts = state['tile_size']
    grid_w, grid_h = d.width // ts, d.height // ts
    mask_np = state['never_relevant']
    if mask_np.size != grid_h * grid_w:
        raise ValueError(f'{d.video}: never-relevant bitmap has {mask_np.size} tiles, video grid has {grid_h}x{grid_w}')
    if not state['optimized']:
        state['model'], _ = select_model_optimization(state['model'], state['benchmark_results'], DEVICE, ts,
                                                      grid_w * grid_h)
        state['optimized'] = True
    mask = torch.from_numpy(mask_np.flatten()).to(DEVICE).to(torch.uint8)
    # Tile positions exactly as p020 builds them.
    y = torch.arange(grid_h, device=DEVICE, dtype=torch.uint8).repeat_interleave(grid_w)
    x = torch.arange(grid_w, device=DEVICE, dtype=torch.uint8).repeat(grid_h)
    positions = torch.stack([y, x], dim=1).float()
    # Positions in the frames block of each sampled frame and of its partner (needed is ascending and unique).
    sampled_pos = np.searchsorted(d.needed, d.sampled)
    partner_pos = np.searchsorted(d.needed, d.partner)
    scores = []
    with torch.no_grad(), frames.attach(d.frames) as m:
        for start in range(0, len(d.sampled), CLASSIFY_BATCH):
            batch = slice(start, start + CLASSIFY_BATCH)
            scores.append(_classify_one_batch(state, m.array, sampled_pos[batch], partner_pos[batch],
                                              grid_w, grid_h, positions, mask))
    return Classified(d, np.concatenate(scores, axis=0))


def _classify_one_batch(state: dict, block: np.ndarray, sampled_pos: np.ndarray, partner_pos: np.ndarray,
                        grid_w: int, grid_h: int, positions: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """One classify_batch call; BGR->RGB copies as p020 makes them (views die with this frame)."""
    rgb = [np.ascontiguousarray(block[k][:, :, ::-1]) for k in sampled_pos]
    prev = [np.ascontiguousarray(block[k][:, :, ::-1]) for k in partner_pos]
    probs, _runtime = classify_batch(grid_w, grid_h, positions, rgb, prev, state['model'], state['tile_size'],
                                     DEVICE, state['normalize_mean'], state['normalize_std'], mask)
    return probs.cpu().numpy()


# ---------------------------------------------------------------- prune

def prune_setup(cfg: Config, worker: Worker) -> dict:
    """p022's per-tile max sampling distance for this tolerance, loaded once."""
    assert cfg.params.mistrack_tolerance is not None and cfg.artifacts.max_rate_table is not None
    table = np.load(cfg.artifacts.max_rate_table)
    if table.ndim != 3:
        raise ValueError(f'max_rate_table must be 3-D (grid_h, grid_w, thresholds), got shape {table.shape}')
    idx = MISTRACK_TOLERANCES.index(cfg.params.mistrack_tolerance)
    if idx >= table.shape[2]:
        raise ValueError(f'max_rate_table has {table.shape[2]} thresholds; tolerance index {idx} is out of range')
    # p022: the table's distance in frames, divided by the sample rate, at least 1.
    distance = table[:, :, idx].copy()
    distance //= cfg.params.sample_rate
    return {'distance': np.maximum(distance, 1), 'relevance_threshold': cfg.params.relevance_threshold}


def prune(state: dict, c: Classified) -> Classified:
    """p022: keep only the polyominoes the ILP selects; scores become 0/255 grids."""
    num_frames, grid_h, grid_w = c.scores.shape
    if state['distance'].shape != (grid_h, grid_w):
        raise ValueError(f'{c.decoded.video}: max_rate_table grid {state["distance"].shape} != video grid {(grid_h, grid_w)}')
    # p022's binarization (note >= floor, unlike p030's >).
    cutoff = np.floor(state['relevance_threshold'] * 255)
    bitmaps = (c.scores >= cutoff).astype(np.uint8)
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(bitmaps, PRUNE_TILEPADDING_MODE)
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)
    result = solve_ilp(tile_to_polyomino_id, polyomino_lengths, state['distance'], grid_h, grid_w,
                       time_limit_seconds=PRUNE_TIME_LIMIT_S)
    # Selected polyomino ids per frame (p022 rebuilds this set per frame; grouping once is equivalent).
    selected: dict[int, list[int]] = {}
    for frame, pid in result.selected:
        selected.setdefault(frame, []).append(pid)
    pruned = np.zeros_like(c.scores)
    for b in range(num_frames):
        ids = tile_to_polyomino_id[b]
        pruned[b] = (np.isin(ids, selected.get(b, [])) & (ids >= 0)).astype(np.uint8) * 255
    return c._replace(scores=pruned)


# ---------------------------------------------------------------- pack

def pack_setup(cfg: Config, prefix: str, worker: Worker) -> dict:
    p = cfg.params
    return {'tile_size': p.tile_size, 'padding': TILEPADDING_MAPS[p.tile_padding], 'canvas_scale': p.canvas_scale,
            'relevance_threshold': p.relevance_threshold, 'prefix': prefix, 'jpeg': p.canvas_format == 'jpeg'}


def pack_video(state: dict, c: Classified) -> Packed:
    """p030: group, best-fit pack, draw every collage into one shared block; then free the frames."""
    d, ts = c.decoded, state['tile_size']
    src_gh, src_gw = d.height // ts, d.width // ts
    dst_gh = max(1, int(round(src_gh * state['canvas_scale'])))
    dst_gw = max(1, int(round(src_gw * state['canvas_scale'])))

    # Group each sampled frame's relevant tiles into polyominoes (p030's > T_r*255).
    stacks = np.empty(len(d.sampled), dtype=np.uint64)
    for i in range(len(d.sampled)):
        bitmap = (c.scores[i] > (state['relevance_threshold'] * 255)).astype(np.uint8)
        assert dtypes.is_bitmap(bitmap), bitmap.shape
        stacks[i] = group_tiles(bitmap, state['padding'])
    collages = pack(stacks, dst_gh, dst_gw, PACK_MODE_BEST_FIT)

    # Tile pixel bounds on the source frame and the canvas, as p030 precomputes them.
    src_bounds = _grid_bounds(src_gh, ts) + _grid_bounds(src_gw, ts)
    dst_bounds = _grid_bounds(dst_gh, ts) + _grid_bounds(dst_gw, ts)
    # Position in the frames block of each sampled frame (needed is ascending and unique).
    sampled_pos = np.searchsorted(d.needed, d.sampled)
    index_maps = np.zeros((len(collages), dst_gh, dst_gw), dtype=np.uint16)
    offset_lookups: list = []
    with frames.attach(d.frames) as fm, \
            frames.create(state['prefix'], (len(collages), dst_gh * ts, dst_gw * ts, 3), 'uint8') as cm:
        cm.array[...] = 0
        for k, collage in enumerate(collages):
            assert len(collage) > 0, f'collage {k} is empty'
            offset_lookups.append(_draw_collage(cm.array[k], index_maps[k], collage, fm.array, sampled_pos,
                                                d.sampled, src_bounds, dst_bounds))
            if state['jpeg']:
                _jpeg_round_trip(cm.array[k])
    # Pack is the frames' last user.
    frames.unlink(d.frames)
    return Packed(d.video, d.width, d.height, d.frame_count, cm.block, index_maps, offset_lookups)


def _jpeg_round_trip(canvas: np.ndarray) -> None:
    """Replace canvas, in place, with what p040 reads back from p030's cv2.imwrite(..., '.jpg')."""
    # Same encoder and default quality as imwrite, same decoder and flags as imread.
    ok, encoded = cv2.imencode('.jpg', canvas)
    if not ok:
        raise RuntimeError('JPEG encoding of a canvas failed')
    canvas[...] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _grid_bounds(n: int, ts: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and end pixel of each of n fixed-size tiles."""
    starts = np.arange(n, dtype=np.int32) * ts
    return starts, starts + ts


def _draw_collage(canvas: np.ndarray, index_map: np.ndarray, collage: list, block: np.ndarray,
                  sampled_pos: np.ndarray, sampled: np.ndarray, src_bounds: tuple, dst_bounds: tuple) -> list:
    """Draw one collage exactly as p030's render loop does; returns its offset lookup."""
    offset_lookup: list = []
    for gid, pos in enumerate(collage, start=1):
        # pack() returns frame as an index into the sampled list; map it to the absolute frame (a plain int,
        # since it goes into the offset lookup).
        frame_idx = int(sampled[pos.frame])
        i_coords, j_coords = pos.shape[:, 0], pos.shape[:, 1]
        bounds = _compute_polyomino_tile_boundaries(pos.oy, pos.ox, pos.py, pos.px, i_coords, j_coords,
                                                    *src_bounds, *dst_bounds)
        _copy_polyomino_tiles_to_canvas(canvas, block[sampled_pos[pos.frame]], *bounds)
        _update_polyomino_metadata(index_map, offset_lookup, gid, pos.py, pos.px, pos.oy, pos.ox, frame_idx,
                                   i_coords, j_coords)
    return offset_lookup


# ---------------------------------------------------------------- detect

def detect_setup(cfg: Config, worker: Worker) -> dict:
    det = cfg.artifacts.detector
    # GPU index 0: the worker's pinned GPU is its only visible device.  num_images is unused by every backend.
    return {'detector': build_detector(det.type, det.options, 0, DETECT_BATCH, 0), 'tile_size': cfg.params.tile_size}


def detect(state: dict, p: Packed) -> Detected:
    """p040 + p050: detect on each canvas, map boxes back to frames; then free the canvases."""
    # p050 keeps a (possibly empty) list for every frame of the video.
    detections: dict[int, list[list[float]]] = {i: [] for i in range(p.frame_count)}
    with torch.no_grad(), frames.attach(p.canvases) as cm:
        num_canvases = cm.array.shape[0]
        for start in range(0, num_canvases, DETECT_BATCH):
            ks = range(start, min(start + DETECT_BATCH, num_canvases))
            outputs = _detect_one_batch(state['detector'], cm.array, ks)
            for k, out in zip(ks, outputs):
                # p050's default: keep boxes centered on empty canvas space (drop_center_outside=False).
                per_frame, _not_in_tile, _center_outside = unpack_detections(
                    out.tolist(), p.index_maps[k], p.offset_lookups[k], state['tile_size'], False)
                for frame_idx, boxes in per_frame.items():
                    detections[frame_idx].extend(boxes)
    # Detect is the canvases' last user.
    frames.unlink(p.canvases)
    return Detected(p.video, p.width, p.height, p.frame_count, detections)


def _detect_one_batch(detector: Any, block: np.ndarray, ks: range) -> list:
    """Detector on copies of the canvases, so no backend can keep a view into shared memory."""
    return detect_batch([block[k].copy() for k in ks], detector)


# ---------------------------------------------------------------- track

def track_setup(cfg: Config, worker: Worker) -> dict:
    return {'tracker': cfg.params.tracker}


def track(state: dict, d: Detected) -> Tracked:
    """p060: tracker over every frame in order, interpolating trajectories."""
    tracker = create_tracker(state['tracker'], img_size=(d.height, d.width))
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    for frame_idx in sorted(d.detections):
        dets = np.array(d.detections[frame_idx])
        dets = dets[:, :5] if dets.size > 0 else np.empty((0, 5))
        tracked = tracker.update(dets)
        register_tracked_detections(tracked, frame_idx, frame_tracks, trajectories, interpolate=True)
    return Tracked(d.video, frame_tracks)


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
