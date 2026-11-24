"""
The pipeline's computations as plain functions with explicit parameters (profiling/PLAN.md 8A).

Nothing here reads a Config or builds a pipeline message, and nothing frees a shared block it did not
create: who frees what is the caller's policy.  execution/stages.py wraps these for one configuration;
profiling calls them for every configuration of a sweep.  Each function matches its p0xx script:

  decode_video        p020/p030 frame reading into one shared BGR block
  classify_frames     p020's classification in batches of CLASSIFY_BATCH
  prune_scores        p022's binarization, tile grouping, and ILP
  pack_collages       p030's grouping, best-fit packing, and collage drawing (optional JPEG round trip)
  detect_canvases     p040's detection in batches of DETECT_BATCH, unpacked as p050 does
  track_detections    p060's tracking with trajectory interpolation
"""

import json
import time
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch

from execution import frames
from polyis import dtypes
from polyis.classify import classify_batch_tensors, load_classifier
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
from polyis.utilities import create_tracker, register_tracked_detections

# p020's BATCH_SIZE, p040's default --batch_size, p022's TILEPADDING_MODE and solve time limit,
# and p030's PackMode.Best_Fit.  tests/execution/test_stages.py checks each against its script.
CLASSIFY_BATCH = 16
DETECT_BATCH = 4
PRUNE_TILEPADDING_MODE = 0
PRUNE_TIME_LIMIT_S = 0.1
PACK_MODE_BEST_FIT = 2

# Workers see their pinned GPU as the only device (pipeline sets CUDA_VISIBLE_DEVICES).
DEVICE = 'cuda:0'


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

class Decoded(NamedTuple):
    video: str                  # path of the video file
    width: int
    height: int
    frame_count: int
    sampled: np.ndarray         # absolute indices of sampled frames, ascending
    partner: np.ndarray         # each sampled frame's diff partner (parallel to sampled)
    needed: np.ndarray          # absolute indices stored in `frames`, ascending and unique (sampled + partners)
    frames: frames.Block        # BGR uint8 [len(needed), height, width, 3]


def decode_video(video: str, prefix: str, sample_rate: int) -> Decoded:
    """Read the video once, keeping only the frames sample_rate needs, into a new shared block (s=1: every frame)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise OSError(f'could not open video {video}')
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sampled, partner, needed = frame_selection(frame_count, sample_rate)
        with frames.create(prefix, (len(needed), height, width, 3), 'uint8') as m:
            read = _read_needed(cap, m.array, needed, video)
        # p020 asserts every reported frame decodes; a short read would leave zero frames in the block.
        if read != frame_count:
            frames.unlink(m.block)
            raise ValueError(f'{video}: read {read} frames but the container reports {frame_count}')
    finally:
        cap.release()
    return Decoded(video, width, height, frame_count, sampled, partner, needed, m.block)


def _read_needed(cap: cv2.VideoCapture, block: np.ndarray, needed: np.ndarray, video: str) -> int:
    """Decode needed frames straight into their block rows, skip the rest; returns frames read.

    read() is grab() + retrieve(): retrieve() converts to BGR directly into the row we pass, and
    grab() alone advances without converting, so stored pixels equal cap.read()'s.  A separate
    function so its views of the block die before the block is closed.
    """
    read = k = 0
    while True:
        # needed is ascending: walk it alongside the frames being read.
        if k < len(needed) and read == needed[k]:
            row = block[k]
            ok, frame = cap.read(row)
            if not ok:
                return read
            # OpenCV writes in place only when the row matches the frame's size; otherwise it silently
            # returns a new array, which means the frame has an unexpected shape.
            if frame.ctypes.data != row.ctypes.data:
                raise ValueError(f'{video}: frame {read} has shape {frame.shape}, expected {row.shape}')
            k += 1
        elif not cap.grab():
            return read
        read += 1


# ---------------------------------------------------------------- classify

def load_classifier_state(weights: str, compilation_benchmark: str, never_relevant_bitmap: str,
                          tile_size: int) -> dict:
    """The optimized classifier and the GPU tensors that depend only on the dataset's tile grid."""
    with open(compilation_benchmark) as f:
        benchmark_results = [json.loads(line) for line in f]
    # The never-relevant bitmap is the dataset's tile grid, (grid_h, grid_w); classify accepts only videos
    # with exactly this grid, so everything below is built once here.
    never_relevant = np.load(never_relevant_bitmap)
    if never_relevant.ndim != 2:
        raise ValueError(f'never-relevant bitmap must be 2-D (grid_h, grid_w), got shape {never_relevant.shape}')
    grid_h, grid_w = never_relevant.shape
    model = load_classifier(weights).to(DEVICE)
    # The grid's tile count, as p020 takes it from its first video.
    model, method = select_model_optimization(model, benchmark_results, DEVICE, tile_size, grid_h * grid_w)
    # load_classifier and select_model_optimization both put the model in eval mode; every method keeps it.
    # Modules that expose .training are checked here; a frozen TorchScript module (the torchscript_optimize
    # method, chosen for caldot1-y05) has none, but torch.jit.freeze only accepts modules in eval mode.
    if getattr(model, 'training', False):
        raise RuntimeError(f'classifier is in training mode after optimization ({method})')
    # Tile positions exactly as p020 builds them.
    y = torch.arange(grid_h, device=DEVICE, dtype=torch.uint8).repeat_interleave(grid_w)
    x = torch.arange(grid_w, device=DEVICE, dtype=torch.uint8).repeat(grid_h)
    return {
        'tile_size': tile_size,
        'grid': (grid_h, grid_w),
        'model': model,
        'mask': torch.from_numpy(never_relevant.flatten()).to(DEVICE).to(torch.uint8),
        'positions': torch.stack([y, x], dim=1).float(),
        'normalize_mean': torch.tensor([0.485, 0.456, 0.406] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1),
        'normalize_std': torch.tensor([0.229, 0.224, 0.225] * 2, device=DEVICE, dtype=torch.float16).view(1, 6, 1, 1),
    }


def classify_frames(state: dict, video: Decoded, time_batches: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """p020's classification of every sampled frame, in batches of CLASSIFY_BATCH.

    Returns the uint8 scores [len(sampled), grid_h, grid_w] and, when time_batches is set, each sampled
    frame's share of its batch's wall time (the batch's seconds split evenly over its frames; the GPU is
    synchronized after each batch so the time belongs to it).  Without time_batches the second value is None.
    """
    ts = state['tile_size']
    grid = (video.height // ts, video.width // ts)
    if grid != state['grid']:
        raise ValueError(f'video grid {grid} does not match the never-relevant bitmap grid {state["grid"]}')
    # Positions in the frames block of each sampled frame and of its partner (needed is ascending and unique).
    sampled_pos = np.searchsorted(video.needed, video.sampled)
    partner_pos = np.searchsorted(video.needed, video.partner)
    scores = []
    shares = np.zeros(len(video.sampled)) if time_batches else None
    # inference_mode: no autograd at all (stricter than no_grad); the numbers are the same.
    with torch.inference_mode(), frames.attach(video.frames) as m:
        for start in range(0, len(video.sampled), CLASSIFY_BATCH):
            batch = slice(start, start + CLASSIFY_BATCH)
            t0 = time.perf_counter()
            scores.append(_classify_one_batch(state, m.array, sampled_pos[batch], partner_pos[batch]))
            if shares is not None:
                torch.cuda.synchronize()
                n = len(sampled_pos[batch])
                shares[batch] = (time.perf_counter() - t0) / n
        # Scores stay on the GPU per batch and come back to the CPU once per video.
        return torch.cat(scores, dim=0).cpu().numpy(), shares


def _classify_one_batch(state: dict, block: np.ndarray, sampled_pos: np.ndarray, partner_pos: np.ndarray) -> torch.Tensor:
    """One batch through p020's classification, with its frames sent to the GPU in one transfer.

    The batch's rows (sampled and partner frames, each row once) are gathered from the block by one
    numpy indexing step (a copy, so no view outlives this call), sent together, and flipped BGR->RGB
    on the GPU: the same uint8 tensors p020 builds frame by frame, without 2x16 CPU flips and transfers.
    Returns the batch's uint8 scores [batch, grid_h, grid_w], still on the GPU.
    """
    rows, where = np.unique(np.concatenate([sampled_pos, partner_pos]), return_inverse=True)
    batch = torch.from_numpy(block[rows]).to(DEVICE).flip(-1)
    where = torch.from_numpy(where).to(DEVICE)
    grid_h, grid_w = state['grid']
    probs, _ = classify_batch_tensors(grid_w, grid_h, state['positions'], batch[where[:len(sampled_pos)]],
                                      batch[where[len(sampled_pos):]], state['model'], state['tile_size'],
                                      DEVICE, state['normalize_mean'], state['normalize_std'], state['mask'])
    return probs


# ---------------------------------------------------------------- prune

def max_sampling_distance(table: np.ndarray, tolerance_index: int, sample_rate: int) -> np.ndarray:
    """p022's per-tile max sampling distance: the table's distance in frames, divided by the sample rate, at least 1."""
    if table.ndim != 3:
        raise ValueError(f'max_rate_table must be 3-D (grid_h, grid_w, thresholds), got shape {table.shape}')
    if tolerance_index >= table.shape[2]:
        raise ValueError(f'max_rate_table has {table.shape[2]} thresholds; tolerance index {tolerance_index} is out of range')
    distance = table[:, :, tolerance_index].copy()
    distance //= sample_rate
    return np.maximum(distance, 1)


def prune_scores(scores: np.ndarray, distance: np.ndarray, relevance_threshold: float,
                 time_limit_s: float | None = None) -> tuple[np.ndarray, Any]:
    """p022: keep only the polyominoes the ILP selects; returns the 0/255 grids and the ILP result.

    time_limit_s defaults to PRUNE_TIME_LIMIT_S, read at call time (not bound when the module loads).
    """
    num_frames, grid_h, grid_w = scores.shape
    if distance.shape != (grid_h, grid_w):
        raise ValueError(f'max_rate_table grid {distance.shape} != video grid {(grid_h, grid_w)}')
    # p022's binarization (note >= floor, unlike p030's >).
    cutoff = np.floor(relevance_threshold * 255)
    bitmaps = (scores >= cutoff).astype(np.uint8)
    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(bitmaps, PRUNE_TILEPADDING_MODE)
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)
    result = solve_ilp(tile_to_polyomino_id, polyomino_lengths, distance, grid_h, grid_w,
                       time_limit_seconds=PRUNE_TIME_LIMIT_S if time_limit_s is None else time_limit_s)
    # Selected polyomino ids per frame (p022 rebuilds this set per frame; grouping once is equivalent).
    selected: dict[int, list[int]] = {}
    for frame, pid in result.selected:
        selected.setdefault(frame, []).append(pid)
    pruned = np.zeros_like(scores)
    for b in range(num_frames):
        ids = tile_to_polyomino_id[b]
        pruned[b] = (np.isin(ids, selected.get(b, [])) & (ids >= 0)).astype(np.uint8) * 255
    return pruned, result


# ---------------------------------------------------------------- pack

class Packed(NamedTuple):
    video: str                  # path of the video file
    width: int
    height: int
    frame_count: int
    canvases: frames.Block      # BGR uint8 [num_collages, canvas_h, canvas_w, 3]
    index_maps: np.ndarray      # uint16 [num_collages, dst_grid_h, dst_grid_w]
    offset_lookups: list        # per collage: list of ((py, px), (oy, ox), frame_idx)


def pack_collages(scores: np.ndarray, video: Decoded, prefix: str, tile_size: int, padding_mode: int,
                  canvas_scale: float, relevance_threshold: float, jpeg: bool) -> Packed:
    """p030: group, best-fit pack, and draw every collage into one new shared block.

    scores is [len(video.sampled), grid_h, grid_w] for the sampled frames it lists (a subset of the video's
    stored frames is fine: positions are looked up in video.needed).  The frames block is only read.
    """
    ts = tile_size
    src_gh, src_gw = video.height // ts, video.width // ts
    dst_gh = max(1, int(round(src_gh * canvas_scale)))
    dst_gw = max(1, int(round(src_gw * canvas_scale)))

    # Group each sampled frame's relevant tiles into polyominoes (p030's > T_r*255).
    stacks = np.empty(len(video.sampled), dtype=np.uint64)
    for i in range(len(video.sampled)):
        bitmap = (scores[i] > (relevance_threshold * 255)).astype(np.uint8)
        assert dtypes.is_bitmap(bitmap), bitmap.shape
        stacks[i] = group_tiles(bitmap, padding_mode)
    collages = pack(stacks, dst_gh, dst_gw, PACK_MODE_BEST_FIT)

    # Tile pixel bounds on the source frame and the canvas, as p030 precomputes them.
    src_bounds = _grid_bounds(src_gh, ts) + _grid_bounds(src_gw, ts)
    dst_bounds = _grid_bounds(dst_gh, ts) + _grid_bounds(dst_gw, ts)
    # Position in the frames block of each sampled frame (needed is ascending and unique).
    sampled_pos = np.searchsorted(video.needed, video.sampled)
    index_maps = np.zeros((len(collages), dst_gh, dst_gw), dtype=np.uint16)
    offset_lookups: list = []
    with frames.attach(video.frames) as fm, \
            frames.create(prefix, (len(collages), dst_gh * ts, dst_gw * ts, 3), 'uint8') as cm:
        cm.array[...] = 0
        for k, collage in enumerate(collages):
            assert len(collage) > 0, f'collage {k} is empty'
            offset_lookups.append(_draw_collage(cm.array[k], index_maps[k], collage, fm.array, sampled_pos,
                                                video.sampled, src_bounds, dst_bounds))
            if jpeg:
                _jpeg_round_trip(cm.array[k])
    return Packed(video.video, video.width, video.height, video.frame_count, cm.block, index_maps, offset_lookups)


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

def load_detector(detector_type: str, options: dict) -> Any:
    """The detector, on the worker's only visible GPU (index 0).  num_images is unused by every backend."""
    return build_detector(detector_type, options, 0, DETECT_BATCH, 0)


def detect_canvases(detector: Any, collages: Packed, tile_size: int) -> dict[int, list[list[float]]]:
    """p040 + p050: detect on each canvas and map boxes back to frames; the canvases block is only read."""
    # p050 keeps a (possibly empty) list for every frame of the video.
    detections: dict[int, list[list[float]]] = {i: [] for i in range(collages.frame_count)}
    with torch.no_grad(), frames.attach(collages.canvases) as cm:
        num_canvases = cm.array.shape[0]
        for start in range(0, num_canvases, DETECT_BATCH):
            ks = range(start, min(start + DETECT_BATCH, num_canvases))
            outputs = _detect_one_batch(detector, cm.array, ks)
            for k, out in zip(ks, outputs):
                # p050's default: keep boxes centered on empty canvas space (drop_center_outside=False).
                per_frame, _not_in_tile, _center_outside = unpack_detections(
                    out.tolist(), collages.index_maps[k], collages.offset_lookups[k], tile_size, False)
                for frame_idx, boxes in per_frame.items():
                    detections[frame_idx].extend(boxes)
    return detections


def _detect_one_batch(detector: Any, block: np.ndarray, ks: range) -> list:
    """Detector on copies of the canvases, so no backend can keep a view into shared memory."""
    return detect_batch([block[k].copy() for k in ks], detector)


# ---------------------------------------------------------------- track

def track_detections(tracker_name: str, width: int, height: int,
                     detections: dict[int, list[list[float]]]) -> dict[int, list[list[float]]]:
    """p060: tracker over every frame in order, interpolating trajectories."""
    tracker = create_tracker(tracker_name, img_size=(height, width))
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_tracks: dict[int, list[list[float]]] = {}
    for frame_idx in sorted(detections):
        dets = np.array(detections[frame_idx])
        dets = dets[:, :5] if dets.size > 0 else np.empty((0, 5))
        tracked = tracker.update(dets)
        register_tracked_detections(tracked, frame_idx, frame_tracks, trajectories, interpolate=True)
    return frame_tracks
