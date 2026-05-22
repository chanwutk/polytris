#!/usr/local/bin/python
"""
Generate data files for the D3 demo visualization.

Pipeline (all run on remote inside Docker via ./run viz/demo/generate_data.py):
  1. Extract `num_frames` consecutive (or sample_rate-spaced) frames from the chosen video; save as PNG.
  2. Compute per-frame relevance bitmaps from groundtruth tracks (same as p021).
  3. Group connected relevant tiles into polyominoes per frame.
  4. Save polyomino cutouts as RGBA PNGs (alpha mask matches polyomino shape) and an outline polyline.
  5. Compute a max_rate_table for the requested M values (mistrack rate tolerance, mapped to accuracy = 1 - M).
     Loads existing accuracy.npy if present; otherwise runs p016 aggregation on the train set
     and writes accuracy.npy back ONLY if it was missing (never overwrites).
  6. For each M: solve the pruning ILP with `time_limit` seconds budget; record discarded (f, i) pairs.
  7. For each M: pack the surviving polyominoes (First-Fit); record canvas positions.
  8. Emit JSON files (polyominoes.json, pruning.json, packing.json, meta.json) under --output-dir.

Refuses to overwrite existing output dir unless --overwrite is passed.
"""

import argparse
import datetime
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Make sibling repo imports work whether we run from the repo root or anywhere else
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyis.io import cache, store
from polyis.pack.adapters import group_tiles_all
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.sample.ilp.c.gurobi import solve_ilp
from polyis.utilities import (
    TILEPADDING_MAPS,
    get_overlapping_tiles,
    load_tracking_results,
    mark_detections,
)


# Sample rates used when p016 builds accuracy.npy. Must match SAMPLE_RATES in p016.
INDEX_SAMPLE_RATES = [1, 2, 4, 8, 16]

# Packing mode integer constants (see polyis/pack/c/pack.h)
PACK_MODE_EASIEST_FIT = 0
PACK_MODE_FIRST_FIT = 1
PACK_MODE_BEST_FIT = 2
PACK_MODE_LOOKUP = {
    'easiest_fit': PACK_MODE_EASIEST_FIT,
    'first_fit': PACK_MODE_FIRST_FIT,
    'best_fit': PACK_MODE_BEST_FIT,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='caldot1-y05')
    parser.add_argument('--video', default=None,
                        help='Video filename. If omitted, picks the first .mp4 alphabetically in the test set.')
    parser.add_argument('--videoset', default='test')
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=64)
    parser.add_argument('--sample-rate', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=60)
    parser.add_argument('--tracker', default='bytetrackcython')
    parser.add_argument('--tilepadding', default='none')
    parser.add_argument('--canvas-scale', type=float, default=1.0)
    parser.add_argument('--relevance-threshold', type=float, default=0.5,
                        help='Unused for groundtruth classification but recorded in meta.json.')
    parser.add_argument('--packing-mode', choices=list(PACK_MODE_LOOKUP), default='first_fit')
    parser.add_argument('--time-limit', type=float, default=10.0,
                        help='ILP solver wall-clock time limit in seconds per M value.')
    parser.add_argument('--m-values', default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated mistrack-rate tolerance values in [0, 1].')
    parser.add_argument('--output-dir', default=str(REPO_ROOT / 'viz' / 'demo' / 'data'))
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow clobbering an existing non-empty output directory.')
    parser.add_argument('--image-scale', type=float, default=0.25,
                        help='Scale factor for saved frame and polyomino PNGs. Coordinates in the '
                             'JSON files remain in original (tile-aligned) pixel space; the SVG '
                             '<image> element scales the lower-res PNGs back up at render time. '
                             'Default 0.25 reduces file size ~16x with negligible visual loss.')
    return parser.parse_args()


def _resolve_video(dataset: str, videoset: str, video: str | None) -> str:
    """Pick the first video alphabetically when not specified."""
    videoset_dir = store.dataset(dataset, videoset)
    assert videoset_dir.exists(), f"Videoset directory {videoset_dir} does not exist"
    if video is not None:
        candidate = videoset_dir / video
        assert candidate.exists(), f"Video file {candidate} does not exist"
        return video
    videos = sorted(f.name for f in videoset_dir.iterdir()
                    if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))
    assert len(videos) > 0, f"No video files found in {videoset_dir}"
    return videos[0]


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"{output_dir} exists and is not a directory")
        # Allow re-running into an empty directory; refuse non-empty without --overwrite
        if any(output_dir.iterdir()):
            if not overwrite:
                raise SystemExit(
                    f"Output directory {output_dir} is not empty. Re-run with --overwrite to clobber it.")
            shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'frames').mkdir(exist_ok=True)
    (output_dir / 'polyominoes').mkdir(exist_ok=True)


def _extract_frames(video_path: Path, frame_indices: list[int]) -> tuple[list[np.ndarray], int, int]:
    """Read frames at the given indices in increasing order. Returns BGR arrays, height, width."""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Could not open video {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr: list[np.ndarray | None] = [None] * len(frame_indices)
    wanted = {idx: pos for pos, idx in enumerate(frame_indices)}
    max_idx = max(frame_indices)
    cur = 0
    # Linear scan is robust across codecs where CAP_PROP_POS_FRAMES seeks are unreliable.
    while cur <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in wanted:
            frames_bgr[wanted[cur]] = frame
        cur += 1
    cap.release()

    missing = [frame_indices[i] for i, f in enumerate(frames_bgr) if f is None]
    assert not missing, f"Failed to read frames {missing} from {video_path}"
    return [f for f in frames_bgr if f is not None], height, width  # type: ignore[return-value]


def _save_frame_pngs(frames_bgr: list[np.ndarray], target_h: int, target_w: int,
                     frame_indices: list[int], out_dir: Path, image_scale: float) -> None:
    """Resize each frame to tile-aligned (target_h, target_w), then optionally scale down for web."""
    save_w = max(1, int(round(target_w * image_scale)))
    save_h = max(1, int(round(target_h * image_scale)))
    for frame_idx, frame in zip(frame_indices, frames_bgr):
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # OpenCV uses BGR; convert to RGB before saving with PIL so PNGs are color-correct.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if (save_w, save_h) != (target_w, target_h):
            img = img.resize((save_w, save_h), Image.LANCZOS)
        img.save(str(out_dir / f'{frame_idx}.png'))


def _resize_detections(detections: list[list[float]], scale_x: float, scale_y: float) -> list[list[float]]:
    """Rescale the bbox portion (last 4 entries) by per-axis scale factors. Preserves any leading id/score."""
    out: list[list[float]] = []
    for det in detections:
        det = list(det)
        det[-4] *= scale_x
        det[-3] *= scale_y
        det[-2] *= scale_x
        det[-1] *= scale_y
        out.append(det)
    return out


def _load_naive_detections(dataset: str, video: str) -> dict[int, list[list[float]]]:
    """
    Load detector outputs from `cache.exec(dataset, 'naive', video, 'detection.jsonl')`.

    Each line is `{"frame_idx": int, "detections": [[x1, y1, x2, y2, score], ...]}`.
    """
    det_path = cache.exec(dataset, 'naive', video, 'detection.jsonl')
    assert det_path.exists(), (
        f"Naive detection file not found: {det_path}. "
        "Run p002_preprocess.../the naive-detection stage for this video first.")
    out: dict[int, list[list[float]]] = {}
    with open(det_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            out[entry['frame_idx']] = entry.get('detections', []) or []
    return out


def _find_polyomino_for_detection(
    bbox: tuple[float, float, float, float],
    tile_to_polyomino_id_frame: np.ndarray,
    tile_size: int,
) -> int | None:
    """
    Return the polyomino_id that covers a detection's tiles, or None if the detection
    lands on background tiles (no surrounding relevance).
    """
    x1, y1, x2, y2 = bbox
    grid_h, grid_w = tile_to_polyomino_id_frame.shape
    row_start, row_end, col_start, col_end = get_overlapping_tiles(
        x1, y1, x2, y2, tile_size, grid_h, grid_w)
    if row_end < row_start or col_end < col_start:
        return None
    tile_ids = tile_to_polyomino_id_frame[row_start:row_end + 1, col_start:col_end + 1]
    valid = tile_ids[tile_ids >= 0]
    if valid.size == 0:
        return None
    # A detection's tile rectangle is connected, so all overlapped tiles should belong to
    # the same polyomino. Use the most frequent id as a robustness measure for edge ties.
    return int(np.bincount(valid).argmax())


def _compute_polyomino_outline(mask: np.ndarray, tile_size: int) -> list[list[int]]:
    """
    Compute a tile-aligned outline of a polyomino as a sequence of edge segments.

    Returns a flat list of [x1, y1, x2, y2] integer pixel coordinates (relative to the
    polyomino bounding-box origin). Each entry is one boundary edge — the consumer
    (D3) draws them as individual line segments, which is robust to disjoint polyominoes
    and simpler than constructing a single closed polyline.
    """
    h, w = mask.shape
    edges: list[list[int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0:
                continue
            # Top edge: emit when the tile above is empty or out of bounds.
            if r == 0 or mask[r - 1, c] == 0:
                x1 = c * tile_size
                y1 = r * tile_size
                edges.append([x1, y1, x1 + tile_size, y1])
            # Bottom edge.
            if r == h - 1 or mask[r + 1, c] == 0:
                x1 = c * tile_size
                y1 = (r + 1) * tile_size
                edges.append([x1, y1, x1 + tile_size, y1])
            # Left edge.
            if c == 0 or mask[r, c - 1] == 0:
                x1 = c * tile_size
                y1 = r * tile_size
                edges.append([x1, y1, x1, y1 + tile_size])
            # Right edge.
            if c == w - 1 or mask[r, c + 1] == 0:
                x1 = (c + 1) * tile_size
                y1 = r * tile_size
                edges.append([x1, y1, x1, y1 + tile_size])
    return edges


def _build_polyominoes(
    frames_rgb: list[np.ndarray],
    bitmaps: np.ndarray,
    tile_to_polyomino_id: np.ndarray,
    polyomino_lengths: list[list[int]],
    frame_indices: list[int],
    tile_size: int,
    out_dir: Path,
    image_scale: float,
) -> list[dict]:
    """
    Cut polyomino PNGs (RGBA with alpha mask), save to disk, return JSON-serializable records.

    Side effect: writes data/polyominoes/{f}_{i}.png for every polyomino in every frame.
    """
    records: list[dict] = []
    num_frames = bitmaps.shape[0]
    grid_h = bitmaps.shape[1]
    grid_w = bitmaps.shape[2]

    for array_idx in range(num_frames):
        frame_idx = frame_indices[array_idx]
        frame_rgb = frames_rgb[array_idx]
        per_frame_ids = tile_to_polyomino_id[array_idx]
        num_polyominoes = len(polyomino_lengths[array_idx])

        for poly_i in range(num_polyominoes):
            # Find the tiles belonging to this polyomino in this frame.
            mask_tiles = (per_frame_ids == poly_i)
            ys, xs = np.where(mask_tiles)
            if ys.size == 0:
                # Defensive: an entry in polyomino_lengths should always map to at least one tile.
                continue
            ty_min, ty_max = int(ys.min()), int(ys.max())
            tx_min, tx_max = int(xs.min()), int(xs.max())
            th = ty_max - ty_min + 1
            tw = tx_max - tx_min + 1

            # Local tile mask within the bounding box.
            local_mask = mask_tiles[ty_min:ty_max + 1, tx_min:tx_max + 1].astype(np.uint8)

            # Pixel coordinates in the source frame.
            py = ty_min * tile_size
            px = tx_min * tile_size
            ph = th * tile_size
            pw = tw * tile_size

            # Cut the bounding-box region from the frame and build the alpha channel.
            crop = frame_rgb[py:py + ph, px:px + pw]
            alpha = np.kron(local_mask, np.ones((tile_size, tile_size), dtype=np.uint8)) * 255
            assert alpha.shape == crop.shape[:2], f"alpha {alpha.shape} != crop {crop.shape[:2]}"

            rgba = np.dstack([crop, alpha])
            img = Image.fromarray(rgba, mode='RGBA')
            if image_scale != 1.0:
                # Use NEAREST on the alpha channel to keep tile-aligned hard edges crisp,
                # but LANCZOS on RGB for natural-looking color downsampling.
                save_w = max(1, int(round(pw * image_scale)))
                save_h = max(1, int(round(ph * image_scale)))
                rgb_part = img.convert('RGB').resize((save_w, save_h), Image.LANCZOS)
                alpha_part = img.getchannel('A').resize((save_w, save_h), Image.NEAREST)
                img = Image.merge('RGBA', (*rgb_part.split(), alpha_part))
            img.save(str(out_dir / f'{frame_idx}_{poly_i}.png'))

            outline = _compute_polyomino_outline(local_mask, tile_size)

            records.append({
                'f': frame_idx,
                'i': poly_i,
                'y': py,
                'x': px,
                'height': ph,
                'width': pw,
                'tile_y': ty_min,
                'tile_x': tx_min,
                'tile_h': th,
                'tile_w': tw,
                'outline': outline,
                'image': f'polyominoes/{frame_idx}_{poly_i}.png',
            })

    return records


def _load_or_build_accuracy(dataset: str, tracker: str, tile_size: int,
                            target_grid_h: int, target_grid_w: int,
                            scratch_dir: Path) -> np.ndarray:
    """
    Load training-set-aggregated accuracy table from cache if present.

    Otherwise fall back to running p016's aggregation, writing all intermediate
    artifacts into `scratch_dir` so we never touch the standard cache.
    """
    standard_dir = cache.index(dataset, 'track_rates', f'{tracker}_{tile_size}')
    accuracy_path = standard_dir / 'accuracy.npy'

    if accuracy_path.exists():
        accuracy = np.load(str(accuracy_path))
        assert accuracy.ndim == 3, f"accuracy.npy must be 3D, got shape {accuracy.shape}"
        assert accuracy.shape[1:] == (target_grid_h, target_grid_w), (
            f"accuracy.npy grid {accuracy.shape[1:]} does not match expected ({target_grid_h}, {target_grid_w}). "
            "Re-run p016 for this dataset/tracker/tile-size or change parameters.")
        return accuracy

    print(f"[max_rate_table] accuracy.npy not at {accuracy_path}; building from train set...",
          flush=True)
    accuracy = _build_accuracy_from_train_set(dataset, tracker, tile_size,
                                              target_grid_h, target_grid_w,
                                              scratch_dir)
    return accuracy


def _build_accuracy_from_train_set(dataset: str, tracker: str, tile_size: int,
                                   target_grid_h: int, target_grid_w: int,
                                   scratch_dir: Path) -> np.ndarray:
    """
    Mirror scripts/p016_tune_track_rate.py:415-476 for one (dataset, tracker, tile_size).
    Writes per-video partial counts into `scratch_dir/partial` and saves the final accuracy
    + counts arrays into `scratch_dir`. Never touches the standard cache location.
    """
    import multiprocessing as mp

    # Lazy import keeps the module importable in environments without these heavy deps.
    from polyis.utilities import ProgressBar
    from scripts.p016_tune_track_rate import process_video_tracker
    from functools import partial as fpartial

    mp.set_start_method('spawn', force=True)

    det_dir = cache.index(dataset, 'det')
    assert det_dir.exists(), (
        f"Detection directory {det_dir} does not exist. Run p011 first to compute detections "
        "for the training set of this dataset.")
    videos = sorted(
        f.stem.replace('.detections', '')
        for f in det_dir.iterdir()
        if f.name.endswith('.detections.jsonl')
    )
    assert len(videos) > 0, f"No detection files found in {det_dir}"
    print(f"[max_rate_table] Aggregating across {len(videos)} train videos.", flush=True)

    partial_dir = scratch_dir / 'partial'
    partial_dir.mkdir(parents=True, exist_ok=True)
    iou_threshold = 0.3

    funcs = [
        fpartial(process_video_tracker, dataset, video, tracker, tile_size,
                 iou_threshold, str(partial_dir))
        for video in videos
    ]
    ProgressBar(num_workers=min(40, len(funcs)), num_tasks=len(funcs)).run_all(funcs)

    total_counts = None
    for video in videos:
        counts_path = partial_dir / f'{video}.npy'
        assert counts_path.exists(), f"Partial counts not found: {counts_path}"
        video_counts = np.load(str(counts_path))
        total_counts = video_counts.copy() if total_counts is None else total_counts + video_counts
    assert total_counts is not None

    correct = total_counts[:, :, :, 0].astype(np.float32)
    incorrect = total_counts[:, :, :, 1].astype(np.float32)
    accuracy = (correct + 1) / (correct + incorrect + 2)
    assert accuracy.shape[1:] == (target_grid_h, target_grid_w), (
        f"Aggregated accuracy grid {accuracy.shape[1:]} does not match expected "
        f"({target_grid_h}, {target_grid_w}).")

    np.save(str(scratch_dir / 'accuracy.npy'), accuracy)
    np.save(str(scratch_dir / 'counts.npy'), total_counts)
    return accuracy


def _build_max_rate_table(accuracy: np.ndarray, m_values: list[float]) -> np.ndarray:
    """
    Mirror p016 lines 484-495 but with our custom thresholds.

    accuracy shape: (num_sample_rates, grid_h, grid_w).
    Returns max_rate_table shape: (grid_h, grid_w, len(m_values)) dtype int32.
    Entry [y, x, m_idx] is the highest sample rate whose accuracy at tile (y, x) meets 1 - M.
    """
    num_rates, grid_h, grid_w = accuracy.shape
    num_m = len(m_values)
    table = np.full((grid_h, grid_w, num_m), INDEX_SAMPLE_RATES[0], dtype=np.int32)
    for m_idx, m_value in enumerate(m_values):
        threshold = 1.0 - m_value
        for rate_idx in range(num_rates):
            meets = accuracy[rate_idx] >= threshold
            table[:, :, m_idx] = np.where(meets, INDEX_SAMPLE_RATES[rate_idx], table[:, :, m_idx])
    return table


def _filter_bitmap_to_selected(bitmap_frame: np.ndarray,
                               per_frame_ids: np.ndarray,
                               selected_ids: set[int]) -> np.ndarray:
    """Zero-out tiles whose polyomino is not in `selected_ids`. Returns a new uint8 bitmap."""
    out = np.zeros_like(bitmap_frame, dtype=np.uint8)
    if not selected_ids:
        return out
    keep_mask = np.isin(per_frame_ids, list(selected_ids)) & (per_frame_ids >= 0)
    out[keep_mask] = 1
    return out


def _pack_for_m(
    bitmaps: np.ndarray,
    tile_to_polyomino_id: np.ndarray,
    polyomino_lengths: list[list[int]],
    selected_per_frame: list[set[int]],
    poly_origin_to_index: dict[tuple[int, int, int], int],
    frame_indices: list[int],
    tilepadding_mode: int,
    dst_grid_h: int,
    dst_grid_w: int,
    tile_size: int,
    packing_mode_int: int,
) -> list[dict]:
    """
    Pack only the selected polyominoes per frame and return JSON-serializable canvas list.

    Mapping back to original polyomino index `i`: each polyomino's (frame, oy, ox) origin is
    unique within its frame, so we look it up in `poly_origin_to_index` to recover `i` after
    re-grouping the filtered bitmap.
    """
    num_frames = bitmaps.shape[0]
    polyominoes_stacks = np.empty(num_frames, dtype=np.uint64)
    for array_idx in range(num_frames):
        filtered = _filter_bitmap_to_selected(
            bitmaps[array_idx], tile_to_polyomino_id[array_idx], selected_per_frame[array_idx])
        polyominoes_stacks[array_idx] = group_tiles(filtered, tilepadding_mode)

    collages = pack(polyominoes_stacks, dst_grid_h, dst_grid_w, packing_mode_int)

    out: list[dict] = []
    for canvas_idx, collage in enumerate(collages):
        entries: list[dict] = []
        for pos in collage:
            # pos.frame is the array index passed into pack (0..num_frames-1).
            frame_idx = frame_indices[int(pos.frame)]
            origin_key = (int(pos.frame), int(pos.oy), int(pos.ox))
            original_i = poly_origin_to_index.get(origin_key)
            assert original_i is not None, (
                f"Could not map packed polyomino origin {origin_key} back to original index. "
                "This indicates a polyomino split during re-grouping which should not happen.")
            entries.append({
                'f': frame_idx,
                'i': original_i,
                'y': int(pos.py) * tile_size,
                'x': int(pos.px) * tile_size,
                'tile_y': int(pos.py),
                'tile_x': int(pos.px),
            })
        out.append({'canvas_idx': canvas_idx, 'polyominoes': entries})
    return out


def main():
    args = parse_args()

    m_values = [float(v.strip()) for v in args.m_values.split(',') if v.strip() != '']
    assert all(0.0 <= v <= 1.0 for v in m_values), f"All M values must be in [0, 1]: {m_values}"
    assert len(m_values) == len(set(m_values)), f"Duplicate M values in {m_values}"

    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    video = _resolve_video(args.dataset, args.videoset, args.video)
    video_path = store.dataset(args.dataset, args.videoset, video)
    print(f"[setup] Dataset={args.dataset} videoset={args.videoset} video={video}", flush=True)
    print(f"[setup] Video path={video_path}", flush=True)

    sampled_indices = [args.frame_start + i * args.sample_rate for i in range(args.num_frames)]
    print(f"[frames] Reading frame indices {sampled_indices[0]}..{sampled_indices[-1]}", flush=True)
    frames_bgr, src_h, src_w = _extract_frames(video_path, sampled_indices)

    tile_size = args.tile_size
    target_h = (src_h // tile_size) * tile_size
    target_w = (src_w // tile_size) * tile_size
    grid_h = target_h // tile_size
    grid_w = target_w // tile_size
    print(f"[frames] src=({src_h},{src_w}) target=({target_h},{target_w}) grid=({grid_h},{grid_w})",
          flush=True)

    # Save tile-aligned frame PNGs (downscaled per --image-scale for faster web loads).
    _save_frame_pngs(frames_bgr, target_h, target_w, sampled_indices,
                     output_dir / 'frames', args.image_scale)

    # Build per-frame relevance bitmaps from groundtruth tracks (same source as p021).
    frame_tracks_all = load_tracking_results(args.dataset, video)
    scale_x = target_w / src_w
    scale_y = target_h / src_h

    bitmaps_list: list[np.ndarray] = []
    for frame_idx in sampled_indices:
        dets = frame_tracks_all.get(frame_idx, [])
        if dets and (scale_x != 1.0 or scale_y != 1.0):
            dets = _resize_detections(dets, scale_x, scale_y)
        bitmap = mark_detections(dets, target_w, target_h, tile_size)
        bitmaps_list.append(bitmap)
    bitmaps = np.stack(bitmaps_list, axis=0).astype(np.uint8)
    assert bitmaps.shape == (args.num_frames, grid_h, grid_w)

    # Group tiles into polyominoes.
    tilepadding_mode = TILEPADDING_MAPS[args.tilepadding]  # type: ignore[index]
    tile_to_polyomino_id_view, polyomino_lengths = group_tiles_all(bitmaps, tilepadding_mode)
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id_view)
    total_polyominoes = sum(len(p) for p in polyomino_lengths)
    print(f"[polyominoes] total={total_polyominoes} across {bitmaps.shape[0]} frames", flush=True)

    # Resize the in-memory RGB frames for use when cutting polyomino crops.
    frames_rgb: list[np.ndarray] = []
    for frame in frames_bgr:
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    polyomino_records = _build_polyominoes(
        frames_rgb, bitmaps, tile_to_polyomino_id, polyomino_lengths,
        sampled_indices, tile_size, output_dir / 'polyominoes', args.image_scale)

    # Build a (array_idx, tile_y, tile_x) -> polyomino index lookup for pack-time mapping.
    poly_origin_to_index: dict[tuple[int, int, int], int] = {}
    for rec in polyomino_records:
        array_idx = sampled_indices.index(rec['f'])
        poly_origin_to_index[(array_idx, rec['tile_y'], rec['tile_x'])] = rec['i']

    # Write polyominoes.json — strip 'tile_*' fields not needed by the viz; keep outline.
    polyominoes_json = [
        {'f': r['f'], 'i': r['i'], 'y': r['y'], 'x': r['x'],
         'height': r['height'], 'width': r['width'],
         'outline': r['outline'], 'image': r['image']}
        for r in polyomino_records
    ]
    with open(output_dir / 'polyominoes.json', 'w') as f:
        json.dump({'polyominoes': polyominoes_json}, f)

    # Build (frame_idx, polyomino_i) -> source pixel origin of the polyomino's bounding
    # box. Used both for detection-on-canvas math below and shared by the unpack step.
    poly_source_origin: dict[tuple[int, int], tuple[int, int]] = {}
    for rec in polyomino_records:
        poly_source_origin[(rec['f'], rec['i'])] = (rec['y'], rec['x'])

    # Load detector outputs and pair each detection with its containing polyomino.
    # Detections whose tiles land on background (no polyomino covers them) are skipped:
    # they would never reach the canvas detector in the real pipeline, so the viz omits them.
    naive_detections = _load_naive_detections(args.dataset, video)
    print(f"[detections] Loaded naive detections for {len(naive_detections)} frames", flush=True)
    detections_json: list[dict] = []
    for array_idx, frame_idx in enumerate(sampled_indices):
        raw_dets = naive_detections.get(frame_idx, [])
        if not raw_dets:
            continue
        # Naive detection format from 002_naive/detection.jsonl: [x1, y1, x2, y2, score].
        # (Unlike tracking results, which are [track_id, x1, y1, x2, y2] — bbox at end.)
        for det_i, det in enumerate(raw_dets):
            x1 = float(det[0]) * scale_x
            y1 = float(det[1]) * scale_y
            x2 = float(det[2]) * scale_x
            y2 = float(det[3]) * scale_y
            poly_i = _find_polyomino_for_detection(
                (x1, y1, x2, y2), tile_to_polyomino_id[array_idx], tile_size)
            if poly_i is None:
                continue  # detection isn't covered by any polyomino — pipeline would drop it
            entry: dict = {
                'f': frame_idx,
                'id': det_i,
                'bbox': [x1, y1, x2, y2],
                'polyomino': {'f': frame_idx, 'i': poly_i},
            }
            if len(det) >= 5:
                entry['score'] = float(det[4])
            detections_json.append(entry)
    print(f"[detections] {len(detections_json)} detections inside polyominoes", flush=True)

    # Build max_rate_table for our 11 M values.
    scratch_dir = output_dir / 'raw_indexing'
    scratch_dir.mkdir(exist_ok=True)
    accuracy = _load_or_build_accuracy(args.dataset, args.tracker, tile_size,
                                       grid_h, grid_w, scratch_dir)
    max_rate_table = _build_max_rate_table(accuracy, m_values)
    np.save(str(output_dir / 'max_rate_table_viz.npy'), max_rate_table)
    print(f"[max_rate_table] shape={max_rate_table.shape}", flush=True)

    # Per-M pruning + packing.
    pruning_out: dict[str, list[list[int]]] = {}
    packing_out: dict[str, list[dict]] = {}
    canvas_counts: dict[str, int] = {}
    # detection_canvas_bboxes[M] is parallel-indexed to detections_json; each entry is either
    # {'canvas_idx': int, 'bbox': [x1, y1, x2, y2]} (canvas-local pixel coords) or None when
    # the covering polyomino was pruned at this M.
    detection_canvas_bboxes: dict[str, list[dict | None]] = {}
    packing_mode_int = PACK_MODE_LOOKUP[args.packing_mode]
    dst_grid_h = max(1, int(round(grid_h * args.canvas_scale)))
    dst_grid_w = max(1, int(round(grid_w * args.canvas_scale)))
    canvas_h = dst_grid_h * tile_size
    canvas_w = dst_grid_w * tile_size

    for m_idx, m_value in enumerate(m_values):
        m_key = f"{m_value:.2f}"
        # Build per-frame max sampling distance for this M.
        max_dist = (max_rate_table[:, :, m_idx] // args.sample_rate).astype(np.float64)
        max_dist = np.maximum(max_dist, 1.0)

        print(f"[M={m_key}] Solving ILP (time_limit={args.time_limit}s)...", flush=True)
        ilp_result = solve_ilp(
            tile_to_polyomino_id, polyomino_lengths, max_dist,
            grid_h, grid_w,
            time_limit_seconds=args.time_limit,
        )
        selected: set[tuple[int, int]] = set(ilp_result.selected)

        # Build per-frame selected set and discarded list.
        selected_per_frame: list[set[int]] = [set() for _ in range(bitmaps.shape[0])]
        for (frame_array_idx, poly_id) in selected:
            selected_per_frame[frame_array_idx].add(int(poly_id))

        discarded: list[list[int]] = []
        for array_idx, frame_idx in enumerate(sampled_indices):
            num_polys = len(polyomino_lengths[array_idx])
            for poly_i in range(num_polys):
                if poly_i not in selected_per_frame[array_idx]:
                    discarded.append([frame_idx, poly_i])
        pruning_out[m_key] = discarded
        print(f"[M={m_key}] discarded={len(discarded)} / {total_polyominoes}", flush=True)

        # Pack the surviving polyominoes.
        canvases = _pack_for_m(
            bitmaps, tile_to_polyomino_id, polyomino_lengths,
            selected_per_frame, poly_origin_to_index,
            sampled_indices, tilepadding_mode,
            dst_grid_h, dst_grid_w, tile_size, packing_mode_int)
        packing_out[m_key] = canvases
        canvas_counts[m_key] = len(canvases)
        print(f"[M={m_key}] canvases={len(canvases)}", flush=True)

        # Build a (f, i) -> (canvas_idx, poly_canvas_y, poly_canvas_x) lookup for this M.
        poly_canvas_lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
        for canvas in canvases:
            canvas_idx = canvas['canvas_idx']
            for p in canvas['polyominoes']:
                poly_canvas_lookup[(p['f'], p['i'])] = (canvas_idx, p['y'], p['x'])

        # Map each detection to its canvas position (or None when its polyomino was pruned).
        per_m: list[dict | None] = []
        for det in detections_json:
            poly_key = (det['polyomino']['f'], det['polyomino']['i'])
            cinfo = poly_canvas_lookup.get(poly_key)
            if cinfo is None:
                per_m.append(None)
                continue
            canvas_idx, poly_canvas_y, poly_canvas_x = cinfo
            poly_src_y, poly_src_x = poly_source_origin[poly_key]
            shift_x = poly_canvas_x - poly_src_x
            shift_y = poly_canvas_y - poly_src_y
            x1, y1, x2, y2 = det['bbox']
            per_m.append({
                'canvas_idx': canvas_idx,
                'bbox': [x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y],
            })
        detection_canvas_bboxes[m_key] = per_m
        kept = sum(1 for v in per_m if v is not None)
        print(f"[M={m_key}] detections kept={kept} / {len(detections_json)}", flush=True)

    with open(output_dir / 'pruning.json', 'w') as f:
        json.dump(pruning_out, f)
    with open(output_dir / 'packing.json', 'w') as f:
        json.dump(packing_out, f)
    with open(output_dir / 'detections.json', 'w') as f:
        json.dump({
            'detections': detections_json,
            'canvas_bboxes': detection_canvas_bboxes,
        }, f)

    meta = {
        'dataset': args.dataset,
        'videoset': args.videoset,
        'video': video,
        'frame_indices': sampled_indices,
        'num_frames': args.num_frames,
        'frame_start': args.frame_start,
        'sample_rate': args.sample_rate,
        'tile_size': tile_size,
        'frame_dims': {'height': target_h, 'width': target_w},
        'grid_dims': {'height': grid_h, 'width': grid_w},
        'canvas_dims': {'height': canvas_h, 'width': canvas_w, 'grid_height': dst_grid_h,
                        'grid_width': dst_grid_w},
        'tracker': args.tracker,
        'tilepadding': args.tilepadding,
        'packing_mode': args.packing_mode,
        'canvas_scale': args.canvas_scale,
        'relevance_threshold': args.relevance_threshold,
        'time_limit_seconds': args.time_limit,
        'image_scale': args.image_scale,
        'm_values': m_values,
        'canvas_counts': canvas_counts,
        'total_polyominoes': total_polyominoes,
        'total_detections': len(detections_json),
        'detections_per_m_kept': {k: sum(1 for v in vs if v is not None)
                                  for k, vs in detection_canvas_bboxes.items()},
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'index_sample_rates': INDEX_SAMPLE_RATES,
    }
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Wrote data to {output_dir}", flush=True)


if __name__ == '__main__':
    main()
