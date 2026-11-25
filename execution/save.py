"""
Optional saving of every stage's non-pixel results, in the scripts' formats at the cache's relative paths
(`execution.main --save <root>`; off by default).  Frames and canvases are never saved.

Under <root>/execution/<video name>/ (with root = <output root>/<dataset>, the result is cache-shaped):

  classify  020_relevancy/<clf>_<ts>_<s>/score/score.jsonl                       (p020's rows)
  prune     022_pruned_polyominoes/<clf>_<ts>_<s>_<tol>_<T_r>_<tracker>/score/score.jsonl   (p022's rows)
            and ilp.json next to score/ (not a script output: the solve's status, time, and kept tiles)
  pack      033_compressed_frames/<param>/{index_maps/*.npy, offset_lookups/*.jsonl}   (p030's names; no images)
  detect    050_uncompressed_detections/<param>/detections.jsonl               (p050's rows, every frame)
  track     060_uncompressed_tracks/<param>/tracking.jsonl                     (p060's format)

Savers run in the pipeline's saver processes, after each worker's process() returns, so they are outside the
stages' work but still share the machine with the timed run.  The fan-out sweep
(experiments/fanout_profiling/save.py) writes through the same functions.
"""

import json
import os
from functools import partial
from typing import Any, Callable

import numpy as np

from execution.pipeline import Stage
from tetris.io.cache import EXEC_STAGE_MAP
from tetris.utilities import build_param_str, save_tracking_results

# Parameter names, in pipeline order, and the build_param_str argument each maps to.
PARAM_KEYS = {
    'classifier': 'classifier', 'tile_size': 'tilesize', 'sample_rate': 'sample_rate',
    'relevance_threshold': 'relevance_threshold', 'mistrack_tolerance': 'tracking_accuracy_threshold',
    'tile_padding': 'tilepadding', 'canvas_scale': 'canvas_scale', 'tracker': 'tracker',
}

# Parameters each stage's directory name uses, as the scripts name them (classifier and tile size always lead).
SCORE_PARAMS = ('sample_rate',)
PRUNE_PARAMS = ('sample_rate', 'tracker', 'mistrack_tolerance', 'relevance_threshold')
PACK_PARAMS = ('sample_rate', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding', 'canvas_scale')
TRACK_PARAMS = PACK_PARAMS + ('tracker',)


def param_str(params: dict[str, Any], *names: str) -> str:
    """The scripts' directory name from `params`, using classifier, tile size, and `names`.

    A name whose value is None (the unpruned tolerance) is left out, as build_param_str leaves it out.
    """
    # Every requested name must be present, so a typo cannot silently drop a part of the name.
    missing = [n for n in ('classifier', 'tile_size', *names) if n not in params]
    if missing:
        raise KeyError(f'parameter set has no {missing}; it has {list(params)}')
    # Map each parameter to build_param_str's argument name.
    return build_param_str(**{PARAM_KEYS[n]: params[n] for n in ('classifier', 'tile_size', *names)})


def pack_param_str(params: dict[str, Any]) -> str:
    """p030/p050's name: the tracker appears only when pruned (the tracker is chosen at prune)."""
    # The tracker joins the name only on pruned configurations.
    names = PACK_PARAMS + (('tracker',) if params['mistrack_tolerance'] is not None else ())
    return param_str(params, *names)


def stage_dir(root: str, video: str, stage: str, name: str) -> str:
    """<root>/execution/<video name>/<stage directory>/<name>, created."""
    # The cache's layout: one directory per video, then per stage, then per parameter string.
    path = os.path.join(root, 'execution', os.path.basename(video), EXEC_STAGE_MAP[stage], name)
    os.makedirs(path, exist_ok=True)
    return path


def write_score_rows(path: str, scores: np.ndarray, sampled: np.ndarray) -> None:
    """<path>/score.jsonl as p020/p022 write it: one row per sampled frame."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'score.jsonl'), 'w') as f:
        # One row per sampled frame: the grid's shape, its bytes in hex, and the frame index.
        for grid, idx in zip(scores, sampled):
            f.write(json.dumps({'classification_size': list(grid.shape),
                                'classification_hex': grid.flatten().tobytes().hex(), 'idx': int(idx)}) + '\n')


def write_ilp(path: str, ilp: dict) -> None:
    """<path>/ilp.json: the solve's status, time, and kept tiles (read by the p022 mirror in equivalence tests)."""
    with open(os.path.join(path, 'ilp.json'), 'w') as f:
        json.dump(ilp, f)


def write_packed(path: str, index_maps: np.ndarray, offset_lookups: list) -> None:
    """<path>/index_maps/*.npy and <path>/offset_lookups/*.jsonl, named as p030 names each collage."""
    os.makedirs(os.path.join(path, 'index_maps'), exist_ok=True)
    os.makedirs(os.path.join(path, 'offset_lookups'), exist_ok=True)
    for k, (index_map, lookup) in enumerate(zip(index_maps, offset_lookups)):
        # p030 names each collage <collage>_<first frame>_<last frame>.
        frame_ids = [frame for _, _, frame in lookup]
        name = f'{k:04d}_{min(frame_ids):04d}_{max(frame_ids):04d}'
        np.save(os.path.join(path, 'index_maps', f'{name}.npy'), index_map)
        # One JSON line per polyomino placement.
        with open(os.path.join(path, 'offset_lookups', f'{name}.jsonl'), 'w') as f:
            for entry in lookup:
                f.write(json.dumps(entry) + '\n')


def write_detections(path: str, detections: dict[int, list[list[float]]]) -> None:
    """<path>/detections.jsonl as p050 writes it: one row per frame, in frame order."""
    with open(os.path.join(path, 'detections.jsonl'), 'w') as f:
        for frame_idx in sorted(detections):
            f.write(json.dumps({'frame_idx': frame_idx, 'bboxes': detections[frame_idx]}) + '\n')


def write_tracks(path: str, frame_tracks: dict[int, list[list[float]]]) -> None:
    """<path>/tracking.jsonl in p060's format."""
    save_tracking_results(frame_tracks, os.path.join(path, 'tracking.jsonl'))


# ---------------------------------------------------------------- execution's savers

def config_params(cfg) -> dict[str, Any]:
    """An execution config's parameters under the names the directory names use."""
    p = cfg.params
    return {'classifier': cfg.artifacts.classifier.name, 'tile_size': p.tile_size, 'sample_rate': p.sample_rate,
            'relevance_threshold': p.relevance_threshold, 'mistrack_tolerance': p.mistrack_tolerance,
            'tile_padding': p.tile_padding, 'canvas_scale': p.canvas_scale, 'tracker': p.tracker}


def save_classified(root: str, params: dict, skip: str | None, msg) -> None:
    """classify's scores (execution.stages.Classified)."""
    # The timing dummy's results are not the run's.
    if msg.decoded.video == skip:
        return
    path = stage_dir(root, msg.decoded.video, 'relevancy', param_str(params, *SCORE_PARAMS))
    write_score_rows(os.path.join(path, 'score'), msg.scores, msg.decoded.sampled)


def save_pruned(root: str, params: dict, skip: str | None, msg) -> None:
    """prune's 0/255 grids and the solve's record (execution.stages.Classified with ilp)."""
    if msg.decoded.video == skip:
        return
    path = stage_dir(root, msg.decoded.video, 'pruned-polyominoes', param_str(params, *PRUNE_PARAMS))
    write_score_rows(os.path.join(path, 'score'), msg.scores, msg.decoded.sampled)
    write_ilp(path, msg.ilp)


def save_packed(root: str, params: dict, skip: str | None, msg) -> None:
    """pack's index maps and offset lookups (execution.operators.Packed; the canvases are never read)."""
    if msg.video == skip:
        return
    write_packed(stage_dir(root, msg.video, 'comp-frames', pack_param_str(params)), msg.index_maps,
                 msg.offset_lookups)


def save_detected(root: str, params: dict, skip: str | None, msg) -> None:
    """detect's per-frame detections in original coordinates (execution.stages.Detected)."""
    if msg.video == skip:
        return
    write_detections(stage_dir(root, msg.video, 'ucomp-dets', pack_param_str(params)), msg.detections)


def save_tracked(root: str, params: dict, skip: str | None, msg) -> None:
    """track's tracks (execution.stages.Tracked)."""
    if msg.video == skip:
        return
    write_tracks(stage_dir(root, msg.video, 'ucomp-tracks', param_str(params, *TRACK_PARAMS)), msg.frame_tracks)


# The saver of each stage that has non-pixel results; decode's output is pixels only.
SAVERS: dict[str, Callable] = {'classify': save_classified, 'prune': save_pruned, 'pack': save_packed,
                               'detect': save_detected, 'track': save_tracked}


def with_savers(stages: list[Stage], cfg, root: str, skip: str | None = None,
                savers: dict[str, Callable] = SAVERS) -> list[Stage]:
    """The same stages, each with a saver (one saver process) where `savers` has one for its name.

    skip names a video whose results are not saved (execution.main's timing dummy).
    """
    params = config_params(cfg)
    # partial of a module-level function, so the saver pickles into its spawned process.
    return [s._replace(save=partial(savers[s.name], root, params, skip), savers=1) if s.name in savers else s
            for s in stages]
