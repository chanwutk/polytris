"""
Savers: every non-pixel result, in the scripts' formats, at the cache's relative paths under output_root
(profiling/PLAN.md 2A, 18B).  Each runs in its stage's saver processes, after the worker's timing.

  classify  020_relevancy/<clf>_<ts>_<s>/score/score.jsonl                       (p020's rows)
  prune     022_pruned_polyominoes/<clf>_<ts>_<s>_<tol>_<T_r>_<tracker>/score/score.jsonl   (p022's rows)
  pack      033_compressed_frames/<param>/{index_maps/*.npy, offset_lookups/*.jsonl}   (p030's names; no images)
  detect    050_uncompressed_detections/<param>/detections.jsonl               (p050's rows, every frame)
  track     060_uncompressed_tracks/<param>/tracking.jsonl                     (p060's format)

p040's per-canvas detections are not saved: detect returns per-frame detections only.
"""

import json
import os

import numpy as np

from polyis.io.cache import EXEC_STAGE_MAP
from polyis.utilities import save_tracking_results

# Branch parameters each stage's directory name uses, as the scripts name them.
SCORE_PARAMS = ('sample_rate',)
PRUNE_PARAMS = ('sample_rate', 'tracker', 'mistrack_tolerance', 'relevance_threshold')
PACK_PARAMS = ('sample_rate', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding', 'canvas_scale')


def stage_dir(root: str, dataset: str, video: str, stage: str, param_str: str) -> str:
    """<root>/<dataset>/execution/<video name>/<stage directory>/<param_str>, created."""
    path = os.path.join(root, dataset, 'execution', os.path.basename(video), EXEC_STAGE_MAP[stage], param_str)
    os.makedirs(path, exist_ok=True)
    return path


def _pack_param_str(branch) -> str:
    """p030/p050's name: the tracker appears only on pruned branches (it is chosen at prune)."""
    names = PACK_PARAMS + (('tracker',) if branch['mistrack_tolerance'] is not None else ())
    return branch.param_str(*names)


def _write_score_rows(path: str, scores: np.ndarray, sampled: np.ndarray) -> None:
    """score.jsonl rows as p020/p022 write them: one per sampled frame."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'score.jsonl'), 'w') as f:
        for grid, idx in zip(scores, sampled):
            f.write(json.dumps({'classification_size': list(grid.shape),
                                'classification_hex': grid.flatten().tobytes().hex(), 'idx': int(idx)}) + '\n')


def save_scores(root: str, dataset: str, msg) -> None:
    path = stage_dir(root, dataset, msg.video, 'relevancy', msg.branch.param_str(*SCORE_PARAMS))
    _write_score_rows(os.path.join(path, 'score'), msg.scores, msg.frames.sampled)


def save_pruned(root: str, dataset: str, msg) -> None:
    # The unpruned branch has nothing new to save (its scores are classify's).
    if msg.branch['mistrack_tolerance'] is None:
        return
    path = stage_dir(root, dataset, msg.video, 'pruned-polyominoes', msg.branch.param_str(*PRUNE_PARAMS))
    _write_score_rows(os.path.join(path, 'score'), msg.scores, msg.frames.sampled)


def save_packed(root: str, dataset: str, msg) -> None:
    path = stage_dir(root, dataset, msg.video, 'comp-frames', _pack_param_str(msg.branch))
    os.makedirs(os.path.join(path, 'index_maps'), exist_ok=True)
    os.makedirs(os.path.join(path, 'offset_lookups'), exist_ok=True)
    for k, (index_map, lookup) in enumerate(zip(msg.collages.index_maps, msg.collages.offset_lookups)):
        # p030 names each collage <collage>_<first frame>_<last frame>.
        frame_ids = [frame for _, _, frame in lookup]
        name = f'{k:04d}_{min(frame_ids):04d}_{max(frame_ids):04d}'
        np.save(os.path.join(path, 'index_maps', f'{name}.npy'), index_map)
        with open(os.path.join(path, 'offset_lookups', f'{name}.jsonl'), 'w') as f:
            for entry in lookup:
                f.write(json.dumps(entry) + '\n')


def save_detections(root: str, dataset: str, msg) -> None:
    path = stage_dir(root, dataset, msg.video, 'ucomp-dets', _pack_param_str(msg.branch))
    with open(os.path.join(path, 'detections.jsonl'), 'w') as f:
        for frame_idx in sorted(msg.detections):
            f.write(json.dumps({'frame_idx': frame_idx, 'bboxes': msg.detections[frame_idx]}) + '\n')


def save_tracks(root: str, dataset: str, msg) -> None:
    names = PACK_PARAMS + ('tracker',)
    path = stage_dir(root, dataset, msg.video, 'ucomp-tracks', msg.branch.param_str(*names))
    save_tracking_results(msg.frame_tracks, os.path.join(path, 'tracking.jsonl'))
