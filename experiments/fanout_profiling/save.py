"""
Savers: every non-pixel result, in the scripts' formats, at the cache's relative paths under output_root
(experiments/fanout_profiling/PLAN.md 2A, 18B).  Each runs in its stage's saver processes, after the worker's timing.

The formats, directory names, and writers are execution/save.py's (one copy, shared with execution --save); these
adapters only take the parameters from each message's Branch and the dataset from the sweep.

p040's per-canvas detections are not saved: detect returns per-frame detections only.
"""

import os

from execution import save as ex


def stage_dir(root: str, dataset: str, video: str, stage: str, param_str: str) -> str:
    """<root>/<dataset>/execution/<video name>/<stage directory>/<param_str>, created."""
    return ex.stage_dir(os.path.join(root, dataset), video, stage, param_str)


def save_scores(root: str, dataset: str, msg) -> None:
    path = stage_dir(root, dataset, msg.video, 'relevancy', msg.branch.param_str(*ex.SCORE_PARAMS))
    ex.write_score_rows(os.path.join(path, 'score'), msg.scores, msg.decoded.sampled)


def save_pruned(root: str, dataset: str, msg) -> None:
    # The unpruned branch has nothing new to save (its scores are classify's).
    if msg.branch['mistrack_tolerance'] is None:
        return
    path = stage_dir(root, dataset, msg.video, 'pruned-polyominoes', msg.branch.param_str(*ex.PRUNE_PARAMS))
    ex.write_score_rows(os.path.join(path, 'score'), msg.scores, msg.decoded.sampled)
    ex.write_ilp(path, msg.ilp)


def save_packed(root: str, dataset: str, msg) -> None:
    path = stage_dir(root, dataset, msg.video, 'comp-frames', ex.pack_param_str(msg.branch.as_dict()))
    ex.write_packed(path, msg.packed.index_maps, msg.packed.offset_lookups)


def save_detections(root: str, dataset: str, msg) -> None:
    ex.write_detections(stage_dir(root, dataset, msg.video, 'ucomp-dets', ex.pack_param_str(msg.branch.as_dict())),
                        msg.detections)


def save_tracks(root: str, dataset: str, msg) -> None:
    ex.write_tracks(stage_dir(root, dataset, msg.video, 'ucomp-tracks', msg.branch.param_str(*ex.TRACK_PARAMS)),
                    msg.frame_tracks)
