"""
The profiling sweep's stages (profiling/PLAN.md "Stages").  Each runs its operator (execution/operators.py)
for every value of its own parameters, given the branch it receives, and emits one message per new branch:

  decode    once per video, every frame                                             1 per video
  classify  once per video on every frame; one output per sample rate (a slice)     x len(sample_rates)
  prune     per relevance threshold x (each tolerance x tracker, or no pruning)     x len(T_r) * (tols*trackers + 1)
  pack      per tile padding x canvas scale; frees the video's frames after its last input
  detect    1:1; frees the canvases
  track     per tracker on unpruned branches (pruned branches fixed theirs at prune)
  evaluate  1:1; TrackEval on one video against its reference tracks, p110's settings

Timing: every stage records its process start and end (PLAN 1).  Charged seconds: classify charges each
sample rate only for its frames (1A); decode and evaluate charge 0 (decode's time is kept, its charging is
undecided; evaluation is not system work); a no-pruning branch's prune record is empty.
"""

import os
import tempfile
import time
from functools import partial

import numpy as np

from execution import frames, operators as ops
from execution.config import MISTRACK_TOLERANCES
from execution.pipeline import Stage, Worker
from profiling import save
from profiling.reduction import sweep_groups
from profiling.completion import VideoCompletion
from profiling.config import SweepConfig
from profiling.messages import Branch, Decoded, Detected, Evaluated, Packed, Scored, StageTiming, Tracked
from tetris.utilities import TILEPADDING_MAPS, is_tile_padding, save_tracking_results


def expected_pack_inputs(cfg: SweepConfig, groups: dict | None = None) -> int:
    """Pack inputs per video: per sample rate, relevance thresholds x (one per tolerance group per tracker,
    plus unpruned).  Without reduction every tolerance is its own group."""
    p = cfg.params
    groups = sweep_groups(cfg) if groups is None else groups
    unpruned = 1 if None in p.mistrack_tolerances else 0
    return sum(len(p.relevance_thresholds) * (unpruned + sum(len(groups.get((tr, s), [])) for tr in p.trackers))
               for s in p.sample_rates)


def expected_results(cfg: SweepConfig, groups: dict | None = None) -> int:
    """Results per video that run: pruned groups x paddings x scales, plus unpruned branches x trackers."""
    p = cfg.params
    groups = sweep_groups(cfg) if groups is None else groups
    per_pack = len(p.tile_paddings) * len(p.canvas_scales)
    unpruned = (len(p.trackers) * per_pack) if None in p.mistrack_tolerances else 0
    return sum(len(p.relevance_thresholds) * (unpruned + per_pack * sum(len(groups.get((tr, s), [])) for tr in p.trackers))
               for s in p.sample_rates)


def _timing(stage: str, start: float, end: float, charged: float | None = None) -> StageTiming:
    """A timing record; charged defaults to the elapsed time."""
    return StageTiming(stage, start, end, end - start if charged is None else charged)


# ---------------------------------------------------------------- decode

def decode_setup(cfg: SweepConfig, prefix: str, worker: Worker) -> dict:
    return {'prefix': prefix, 'branch': Branch().with_(classifier=cfg.params.classifier,
                                                       tile_size=cfg.params.tile_size)}


def decode(state: dict, video: str) -> Decoded:
    """Every frame of the video into one shared block (sample rate 1 needs all of them)."""
    start = time.time()
    decoded = ops.decode_video(video, state['prefix'], 1)
    # Decode's time is recorded on every branch; whether it counts toward runtime is decided later.
    return Decoded(video, state['branch'], (_timing('decode', start, time.time(), charged=0.0),), decoded)


# ---------------------------------------------------------------- classify

def classify_setup(cfg: SweepConfig, worker: Worker) -> dict:
    a = cfg.artifacts
    return {'model': ops.load_classifier_state(a.classifier_weights, a.compilation_benchmark,
                                               a.never_relevant_bitmap, cfg.params.tile_size),
            'sample_rates': cfg.params.sample_rates}


def classify(state: dict, d: Decoded) -> list[Scored]:
    """Classify every frame once; one output per sample rate with that rate's frames and charged share."""
    v = d.decoded
    if not np.array_equal(v.sampled, np.arange(v.frame_count)):
        raise ValueError(f'{d.video}: classify expects every frame decoded (sample rate 1)')
    start = time.time()
    try:
        scores, shares = ops.classify_frames(state['model'], v, time_batches=True)
    except ValueError as e:
        raise ValueError(f'{d.video}: {e}') from e
    end = time.time()
    outputs = []
    for s in state['sample_rates']:
        sampled, partner, _ = ops.frame_selection(v.frame_count, s)
        # The all-frames block keeps every frame, so this rate's frames are found in it by index.
        view = v._replace(sampled=sampled, partner=partner)
        # Each rate is charged for the frames it samples (sampled index == frame index at rate 1).
        timing = _timing('classify', start, end, charged=float(shares[sampled].sum()))
        outputs.append(Scored(d.video, d.branch.with_(sample_rate=s), d.timings + (timing,), view, scores[sampled]))
    return outputs


# ---------------------------------------------------------------- prune

def prune_setup(cfg: SweepConfig, worker: Worker) -> dict:
    p = cfg.params
    return {'tables': {t: np.load(path) for t, path in cfg.artifacts.max_rate_tables.items()},
            'relevance_thresholds': p.relevance_thresholds, 'tolerances': p.mistrack_tolerances,
            'trackers': p.trackers, 'distances': {}, 'time_limit_s': p.prune_time_limit_s,
            'groups': sweep_groups(cfg), 'solve': solve_ilp}


def _distance(state: dict, tracker: str, tolerance: float, sample_rate: int) -> np.ndarray:
    """p022's max sampling distance for (tracker, tolerance, sample rate), computed once per worker."""
    key = (tracker, tolerance, sample_rate)
    if key not in state['distances']:
        state['distances'][key] = ops.max_sampling_distance(state['tables'][tracker],
                                                            MISTRACK_TOLERANCES.index(tolerance), sample_rate)
    return state['distances'][key]


def solve_ilp(state: dict, sc: Scored, tracker: str, tolerance: float, relevance_threshold: float) -> tuple:
    """p022's ILP for one branch: the 0/255 pruned grids and the solve's status and time.

    prune calls it through state['solve'], which equivalence tests replace with a reader of the scripts' grids.
    """
    s = sc.branch['sample_rate']
    try:
        pruned, result = ops.prune_scores(sc.scores, _distance(state, tracker, tolerance, s), relevance_threshold,
                                          state['time_limit_s'])
    except ValueError as e:
        raise ValueError(f'{sc.video}: {e}') from e
    return pruned, {'status': result.status, 'solve_ms': result.solve_ms}


def prune(state: dict, sc: Scored) -> list[Scored]:
    """One output per relevance threshold x (tolerance group x tracker, or no pruning).

    A group's representative runs (its branch carries that tolerance); the aggregator expands each group's
    result to all of its tolerances.
    """
    outputs = []
    s = sc.branch['sample_rate']
    for t_r in state['relevance_thresholds']:
        if None in state['tolerances']:
            # No pruning: scores pass through; pack binarizes them with this T_r (p030's rule).
            now = time.time()
            outputs.append(Scored(sc.video, sc.branch.with_(relevance_threshold=t_r, mistrack_tolerance=None),
                                  sc.timings + (_timing('prune', now, now),), sc.decoded, sc.scores))
        for tracker in state['trackers']:
            for group in state['groups'].get((tracker, s), []):
                tolerance = group.representative
                start = time.time()
                pruned, ilp = state['solve'](state, sc, tracker, tolerance, t_r)
                timing = _timing('prune', start, time.time())
                branch = sc.branch.with_(relevance_threshold=t_r, mistrack_tolerance=tolerance, tracker=tracker)
                outputs.append(Scored(sc.video, branch, sc.timings + (timing,), sc.decoded, pruned,
                                      ilp={**ilp, 'kept_tiles': int(np.count_nonzero(pruned)),
                                           'group': list(group.tolerances)}))
    return outputs


# ---------------------------------------------------------------- pack

def pack_setup(cfg: SweepConfig, prefix: str, completion: VideoCompletion, worker: Worker) -> dict:
    p = cfg.params
    for padding in p.tile_paddings:
        if not is_tile_padding(padding):
            raise ValueError(f'unknown tile padding {padding!r}')
    return {'prefix': prefix, 'tile_size': p.tile_size, 'paddings': p.tile_paddings, 'scales': p.canvas_scales,
            'jpeg': p.canvas_format == 'jpeg', 'completion': completion, 'expected': expected_pack_inputs(cfg)}


def pack(state: dict, sc: Scored) -> list[Packed]:
    """One output per tile padding x canvas scale; frees the video's frames after its last pack input."""
    v = sc.decoded
    outputs = []
    t_r = sc.branch['relevance_threshold']
    for padding in state['paddings']:
        for scale in state['scales']:
            start = time.time()
            collages = ops.pack_collages(sc.scores, v, state['prefix'], state['tile_size'], TILEPADDING_MAPS[padding],
                                         scale, t_r, state['jpeg'])
            timing = _timing('pack', start, time.time())
            outputs.append(Packed(sc.video, sc.branch.with_(tile_padding=padding, canvas_scale=scale),
                                  sc.timings + (timing,), collages))
    # Pack is the frames' last user: free them once every pack input of this video is done.
    if state['completion'].finish(sc.video, state['expected']):
        frames.unlink(v.frames)
    return outputs


# ---------------------------------------------------------------- detect

def detect_setup(cfg: SweepConfig, worker: Worker) -> dict:
    det = cfg.artifacts.detector
    return {'detector': ops.load_detector(det.type, det.options), 'tile_size': cfg.params.tile_size}


def detect(state: dict, p: Packed) -> Detected:
    """Detect on the branch's canvases, map boxes back to frames, free the canvases."""
    start = time.time()
    detections = ops.detect_canvases(state['detector'], p.packed, state['tile_size'])
    timing = _timing('detect', start, time.time())
    # Detect is the canvases' last user.
    frames.unlink(p.packed.canvases)
    q = p.packed
    return Detected(p.video, p.branch, p.timings + (timing,), q.width, q.height, q.frame_count, detections)


# ---------------------------------------------------------------- track

def track_setup(cfg: SweepConfig, worker: Worker) -> dict:
    return {'trackers': cfg.params.trackers}


def track(state: dict, d: Detected) -> list[Tracked]:
    """Unpruned branches: one output per tracker.  Pruned branches: their prune tracker only."""
    pruned = d.branch['mistrack_tolerance'] is not None
    outputs = []
    for tracker in ([d.branch['tracker']] if pruned else state['trackers']):
        start = time.time()
        tracks = ops.track_detections(tracker, d.width, d.height, d.detections)
        timing = _timing('track', start, time.time())
        branch = d.branch if pruned else d.branch.with_(tracker=tracker)
        outputs.append(Tracked(d.video, branch, d.timings + (timing,), tracks))
    return outputs


# ---------------------------------------------------------------- evaluate

def evaluate_setup(cfg: SweepConfig, worker: Worker) -> dict:
    # p110's metric objects and evaluator (imported here: p110 loads TrackEval and the repo config).
    from evaluation.p110_accuracy_compute import build_metric_objects, run_trackeval
    from trackeval.metrics import Count
    scratch = tempfile.mkdtemp(prefix=f'profiling-evaluate-{worker.index}-')
    return {'scratch': scratch, 'reference_dir': cfg.evaluation.reference_dir,
            'reference_file': cfg.evaluation.reference_file, 'metrics': build_metric_objects(['HOTA']),
            'count': Count(), 'run_trackeval': run_trackeval}


def evaluate(state: dict, t: Tracked) -> Evaluated:
    """TrackEval on this video's tracks vs its reference tracks, with p110's settings; raw per-video result."""
    start = time.time()
    seq = os.path.basename(t.video)
    track_dir = os.path.join(state['scratch'], 'tracks')
    os.makedirs(os.path.join(track_dir, seq), exist_ok=True)
    save_tracking_results(t.frame_tracks, os.path.join(track_dir, seq, 'tracking.jsonl'))
    dataset_config = {
        'output_fol': os.path.join(state['scratch'], 'trackeval'), 'output_sub_fol': 'profiling',
        'input_gt': state['reference_file'], 'input_track': 'tracking.jsonl', 'skip': 1, 'tracker': 'profiling',
        'seq_list': [seq], 'input_dir': track_dir, 'input_gt_dir': state['reference_dir'],
        'input_track_dir': track_dir,
    }
    results = state['run_trackeval'](dataset_config, state['metrics'], dataset_config['output_fol'], 1)
    vehicle = results[seq]['vehicle']
    metrics = {m.get_name(): vehicle[m.get_name()] for m in [*state['metrics'], state['count']]}
    # Evaluation is measurement, not system work: recorded, not charged.
    return Evaluated(t.video, t.branch, t.timings + (_timing('evaluate', start, time.time(), charged=0.0),), metrics)


# ---------------------------------------------------------------- wiring

def build_stages(cfg: SweepConfig, prefix: str, completion: VideoCompletion) -> list[Stage]:
    """The sweep pipeline: process counts, GPUs, and savers from cfg.system; prune only when swept."""
    s = cfg.system.stages
    root, ds = cfg.output_root, cfg.dataset

    def stage(name, setup, process, fan_out=False, saver=None):
        sys_ = s[name]
        return Stage(name, setup, process, sys_.procs, sys_.gpus or None, fan_out,
                     partial(saver, root, ds) if saver else None, sys_.savers)

    stages = [stage('decode', partial(decode_setup, cfg, prefix), decode),
              stage('classify', partial(classify_setup, cfg), classify, fan_out=True, saver=save.save_scores)]
    stages.append(stage('prune', partial(prune_setup, cfg), prune, fan_out=True, saver=save.save_pruned)
                  if cfg.uses_prune else
                  Stage('prune', partial(prune_setup, cfg), prune, 1, None, True))
    stages += [stage('pack', partial(pack_setup, cfg, prefix, completion), pack, fan_out=True, saver=save.save_packed),
               stage('detect', partial(detect_setup, cfg), detect, saver=save.save_detections),
               stage('track', partial(track_setup, cfg), track, fan_out=True, saver=save.save_tracks),
               stage('evaluate', partial(evaluate_setup, cfg), evaluate)]
    return stages
