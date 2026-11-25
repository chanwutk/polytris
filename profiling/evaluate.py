#!/usr/local/bin/python
"""
Score a profiling sweep's tracks against each video's reference tracks with p110's settings, after the timed runs
(profiling/README.md "Steps").

    python -m profiling.evaluate --config <sweep yaml> [--procs N] [--force]

For every configuration that ran (runs/<id>/result.json without duplicate_of) and has no evaluation.json yet (all of
them with --force), TrackEval scores its videos in one call and writes runs/<id>/evaluation.json: video file ->
{'HOTA': ..., 'Count': ...}, TrackEval's raw per-sequence results.  Duplicates are scored through the configuration
that ran for them.  Configurations are scored in parallel, and never while a timed run is going, so evaluation
cannot affect runtime_s.  A configuration that fails gets runs/<id>/evaluation-error.txt; the others continue and
the exit code is 1.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback

import numpy as np

from execution.config import ConfigError
from profiling.config import SweepConfig, load_sweep_config


def parse_args():
    parser = argparse.ArgumentParser(description='Score a profiling sweep\'s tracks against the reference tracks.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def to_json(value):
    """TrackEval results hold numpy arrays and scalars; JSON gets lists and Python numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def missing_references(reference_dir: str, reference_file: str, video_files: list[str]) -> list[str]:
    """The reference track files that do not exist, one per video lacking its reference."""
    paths = [os.path.join(reference_dir, os.path.basename(v), reference_file) for v in video_files]
    return [p for p in paths if not os.path.exists(p)]


def trackeval_sequences(track_dir: str, seqs: list[str], reference_dir: str, reference_file: str,
                        scratch: str) -> dict[str, dict]:
    """TrackEval with p110's settings on <track_dir>/<seq>/tracking.jsonl for each seq; seq -> {metric: result}.

    Metrics are p110's default (HOTA) plus Count, as p110 stores them.  The fan-out sweep's evaluate stage calls
    this too, so both score tracks identically.
    """
    # Imported here: TrackEval's path setup and the repo config load only where scoring happens.
    from tetris.trackeval.evaluate import build_metric_objects, run_trackeval
    from trackeval.metrics import Count
    metrics = build_metric_objects(['HOTA'])
    count = Count()
    dataset_config = {
        'output_fol': os.path.join(scratch, 'trackeval'), 'output_sub_fol': 'profiling',
        'input_gt': reference_file, 'input_track': 'tracking.jsonl', 'skip': 1, 'tracker': 'profiling',
        'seq_list': list(seqs), 'input_dir': track_dir, 'input_gt_dir': reference_dir, 'input_track_dir': track_dir,
    }
    # TrackEval always runs Count itself, so only HOTA is passed (as p110 and the fan-out sweep did).
    results = run_trackeval(dataset_config, metrics, dataset_config['output_fol'], len(seqs))
    # Each sequence's vehicle-class result per metric.
    return {seq: {m.get_name(): results[seq]['vehicle'][m.get_name()] for m in [*metrics, count]} for seq in seqs}


def evaluate_run(run_dir: str, reference_dir: str, reference_file: str) -> dict[str, dict]:
    """Score one configuration's tracks (runs/<id>/tracks/<stem>.jsonl) for every video it ran."""
    result = json.load(open(os.path.join(run_dir, 'result.json')))
    with tempfile.TemporaryDirectory(prefix='profiling-evaluate-') as scratch:
        # TrackEval reads <dir>/<video file>/tracking.jsonl; link each track file into that layout.
        track_dir = os.path.join(scratch, 'tracks')
        for video in result['video_files']:
            os.makedirs(os.path.join(track_dir, video))
            stem = os.path.splitext(video)[0]
            os.symlink(os.path.abspath(os.path.join(run_dir, 'tracks', f'{stem}.jsonl')),
                       os.path.join(track_dir, video, 'tracking.jsonl'))
        return to_json(trackeval_sequences(track_dir, result['video_files'], reference_dir, reference_file, scratch))


def _write_json_atomically(path: str, value) -> None:
    """Write to a temporary file and rename, so a reader never sees half a file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(value, f)
    os.replace(tmp, path)


def _evaluate_one(args: tuple[str, str, str]) -> tuple[str, str | None]:
    """Pool task: score one run directory; returns (run_dir, error text or None)."""
    run_dir, reference_dir, reference_file = args
    try:
        evaluation = evaluate_run(run_dir, reference_dir, reference_file)
        _write_json_atomically(os.path.join(run_dir, 'evaluation.json'), evaluation)
        return run_dir, None
    except Exception:
        error = traceback.format_exc()
        with open(os.path.join(run_dir, 'evaluation-error.txt'), 'w') as f:
            f.write(error)
        return run_dir, error


def pending_runs(cfg: SweepConfig, force: bool = False) -> list[str]:
    """Run directories that ran (not duplicates) and still need scoring."""
    runs = os.path.join(cfg.profiling_dir, 'runs')
    if not os.path.isdir(runs):
        return []
    out = []
    for name in sorted(os.listdir(runs)):
        run_dir = os.path.join(runs, name)
        result_path = os.path.join(run_dir, 'result.json')
        if not os.path.exists(result_path) or json.load(open(result_path)).get('duplicate_of') is not None:
            continue
        if force or not os.path.exists(os.path.join(run_dir, 'evaluation.json')):
            out.append(run_dir)
    return out


def evaluate_sweep(cfg: SweepConfig, procs: int = 1, force: bool = False) -> dict:
    """Score every pending run; returns {'evaluated': n, 'failed': [run ids]}."""
    runs = pending_runs(cfg, force)
    tasks = [(r, cfg.evaluation.reference_dir, cfg.evaluation.reference_file) for r in runs]
    # Earlier failures are retried; their old error files would otherwise linger beside a new evaluation.json.
    for r in runs:
        if os.path.exists(os.path.join(r, 'evaluation-error.txt')):
            os.remove(os.path.join(r, 'evaluation-error.txt'))
    if procs <= 1 or len(tasks) <= 1:
        outcomes = [_evaluate_one(t) for t in tasks]
    else:
        with mp.get_context('spawn').Pool(procs) as pool:
            outcomes = pool.map(_evaluate_one, tasks)
    failed = [os.path.basename(r) for r, error in outcomes if error is not None]
    return {'evaluated': len(tasks) - len(failed), 'failed': failed}


def main() -> int:
    args = parse_args()
    try:
        cfg = load_sweep_config(args.config)
    except ConfigError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    summary = evaluate_sweep(cfg, args.procs, args.force)
    print(json.dumps(summary))
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
