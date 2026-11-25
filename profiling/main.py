#!/usr/local/bin/python
"""
Run a profiling sweep: every configuration of a dataset's split, one at a time, each as a whole execution run
(profiling/README.md).

    python -m profiling.main --config <sweep yaml> [--save]

For each configuration that runs (dedup_pruning leaves out duplicates), `python -m execution.main` runs over every
video of the split, so runtime_s is execution's own end-to-end measurement, timing dummy included.  Under
<output_root>/<dataset>/profiling/:

  sweep.json                 the sweep document and its hash; a different sweep is refused in the same output
  runs/<id>/config.yaml      the configuration's execution config
  runs/<id>/tracks/          execution's --out: <video stem>.jsonl per video and summary.json
  runs/<id>/stdout.log, stderr.log
  runs/<id>/result.json      written last: parameters, runtime_s, frames, video files, saved, duplicate_of
  runs/<id>/error.txt        instead of result.json when the run failed (exit code and the end of stderr)

A rerun skips configurations that have result.json, cleans and retries the rest, and copies each finished
configuration's result to its duplicates (duplicate_of names it; their runtime is the one measured for it).
--save passes --save <output_root>/<dataset> to execution (every stage's non-pixel outputs, cache-shaped); such
runs are marked saved, since saving shares the machine with the timed run.  Evaluation and Pareto selection are
separate steps (profiling.evaluate, profiling.aggregate).  Exit code 1 when any configuration failed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import yaml

from execution.config import ConfigError
from execution.main import StartupError, list_videos
from profiling.config import Configuration, SweepConfig, configurations, load_sweep_config, sweep_hash
from profiling.evaluate import missing_references
from profiling.reduction import duplicates

REPO = Path(__file__).resolve().parents[1]
# How much of a failed run's stderr goes into error.txt.
STDERR_TAIL_CHARS = 20000

# Runs one execution: (config path, videos dir, out dir, save root or None, log dir) -> exit code.
Runner = Callable[[str, str, str, str | None, str], int]


def parse_args():
    parser = argparse.ArgumentParser(description='Run a profiling sweep: every configuration as one execution run.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


def run_execution(config_path: str, videos_dir: str, out_dir: str, save_root: str | None, log_dir: str,
                  module: str = 'execution.main') -> int:
    """`python -m execution.main` in a fresh interpreter from the repo root; stdout and stderr go to log files.

    module replaces execution.main only in tests (a stand-in with the same command line).
    """
    command = [sys.executable, '-m', module, '--config', config_path, '--videos', videos_dir, '--out', out_dir]
    if save_root is not None:
        command += ['--save', save_root]
    with open(os.path.join(log_dir, 'stdout.log'), 'w') as out, open(os.path.join(log_dir, 'stderr.log'), 'w') as err:
        return subprocess.run(command, cwd=REPO, stdout=out, stderr=err).returncode


def videos_dir(cfg: SweepConfig) -> str:
    """The split's videos (the repo's dataset layout)."""
    # Imported here: tetris.io.store reads the repo config on import.
    from tetris.io import store
    return str(store.dataset(cfg.dataset, cfg.videoset))


def write_json_atomically(path: str, value) -> None:
    """Write to a temporary file and rename, so result.json exists only when complete."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def claim_output(cfg: SweepConfig) -> None:
    """Create the sweep's directory, or accept it when it holds the same sweep (a resume); refuse otherwise."""
    path = os.path.join(cfg.profiling_dir, 'sweep.json')
    digest = sweep_hash(cfg)
    if os.path.exists(path):
        recorded = json.load(open(path))['hash']
        if recorded != digest:
            raise StartupError(f'{cfg.profiling_dir} holds a different sweep (hash {recorded[:12]}, this one '
                               f'{digest[:12]}); choose another output_root so its results are kept')
        return
    os.makedirs(os.path.join(cfg.profiling_dir, 'runs'), exist_ok=True)
    write_json_atomically(path, {'hash': digest, 'sweep': cfg.raw})


def run_configuration(cfg: SweepConfig, c: Configuration, videos: str, video_files: list[str], save: bool,
                      runner: Runner) -> str | None:
    """Run one configuration into a clean runs/<id>/; returns None on success or the error text."""
    run_dir = os.path.join(cfg.profiling_dir, 'runs', c.id)
    # A directory without result.json is a crashed or failed attempt: start it over.
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(c.execution, f, sort_keys=False)
    tracks = os.path.join(run_dir, 'tracks')
    start = time.time()
    code = runner(config_path, videos, tracks, cfg.save_root if save else None, run_dir)
    wall = time.time() - start
    summary_path = os.path.join(tracks, 'summary.json')
    if code != 0 or not os.path.exists(summary_path):
        stderr_path = os.path.join(run_dir, 'stderr.log')
        tail = open(stderr_path).read()[-STDERR_TAIL_CHARS:] if os.path.exists(stderr_path) else ''
        error = f'execution exited with {code}' + ('' if os.path.exists(summary_path) else ' and wrote no summary.json')
        with open(os.path.join(run_dir, 'error.txt'), 'w') as f:
            f.write(f'{error}\n\n{tail}')
        return error
    summary = json.load(open(summary_path))
    write_json_atomically(os.path.join(run_dir, 'result.json'), {
        'id': c.id, 'params': c.params, 'runtime_s': summary['runtime_s'], 'frames': summary['frames'],
        'videos': summary['videos'], 'video_files': video_files, 'saved': summary['saved'], 'duplicate_of': None,
        # Process wall clock (startup, model loading, the timed run, writing); not the measured runtime.
        'wall_s': wall,
    })
    return None


def copy_to_duplicate(cfg: SweepConfig, c: Configuration, source_id: str) -> bool:
    """Write a duplicate's result.json from the configuration that ran for it; False when that has no result yet."""
    source = os.path.join(cfg.profiling_dir, 'runs', source_id, 'result.json')
    if not os.path.exists(source):
        return False
    run_dir = os.path.join(cfg.profiling_dir, 'runs', c.id)
    os.makedirs(run_dir, exist_ok=True)
    result = json.load(open(source))
    write_json_atomically(os.path.join(run_dir, 'result.json'),
                          {**result, 'id': c.id, 'params': c.params, 'duplicate_of': source_id})
    return True


def sweep(cfg: SweepConfig, save: bool = False, runner: Runner = run_execution, videos: str | None = None) -> dict:
    """Run every pending configuration; returns the sweep's summary (failed lists configuration ids)."""
    videos = videos if videos is not None else videos_dir(cfg)
    video_files = [os.path.basename(v) for v in list_videos(videos)]
    # Every video needs its reference tracks, or the evaluation step would fail after hours of runs.
    missing = missing_references(cfg.evaluation.reference_dir, cfg.evaluation.reference_file, video_files)
    if missing:
        raise StartupError(f'reference tracks missing for {len(missing)} videos, e.g. {missing[0]}')
    claim_output(cfg)
    configs = configurations(cfg)
    dup = duplicates(cfg, configs)
    ran, skipped, failed = 0, 0, []
    for c in configs:
        if c.id in dup:
            continue
        if os.path.exists(os.path.join(cfg.profiling_dir, 'runs', c.id, 'result.json')):
            skipped += 1
            continue
        error = run_configuration(cfg, c, videos, video_files, save, runner)
        ran += 1
        if error is not None:
            failed.append(c.id)
            print(f'{c.id}: {error}', file=sys.stderr, flush=True)
    # Duplicates follow the configuration that ran for them (a failed one leaves its duplicates pending).
    copied = sum(copy_to_duplicate(cfg, c, dup[c.id]) for c in configs if c.id in dup
                 and not os.path.exists(os.path.join(cfg.profiling_dir, 'runs', c.id, 'result.json')))
    return {'configurations': len(configs), 'duplicates': len(dup), 'ran': ran, 'skipped': skipped,
            'duplicates_copied': copied, 'failed': failed}


def main() -> int:
    args = parse_args()
    try:
        summary = sweep(load_sweep_config(args.config), save=args.save)
    except (StartupError, ConfigError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
