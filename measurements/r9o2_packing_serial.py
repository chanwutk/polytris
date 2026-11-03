#!/usr/local/bin/python
"""R9.O2: p030 re-run serially on the validation sweep's packing units, timed from start to end of compress().

The sweep ran p030 with 20 concurrent workers, so its per-unit spans include slowdown from sharing CPUs and disk. This
script runs every unit of one dataset (the units of measurements/r9o2_packing_untimed.py) one after another in a single
process pinned to --cpus, and records compress()'s own whole-unit time (compress_total.json) and its recorded ops
(runtime.jsonl). Outputs go to --scratch instead of the cache, so the sweep's cached outputs and runtimes are untouched,
and each unit's output is deleted after it is read. Records append to measurements-results/r9o2_packing_serial/
<dataset>.jsonl and finished units are skipped, so an interrupted run resumes.
"""
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from evaluation.p120_throughput_gather import resolve_upstream_tracker
from measurements.common import results_dir
from measurements.r9o2_packing_untimed import packing_units
from polyis.io import cache


def parse_args():
    parser = argparse.ArgumentParser(description='R9.O2: p030 re-run serially on the validation sweep')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--scratch', type=Path, default=Path('/polyis-cache/scratch/r9o2_packing_serial'))
    parser.add_argument('--limit', type=int)
    return parser.parse_args()


def load_p030():
    """scripts/p030_exec_compress.py as a module (its file name is not importable)."""
    # Load the script from its path, as scripts/ is not a package.
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'p030_exec_compress.py'
    spec = importlib.util.spec_from_file_location('p030_exec_compress', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    # Pin to the requested CPUs so the run neither waits on nor disturbs other work.
    os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    p030 = load_p030()
    # Redirect only p030's outputs to scratch; its inputs still come from the cache.
    scratch = args.scratch / args.dataset
    cache_exec = cache.exec
    def scratch_exec(dataset, stage, video, *rest):
        path = cache_exec(dataset, stage, video, *rest)
        return scratch / path.relative_to(cache.CACHE_DIR) if stage == 'comp-frames' else path
    p030.cache.exec = scratch_exec
    # The dataset's units and the ones a previous run finished.
    units, _ = packing_units(args.dataset)
    out = results_dir(__file__) / f'{args.dataset}.jsonl'
    done = {json.loads(line)['runtime_file'] for line in out.open()} if out.exists() else set()
    todo = units[~units['runtime_file'].isin(done)]
    todo = todo if args.limit is None else todo.head(args.limit)
    print(f'{args.dataset}: {len(units)} units, {len(done)} done, running {len(todo)}', flush=True)
    for i, (_, row) in enumerate(todo.iterrows()):
        threshold = None if pd.isna(row['tracking_accuracy_threshold']) else float(row['tracking_accuracy_threshold'])
        # Run the unit exactly as p030's sweep task does, with p030's default packing mode.
        p030.compress(row['dataset'], 'valid', row['video'], row['classifier'], int(row['tilesize']),
                      int(row['sample_rate']), row['tilepadding'], float(row['canvas_scale']),
                      resolve_upstream_tracker(row), threshold, float(row['relevance_threshold']),
                      p030.PackMode.Best_Fit)
        # Read its whole-unit time and recorded ops, then free the scratch output.
        output_dir = Path(scratch_exec(row['dataset'], 'comp-frames', row['video'])) / Path(row['runtime_file']).parent.name
        total_ms = json.loads((output_dir / 'compress_total.json').read_text())['compress_total_ms']
        ops = {}
        for line in (output_dir / 'runtime.jsonl').open():
            for op in json.loads(line)['runtime']:
                ops[op['op']] = ops.get(op['op'], 0) + op['time']
        shutil.rmtree(output_dir)
        with out.open('a') as f:
            f.write(json.dumps(dict(runtime_file=row['runtime_file'], video=row['video'], total_ms=total_ms,
                                    ops=ops)) + '\n')
        if i % 500 == 0:
            print(f'{args.dataset}: {i + 1}/{len(todo)} units', flush=True)


if __name__ == '__main__':
    main()
