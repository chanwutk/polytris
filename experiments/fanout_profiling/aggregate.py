"""
Aggregate a profiling sweep into per-configuration results, the Pareto front, and an interactive page
(experiments/fanout_profiling/PLAN.md "Aggregator", 5A, 7A).

    python -m experiments.fanout_profiling.aggregate --config experiments/fanout_profiling/configs/<dataset>.yaml

Reads <output_root>/<dataset>/profiling/{records.jsonl, run.json}; writes next to them:
  configurations.json   every configuration: parameters, dataset HOTA, charged time, throughput (with and
                        without decode), and its execution YAML
  pareto.json           the configurations on any ablation condition's Pareto front (p135's union of fronts:
                        minimize time, maximize HOTA), with the conditions each is on
  pareto.html           a self-contained Vega-Lite page: HOTA vs throughput; clicking a point shows that
                        configuration's execution YAML (copy / download)

Dataset HOTA is TrackEval's: each configuration's per-video results are combined with HOTA's own
combine_sequences (summed counts, recomputed scores), then averaged over localization thresholds as p111 does.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd
import yaml

# The HOTA combination, Pareto union, and page are profiling/aggregate.py's (one copy, shared with the profiling sweep).
from profiling.aggregate import combine_hota, hota_score, page, pareto_union  # noqa: F401  (re-exported for tests)
from experiments.fanout_profiling.config import SweepConfig, load_sweep_config
from experiments.fanout_profiling.reduction import sweep_groups
from profiling.aggregate import COLUMNS  # noqa: F401  (Branch keys -> the column names p135 / tetris.pareto use)
from profiling.config import Artifacts as ProfilingArtifacts, execution_config as profiling_execution_config


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate a profiling sweep into results, a Pareto front, and a page.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def branch_key(branch: dict) -> str:
    """A stable identifier for a configuration (its parameters as sorted JSON)."""
    return json.dumps(branch, sort_keys=True)


def times_per_configuration(records_path: str) -> dict[str, dict]:
    """Charged seconds (all stages; decode and evaluate charge 0) and decode seconds, summed over videos."""
    con = duckdb.connect()
    rows = con.execute(f"""
        WITH r AS (SELECT to_json(branch)::VARCHAR AS key, video, unnest(timings) AS t
                   FROM read_json_auto('{records_path}', maximum_object_size=1000000000))
        SELECT key, count(DISTINCT video) AS videos,
               sum(t.charged) AS charged_s,
               sum(CASE WHEN t.stage = 'decode' THEN t."end" - t.start ELSE 0 END) AS decode_s
        FROM r GROUP BY key
    """).fetchall()
    # DuckDB's to_json keeps the branch's key order; re-key by sorted JSON to match branch_key().
    return {branch_key(json.loads(k)): {'videos': v, 'charged_s': c, 'decode_s': d} for k, v, c, d in rows}


# The execution config the page offers for each configuration: one GPU, as the paper's execution runs.
ONE_GPU_SYSTEM = {'queue_maxsize': 2, 'stages': {
    'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpu': 0}, 'prune': {'procs': 1}, 'pack': {'procs': 2},
    'detect': {'procs': 1, 'gpu': 0}, 'track': {'procs': 2}}}


def execution_config(cfg: SweepConfig, branch: dict) -> dict:
    """The execution config for one configuration: the sweep's artifacts, one GPU, JPEG/raw as swept.

    Built by profiling.config.execution_config (one copy of the document's shape).
    """
    a = cfg.artifacts
    artifacts = ProfilingArtifacts(a.classifier_weights, a.compilation_benchmark, a.never_relevant_bitmap,
                                   a.max_rate_tables, {'type': a.detector.type, **a.detector.options})
    return profiling_execution_config(branch, artifacts, cfg.params.canvas_format, ONE_GPU_SYSTEM)


def aggregate(cfg: SweepConfig) -> tuple[list[dict], list[dict]]:
    """(every configuration, the Pareto configurations), and writes the JSON files and the page."""
    out_dir = os.path.join(cfg.output_root, cfg.dataset, 'profiling')
    records_path = os.path.join(out_dir, 'records.jsonl')
    run = json.load(open(os.path.join(out_dir, 'run.json')))
    frames_total = sum(run['video_frames'].values())
    # Per-video HOTA results grouped by configuration (TrackEval's combine needs the raw per-video results).
    per_config: dict[str, list[dict]] = defaultdict(list)
    branches: dict[str, dict] = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            key = branch_key(r['branch'])
            branches[key] = r['branch']
            per_config[key].append(r['metrics']['HOTA'])
    times = times_per_configuration(records_path)
    configs = []
    for key, branch in branches.items():
        if len(per_config[key]) != run['videos']:
            raise RuntimeError(f'{key}: {len(per_config[key])} videos recorded, expected {run["videos"]}')
        t = times[key]
        config = {**{COLUMNS[k]: v for k, v in branch.items()},
                  'HOTA_HOTA': hota_score(combine_hota(per_config[key])), 'time': t['charged_s'],
                  'decode_s': t['decode_s'], 'frames': frames_total,
                  'throughput_fps': frames_total / t['charged_s'] if t['charged_s'] > 0 else None,
                  'throughput_fps_with_decode': frames_total / (t['charged_s'] + t['decode_s'])}
        configs.append(config)
    # Runs from before the flag existed ran every tolerance.
    if run.get('reduce_search_space', False):
        configs = expand_tolerance_groups(cfg, configs)
    else:
        configs = [{**c, 'ran_as': c['tracking_accuracy_threshold']} for c in configs]
    for i, config in enumerate(configs):
        branch = {k: config[c] for k, c in COLUMNS.items()}
        config['id'] = i
        config['execution_yaml'] = yaml.safe_dump(execution_config(cfg, branch), sort_keys=False)
    pareto = pareto_union(configs)
    with open(os.path.join(out_dir, 'configurations.json'), 'w') as f:
        json.dump(configs, f, indent=1)
    with open(os.path.join(out_dir, 'pareto.json'), 'w') as f:
        json.dump(pareto, f, indent=1)
    title = f'{cfg.dataset} ({cfg.videoset}): HOTA vs throughput, {len(configs)} configurations'
    with open(os.path.join(out_dir, 'pareto.html'), 'w') as f:
        f.write(page(title, cfg.dataset, 'throughput (frames/s, charged time, decode excluded)', configs, pareto))
    return configs, pareto


def expand_tolerance_groups(cfg: SweepConfig, configs: list[dict]) -> list[dict]:
    """One row per configuration of the grid: each pruned representative's result is copied to every other
    tolerance of its group (identical ILP, so identical results; experiments/fanout_profiling/reduction.py).  `ran_as` names the
    tolerance that ran; duplicates carry its measured time (DECISIONS.md Q2)."""
    # The run merged groups (run.json says so), whatever the config's flag says now.
    groups = sweep_groups(replace(cfg, params=replace(cfg.params, reduce_search_space=True)))
    out = []
    for c in configs:
        tol = c['tracking_accuracy_threshold']
        out.append({**c, 'ran_as': tol})
        if tol is None:
            continue
        group = next(g for g in groups[(c['tracker'], c['sample_rate'])] if g.representative == tol)
        out += [{**c, 'tracking_accuracy_threshold': other, 'ran_as': tol} for other in group.tolerances[1:]]
    return out


def main() -> int:
    args = parse_args()
    configs, pareto = aggregate(load_sweep_config(args.config))
    print(json.dumps({'configurations': len(configs), 'pareto': len(pareto)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
