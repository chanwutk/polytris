"""
Aggregate a profiling sweep into per-configuration results, the Pareto front, and an interactive page
(profiling/PLAN.md "Aggregator", 5A, 7A).

    python -m profiling.aggregate --config profiling/configs/<dataset>.yaml

Reads <output_root>/<dataset>/profiling/{records.jsonl, run.json}; writes next to them:
  configurations.json   every configuration: parameters, dataset HOTA, charged time, throughput (with and
                        without decode), and its execution2 YAML
  pareto.json           the configurations on any ablation condition's Pareto front (p135's union of fronts:
                        minimize time, maximize HOTA), with the conditions each is on
  pareto.html           a self-contained Vega-Lite page: HOTA vs throughput; clicking a point shows that
                        configuration's execution2 YAML (copy / download)

Dataset HOTA is TrackEval's: each configuration's per-video results are combined with HOTA's own
combine_sequences (summed counts, recomputed scores), then averaged over localization thresholds as p111 does.
"""

import argparse
import json
import os
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import yaml

from evaluation.ablation import ABLATION_CONDITIONS, filter_by_ablation_condition
from polyis.pareto import compute_pareto_front
from profiling.config import SweepConfig, load_sweep_config
from profiling.reduction import sweep_groups

# Branch keys -> the column names p135 / polyis.pareto use.
COLUMNS = {'classifier': 'classifier', 'tile_size': 'tilesize', 'sample_rate': 'sample_rate',
           'mistrack_tolerance': 'tracking_accuracy_threshold', 'relevance_threshold': 'relevance_threshold',
           'tile_padding': 'tilepadding', 'canvas_scale': 'canvas_scale', 'tracker': 'tracker'}

# Pinned Vega releases for the page.
VEGA = 'https://cdn.jsdelivr.net/npm/vega@5.30.0'
VEGA_LITE = 'https://cdn.jsdelivr.net/npm/vega-lite@5.21.0'
VEGA_EMBED = 'https://cdn.jsdelivr.net/npm/vega-embed@6.26.0'


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate a profiling sweep into results, a Pareto front, and a page.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def branch_key(branch: dict) -> str:
    """A stable identifier for a configuration (its parameters as sorted JSON)."""
    return json.dumps(branch, sort_keys=True)


def combine_hota(per_video: list[dict]) -> dict:
    """TrackEval's combined HOTA over videos, from each video's raw HOTA result (lists from JSON)."""
    # Through p110, which puts modules/TrackEval on sys.path before importing trackeval.
    from evaluation.p110_accuracy_compute import HOTA
    metric = HOTA()
    arrays = {f'v{i}': {k: np.asarray(v, dtype=float) for k, v in r.items()} for i, r in enumerate(per_video)}
    return metric.combine_sequences(arrays)


def hota_score(combined: dict) -> float:
    """p111's HOTA_HOTA: the mean over localization thresholds."""
    values = list(np.asarray(combined['HOTA'], dtype=float))
    return float(sum(values) / len(values))


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


def execution2_config(cfg: SweepConfig, branch: dict) -> dict:
    """The execution2 config for one configuration: the sweep's artifacts, one GPU, JPEG/raw as swept."""
    a, tol = cfg.artifacts, branch['mistrack_tolerance']
    artifacts = {
        'classifier': {'name': branch['classifier'], 'weights': a.classifier_weights,
                       'compilation_benchmark': a.compilation_benchmark},
        'never_relevant_bitmap': a.never_relevant_bitmap,
        'detector': {'type': a.detector.type, **a.detector.options},
    }
    stages = {'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpu': 0}, 'pack': {'procs': 2},
              'detect': {'procs': 1, 'gpu': 0}, 'track': {'procs': 2}}
    if tol is not None:
        artifacts['max_rate_table'] = a.max_rate_tables[branch['tracker']]
        stages['prune'] = {'procs': 1}
    params = {'sample_rate': branch['sample_rate'], 'relevance_threshold': branch['relevance_threshold'],
              'mistrack_tolerance': tol, 'tile_padding': branch['tile_padding'], 'tracker': branch['tracker'],
              'tile_size': branch['tile_size'], 'canvas_scale': branch['canvas_scale'],
              'canvas_format': cfg.params.canvas_format}
    return {'params': params, 'artifacts': artifacts, 'system': {'queue_maxsize': 2, 'stages': stages}}


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
        config['execution2_yaml'] = yaml.safe_dump(execution2_config(cfg, branch), sort_keys=False)
    pareto = pareto_union(configs)
    with open(os.path.join(out_dir, 'configurations.json'), 'w') as f:
        json.dump(configs, f, indent=1)
    with open(os.path.join(out_dir, 'pareto.json'), 'w') as f:
        json.dump(pareto, f, indent=1)
    with open(os.path.join(out_dir, 'pareto.html'), 'w') as f:
        f.write(page(cfg, configs, pareto))
    return configs, pareto


def expand_tolerance_groups(cfg: SweepConfig, configs: list[dict]) -> list[dict]:
    """One row per configuration of the grid: each pruned representative's result is copied to every other
    tolerance of its group (identical ILP, so identical results; profiling/reduction.py).  `ran_as` names the
    tolerance that ran; duplicates carry its measured time (DECISIONS.md Q2)."""
    groups = sweep_groups(cfg)
    out = []
    for c in configs:
        tol = c['tracking_accuracy_threshold']
        out.append({**c, 'ran_as': tol})
        if tol is None:
            continue
        group = next(g for g in groups[(c['tracker'], c['sample_rate'])] if g.representative == tol)
        out += [{**c, 'tracking_accuracy_threshold': other, 'ran_as': tol} for other in group.tolerances[1:]]
    return out


def pareto_union(configs: list[dict]) -> list[dict]:
    """p135's rule: the union of each ablation condition's front (minimize time, maximize HOTA)."""
    df = pd.DataFrame(configs)
    on: dict[int, list[str]] = defaultdict(list)
    for condition in ABLATION_CONDITIONS:
        subset = filter_by_ablation_condition(df, condition)
        if subset.empty:
            continue
        front = compute_pareto_front(subset, 'time', 'HOTA_HOTA', minx=True, miny=False)
        for i in front['id']:
            on[int(i)].append(condition.name)
    return [{**c, 'pareto_conditions': on[c['id']]} for c in configs if c['id'] in on]


def page(cfg: SweepConfig, configs: list[dict], pareto: list[dict]) -> str:
    """The interactive HTML page (data and YAML embedded; Vega from a CDN)."""
    on = {p['id']: p['pareto_conditions'] for p in pareto}
    rows = [{**{k: v for k, v in c.items() if k != 'execution2_yaml'},
             'front': 'full Tetris front' if 'full' in on.get(c['id'], []) else
                      ('other ablation front' if c['id'] in on else 'not on a front')} for c in configs]
    yamls = {c['id']: c['execution2_yaml'] for c in configs}
    spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'title': f'{cfg.dataset} ({cfg.videoset}): HOTA vs throughput, {len(configs)} configurations',
        'width': 760, 'height': 460, 'data': {'values': rows},
        'params': [{'name': 'pick', 'select': {'type': 'point', 'fields': ['id'], 'on': 'click'}}],
        'mark': {'type': 'point', 'filled': True, 'size': 60, 'cursor': 'pointer'},
        'encoding': {
            'x': {'field': 'throughput_fps', 'type': 'quantitative', 'scale': {'type': 'log'},
                  'title': 'throughput (frames/s, charged time, decode excluded)'},
            'y': {'field': 'HOTA_HOTA', 'type': 'quantitative', 'scale': {'zero': False}, 'title': 'HOTA'},
            'color': {'field': 'front', 'type': 'nominal',
                      'scale': {'domain': ['full Tetris front', 'other ablation front', 'not on a front'],
                                'range': ['#d62728', '#ff9896', '#c7c7c7']}},
            'opacity': {'condition': {'param': 'pick', 'value': 1}, 'value': 0.85},
            'tooltip': [{'field': f} for f in ('sample_rate', 'relevance_threshold', 'tracking_accuracy_threshold',
                                                 'tilepadding', 'canvas_scale', 'tracker', 'HOTA_HOTA',
                                                 'throughput_fps', 'throughput_fps_with_decode')],
        },
    }
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{cfg.dataset} profiling</title>
<script src="{VEGA}"></script><script src="{VEGA_LITE}"></script><script src="{VEGA_EMBED}"></script>
<style>body{{font-family:sans-serif;margin:16px}}pre{{background:#f5f5f5;padding:8px;max-height:420px;overflow:auto}}
button{{margin-right:8px}}</style></head>
<body><div id="vis"></div>
<h3 id="title">Click a point to get its execution2 config.</h3>
<div><button id="copy" disabled>Copy YAML</button><button id="download" disabled>Download YAML</button></div>
<pre id="yaml"></pre>
<script>
const SPEC = {json.dumps(spec)};
const YAML = {json.dumps(yamls)};
const ROWS = Object.fromEntries(SPEC.data.values.map(r => [r.id, r]));
vegaEmbed('#vis', SPEC).then(result => {{
  result.view.addSignalListener('pick', (name, value) => {{
    if (!value || !value.id) return;
    const id = value.id[0], r = ROWS[id], text = YAML[id];
    document.getElementById('title').textContent =
      `s=${{r.sample_rate}} T_r=${{r.relevance_threshold}} tol=${{r.tracking_accuracy_threshold}} ` +
      `P=${{r.tilepadding}} scale=${{r.canvas_scale}} ${{r.tracker}}: HOTA ${{r.HOTA_HOTA.toFixed(4)}}, ` +
      `${{r.throughput_fps.toFixed(1)}} fps`;
    document.getElementById('yaml').textContent = text;
    const copy = document.getElementById('copy'), dl = document.getElementById('download');
    copy.disabled = dl.disabled = false;
    copy.onclick = () => navigator.clipboard.writeText(text);
    dl.onclick = () => {{
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([text], {{type: 'text/yaml'}}));
      a.download = `{cfg.dataset}-config-${{id}}.yaml`; a.click();
    }};
  }});
}});
</script></body></html>
"""


def main() -> int:
    args = parse_args()
    configs, pareto = aggregate(load_sweep_config(args.config))
    print(json.dumps({'configurations': len(configs), 'pareto': len(pareto)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
