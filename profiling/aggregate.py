#!/usr/local/bin/python
"""
Aggregate a profiling sweep into per-configuration results, the Pareto fronts, and an interactive page
(profiling/README.md "Steps").

    python -m profiling.aggregate --config <sweep yaml>

Reads every configuration's runs/<id>/result.json and its scored evaluation.json (a duplicate's comes from the
configuration that ran for it); writes under <output_root>/<dataset>/profiling/:

  configurations.json   every configuration: parameters, dataset HOTA, time (execution's end-to-end runtime_s),
                        throughput, saved, duplicate_of, and its execution YAML
  pareto.json           the configurations on any ablation condition's Pareto front (p135's union of fronts:
                        minimize time, maximize HOTA), with the conditions each is on
  pareto.html           a self-contained Vega-Lite page: HOTA vs throughput; clicking a point shows that
                        configuration's execution YAML (copy / download)

Dataset HOTA is TrackEval's: each configuration's per-video results are combined with HOTA's own
combine_sequences (summed counts, recomputed scores), then averaged over localization thresholds as p111 does.
Every configuration must have a result and an evaluation; missing ones are listed and nothing is written.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from execution.config import ConfigError
from profiling.config import SweepConfig, configurations, load_sweep_config
from tetris.ablation import ABLATION_CONDITIONS, filter_by_ablation_condition
from tetris.pareto import compute_pareto_front

# Configuration parameter names -> the column names p135 / tetris.pareto use.
COLUMNS = {'classifier': 'classifier', 'tile_size': 'tilesize', 'sample_rate': 'sample_rate',
           'mistrack_tolerance': 'tracking_accuracy_threshold', 'relevance_threshold': 'relevance_threshold',
           'tile_padding': 'tilepadding', 'canvas_scale': 'canvas_scale', 'tracker': 'tracker'}

# Pinned Vega releases for the page.
VEGA = 'https://cdn.jsdelivr.net/npm/vega@5.30.0'
VEGA_LITE = 'https://cdn.jsdelivr.net/npm/vega-lite@5.21.0'
VEGA_EMBED = 'https://cdn.jsdelivr.net/npm/vega-embed@6.26.0'


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate a profiling sweep into results, Pareto fronts, and a page.')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def combine_hota(per_video: list[dict]) -> dict:
    """TrackEval's combined HOTA over videos, from each video's raw HOTA result (lists from JSON)."""
    # tetris.trackeval.evaluate puts modules/TrackEval on sys.path before importing trackeval.
    from tetris.trackeval.evaluate import HOTA
    metric = HOTA()
    arrays = {f'v{i}': {k: np.asarray(v, dtype=float) for k, v in r.items()} for i, r in enumerate(per_video)}
    return metric.combine_sequences(arrays)


def hota_score(combined: dict) -> float:
    """p111's HOTA_HOTA: the mean over localization thresholds."""
    values = list(np.asarray(combined['HOTA'], dtype=float))
    return float(sum(values) / len(values))


def pareto_union(configs: list[dict]) -> list[dict]:
    """p135's rule: the union of each ablation condition's front (minimize time, maximize HOTA).

    Each config needs an integer 'id', 'time', 'HOTA_HOTA', 'sample_rate', and 'tracking_accuracy_threshold'.
    """
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


def page(title: str, file_stem: str, x_title: str, configs: list[dict], pareto: list[dict]) -> str:
    """The interactive HTML page (data and YAML embedded; Vega from a CDN).  Configs carry 'execution_yaml'."""
    on = {p['id']: p['pareto_conditions'] for p in pareto}
    rows = [{**{k: v for k, v in c.items() if k != 'execution_yaml'},
             'front': 'full Tetris front' if 'full' in on.get(c['id'], []) else
                      ('other ablation front' if c['id'] in on else 'not on a front')} for c in configs]
    yamls = {c['id']: c['execution_yaml'] for c in configs}
    spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'title': title, 'width': 760, 'height': 460, 'data': {'values': rows},
        'params': [{'name': 'pick', 'select': {'type': 'point', 'fields': ['id'], 'on': 'click'}}],
        'mark': {'type': 'point', 'filled': True, 'size': 60, 'cursor': 'pointer'},
        'encoding': {
            'x': {'field': 'throughput_fps', 'type': 'quantitative', 'scale': {'type': 'log'}, 'title': x_title},
            'y': {'field': 'HOTA_HOTA', 'type': 'quantitative', 'scale': {'zero': False}, 'title': 'HOTA'},
            'color': {'field': 'front', 'type': 'nominal',
                      'scale': {'domain': ['full Tetris front', 'other ablation front', 'not on a front'],
                                'range': ['#d62728', '#ff9896', '#c7c7c7']}},
            'opacity': {'condition': {'param': 'pick', 'value': 1}, 'value': 0.85},
            'tooltip': [{'field': f} for f in ('sample_rate', 'relevance_threshold', 'tracking_accuracy_threshold',
                                                 'tilepadding', 'canvas_scale', 'tracker', 'HOTA_HOTA',
                                                 'throughput_fps')],
        },
    }
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{file_stem} profiling</title>
<script src="{VEGA}"></script><script src="{VEGA_LITE}"></script><script src="{VEGA_EMBED}"></script>
<style>body{{font-family:sans-serif;margin:16px}}pre{{background:#f5f5f5;padding:8px;max-height:420px;overflow:auto}}
button{{margin-right:8px}}</style></head>
<body><div id="vis"></div>
<h3 id="title">Click a point to get its execution config.</h3>
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
      a.download = `{file_stem}-config-${{id}}.yaml`; a.click();
    }};
  }});
}});
</script></body></html>
"""


def configuration_rows(cfg: SweepConfig) -> list[dict]:
    """One row per configuration: parameters, dataset HOTA, end-to-end time, throughput, provenance."""
    runs = os.path.join(cfg.profiling_dir, 'runs')
    rows, missing = [], []
    for i, c in enumerate(configurations(cfg)):
        result_path = os.path.join(runs, c.id, 'result.json')
        if not os.path.exists(result_path):
            missing.append(f'{c.id}: no result.json')
            continue
        result = json.load(open(result_path))
        # A duplicate is scored through the configuration that ran for it.
        scored_id = result['duplicate_of'] or c.id
        evaluation_path = os.path.join(runs, scored_id, 'evaluation.json')
        if not os.path.exists(evaluation_path):
            missing.append(f'{c.id}: no evaluation.json in {scored_id}')
            continue
        evaluation = json.load(open(evaluation_path))
        if sorted(evaluation) != sorted(result['video_files']):
            missing.append(f'{c.id}: evaluation covers {len(evaluation)} of {len(result["video_files"])} videos')
            continue
        rows.append({'id': i, 'config_id': c.id, **{COLUMNS[k]: v for k, v in c.params.items()},
                     'HOTA_HOTA': hota_score(combine_hota([evaluation[v]['HOTA'] for v in result['video_files']])),
                     'time': result['runtime_s'], 'frames': result['frames'],
                     'throughput_fps': result['frames'] / result['runtime_s'] if result['runtime_s'] > 0 else None,
                     'saved': result['saved'], 'duplicate_of': result['duplicate_of'],
                     'execution_yaml': yaml.safe_dump(c.execution, sort_keys=False)})
    if missing:
        raise RuntimeError(f'{len(missing)} configurations are not ready for aggregation, e.g. ' + '; '.join(missing[:5]))
    return rows


def aggregate(cfg: SweepConfig) -> tuple[list[dict], list[dict]]:
    """(every configuration, the Pareto configurations); writes configurations.json, pareto.json, pareto.html."""
    configs = configuration_rows(cfg)
    pareto = pareto_union(configs)
    saved = [c['config_id'] for c in configs if c['saved']]
    if saved:
        # Saving shares the machine with the timed run, so these runtimes overstate the cost.
        print(f'warning: {len(saved)} configurations ran with --save; their runtimes include saving, e.g. {saved[0]}',
              file=sys.stderr)
    out = cfg.profiling_dir
    with open(os.path.join(out, 'configurations.json'), 'w') as f:
        json.dump(configs, f, indent=1)
    with open(os.path.join(out, 'pareto.json'), 'w') as f:
        json.dump(pareto, f, indent=1)
    title = f'{cfg.dataset} ({cfg.videoset}): HOTA vs throughput, {len(configs)} configurations'
    with open(os.path.join(out, 'pareto.html'), 'w') as f:
        f.write(page(title, cfg.dataset, 'throughput (frames/s, end-to-end runtime)', configs, pareto))
    return configs, pareto


def main() -> int:
    args = parse_args()
    try:
        cfg = load_sweep_config(args.config)
    except ConfigError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    configs, pareto = aggregate(cfg)
    print(json.dumps({'configurations': len(configs), 'pareto': len(pareto)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
