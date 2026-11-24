"""
Write one execution2 config per operating point the paper reports (the fastest Tetris test configuration
within 5% and within 10% HOTA loss of the reference, per dataset), as chosen by R1.O3's clean timing job.

Reads measurements-results/r1o3_clean_extraction_timing/{throughput.json, raw/plan.json} and
configs/detectors.yaml; writes execution/configs/paper/<dataset>-<loss>.yaml.  Run in the polyis container:

  python -m execution.tools.make_paper_configs
"""

import json
import re
from pathlib import Path

import yaml

from execution.config import DETECTOR_FIELDS

ROOT = Path(__file__).resolve().parents[2]
R1O3 = ROOT / 'measurements-results' / 'r1o3_clean_extraction_timing'
OUT = ROOT / 'execution' / 'configs' / 'paper'
CACHE = '/polyis-cache'
RETINANET_CONFIG = '/polyis/configs/retinanet.json'

# <classifier>_<tile>_<s>[_<tolerance x100>]_r<T_r x100>_<padding>_s<canvas scale x100>_<tracker>
NAME = re.compile(r'^(?P<classifier>[^_]+)_(?P<tile>\d+)_(?P<s>\d+)(?:_(?P<tol>\d{3}))?_r(?P<tr>\d{3})_'
                  r'(?P<padding>[a-z]+)_s(?P<scale>\d+)_(?P<tracker>\w+)$')


def parse_name(name: str) -> dict:
    """The parameters encoded in a cache configuration name (polyis.utilities.build_param_str)."""
    m = NAME.match(name)
    if m is None:
        raise ValueError(f'cannot parse configuration name {name!r}')
    return {'classifier': m['classifier'], 'tile_size': int(m['tile']), 'sample_rate': int(m['s']),
            'mistrack_tolerance': int(m['tol']) / 100 if m['tol'] else None,
            'relevance_threshold': int(m['tr']) / 100, 'tile_padding': m['padding'],
            'canvas_scale': int(m['scale']) / 100, 'tracker': m['tracker']}


def detector_spec(dataset: str, detectors: dict) -> dict:
    """The dataset's detector from configs/detectors.yaml, with only the fields execution2 takes."""
    entry = detectors['dataset_detector_mapping'][detectors['dataset_name_mapping'][dataset]]
    kind = entry['detector']
    if kind == 'retina':
        return {'type': 'retina', 'config_path': RETINANET_CONFIG}
    return {'type': kind, **{k: entry[k] for k in DETECTOR_FIELDS[kind]}}


def build(dataset: str, name: str, detectors: dict) -> dict:
    """An execution2 config for the operating point a cache configuration name encodes."""
    return build_params(dataset, parse_name(name), detectors)


def build_params(dataset: str, p: dict, detectors: dict) -> dict:
    """An execution2 config for one operating point (parse_name's keys): JPEG canvases (as the paper's runs), one GPU."""
    index = f'{CACHE}/{dataset}/indexing'
    artifacts = {
        'classifier': {'name': p['classifier'],
                       'weights': f'{index}/training/results/{p["classifier"]}_{p["tile_size"]}/model_best.pth',
                       'compilation_benchmark': f'{index}/training/results/{p["classifier"]}_{p["tile_size"]}/model_compilation.jsonl'},
        'never_relevant_bitmap': f'{index}/always_relevant/{p["tile_size"]}_all.npy',
        'detector': detector_spec(dataset, detectors),
    }
    stages = {'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpu': 0}, 'pack': {'procs': 2},
              'detect': {'procs': 1, 'gpu': 0}, 'track': {'procs': 2}}
    if p['mistrack_tolerance'] is not None:
        artifacts['max_rate_table'] = f'{index}/track_rate/{p["tracker"]}_{p["tile_size"]}/max_rate_table.npy'
        stages['prune'] = {'procs': 1}
    params = {k: p[k] for k in ('sample_rate', 'relevance_threshold', 'mistrack_tolerance', 'tile_padding',
                                'tracker', 'tile_size', 'canvas_scale')}
    return {'params': {**params, 'canvas_format': 'jpeg'}, 'artifacts': artifacts,
            'system': {'queue_maxsize': 2, 'stages': stages}}


def main() -> None:
    throughput = json.loads((R1O3 / 'throughput.json').read_text())
    labels = {v['label']: k for k, v in json.loads((R1O3 / 'raw' / 'plan.json').read_text())['datasets'].items()}
    detectors = yaml.safe_load((ROOT / 'configs' / 'detectors.yaml').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for label, levels in throughput['datasets'].items():
        dataset = labels[label]
        for loss, rec in levels.items():
            path = OUT / f'{dataset}-{loss}.yaml'
            header = (f'# {label} ({dataset}): the paper\'s fastest configuration within {float(loss):.0%} HOTA loss,\n'
                      f'# {rec["config"]} (from measurements-results/r1o3_clean_extraction_timing).\n')
            path.write_text(header + yaml.safe_dump(build(dataset, rec['config'], detectors), sort_keys=False))
            print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
