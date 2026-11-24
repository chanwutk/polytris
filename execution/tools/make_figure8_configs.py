"""
Write one execution config per full-Tetris point of the paper's Figure 8 (fig:e2e-results).

Reads measurements-results/execution-figure8/figure8_points.json (the "Tetris" rows of Figure 8's
Vega-Lite data, with the throughput and HOTA the figure plots) and configs/detectors.yaml; writes
execution/configs/figure8/<dataset>/<configuration name>.yaml.  Run in the polyis container:

  python -m execution.tools.make_figure8_configs
"""

import json
from pathlib import Path

import yaml

from execution.tools.make_paper_configs import build_params
from polyis.utilities import build_param_str

ROOT = Path(__file__).resolve().parents[2]
POINTS = ROOT / 'measurements-results' / 'execution-figure8' / 'figure8_points.json'
OUT = ROOT / 'execution' / 'configs' / 'figure8'


def params_of(point: dict) -> dict:
    """make_paper_configs.build_params's keys from one Figure 8 row."""
    tol = point['tracking_accuracy_threshold']
    return {'classifier': point['classifier'], 'tile_size': 60, 'sample_rate': int(point['sample_rate']),
            'mistrack_tolerance': None if tol is None else float(tol),
            'relevance_threshold': float(point['relevance_threshold']), 'tile_padding': point['tilepadding'],
            'canvas_scale': float(point['canvas_scale']), 'tracker': point['tracker']}


def config_name(p: dict) -> str:
    """The cache's name for this configuration (the scripts' p060 directory)."""
    return build_param_str(classifier=p['classifier'], tilesize=p['tile_size'], sample_rate=p['sample_rate'],
                           tilepadding=p['tile_padding'], canvas_scale=p['canvas_scale'], tracker=p['tracker'],
                           tracking_accuracy_threshold=p['mistrack_tolerance'],
                           relevance_threshold=p['relevance_threshold'])


def main() -> None:
    points = json.loads(POINTS.read_text())['points']
    detectors = yaml.safe_load((ROOT / 'configs' / 'detectors.yaml').read_text())
    written = 0
    for point in points:
        p = params_of(point)
        path = OUT / point['dataset'] / f'{config_name(p)}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Figure 8 full-Tetris point, {point['dataset_display']} ({point['dataset']}): reported "
                  f"{point['throughput_fps']:.1f} fps, HOTA {point['HOTA_HOTA']:.4f}.\n")
        path.write_text(header + yaml.safe_dump(build_params(point['dataset'], p, detectors), sort_keys=False))
        written += 1
    assert written == len({(pt['dataset'], config_name(params_of(pt))) for pt in points}), 'duplicate configurations'
    print(f'wrote {written} configs under {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
