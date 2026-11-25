"""
Write one profiling sweep config per dataset of configs/global.yaml's EXEC.DATASETS (profiling/README.md): the
validation grid from global.yaml, artifacts at the cache's standard paths, the dataset's detector from
configs/detectors.yaml, and the paper's execution system section, so validation runtimes are measured as the final
execution runs are.  Run in the polyis container (each config is validated, artifacts included):

    python -m experiments.tools.profiling.make_configs
"""

from pathlib import Path

import yaml

from experiments.tools.execution.make_paper_configs import CACHE, detector_spec, one_gpu_system
from profiling.config import load_sweep_config

ROOT = Path(__file__).resolve().parents[2]   # experiments/
REPO = ROOT.parent
OUT = ROOT / 'configs' / 'profiling'
# Results (runs, tracks, evaluations, Pareto fronts) go to <OUTPUT_ROOT>/<dataset>/profiling on the data volume.
OUTPUT_ROOT = '/polyis-data/profiling'


def sweep_config(dataset: str, grid: dict, detectors: dict) -> dict:
    """One dataset's sweep: global.yaml's grid (one classifier, one tile size), standard artifact paths."""
    classifier, tile = grid['CLASSIFIERS'][0], grid['TILE_SIZES'][0]
    index = f'{CACHE}/{dataset}/indexing'
    return {
        'dataset': dataset, 'videoset': 'valid', 'output_root': OUTPUT_ROOT,
        'params': {'classifier': classifier, 'tile_size': tile, 'canvas_format': 'jpeg',
                   'sample_rates': grid['SAMPLE_RATES'], 'relevance_thresholds': grid['RELEVANCE_THRESHOLDS'],
                   'mistrack_tolerances': grid['TRACKING_ACCURACY_THRESHOLDS'],
                   'tile_paddings': grid['TILEPADDING_MODES'], 'canvas_scales': [float(c) for c in grid['CANVAS_SCALE']],
                   'trackers': grid['TRACKERS'], 'dedup_pruning': True},
        'artifacts': {
            'classifier_weights': f'{index}/training/results/{classifier}_{tile}/model_best.pth',
            'compilation_benchmark': f'{index}/training/results/{classifier}_{tile}/model_compilation.jsonl',
            'never_relevant_bitmap': f'{index}/always_relevant/{tile}_all.npy',
            'max_rate_tables': {t: f'{index}/track_rate/{t}_{tile}/max_rate_table.npy' for t in grid['TRACKERS']},
            'detector': detector_spec(dataset, detectors)},
        'evaluation': {'reference_dir': f'{CACHE}/{dataset}/execution', 'reference_file': '002_naive/tracking.jsonl'},
        'system': one_gpu_system(pruned=True),
    }


def main() -> None:
    grid = yaml.safe_load((REPO / 'configs' / 'global.yaml').read_text())['EXEC']
    detectors = yaml.safe_load((REPO / 'configs' / 'detectors.yaml').read_text())
    if len(grid['CLASSIFIERS']) != 1 or len(grid['TILE_SIZES']) != 1:
        raise SystemExit('a sweep covers one classifier and one tile size')
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in grid['DATASETS']:
        path = OUT / f'{dataset}.yaml'
        header = (f'# {dataset} validation sweep: the grid of configs/global.yaml '
                  f'(written by experiments/tools/profiling/make_configs.py).\n')
        path.write_text(header + yaml.safe_dump(sweep_config(dataset, grid, detectors), sort_keys=False))
        # Every artifact must exist and every configuration must be a valid execution config.
        load_sweep_config(str(path))
        print(path.relative_to(REPO))


if __name__ == '__main__':
    main()
