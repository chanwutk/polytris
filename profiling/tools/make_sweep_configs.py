"""
Write one profiling sweep config per dataset of configs/global.yaml's EXEC.DATASETS: the validation grid from
global.yaml, artifacts at the cache's standard paths, the dataset's detector from configs/detectors.yaml, and
the ace process split (profiling/PLAN.md 19A).

    python -m profiling.tools.make_sweep_configs          (in the polyis container; validates each config)
"""

from pathlib import Path

import yaml

from execution2.tools.make_paper_configs import detector_spec
from profiling.config import load_sweep_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'profiling' / 'configs'
CACHE = '/polyis-cache'
SYSTEM = {'queue_maxsize': 4, 'save_queue_maxsize': 256, 'max_videos_in_flight': 4, 'stages': {
    'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpus': [0]}, 'prune': {'procs': 1}, 'pack': {'procs': 4},
    'detect': {'procs': 7, 'gpus': [1, 2, 3, 4, 5, 6, 7]}, 'track': {'procs': 4}, 'evaluate': {'procs': 4}}}


def sweep_config(dataset: str, grid: dict, detectors: dict) -> dict:
    """One dataset's sweep: global.yaml's grid (one classifier, one tile size), standard artifact paths."""
    classifier, tile = grid['CLASSIFIERS'][0], grid['TILE_SIZES'][0]
    index = f'{CACHE}/{dataset}/indexing'
    return {
        'dataset': dataset, 'videoset': 'valid', 'output_root': '/polyis-profiling',
        'params': {'classifier': classifier, 'tile_size': tile, 'canvas_format': 'jpeg',
                   'sample_rates': grid['SAMPLE_RATES'], 'relevance_thresholds': grid['RELEVANCE_THRESHOLDS'],
                   'mistrack_tolerances': grid['TRACKING_ACCURACY_THRESHOLDS'],
                   'tile_paddings': grid['TILEPADDING_MODES'], 'canvas_scales': [float(c) for c in grid['CANVAS_SCALE']],
                   'trackers': grid['TRACKERS'],
                   # Merge tolerances with identical ILPs (the aggregator restores every configuration).
                   'reduce_search_space': True},
        'artifacts': {
            'classifier_weights': f'{index}/training/results/{classifier}_{tile}/model_best.pth',
            'compilation_benchmark': f'{index}/training/results/{classifier}_{tile}/model_compilation.jsonl',
            'never_relevant_bitmap': f'{index}/always_relevant/{tile}_all.npy',
            'max_rate_tables': {t: f'{index}/track_rate/{t}_{tile}/max_rate_table.npy' for t in grid['TRACKERS']},
            'detector': detector_spec(dataset, detectors)},
        'evaluation': {'reference_dir': f'{CACHE}/{dataset}/execution', 'reference_file': '002_naive/tracking.jsonl'},
        'system': SYSTEM,
    }


def main() -> None:
    grid = yaml.safe_load((ROOT / 'configs' / 'global.yaml').read_text())['EXEC']
    detectors = yaml.safe_load((ROOT / 'configs' / 'detectors.yaml').read_text())
    if len(grid['CLASSIFIERS']) != 1 or len(grid['TILE_SIZES']) != 1:
        raise SystemExit('a sweep covers one classifier and one tile size (DECISIONS.md D4)')
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in grid['DATASETS']:
        path = OUT / f'{dataset}.yaml'
        header = (f'# {dataset} validation sweep: the grid of configs/global.yaml (written by '
                  f'profiling/tools/make_sweep_configs.py).\n# Run with the real cache read-only at /polyis-cache and a '
                  f'scratch directory at /polyis-profiling.\n')
        path.write_text(header + yaml.safe_dump(sweep_config(dataset, grid, detectors), sort_keys=False))
        load_sweep_config(str(path))   # every artifact must exist
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
