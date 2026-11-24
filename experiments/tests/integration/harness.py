"""
Shared setup for the scripts-vs-pipelines integration tests (tests/integration).

Each test builds a scratch tree from the committed fixtures (tests/data/equivalence: short clips and each dataset's
small trained artifacts) and never touches /polyis-data or /polyis-cache:

  <tmp>/data/datasets/<dataset>/<split>/*.mp4     the fixture clips of one split
  <tmp>/cache/<dataset>/indexing/...              the fixture artifacts (classifier, never-relevant, sampling limits)
  <tmp>/global.yaml                               configs/global.yaml with the test's grid and the scratch paths

The scripts, the profiling sweep, execution and the p022 mirror each run as a subprocess with CONFIG pointing at that
file: the scripts and several tetris modules read the config (dataset and cache paths, the grid) when imported.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / 'tests' / 'data' / 'equivalence'
REFERENCE_FILE = '002_naive/tracking.jsonl'
# The scripts' config keys for each grid dimension (profiling's sweep config names).
GRID_KEYS = {'sample_rates': 'SAMPLE_RATES', 'relevance_thresholds': 'RELEVANCE_THRESHOLDS',
             'mistrack_tolerances': 'TRACKING_ACCURACY_THRESHOLDS', 'tile_paddings': 'TILEPADDING_MODES',
             'canvas_scales': 'CANVAS_SCALE', 'trackers': 'TRACKERS'}


class Tree:
    """The scratch tree of one test: paths and the subprocess environment that points the code at them."""

    def __init__(self, root: Path, dataset: str, split: str):
        self.root, self.dataset, self.split = root, dataset, split
        self.data = root / 'data'
        self.cache = root / 'cache'
        self.config = root / 'global.yaml'
        self.clips = self.data / 'datasets' / dataset / split

    @property
    def videos(self) -> list[str]:
        """The split's clip names, sorted."""
        return sorted(p.name for p in self.clips.glob('*.mp4'))

    @property
    def execution(self) -> Path:
        """The scripts' per-video output root (cache.exec's parent)."""
        return self.cache / self.dataset / 'execution'

    def env(self, **extra: str) -> dict:
        """The environment every subprocess of this test runs with."""
        return {**os.environ, 'CONFIG': str(self.config), 'POLYIS_DATASETS': self.dataset,
                'PYTHONPATH': str(REPO), **extra}

    def run(self, *cmd: str, **extra: str) -> None:
        """One subprocess from the repository root; its output goes to the test log, and failure fails the test."""
        print('>>>', ' '.join(cmd), flush=True)
        subprocess.run([sys.executable, *cmd], cwd=REPO, env=self.env(**extra), check=True)


def make_tree(tmp: Path, dataset: str, split: str, grid: dict) -> Tree:
    """Copy one split's clips and the dataset's artifacts into tmp, and write the scripts' config for the grid."""
    tree = Tree(tmp, dataset, split)
    fixture = FIXTURES / dataset
    shutil.copytree(fixture / split, tree.clips)
    shutil.copytree(fixture / 'indexing', tree.cache / dataset / 'indexing')
    config = yaml.safe_load((REPO / 'configs' / 'global.yaml').read_text())
    # One dataset, one split, the test's grid; the rest of configs/global.yaml (classifier, tile size) unchanged.
    config['EXEC']['DATASETS'] = [dataset]
    config['EXEC']['VIDEO_SETS'] = [split]
    for key, name in GRID_KEYS.items():
        config['EXEC'][name] = list(grid[key])
    config['DATA']['DATASETS_DIR'] = str(tree.data / 'datasets')
    config['DATA']['CACHE_DIR'] = str(tree.cache)
    tree.config.write_text(yaml.safe_dump(config, sort_keys=False))
    return tree


def require_detector(dataset: str) -> None:
    """Skip unless the dataset's detector checkpoint is in place (fetched and verified by scripts/weights.py)."""
    from scripts import weights
    detectors = yaml.safe_load((REPO / 'configs' / 'detectors.yaml').read_text())
    entry = detectors['dataset_detector_mapping'][detectors['dataset_name_mapping'][dataset]]
    path = '/polyis/weights/retinanet_b3d.pth' if entry['detector'] == 'retina' else entry['model_path']
    if path not in weights.WEIGHTS:
        pytest.skip(f'{dataset}: no checkpoint entry for {path} in scripts/weights.py')
    if os.path.exists(path) and weights.sha256(path) == weights.WEIGHTS[path]['sha256']:
        return
    # Fetch every checkpoint once (scripts/weights.py skips the ones already in place).
    try:
        weights.download()
    except Exception as e:  # noqa: BLE001 - any download failure means the test cannot run here
        pytest.skip(f'{dataset}: detector checkpoint {path} unavailable ({e}); run python scripts/weights.py download')


def scripts(tree: Tree, *names: str, **extra: str) -> None:
    """Run scripts/<name>.py for the tree's split, restricted to its clips."""
    flag = f'--{tree.split}'
    for name in names:
        tree.run(f'scripts/{name}.py', flag, '--videos', *tree.videos, **extra)


def reference(tree: Tree) -> None:
    """The reference tracks (full-frame detector and tracker) the profiling sweep and HOTA score against."""
    flag = f'--{tree.split}'
    tree.run('preprocess/p002_preprocess_naive_detection.py', flag)
    tree.run('preprocess/p002_preprocess_naive_tracking.py', flag)


def mirror_p022(tree: Tree, pipeline_execution: Path | None, report: Path) -> None:
    """tests/integration/mirror_p022.py: p022's solve per grid combination, one at a time (one Gurobi session);
    with pipeline_execution, a pair where either side's ILP hit its limit takes the pipeline's pruned grid."""
    args = ['-m', 'tests.integration.mirror_p022', '--split', tree.split, '--videos', *tree.videos,
            '--report', str(report)]
    if pipeline_execution is not None:
        args += ['--pipeline-execution', str(pipeline_execution)]
    tree.run(*args)
