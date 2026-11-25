"""
Write one test-split sweep config per dataset whose validation sweep is aggregated (profiling/README.md "Test split"):
the validation sweep config (experiments/configs/profiling/<dataset>.yaml) with videoset 'test', results under
TEST_OUTPUT_ROOT, and `configurations` listing the validation Pareto configurations (pareto.json: the union of the
fronts of every ablation condition).  Run in the polyis container after profiling.aggregate:

    python -m experiments.tools.profiling.make_test_configs [--datasets caldot1-y05 ...]
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from profiling.config import load_sweep_config

ROOT = Path(__file__).resolve().parents[2]   # experiments/
VALID = ROOT / 'configs' / 'profiling'
OUT = ROOT / 'configs' / 'profiling-test'
# Test runs go to <TEST_OUTPUT_ROOT>/<dataset>/profiling, beside the validation sweeps' <dataset> directories.
TEST_OUTPUT_ROOT = '/polyis-data/profiling/test'


def parse_args():
    parser = argparse.ArgumentParser(description='Write test-split sweep configs of the validation Pareto configurations.')
    parser.add_argument('--datasets', nargs='+', default=None)
    return parser.parse_args()


def on_test_split(valid: dict, pareto: list[dict]) -> dict:
    """The validation sweep document rerun on the test split for the Pareto configurations only."""
    ids = [p['config_id'] for p in pareto]
    if not ids:
        raise ValueError('the validation Pareto front is empty')
    # Everything else (grid, artifacts, references, system) stays the validation sweep's, so each test run uses
    # exactly the execution config that was profiled.
    return {**valid, 'videoset': 'test', 'output_root': TEST_OUTPUT_ROOT, 'configurations': sorted(ids)}


def main() -> None:
    args = parse_args()
    datasets = args.datasets or sorted(p.stem for p in VALID.glob('*.yaml'))
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        valid_path = VALID / f'{dataset}.yaml'
        pareto_path = os.path.join(load_sweep_config(str(valid_path)).profiling_dir, 'pareto.json')
        if not os.path.exists(pareto_path):
            raise SystemExit(f'{dataset}: no {pareto_path}; run profiling.aggregate on its validation sweep first')
        pareto = json.load(open(pareto_path))
        doc = on_test_split(yaml.safe_load(valid_path.read_text()), pareto)
        path = OUT / f'{dataset}.yaml'
        header = (f'# {dataset} test split: the {len(pareto)} validation Pareto configurations of {valid_path.name} '
                  f'(written by experiments/tools/profiling/make_test_configs.py from {pareto_path}).\n')
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
        # Every id must be a configuration of the grid, and every artifact must exist.
        load_sweep_config(str(path))
        print(f'{path.relative_to(ROOT.parent)}: {len(pareto)} configurations')


if __name__ == '__main__':
    main()
