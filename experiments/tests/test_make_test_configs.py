"""experiments/tools/profiling/make_test_configs.py: validation Pareto configurations -> a test-split sweep config."""

import json
import sys

import pytest
import yaml

from experiments.tools.profiling import make_test_configs as m
from profiling.config import configurations, load_sweep_config, parse_sweep_config
from tests.profiling.fixtures import sweep_doc

PARETO = [{'id': 7, 'config_id': 'C_60_4_040_r050_none_s100_sortcython'},
          {'id': 0, 'config_id': 'C_60_1_r050_none_s100_sortcython'}]


def test_the_test_config_is_the_validation_sweep_on_the_test_split_for_the_pareto_ids(tmp_path):
    valid = sweep_doc(tmp_path)
    doc = m.on_test_split(valid, PARETO)
    assert doc['videoset'] == 'test' and doc['output_root'] == m.TEST_OUTPUT_ROOT
    assert doc['configurations'] == sorted(p['config_id'] for p in PARETO)
    # Everything else is the validation sweep's, so each test run uses the profiled execution config.
    assert {k: v for k, v in doc.items() if k not in ('videoset', 'output_root', 'configurations')} == \
        {k: v for k, v in valid.items() if k not in ('videoset', 'output_root')}
    cfg = parse_sweep_config({**doc, 'output_root': str(tmp_path / 'test-out')})
    valid_executions = {c.id: c.execution for c in configurations(parse_sweep_config(valid))}
    assert {c.id: c.execution for c in configurations(cfg)} == {p['config_id']: valid_executions[p['config_id']]
                                                               for p in PARETO}
    # The input is left unchanged.
    assert valid['videoset'] == 'valid' and 'configurations' not in valid


def test_an_empty_front_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='empty'):
        m.on_test_split(sweep_doc(tmp_path), [])


def test_main_writes_a_loadable_config_per_aggregated_dataset(tmp_path, monkeypatch):
    valid_dir, out_dir = tmp_path / 'valid', tmp_path / 'test'
    valid_dir.mkdir()
    (valid_dir / 'ds.yaml').write_text(yaml.safe_dump(sweep_doc(tmp_path)))
    pareto = tmp_path / 'out' / 'ds' / 'profiling' / 'pareto.json'
    pareto.parent.mkdir(parents=True)
    pareto.write_text(json.dumps(PARETO))
    monkeypatch.setattr(m, 'VALID', valid_dir)
    monkeypatch.setattr(m, 'OUT', out_dir)
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'experiments')
    monkeypatch.setattr(sys, 'argv', ['make_test_configs'])
    m.main()
    cfg = load_sweep_config(str(out_dir / 'ds.yaml'))
    assert cfg.videoset == 'test' and cfg.selected == frozenset(p['config_id'] for p in PARETO)
    assert (out_dir / 'ds.yaml').read_text().startswith('# ds test split: the 2 validation Pareto configurations')


def test_main_stops_when_a_dataset_is_not_aggregated(tmp_path, monkeypatch):
    valid_dir = tmp_path / 'valid'
    valid_dir.mkdir()
    (valid_dir / 'ds.yaml').write_text(yaml.safe_dump(sweep_doc(tmp_path)))
    monkeypatch.setattr(m, 'VALID', valid_dir)
    monkeypatch.setattr(m, 'OUT', tmp_path / 'test')
    monkeypatch.setattr(sys, 'argv', ['make_test_configs', '--datasets', 'ds'])
    with pytest.raises(SystemExit, match='run profiling.aggregate'):
        m.main()
