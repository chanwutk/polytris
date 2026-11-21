"""Sweep config validation (profiling/PLAN.md 9A)."""

import copy

import pytest
import yaml

from execution2.config import ConfigError
from profiling.config import load_sweep_config, parse_sweep_config


@pytest.fixture
def files(tmp_path):
    out = {}
    for n in ['w.pth', 'bench.jsonl', 'never.npy', 'sort.npy', 'byte.npy', 'model.pt']:
        (tmp_path / n).write_text('x')
        out[n] = str(tmp_path / n)
    (tmp_path / 'ref').mkdir()
    out['ref'] = str(tmp_path / 'ref')
    out['out'] = str(tmp_path / 'out')
    return out


@pytest.fixture
def valid(files):
    return {
        'dataset': 'caldot1-y05', 'videoset': 'valid', 'output_root': files['out'],
        'params': {'classifier': 'ShuffleNet05', 'tile_size': 60, 'canvas_format': 'jpeg',
                   'sample_rates': [1, 2, 4, 8, 16], 'relevance_thresholds': [0.25, 0.5, 0.75],
                   'mistrack_tolerances': [None, 0.4, 0.6, 0.8], 'tile_paddings': ['none', 'tr', 'bl', 'square'],
                   'canvas_scales': [1.0], 'trackers': ['sortcython', 'bytetrackcython']},
        'artifacts': {'classifier_weights': files['w.pth'], 'compilation_benchmark': files['bench.jsonl'],
                      'never_relevant_bitmap': files['never.npy'],
                      'max_rate_tables': {'sortcython': files['sort.npy'], 'bytetrackcython': files['byte.npy']},
                      'detector': {'type': 'ultralytics', 'model_path': files['model.pt']}},
        'evaluation': {'reference_dir': files['ref'], 'reference_file': '002_naive/tracking.jsonl'},
        'system': {'queue_maxsize': 2, 'save_queue_maxsize': 64, 'max_videos_in_flight': 4, 'stages': {
            'decode': {'procs': 2}, 'classify': {'procs': 1, 'gpus': [0]}, 'prune': {'procs': 1},
            'pack': {'procs': 4}, 'detect': {'procs': 7, 'gpus': [1, 2, 3, 4, 5, 6, 7]},
            'track': {'procs': 4}, 'evaluate': {'procs': 4, 'savers': 2}}},
    }


def test_valid_config_loads(valid, tmp_path):
    path = tmp_path / 's.yaml'
    path.write_text(yaml.safe_dump(valid))
    cfg = load_sweep_config(str(path))
    assert cfg.uses_prune and cfg.params.mistrack_tolerances == (None, 0.4, 0.6, 0.8)
    assert cfg.system.stages['detect'].gpus == (1, 2, 3, 4, 5, 6, 7) and cfg.system.stages['decode'].gpus == ()
    assert cfg.system.stages['evaluate'].savers == 2 and cfg.system.stages['pack'].savers == 1
    assert set(cfg.artifacts.max_rate_tables) == {'sortcython', 'bytetrackcython'}


def test_without_tolerances_prune_and_tables_are_omitted(valid):
    v = copy.deepcopy(valid)
    v['params']['mistrack_tolerances'] = [None]
    del v['artifacts']['max_rate_tables']
    del v['system']['stages']['prune']
    cfg = parse_sweep_config(v)
    assert not cfg.uses_prune and cfg.artifacts.max_rate_tables == {} and 'prune' not in cfg.system.stages


@pytest.mark.parametrize('mutate, message', [
    (lambda v: v['params'].update(sample_rates=[]), 'non-empty list'),
    (lambda v: v['params'].update(sample_rates=[4, 4]), 'distinct'),
    (lambda v: v['params'].update(sample_rates=[0]), 'must be >= 1'),
    (lambda v: v['params'].update(relevance_thresholds=[1.5]), r'in \[0, 1\]'),
    (lambda v: v['params'].update(mistrack_tolerances=[None, 0.2]), 'each must be null or one of'),
    (lambda v: v['params'].update(tile_paddings=['diag']), 'each must be one of'),
    (lambda v: v['params'].update(canvas_format='png'), 'canvas_format'),
    (lambda v: v['artifacts']['max_rate_tables'].pop('bytetrackcython'), 'missing keys'),
    (lambda v: v['system']['stages']['prune'].update(procs=2), 'must be 1'),
    (lambda v: v['system']['stages']['classify'].pop('gpus'), 'gpus: required'),
    (lambda v: v['system']['stages']['pack'].update(gpus=[0]), 'gpus: not allowed'),
    (lambda v: v['system']['stages']['detect'].update(gpus=[]), 'non-empty list'),
    (lambda v: v['system']['stages']['track'].update(savers=0), 'must be >= 1'),
    (lambda v: v['system'].update(max_videos_in_flight=0), 'must be >= 1'),
    (lambda v: v['system']['stages'].pop('evaluate'), 'missing keys'),
    (lambda v: v['params'].update(extra=1), 'unknown keys'),
])
def test_bad_configs_are_errors(valid, mutate, message):
    mutate(valid)
    with pytest.raises(ConfigError, match=message):
        parse_sweep_config(valid)


def test_output_root_may_not_contain_or_sit_inside_the_reference(valid, files):
    for root in [files['ref'] + '/..', files['ref'], files['ref'] + '/sub']:
        v = copy.deepcopy(valid)
        v['output_root'] = root
        with pytest.raises(ConfigError, match='contain one another'):
            parse_sweep_config(v)
