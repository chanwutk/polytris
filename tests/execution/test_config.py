"""--config validation tests (PLAN.md "Config")."""

import copy

import pytest
import yaml

from execution.config import MISTRACK_TOLERANCES, ConfigError, load_config, parse_config


@pytest.fixture
def paths(tmp_path):
    """Real files for every artifact path."""
    names = ['weights.pth', 'bench.jsonl', 'never.npy', 'rates.npy', 'model.pt', 'det.cfg', 'retina.json']
    out = {}
    for n in names:
        (tmp_path / n).write_text('x')
        out[n] = str(tmp_path / n)
    return out


@pytest.fixture
def valid(paths):
    """A pruning config with every field set."""
    return {
        'params': {'sample_rate': 4, 'relevance_threshold': 0.5, 'mistrack_tolerance': 0.4, 'tile_padding': 'none',
                   'tracker': 'sortcython', 'tile_size': 60, 'canvas_scale': 1.0,
                   'canvas_format': 'jpeg'},
        'artifacts': {
            'classifier': {'name': 'ShuffleNet05', 'weights': paths['weights.pth'],
                           'compilation_benchmark': paths['bench.jsonl']},
            'never_relevant_bitmap': paths['never.npy'],
            'max_rate_table': paths['rates.npy'],
            'detector': {'type': 'ultralytics', 'model_path': paths['model.pt']},
        },
        'system': {'queue_maxsize': 2, 'stages': {
            'decode': {'procs': 4}, 'classify': {'procs': 1, 'gpu': 0}, 'prune': {'procs': 1},
            'pack': {'procs': 2}, 'detect': {'procs': 1, 'gpu': 1}, 'track': {'procs': 2}}},
    }


def no_prune(cfg):
    """The same config with pruning off."""
    cfg = copy.deepcopy(cfg)
    cfg['params']['mistrack_tolerance'] = None
    del cfg['artifacts']['max_rate_table']
    del cfg['system']['stages']['prune']
    return cfg


def test_valid_config_loads_from_yaml(valid, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump(valid))
    cfg = load_config(str(path))
    assert cfg.use_prune and cfg.params.mistrack_tolerance == 0.4
    assert cfg.system.stages['classify'].gpu == 0 and cfg.system.stages['decode'].gpu is None
    assert cfg.artifacts.detector.options == {'model_path': valid['artifacts']['detector']['model_path']}


def test_no_prune_config_loads(valid):
    cfg = parse_config(no_prune(valid))
    assert not cfg.use_prune and cfg.artifacts.max_rate_table is None and 'prune' not in cfg.system.stages


def test_ints_are_accepted_as_floats(valid):
    valid['params']['canvas_scale'] = 1
    cfg = parse_config(valid)
    assert isinstance(cfg.params.canvas_scale, float) and cfg.params.canvas_scale == 1.0


def test_config_is_frozen(valid):
    cfg = parse_config(valid)
    with pytest.raises(Exception):
        cfg.params.sample_rate = 8  # type: ignore[misc]


@pytest.mark.parametrize('section, key', [
    ('params', 'tracker'), ('artifacts', 'never_relevant_bitmap'), ('system', 'queue_maxsize')])
def test_missing_keys_are_errors(valid, section, key):
    del valid[section][key]
    with pytest.raises(ConfigError, match=f'{section}: missing keys .*{key}'):
        parse_config(valid)


@pytest.mark.parametrize('where', [[], ['params'], ['artifacts', 'classifier'], ['system', 'stages', 'pack']])
def test_unknown_keys_are_errors(valid, where):
    node = valid
    for k in where:
        node = node[k]
    node['typo'] = 1
    with pytest.raises(ConfigError, match='unknown keys'):
        parse_config(valid)


@pytest.mark.parametrize('key, value, message', [
    ('sample_rate', 4.0, 'expected int, got float'),
    ('sample_rate', True, 'got bool'),
    ('sample_rate', 0, 'must be >= 1'),
    ('relevance_threshold', 1.5, r'must be in \[0, 1\]'),
    ('relevance_threshold', '0.5', 'expected float, got str'),
    ('mistrack_tolerance', 0.2, 'must be null or one of'),
    ('tile_padding', 'diagonal', 'must be one of'),
    ('tile_size', -60, 'must be >= 1'),
    ('canvas_scale', 0, 'must be > 0'),
    ('tracker', None, 'expected str, got NoneType'),
    ('canvas_format', 'png', 'must be one of'),
])
def test_bad_params_are_errors(valid, key, value, message):
    valid['params'][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(valid)


def test_missing_artifact_path_is_an_error(valid):
    valid['artifacts']['classifier']['weights'] = '/nonexistent/model.pth'
    with pytest.raises(ConfigError, match='artifacts.classifier.weights: /nonexistent/model.pth does not exist'):
        parse_config(valid)


def test_max_rate_table_required_exactly_when_pruning(valid):
    missing = copy.deepcopy(valid)
    del missing['artifacts']['max_rate_table']
    with pytest.raises(ConfigError, match='required when params.mistrack_tolerance is set'):
        parse_config(missing)
    extra = no_prune(valid)
    extra['artifacts']['max_rate_table'] = valid['artifacts']['max_rate_table']
    with pytest.raises(ConfigError, match='must be omitted'):
        parse_config(extra)


def test_prune_stage_required_exactly_when_pruning(valid):
    missing = copy.deepcopy(valid)
    del missing['system']['stages']['prune']
    with pytest.raises(ConfigError, match='missing keys .*prune'):
        parse_config(missing)
    extra = no_prune(valid)
    extra['system']['stages']['prune'] = {'procs': 1}
    with pytest.raises(ConfigError, match='unknown keys .*prune'):
        parse_config(extra)


def test_prune_is_limited_to_one_worker(valid):
    valid['system']['stages']['prune']['procs'] = 2
    with pytest.raises(ConfigError, match='prune.procs: must be 1'):
        parse_config(valid)


@pytest.mark.parametrize('stage, entry, message', [
    ('classify', {'procs': 1}, 'gpu: required for classify'),
    ('detect', {'procs': 1}, 'gpu: required for detect'),
    ('decode', {'procs': 1, 'gpu': 0}, 'gpu: not allowed for decode'),
    ('pack', {'procs': 0}, 'procs: must be >= 1'),
    ('classify', {'procs': 1, 'gpu': -1}, 'gpu: must be >= 0'),
])
def test_bad_stage_entries_are_errors(valid, stage, entry, message):
    valid['system']['stages'][stage] = entry
    with pytest.raises(ConfigError, match=message):
        parse_config(valid)


def test_queue_maxsize_must_be_positive(valid):
    valid['system']['queue_maxsize'] = 0
    with pytest.raises(ConfigError, match='queue_maxsize: must be >= 1'):
        parse_config(valid)


def test_each_detector_type_takes_its_own_fields(valid, paths):
    for spec in [
        {'type': 'retina', 'config_path': paths['retina.json']},
        {'type': 'yolov3', 'config_path': paths['det.cfg'], 'model_path': paths['model.pt'],
         'detector_label': 'caldot', 'width': 704, 'height': 480},
        {'type': 'torchvision', 'model_path': paths['model.pt'], 'model_type': 'fasterrcnn'},
    ]:
        valid['artifacts']['detector'] = spec
        assert parse_config(valid).artifacts.detector.type == spec['type']


@pytest.mark.parametrize('spec, message', [
    ({'type': 'yolo99', 'model_path': 'x'}, 'type: must be one of'),
    ({'model_path': 'x'}, 'with a string "type"'),
    ({'type': 'ultralytics'}, 'missing keys .*model_path'),
    ({'type': 'ultralytics', 'model_path': '/nope.pt'}, 'does not exist'),
])
def test_bad_detector_specs_are_errors(valid, spec, message):
    valid['artifacts']['detector'] = spec
    with pytest.raises(ConfigError, match=message):
        parse_config(valid)


def test_yolov3_numeric_fields_are_typed(valid, paths):
    valid['artifacts']['detector'] = {'type': 'yolov3', 'config_path': paths['det.cfg'], 'model_path': paths['model.pt'],
                                      'detector_label': 'caldot', 'width': '704', 'height': 480}
    with pytest.raises(ConfigError, match='detector.width: expected int, got str'):
        parse_config(valid)


def test_non_mapping_documents_are_errors(tmp_path):
    for text in ['', '- a\n- b\n', 'just a string']:
        path = tmp_path / 'c.yaml'
        path.write_text(text)
        with pytest.raises(ConfigError, match='expected a mapping'):
            load_config(str(path))


def test_invalid_yaml_is_a_config_error(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('params: [unclosed\n')
    with pytest.raises(ConfigError, match='invalid YAML'):
        load_config(str(path))


def test_mistrack_tolerances_match_p022():
    # The config's list must stay the max_rate_table axis p022 indexes into.
    from scripts.p022_exec_prune_polyominoes import _ALL_ACCURACY_THRESHOLDS
    assert list(MISTRACK_TOLERANCES) == _ALL_ACCURACY_THRESHOLDS
