"""profiling/config.py: expansion into execution configs, validation through execution, and the sweep hash."""

import copy

import pytest
import yaml

from execution.config import ConfigError, parse_config
from profiling.config import configurations, load_sweep_config, parse_sweep_config, sweep_hash
from tests.profiling.fixtures import sweep_doc


def test_expansion_covers_the_grid_with_unique_ids_and_valid_execution_configs(tmp_path):
    cfg = parse_sweep_config(sweep_doc(tmp_path))
    configs = configurations(cfg)
    # 2 sample rates x 1 T_r x 4 tolerances x 1 padding x 1 scale x 2 trackers.
    assert len(configs) == 16
    assert len({c.id for c in configs}) == 16
    # Each one is an execution config execution accepts, with its own parameters.
    for c in configs:
        ex = parse_config(c.execution)
        assert (ex.params.sample_rate, ex.params.mistrack_tolerance, ex.params.tracker) == (
            c.params['sample_rate'], c.params['mistrack_tolerance'], c.params['tracker'])


def test_ids_are_the_scripts_tracks_directory_names(tmp_path):
    ids = [c.id for c in configurations(parse_sweep_config(sweep_doc(tmp_path)))]
    assert 'C_60_1_r050_none_s100_sortcython' in ids          # no pruning: the tolerance is left out
    assert 'C_60_4_040_r050_none_s100_bytetrackcython' in ids


def test_unpruned_configurations_drop_the_table_and_the_prune_stage(tmp_path):
    configs = configurations(parse_sweep_config(sweep_doc(tmp_path)))
    unpruned = [c for c in configs if c.params['mistrack_tolerance'] is None]
    pruned = [c for c in configs if c.params['mistrack_tolerance'] is not None]
    assert unpruned and pruned
    for c in unpruned:
        assert 'max_rate_table' not in c.execution['artifacts'] and 'prune' not in c.execution['system']['stages']
    for c in pruned:
        # The configuration's own tracker's table.
        assert c.execution['artifacts']['max_rate_table'].endswith(f"{c.params['tracker']}.npy")
        assert c.execution['system']['stages']['prune'] == {'procs': 1}


def test_expansion_does_not_share_mutable_sections_between_configurations(tmp_path):
    configs = configurations(parse_sweep_config(sweep_doc(tmp_path)))
    configs[0].execution['system']['stages']['decode']['procs'] = 99
    assert configs[1].execution['system']['stages']['decode']['procs'] == 2


def test_a_bad_configuration_is_reported_by_its_id(tmp_path):
    doc = sweep_doc(tmp_path)
    # Execution requires a prune stage for pruned configurations.
    del doc['system']['stages']['prune']
    with pytest.raises(ConfigError, match=r'configuration C_60_1_040_.*prune'):
        parse_sweep_config(doc)


def test_execution_range_checks_apply_to_every_value(tmp_path):
    with pytest.raises(ConfigError, match='relevance_threshold'):
        parse_sweep_config(sweep_doc(tmp_path, relevance_thresholds=[0.5, 1.5]))
    with pytest.raises(ConfigError, match='mistrack_tolerance'):
        parse_sweep_config(sweep_doc(tmp_path, mistrack_tolerances=[0.45]))


@pytest.mark.parametrize('key,value,match', [
    ('sample_rates', [], 'non-empty list'),
    ('sample_rates', [1, 1], 'distinct'),
    ('trackers', 'sortcython', 'non-empty list'),
    ('dedup_pruning', 1, 'expected bool'),
])
def test_swept_lists_and_the_dedup_flag_are_checked(tmp_path, key, value, match):
    with pytest.raises(ConfigError, match=match):
        parse_sweep_config(sweep_doc(tmp_path, **{key: value}))


def test_unknown_keys_and_missing_tables_are_rejected(tmp_path):
    doc = sweep_doc(tmp_path)
    doc['params']['extra'] = 1
    with pytest.raises(ConfigError, match='extra'):
        parse_sweep_config(doc)
    doc = sweep_doc(tmp_path)
    del doc['artifacts']['max_rate_tables']['bytetrackcython']
    with pytest.raises(ConfigError, match='max_rate_tables'):
        parse_sweep_config(doc)
    # Tables without any swept tolerance are a mistake too.
    doc = sweep_doc(tmp_path, mistrack_tolerances=[None])
    doc['artifacts']['max_rate_tables'] = {'sortcython': doc['artifacts']['never_relevant_bitmap']}
    with pytest.raises(ConfigError, match='must be omitted'):
        parse_sweep_config(doc)


def test_output_root_may_not_contain_or_sit_inside_the_references(tmp_path):
    doc = sweep_doc(tmp_path)
    doc['output_root'] = doc['evaluation']['reference_dir']
    with pytest.raises(ConfigError, match='contain one another'):
        parse_sweep_config(doc)
    doc['output_root'] = str(tmp_path)
    with pytest.raises(ConfigError, match='contain one another'):
        parse_sweep_config(doc)


def test_the_hash_is_stable_and_changes_with_the_sweep(tmp_path):
    doc = sweep_doc(tmp_path)
    a, b = parse_sweep_config(doc), parse_sweep_config(copy.deepcopy(doc))
    assert sweep_hash(a) == sweep_hash(b)
    changed = copy.deepcopy(doc)
    changed['params']['sample_rates'] = [1]
    assert sweep_hash(parse_sweep_config(changed)) != sweep_hash(a)


def test_load_reads_yaml_and_reports_bad_yaml(tmp_path):
    path = tmp_path / 'sweep.yaml'
    path.write_text(yaml.safe_dump(sweep_doc(tmp_path)))
    assert load_sweep_config(str(path)).dataset == 'ds'
    path.write_text('params: [unclosed')
    with pytest.raises(ConfigError, match='invalid YAML'):
        load_sweep_config(str(path))
