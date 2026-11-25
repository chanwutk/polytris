"""profiling/reduction.py: tolerance groups and the duplicate map the sweep's dedup uses."""

import numpy as np
import pytest

from profiling.config import configurations, parse_sweep_config
from profiling.reduction import duplicates, group_tolerances
from tests.profiling.fixtures import GRID_H, GRID_W, rate_table, sweep_doc


def test_groups_follow_identical_gaps_on_mask_tiles():
    table = rate_table({0.4: 8, 0.6: 6, 0.8: 2})
    mask = np.ones((GRID_H, GRID_W), dtype=bool)
    # At s=1 every tolerance's gaps differ; at s=4 0.6 and 0.8 both become 1 everywhere.
    assert [g.tolerances for g in group_tolerances(table, mask, [0.4, 0.6, 0.8], 1)] == [(0.4,), (0.6,), (0.8,)]
    groups = group_tolerances(table, mask, [0.4, 0.6, 0.8], 4)
    assert [g.tolerances for g in groups] == [(0.4,), (0.6, 0.8)]
    assert [g.all_ones for g in groups] == [False, True]


def test_tiles_outside_the_mask_do_not_split_groups():
    table = rate_table({0.4: 8, 0.6: 8})
    table[0, 0, :] = [1, 3, 5, 7, 9, 11, 13, 15, 17]   # tile (0, 0) differs per tolerance
    mask = np.ones((GRID_H, GRID_W), dtype=bool)
    assert len(group_tolerances(table, mask, [0.4, 0.6], 1)) == 2
    mask[0, 0] = False
    assert [g.tolerances for g in group_tolerances(table, mask, [0.4, 0.6], 1)] == [(0.4, 0.6)]


def test_duplicates_map_each_merged_configuration_to_its_representative(tmp_path):
    cfg = parse_sweep_config(sweep_doc(tmp_path))
    configs = configurations(cfg)
    dup = duplicates(cfg, configs)
    # s=4 with tolerance 0.8 duplicates s=4 with 0.6, once per tracker; nothing else merges.
    assert dup == {'C_60_4_080_r050_none_s100_sortcython': 'C_60_4_060_r050_none_s100_sortcython',
                   'C_60_4_080_r050_none_s100_bytetrackcython': 'C_60_4_060_r050_none_s100_bytetrackcython'}
    # Representatives are configurations of the sweep, and none of them is itself a duplicate.
    ids = {c.id for c in configs}
    assert set(dup.values()) <= ids and not set(dup.values()) & set(dup)


def test_trackers_are_grouped_with_their_own_tables(tmp_path):
    tables = {'sortcython': rate_table({0.4: 8, 0.6: 6, 0.8: 2}), 'bytetrackcython': rate_table({0.4: 3, 0.6: 3, 0.8: 3})}
    cfg = parse_sweep_config(sweep_doc(tmp_path, tables=tables, sample_rates=[1]))
    dup = duplicates(cfg, configurations(cfg))
    # Only bytetrack's table makes every tolerance identical (at s=1).
    assert dup == {'C_60_1_060_r050_none_s100_bytetrackcython': 'C_60_1_040_r050_none_s100_bytetrackcython',
                   'C_60_1_080_r050_none_s100_bytetrackcython': 'C_60_1_040_r050_none_s100_bytetrackcython'}


def test_no_duplicates_when_dedup_is_off_or_nothing_prunes(tmp_path):
    off = parse_sweep_config(sweep_doc(tmp_path, dedup_pruning=False))
    assert duplicates(off, configurations(off)) == {}
    unpruned = parse_sweep_config(sweep_doc(tmp_path, mistrack_tolerances=[None]))
    assert duplicates(unpruned, configurations(unpruned)) == {}


def test_a_table_with_the_wrong_grid_is_an_error(tmp_path):
    doc = sweep_doc(tmp_path)
    np.save(doc['artifacts']['max_rate_tables']['sortcython'], np.ones((5, 5, 9), dtype=np.int64))
    cfg = parse_sweep_config(doc)
    with pytest.raises(ValueError, match='grid'):
        duplicates(cfg, configurations(cfg))


def test_a_selection_without_the_representative_runs_its_first_selected_member(tmp_path):
    # At s=1 every bytetrack tolerance is one group; its representative (0.4) is not selected.
    tables = {'sortcython': rate_table({0.4: 8, 0.6: 6, 0.8: 2}), 'bytetrackcython': rate_table({0.4: 3, 0.6: 3, 0.8: 3})}
    doc = sweep_doc(tmp_path, tables=tables, sample_rates=[1])
    cfg = parse_sweep_config({**doc, 'configurations': ['C_60_1_080_r050_none_s100_bytetrackcython',
                                                        'C_60_1_060_r050_none_s100_bytetrackcython',
                                                        'C_60_1_080_r050_none_s100_sortcython']})
    # 0.6 runs (first in grid order) and 0.8 follows it; sort's 0.8 is alone in its group.
    assert duplicates(cfg, configurations(cfg)) == {
        'C_60_1_080_r050_none_s100_bytetrackcython': 'C_60_1_060_r050_none_s100_bytetrackcython'}


def test_a_selected_duplicate_without_its_representative_runs_itself(tmp_path):
    cfg = parse_sweep_config({**sweep_doc(tmp_path), 'configurations': ['C_60_4_080_r050_none_s100_sortcython']})
    assert duplicates(cfg, configurations(cfg)) == {}
