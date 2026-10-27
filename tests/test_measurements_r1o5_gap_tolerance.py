import argparse
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

import measurements.common as common
from measurements import r1o5_gap_tolerance as gt
from polyis.io import cache, store


# p016's candidate gaps and stored thresholds (percent).
GAPS = [1, 2, 4, 8, 16]
THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 95, 100]


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    # Point the cache, the dataset store, and the results root at a scratch directory.
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(store, 'DATASETS_DIR', str(tmp_path / 'datasets'))
    monkeypatch.setattr(common, 'RESULTS_ROOT', tmp_path / 'results')
    return tmp_path


def p016_accuracy(counts):
    # p016's float32 Laplace-smoothed accuracy.
    correct = counts[..., 0].astype(np.float32)
    incorrect = counts[..., 1].astype(np.float32)
    return (correct + 1) / (correct + incorrect + 2)


def pipeline_table(counts, thresholds=THRESHOLDS, gaps=GAPS):
    # p016's own rule (scripts/p016_tune_track_rate.py): highest gap meeting t on the float32 accuracy, default 1.
    accuracy = p016_accuracy(counts)
    table = np.full((*counts.shape[1:3], len(thresholds)), gaps[0], dtype=np.int32)
    for k, pct in enumerate(thresholds):
        for g in range(len(gaps)):
            table[:, :, k] = np.where(accuracy[g] >= pct / 100.0, gaps[g], table[:, :, k])
    return table


def tile_counts(*tiles):
    # Counts array (gaps, 1, n tiles, 2) from one [[correct, incorrect] per gap] list per tile.
    return np.array(tiles, dtype=np.int64).transpose(1, 0, 2)[:, None, :, :]


def make_profile(counts, thresholds=THRESHOLDS, table=None):
    # A profile dict as load_profile returns it (without input records); the table defaults to p016's rule.
    return dict(gaps=GAPS, counts=counts, thresholds=thresholds, accuracy=p016_accuracy(counts),
                table=pipeline_table(counts, thresholds) if table is None else table)


def write_profile(directory, counts, thresholds=THRESHOLDS, table=None, gaps=GAPS, partials=None):
    # Write a p016 profile directory: sample_rates.json, counts/accuracy/max_rate_table .npy, max_rate_XXX.txt, partial/.
    directory.mkdir(parents=True, exist_ok=True)
    table = pipeline_table(counts, thresholds, gaps) if table is None else table
    (directory / 'sample_rates.json').write_text(json.dumps(gaps))
    np.save(directory / 'counts.npy', counts)
    np.save(directory / 'accuracy.npy', p016_accuracy(counts))
    np.save(directory / 'max_rate_table.npy', table)
    for k, pct in enumerate(thresholds):
        np.savetxt(directory / f'max_rate_{pct:03d}.txt', table[:, :, k], fmt='%2d')
    for i, part in enumerate(partials or []):
        (directory / 'partial').mkdir(exist_ok=True)
        np.save(directory / 'partial' / f'v{i}.npy', part)
    return directory


def write_scores(dataset, stage, video, name, grids):
    # A 020 or 022 score.jsonl at its cache path: one line per frame with size, hex, and frame index.
    path = cache.exec(dataset, stage, video, name, 'score', 'score.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = np.asarray(grids, dtype=np.uint8)
    path.write_text(''.join(json.dumps(dict(classification_size=list(g.shape), classification_hex=g.tobytes().hex(),
                                            idx=i)) + '\n' for i, g in enumerate(grids)))
    return path


# Three tiles used across tests:
# never relevant (no associations at any gap);
NEVER = [[0, 0]] * 5
# tracked well at every gap (smoothed accuracy above 0.95 everywhere);
EASY = [[1000, 0], [500, 0], [250, 0], [125, 0], [62, 0]]
# tracked well at gaps 1-2 (accuracy 0.99), at 0.75 at gap 4, at 0.5 at gap 8, at 0.1 at gap 16.
HARD = [[100, 0], [99, 0], [29, 9], [9, 9], [0, 8]]


def tile_gaps(con):
    # {(col, threshold_pct): row of tile_gaps} for a one-row grid.
    return {(r['col'], r['threshold_pct']): r for r in gt.query_dicts(con, 'SELECT * FROM tile_gaps')}


def test_tolerance_key_is_exact():
    # The tolerance is 1 - t, formatted without float noise.
    assert [gt.tolerance_key(t) for t in (30, 80, 95, 100)] == ['0.70', '0.20', '0.05', '0.00']


def test_gap_key_sorts_numerically():
    # Zero-padded keys keep numeric order under JSON's sorted string keys.
    keys = [gt.gap_key(g) for g in GAPS]
    assert keys == ['01', '02', '04', '08', '16'] and sorted(keys) == keys


def test_load_profile_reads_thresholds_from_file_names(tmp_path):
    # Thresholds come from the slice file names (any subset p016 stored), and the arrays round-trip.
    counts = tile_counts(EASY, HARD)
    directory = write_profile(tmp_path / 'p', counts, thresholds=[30, 80, 100])
    profile = gt.load_profile(directory)
    assert profile['thresholds'] == [30, 80, 100]
    assert profile['gaps'] == GAPS and np.array_equal(profile['counts'], counts)
    # Input records name the files read and their versions; no partial/ directory means no check.
    inputs = profile['inputs']
    assert inputs['max_rate_table']['path'].endswith('max_rate_table.npy') and len(inputs['counts']['sha256']) == 64
    assert inputs['partial_files'] == 0 and inputs['counts_equal_partial_sum'] is None


def test_load_profile_records_partial_sum_check(tmp_path):
    # counts.npy equal to the sum of the per-video partials is recorded as True, a stale partial as False.
    counts = tile_counts(EASY, HARD)
    half = counts // 2
    good = gt.load_profile(write_profile(tmp_path / 'good', counts, partials=[half, counts - half]))
    assert good['inputs']['partial_files'] == 2 and good['inputs']['counts_equal_partial_sum'] is True
    stale = gt.load_profile(write_profile(tmp_path / 'stale', counts, partials=[half, counts - half, half]))
    assert stale['inputs']['counts_equal_partial_sum'] is False


def test_load_profile_rejects_table_not_in_ascending_threshold_order(tmp_path):
    # p016 stores slices in ascending threshold order; a table written as 80, 30, 100 disagrees with its files.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY, HARD), thresholds=[80, 30, 100])
    with pytest.raises(AssertionError, match='disagree'):
        gt.load_profile(directory)


@pytest.mark.parametrize('name', ['counts.npy', 'accuracy.npy', 'max_rate_table.npy', 'sample_rates.json'])
def test_load_profile_missing_file(tmp_path, name):
    # A profile without one of its files fails with the p016 hint, naming the file.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY))
    (directory / name).unlink()
    with pytest.raises(AssertionError, match=f'p016_tune_track_rate.py.*missing.*{name}'):
        gt.load_profile(directory)


def test_load_profile_without_slice_files(tmp_path):
    # No max_rate_XXX.txt files means the slice order is unknown.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY))
    for path in directory.glob('max_rate_*.txt'):
        path.unlink()
    with pytest.raises(AssertionError, match='bad threshold files'):
        gt.load_profile(directory)


def test_load_profile_slice_file_disagrees_with_table(tmp_path):
    # A slice file that differs from its table slice means the order cannot be trusted.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY, HARD))
    np.savetxt(directory / 'max_rate_080.txt', np.array([[1, 1]]), fmt='%2d')
    with pytest.raises(AssertionError, match='disagree'):
        gt.load_profile(directory)


def test_load_profile_rejects_gap_outside_candidates(tmp_path):
    # A cached gap of 3 is not a candidate gap.
    counts = tile_counts(EASY)
    table = pipeline_table(counts)
    table[0, 0, 0] = 3
    directory = write_profile(tmp_path / 'p', counts, table=table)
    with pytest.raises(AssertionError, match='outside'):
        gt.load_profile(directory)


@pytest.mark.parametrize('gaps', [[2, 4, 8, 16, 32], [1, 4, 2, 8, 16], [1, 1, 2, 4, 8],
                                  [1.0, 2.0, 4.0, 8.0, 16.0], [True, 2, 4, 8, 16]])
def test_load_profile_rejects_bad_candidate_gaps(tmp_path, gaps):
    # Gaps must be integers that start at 1 and strictly increase (floats and booleans are not gaps).
    directory = write_profile(tmp_path / 'p', tile_counts(EASY), gaps=[1, 2, 4, 8, 16])
    (directory / 'sample_rates.json').write_text(json.dumps(gaps))
    with pytest.raises(AssertionError, match='bad candidate gaps'):
        gt.load_profile(directory)


def test_load_profile_rejects_float_counts(tmp_path):
    # Exact integer arithmetic needs integer tallies.
    counts = tile_counts(EASY)
    directory = write_profile(tmp_path / 'p', counts)
    np.save(directory / 'counts.npy', counts.astype(np.float64))
    with pytest.raises(AssertionError, match='not an integer type'):
        gt.load_profile(directory)


def test_load_profile_rejects_accuracy_not_from_counts(tmp_path):
    # accuracy.npy (what Fig. 7 reads) must be p016's formula on counts.npy.
    counts = tile_counts(EASY, HARD)
    directory = write_profile(tmp_path / 'p', counts)
    np.save(directory / 'accuracy.npy', p016_accuracy(counts) * np.float32(0.99))
    with pytest.raises(AssertionError, match='accuracy.npy'):
        gt.load_profile(directory)


def test_load_profile_rejects_counts_shape_mismatch(tmp_path):
    # Counts with four gaps cannot match five candidate gaps.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY))
    np.save(directory / 'counts.npy', np.zeros((4, 1, 1, 2), dtype=np.int64))
    with pytest.raises(AssertionError, match='counts shape'):
        gt.load_profile(directory)


def test_load_profile_rejects_table_shape_mismatch(tmp_path):
    # A table over a 1x2 grid does not match counts over a 1x1 grid.
    directory = write_profile(tmp_path / 'p', tile_counts(EASY))
    np.save(directory / 'max_rate_table.npy', np.ones((1, 2, len(THRESHOLDS)), dtype=np.int32))
    with pytest.raises(AssertionError, match='table shape'):
        gt.load_profile(directory)


def test_load_profile_rejects_negative_counts(tmp_path):
    # Counts are tallies and cannot be negative.
    counts = tile_counts(EASY)
    counts[0, 0, 0, 0] = -1
    directory = write_profile(tmp_path / 'p', counts, table=np.ones((1, 1, len(THRESHOLDS)), dtype=np.int32))
    with pytest.raises(AssertionError, match='negative'):
        gt.load_profile(directory)


@pytest.mark.parametrize('transpose', [False, True])
def test_load_profile_single_row_and_single_column_grids(tmp_path, transpose):
    # One-row and one-column grids survive the text round trip (savetxt/loadtxt drop dimensions otherwise).
    counts = tile_counts(EASY, HARD, NEVER)
    if transpose:
        # Transpose to a 3x1 grid.
        counts = np.ascontiguousarray(counts.transpose(0, 2, 1, 3))
    directory = write_profile(tmp_path / 'p', counts)
    assert gt.load_profile(directory)['table'].shape[:2] == counts.shape[1:3]


def test_create_tables_requires_profiles(con):
    # Nothing to measure is an error, not an empty report.
    with pytest.raises(AssertionError, match='no p016 profiles'):
        gt.create_tables(con, {})


def test_gaps_follow_rule_and_tile_classes(con):
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(NEVER, EASY, HARD))})
    rows = tile_gaps(con)
    # Never-relevant tiles sit at the 0.5 prior: gap 16 when t <= 0.5, gap 1 above.
    assert [rows[(0, t)]['gap'] for t in THRESHOLDS] == [16, 16, 16, 1, 1, 1, 1, 1, 1]
    assert rows[(0, 30)]['tile_class'] == 'never_relevant' and rows[(0, 30)]['associations'] == 0
    # The easy tile gets the largest gap at every threshold below 1; smoothing never reaches accuracy 1,
    # so t = 1 puts every tile at gap 1.
    assert {rows[(1, t)]['gap'] for t in THRESHOLDS[:-1]} == {16}
    assert {r['gap'] for (col, t), r in rows.items() if t == 100} == {1}
    # The hard tile: 16 never qualifies, 8 at t <= 0.5, 4 at t <= 0.75, 2 up to t = 0.95, and 1 at t = 1.
    assert [rows[(2, t)]['gap'] for t in THRESHOLDS] == [8, 8, 8, 4, 4, 2, 2, 2, 1]
    assert rows[(2, 30)]['tile_class'] == 'relevant' and rows[(2, 30)]['associations'] == 100
    # The observations behind the chosen gap: the hard tile's gap 8 rests on 18, its gap 2 on 99.
    assert rows[(2, 30)]['chosen_observations'] == 18 and rows[(2, 80)]['chosen_observations'] == 99
    # p016's table agrees with the exact recomputation everywhere here.
    assert all(r['mismatches'] == 0 for r in gt.recompute_mismatches(con))


def test_exact_tie_meets_threshold_like_p016(con):
    # At gap 2 the smoothed accuracy is exactly 7 / 10 = 0.7; larger gaps fail every stored threshold.
    counts = tile_counts([[50, 0], [6, 2], [0, 9], [0, 9], [0, 9]])
    gt.create_tables(con, {('d', 'trk'): make_profile(counts)})
    rows = tile_gaps(con)
    # The paper's rule (MissRate <= 0.3, inclusive) admits gap 2 at t = 0.7, and so does p016
    # (NumPy compares the float32 accuracy with the threshold in float32, so the tie holds).
    assert rows[(0, 70)]['exact_gap'] == 2 and rows[(0, 70)]['gap'] == 2
    # Just above the tie, gap 2 fails.
    assert rows[(0, 80)]['exact_gap'] == 1
    assert all(r['mismatches'] == 0 for r in gt.recompute_mismatches(con))


def test_recompute_reports_planted_mismatch(con):
    # A cached gap that disagrees with the counts is counted, not hidden.
    counts = tile_counts(EASY, HARD)
    table = pipeline_table(counts)
    table[0, 1, THRESHOLDS.index(80)] = 16
    gt.create_tables(con, {('d', 'trk'): make_profile(counts, table=table)})
    mismatches = {r['threshold_pct']: r['mismatches'] for r in gt.recompute_mismatches(con)}
    assert mismatches[80] == 1 and sum(mismatches.values()) == 1
    # The report keeps the cached gap (what the ILP read) and the exact one side by side.
    row = tile_gaps(con)[(1, 80)]
    assert row['gap'] == 16 and row['exact_gap'] == 2


def test_profiles_stay_isolated(con):
    # Two profiles with different grids and counts: a join across profiles would mix their tiles.
    easy_row = make_profile(tile_counts(EASY, EASY, EASY))
    hard_square = np.stack([np.array(HARD, dtype=np.int64)] * 4, axis=1).reshape(5, 2, 2, 2)
    gt.create_tables(con, {('a', 'trk'): easy_row, ('b', 'other'): make_profile(hard_square)})
    rows = gt.query_dicts(con, """
        SELECT dataset, tracker, count(*) AS tiles, min(exact_gap) AS lo, max(exact_gap) AS hi
        FROM tile_gaps WHERE threshold_pct = 80 GROUP BY ALL ORDER BY ALL
    """)
    # Three easy tiles at gap 16 in one profile, four hard tiles at gap 2 in the other.
    assert rows == [dict(dataset='a', tracker='trk', tiles=3, lo=16, hi=16),
                    dict(dataset='b', tracker='other', tiles=4, lo=2, hi=2)]
    assert all(r['mismatches'] == 0 for r in gt.recompute_mismatches(con))
    # The distribution keeps each profile's tile count.
    groups = {(r['dataset'], r['tile_class']): r['tiles'] for r in gt.gap_summary(con) if r['threshold_pct'] == 80}
    assert groups == {('a', 'relevant'): 3, ('a', 'all'): 3, ('b', 'relevant'): 4, ('b', 'all'): 4}


def test_low_evidence_and_skipped_gap_flags(con):
    # Relevant tile with no observations at gap 16: its 0.5 prior admits gap 16 at t <= 0.5.
    sparse = [[3, 0], [2, 0], [1, 0], [1, 0], [0, 0]]
    # Tile that fails at gap 2 but passes at gap 4 (non-monotone rates), so the max rule skips gap 2.
    bumpy = [[100, 0], [10, 90], [100, 0], [0, 50], [0, 50]]
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(sparse, bumpy, NEVER))})
    rows = tile_gaps(con)
    assert rows[(0, 50)]['gap'] == 16 and rows[(0, 50)]['chosen_observations'] == 0
    assert rows[(1, 80)]['gap'] == 4 and rows[(1, 80)]['skipped_gap']
    summary = {(r['threshold_pct'], r['tile_class']): r for r in gt.gap_summary(con)}
    # Only the relevant sparse tile counts as unobserved; never-relevant tiles are a class of their own.
    assert summary[(50, 'relevant')]['unobserved_gap_tiles'] == 1
    assert summary[(50, 'never_relevant')]['unobserved_gap_tiles'] == 0
    # At t = 0.6 the sparse tile drops to gap 8 (accuracy 2/3), observed once: few observations, not unobserved.
    assert rows[(0, 60)]['gap'] == 8
    assert summary[(60, 'relevant')]['unobserved_gap_tiles'] == 0
    assert summary[(60, 'relevant')]['few_observation_tiles'] == 1
    # The bumpy tile's gap 4 rests on 100 observations, so it is not low evidence.
    assert summary[(80, 'relevant')]['few_observation_tiles'] == 0
    assert summary[(80, 'relevant')]['skipped_gap_tiles'] == 1


def test_few_observation_boundary(con):
    # Exactly LOW_EVIDENCE_OBSERVATIONS observations at the chosen gap is enough; one fewer is not.
    k = gt.LOW_EVIDENCE_OBSERVATIONS
    at_k = [[100, 0], [0, 50], [0, 50], [k, 0], [0, 50]]
    below_k = [[100, 0], [0, 50], [0, 50], [k - 1, 0], [0, 50]]
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(at_k, below_k))})
    rows = tile_gaps(con)
    # Both tiles get gap 8 at t = 0.8 ((k + 1) / (k + 2) and k / (k + 1) are above 0.8 for k = 10).
    assert rows[(0, 80)]['gap'] == 8 and rows[(1, 80)]['gap'] == 8
    summary = {(r['threshold_pct'], r['tile_class']): r for r in gt.gap_summary(con)}
    assert summary[(80, 'relevant')]['few_observation_tiles'] == 1


def test_summary_statistics_and_weighted_bound(con):
    # Two relevant tiles at gaps 2 (100 associations) and 4 (300), plus one never-relevant tile.
    gap2 = [[100, 0], [50, 0], [0, 25], [0, 12], [0, 6]]
    gap4 = [[300, 0], [150, 0], [75, 0], [0, 37], [0, 19]]
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(gap2, gap4, NEVER))})
    summary = {(r['threshold_pct'], r['tile_class']): r for r in gt.gap_summary(con)}
    rel = summary[(80, 'relevant')]
    assert rel['tiles'] == 2 and rel['mean_gap'] == pytest.approx(3.0)
    # The lower median of {2, 4} is 2, a candidate gap.
    assert rel['median_gap'] == 2
    # Unweighted floor (1/2 + 1/4) / 2; weighted (100/2 + 300/4) / 400.
    assert rel['kept_fraction_lower_bound'] == pytest.approx(0.375)
    assert rel['kept_fraction_lower_bound_weighted'] == pytest.approx(0.3125)
    # Never-relevant tiles have no relevant tile-frames, so neither floor is defined for them.
    never = summary[(80, 'never_relevant')]
    assert never['kept_fraction_lower_bound'] is None and never['kept_fraction_lower_bound_weighted'] is None
    # 'all' covers every tile for the gap statistics, but its floors equal the relevant ones.
    everything = summary[(80, 'all')]
    assert everything['tiles'] == 3 and everything['mean_gap'] == pytest.approx(7 / 3)
    assert everything['kept_fraction_lower_bound'] == pytest.approx(0.375)
    assert everything['kept_fraction_lower_bound_weighted'] == pytest.approx(0.3125)


def test_distribution_includes_zero_gaps_and_sums_to_one(con):
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(NEVER, EASY, HARD))})
    dist = gt.gap_distribution(con)
    # Every (threshold, class) group lists all five candidate gaps.
    groups = {(r['threshold_pct'], r['tile_class']) for r in dist}
    assert len(dist) == 5 * len(groups)
    # Shares within each group sum to one.
    for key in groups:
        assert sum(r['share'] for r in dist if (r['threshold_pct'], r['tile_class']) == key) == pytest.approx(1.0)
    rel = {r['gap']: r for r in dist if r['threshold_pct'] == 80 and r['tile_class'] == 'relevant'}
    # At t = 0.8: hard tile at 2, easy at 16, nothing at 1, 4, or 8.
    assert {g: r['tiles'] for g, r in rel.items()} == {1: 0, 2: 1, 4: 0, 8: 0, 16: 1}
    assert rel[2]['share'] == pytest.approx(0.5)


def test_profile_without_never_relevant_tiles(con):
    # A grid where every tile is relevant has no never-relevant group, and the report omits it.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD))}
    gt.create_tables(con, profiles)
    summary = gt.gap_summary(con)
    assert {r['tile_class'] for r in summary} == {'relevant', 'all'}
    report, rows = gt.assemble(profiles, summary, gt.gap_distribution(con), gt.sample_rate_effect(con, [1]), [],
                               gt.recompute_mismatches(con))
    level = report['d']['trackers']['trk']['tolerances']['0.20']
    assert 'never_relevant' not in level and level['relevant']['tiles'] == 2
    assert {r['tile_class'] for r in rows} == {'relevant', 'all'}


def test_sample_rate_effect_divides_gaps(con):
    # One relevant tile at gap 8 for t = 0.5 (accuracy 0.5 at gap 8 meets 0.5; gap 16 fails).
    tile = [[100, 0], [99, 0], [80, 0], [9, 9], [0, 8]]
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(tile, NEVER))})
    effect = {r['sample_rate']: r for r in gt.sample_rate_effect(con, [1, 4, 16]) if r['threshold_pct'] == 50}
    # Only relevant tiles enter; gap 8 becomes 8, 2, and 1 sampled frames.
    assert effect[1]['kept_fraction_lower_bound_weighted'] == pytest.approx(1 / 8)
    assert effect[4]['kept_fraction_lower_bound_weighted'] == pytest.approx(1 / 2)
    assert effect[16]['kept_fraction_lower_bound_weighted'] == pytest.approx(1.0)
    # Effective gap in frames: 8, 4 * 2, and 16 * 1 (whole-frame sampling dominates).
    assert [effect[s]['mean_effective_gap'] for s in (1, 4, 16)] == [8, 8, 16]
    assert [effect[s]['prunable_tile_share'] for s in (1, 4, 16)] == [1, 1, 0]
    # The same rows are kept as a table for the per-setting comparison.
    assert con.execute('SELECT count(*) FROM sample_rate_effect').fetchone()[0] == 3 * len(THRESHOLDS)


@pytest.mark.parametrize('rates', [[], [0], [2.0], [True]])
def test_sample_rate_effect_rejects_bad_rates(con, rates):
    # Empty, zero, non-integer, or boolean sampling rates are errors.
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(EASY))})
    with pytest.raises(AssertionError, match='bad sample rates'):
        gt.sample_rate_effect(con, rates)


def test_create_tables_twice_replaces_data(con):
    # A second call reads the new profiles, not the tables of the first.
    gt.create_tables(con, {('a', 'trk'): make_profile(tile_counts(EASY))})
    gt.create_tables(con, {('b', 'trk'): make_profile(tile_counts(EASY, HARD))})
    assert con.execute('SELECT DISTINCT dataset FROM tile_gaps').fetchall() == [('b',)]
    assert con.execute('SELECT count(*) FROM rates').fetchone()[0] == 2 * 5


def validated_frame(rows):
    # Validated configurations as rows of VALIDATED_COLUMNS.
    return pd.DataFrame(rows, columns=gt.VALIDATED_COLUMNS)


def config(variant_id, tracker='sortcython', tilesize=60.0, sample_rate=1.0, threshold=0.8, relevance=0.25,
           padding='none', scale=1.0, dataset='d', classifier='ShuffleNet05'):
    # One validated row; the defaults match ShuffleNet05_60_1_080_r025_none_s100_sortcython.
    return (dataset, variant_id, classifier, tilesize, sample_rate, threshold, relevance, padding, scale, tracker)


PROFILE_THRESHOLDS = {('d', 'sortcython'): THRESHOLDS}


def test_pipeline_ids_builds_variant_and_prune_ids():
    frame = validated_frame([config('x'), config('y', threshold=np.nan), config('z', tilesize=60.5),
                             config('w', tracker=None)])
    ids = gt.pipeline_ids(frame)
    # A pruned row gets both ids; the 022 name drops the packing settings.
    assert ids['expected_id'][0] == 'ShuffleNet05_60_1_080_r025_none_s100_sortcython'
    assert ids['prune_id'][0] == 'ShuffleNet05_60_1_080_r025_sortcython'
    # No pruning: no threshold token and no 022 directory.
    assert ids['expected_id'][1] == 'ShuffleNet05_60_1_r025_none_s100_sortcython' and ids['prune_id'][1] is None
    # A fractional tile size or a missing tracker has no pipeline id.
    assert ids['expected_id'][2] is None and ids['expected_id'][3] is None


def test_validated_configs_counts_rows_and_pruning_settings(con):
    frame = validated_frame([
        config('ShuffleNet05_60_1_080_r025_none_s100_sortcython'),
        # A tile-padding variant of the same pruning setting.
        config('ShuffleNet05_60_1_080_r025_tr_s100_sortcython', padding='tr'),
        # Another pruning setting at the same threshold and rate (different T_r).
        config('ShuffleNet05_60_1_080_r050_none_s100_sortcython', relevance=0.5),
        config('ShuffleNet05_60_4_040_r025_none_s100_sortcython', sample_rate=4.0, threshold=0.4),
        # No pruning: NaN threshold and no 3-digit token.
        config('ShuffleNet05_60_2_r025_none_s100_sortcython', sample_rate=2.0, threshold=np.nan),
    ])
    rows, other = gt.validated_configs(con, frame, 60, PROFILE_THRESHOLDS)
    assert other == 0
    assert {(r['threshold_pct'], r['sample_rate']): (r['configs'], r['pruning_settings']) for r in rows} == \
        {(80, 1): (3, 2), (40, 4): (1, 1), (None, 2): (1, 0)}
    # Distinct pruning settings, with the rows sharing each.
    settings = {s['prune_id']: s['configs'] for s in gt.pruning_settings(con)}
    assert settings == {'ShuffleNet05_60_1_080_r025_sortcython': 2, 'ShuffleNet05_60_1_080_r050_sortcython': 1,
                        'ShuffleNet05_60_4_040_r025_sortcython': 1}
    # Totals per kind and rate.
    totals = {(t['kind'], t['sample_rate']): (t['configs'], t['pruning_settings']) for t in gt.validated_totals(con)}
    assert totals == {('pruned', 1): (3, 2), ('pruned', 4): (1, 1), ('none', 2): (1, 0)}


@pytest.mark.parametrize('row', [
    # The threshold column says 0.6 but the id says 080.
    config('ShuffleNet05_60_1_080_r025_none_s100_sortcython', threshold=0.6),
    # A pruned id with no threshold in the column.
    config('ShuffleNet05_60_1_080_r025_none_s100_sortcython', threshold=np.nan),
    # The tracker column disagrees with the id's suffix.
    config('ShuffleNet05_60_1_080_r025_none_s100_sortcython', tracker='bytetrackcython'),
    # The sampling-rate column says 1 but the id says 4.
    config('ShuffleNet05_60_4_080_r025_none_s100_sortcython'),
    # The relevance threshold or padding disagrees with the id.
    config('ShuffleNet05_60_1_080_r050_none_s100_sortcython'),
    config('ShuffleNet05_60_1_080_r025_tr_s100_sortcython'),
    # Missing tracker or tile size: NULL must not slip through the check.
    config('ShuffleNet05_60_1_080_r025_none_s100_sortcython', tracker=None),
    config('ShuffleNet05_60_1_080_r025_none_s100_sortcython', tilesize=np.nan),
])
def test_validated_configs_rejects_inconsistent_rows(con, row):
    with pytest.raises(AssertionError, match='ids the pipeline builds'):
        gt.validated_configs(con, validated_frame([row]), 60, PROFILE_THRESHOLDS)


def test_validated_configs_drops_other_tile_sizes(con):
    # A tile-size sweep row belongs to another profile: dropped and counted, not an error.
    frame = validated_frame([config('ShuffleNet05_30_1_080_r025_none_s100_sortcython', tilesize=30.0),
                             config('ShuffleNet05_60_1_080_r025_none_s100_sortcython')])
    rows, other = gt.validated_configs(con, frame, 60, PROFILE_THRESHOLDS)
    assert other == 1 and [r['configs'] for r in rows] == [1]


def test_validated_configs_rejects_threshold_missing_from_profile(con):
    # A pruned row of a measured profile must use a threshold that profile stores.
    frame = validated_frame([config('ShuffleNet05_60_1_085_r025_none_s100_sortcython', threshold=0.85)])
    with pytest.raises(AssertionError, match='missing from the p016'):
        gt.validated_configs(con, frame, 60, PROFILE_THRESHOLDS)


def test_validated_configs_skips_unmeasured_profiles(con):
    # A threshold check needs the profile; configurations of trackers not measured pass through,
    # but they enter neither the pruning settings nor the totals.
    frame = validated_frame([config('ShuffleNet05_60_1_085_r025_none_s100_bytetrackcython', tracker='bytetrackcython',
                                    threshold=0.85)])
    rows, _ = gt.validated_configs(con, frame, 60, PROFILE_THRESHOLDS)
    assert rows[0]['configs'] == 1
    assert gt.pruning_settings(con) == [] and gt.validated_totals(con) == []


def test_validated_configs_empty_table(con):
    # A dataset without Tetris rows reports no configurations.
    assert gt.validated_configs(con, validated_frame([]), 60, PROFILE_THRESHOLDS) == ([], 0)


def test_list_test_videos(tmp_cache):
    # Missing videoset, an empty one, and a listing that keeps only video files in sorted order.
    with pytest.raises(AssertionError, match='does not exist'):
        gt.list_test_videos('d')
    directory = store.dataset('d', 'test')
    directory.mkdir(parents=True)
    with pytest.raises(AssertionError, match='no test videos'):
        gt.list_test_videos('d')
    for name in ('te02.mp4', 'te01.avi', 'notes.txt'):
        (directory / name).write_text('')
    assert gt.list_test_videos('d') == ['te01.avi', 'te02.mp4']


# One pruning setting of profile ('d', 'trk') at t = 0.8, s = 1, T_r = 0.25.
SETTING = dict(dataset='d', tracker='trk', classifier='C', tilesize=60, sample_rate=1, threshold_pct=80,
               relevance_threshold=0.25, prune_id='C_60_1_080_r025_trk', configs=2)
# Classifier scores of four frames on a 1x3 grid: 63 is the floor(0.25 * 255) boundary.
SCORES = [[[63, 200, 0]], [[64, 200, 0]], [[10, 63, 100]], [[200, 0, 64]]]
# Kept tiles: all inside the floor-rule input except the third tile of frame 1 (score 0).
KEPT = [[[255, 255, 0]], [[0, 0, 255]], [[0, 255, 0]], [[255, 0, 255]]]


def test_kept_counts_measures_ilp_input_and_output(tmp_cache):
    # Profile gaps at t = 0.8: easy tile 16, hard tile 2, never-relevant tile 1.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD, NEVER))}
    write_scores('d', 'relevancy', 'v0.mp4', 'C_60_1', SCORES)
    write_scores('d', 'pruned-polyominoes', 'v0.mp4', SETTING['prune_id'], KEPT)
    (row,) = gt.kept_counts('d', [SETTING], profiles, ['v0.mp4'])
    assert (row['frames'], row['tiles']) == (4, 3)
    # Floor rule (>= 63): 8 input tile-frames; current rule (> 63.75): 6, since the two 63s drop.
    assert row['covered'] == 8 and row['covered_current_rule'] == 6
    # 6 kept, one of them outside the input.
    assert row['kept'] == 6 and row['kept_outside'] == 1
    # Per-tile input counts 3, 3, 2 over gaps 16, 2, 1.
    assert row['bound_tile_frames'] == pytest.approx(3 / 16 + 3 / 2 + 2 / 1)
    assert row['relevancy_mtime'] <= row['pruned_mtime']


def test_kept_counts_divides_gaps_by_sample_rate(tmp_cache):
    # At s = 2 the gaps 16, 2, 1 become 8, 1, 1 sampled frames.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD, NEVER))}
    setting = SETTING | dict(sample_rate=2, prune_id='C_60_2_080_r025_trk')
    write_scores('d', 'relevancy', 'v0.mp4', 'C_60_2', SCORES)
    write_scores('d', 'pruned-polyominoes', 'v0.mp4', setting['prune_id'], KEPT)
    (row,) = gt.kept_counts('d', [setting], profiles, ['v0.mp4'])
    assert row['bound_tile_frames'] == pytest.approx(3 / 8 + 3 / 1 + 2 / 1)


def test_kept_counts_rejects_frame_mismatch(tmp_cache):
    # A 022 output with fewer frames than its 020 input is not this ILP's output.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD, NEVER))}
    write_scores('d', 'relevancy', 'v0.mp4', 'C_60_1', SCORES)
    write_scores('d', 'pruned-polyominoes', 'v0.mp4', SETTING['prune_id'], KEPT[:3])
    with pytest.raises(AssertionError, match='differ from 020'):
        gt.kept_counts('d', [SETTING], profiles, ['v0.mp4'])


def test_kept_counts_rejects_grid_unlike_profile(tmp_cache):
    # Scores on a 1x3 grid cannot be judged with a profile over a 1x2 grid.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD))}
    write_scores('d', 'relevancy', 'v0.mp4', 'C_60_1', SCORES)
    write_scores('d', 'pruned-polyominoes', 'v0.mp4', SETTING['prune_id'], KEPT)
    with pytest.raises(AssertionError, match="profile's"):
        gt.kept_counts('d', [SETTING], profiles, ['v0.mp4'])


def test_kept_counts_missing_outputs(tmp_cache):
    # A validated setting without a 022 output on a test video fails rather than shrinking the sample.
    profiles = {('d', 'trk'): make_profile(tile_counts(EASY, HARD, NEVER))}
    write_scores('d', 'relevancy', 'v0.mp4', 'C_60_1', SCORES)
    with pytest.raises(FileNotFoundError):
        gt.kept_counts('d', [SETTING], profiles, ['v0.mp4'])
    # An empty score file has no frames.
    write_scores('d', 'pruned-polyominoes', 'v0.mp4', SETTING['prune_id'], np.zeros((0, 1, 3)))
    with pytest.raises(AssertionError, match='no frames'):
        gt.kept_counts('d', [SETTING], profiles, ['v0.mp4'])


def counts_row(prune_id, video, covered, kept, kept_outside=0, bound=0.0, frames=4, tiles=3, current=None):
    # One kept_counts row with fixed timestamps.
    return dict(dataset='d', prune_id=prune_id, video=video, frames=frames, tiles=tiles, covered=covered,
                covered_current_rule=covered if current is None else current, kept=kept, kept_outside=kept_outside,
                bound_tile_frames=bound, relevancy_mtime='2026-04-10T00:00:00+00:00',
                pruned_mtime='2026-04-19T00:00:00+00:00')


def test_kept_summary_pools_videos_and_settings(con):
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(EASY, HARD, NEVER))})
    gt.sample_rate_effect(con, [1])
    other = SETTING | dict(relevance_threshold=0.5, prune_id='C_60_1_080_r050_trk', configs=1)
    counts = [counts_row(SETTING['prune_id'], 'v0', covered=8, kept=4, bound=2.0),
              counts_row(SETTING['prune_id'], 'v1', covered=2, kept=2, bound=1.0, current=1),
              counts_row(other['prune_id'], 'v0', covered=10, kept=2, bound=1.5)]
    per_setting, per_rate, per_profile = gt.kept_summary(con, counts, [SETTING, other])
    by_id = {r['prune_id']: r for r in per_setting}
    # Per setting: kept over input, pooled over videos; the floor weighted by the same input.
    assert by_id[SETTING['prune_id']]['videos'] == 2 and by_id[SETTING['prune_id']]['kept_fraction'] == pytest.approx(0.6)
    assert by_id[SETTING['prune_id']]['bound_test_weighted'] == pytest.approx(0.3)
    assert by_id[other['prune_id']]['kept_fraction'] == pytest.approx(0.2)
    # The training-weighted floor at the same threshold and rate comes from sample_rate_effect.
    assert by_id[other['prune_id']]['bound_train_weighted'] is not None
    # Per rate: pooled 8 / 20, with the per-setting range and the setting and row counts.
    (rate,) = per_rate
    assert rate['settings'] == 2 and rate['configs'] == 3
    assert rate['kept_fraction'] == pytest.approx(0.4)
    assert (rate['kept_fraction_min'], rate['kept_fraction_max']) == (pytest.approx(0.2), pytest.approx(0.6))
    assert rate['covered_share'] == pytest.approx(20 / 36)
    # Per profile: input timing and the share of input the current rule would keep (19 of 20).
    (profile,) = per_profile
    assert profile['earliest_pruned_output'] == '2026-04-19T00:00:00+00:00'
    assert profile['relevancy_newer_than_pruned'] == 0
    assert profile['covered_current_rule_share'] == pytest.approx(19 / 20)


def test_kept_summary_without_settings(con):
    # No validated pruning settings means nothing measured.
    assert gt.kept_summary(con, [], []) == ([], [], [])


def test_kept_summary_rejects_unmeasured_setting(con):
    # A setting with no measured video would silently vanish from the per-rate pool.
    gt.create_tables(con, {('d', 'trk'): make_profile(tile_counts(EASY))})
    gt.sample_rate_effect(con, [1])
    other = SETTING | dict(prune_id='C_60_1_080_r050_trk', relevance_threshold=0.5)
    with pytest.raises(AssertionError, match='without measured videos'):
        gt.kept_summary(con, [counts_row(SETTING['prune_id'], 'v0', covered=1, kept=1)], [SETTING, other])


def test_figure_crop_values_and_bounds():
    profile = make_profile(tile_counts(NEVER, EASY, HARD))
    crop = gt.figure_crop(profile, (0, 1), (1, 3))
    # Gaps of the easy and hard tiles at t = 0.8 (tolerance 0.20) and t = 0.4 (0.60).
    assert crop['gaps_by_tolerance']['0.20'] == [[16, 2]] and crop['gaps_by_tolerance']['0.60'] == [[16, 8]]
    # Smoothed mistrack percent from accuracy.npy: easy tile 1 / 64 at gap 16, hard tile (8 + 1) / (8 + 2).
    assert crop['mistrack_pct']['16'] == [[2, 90]] and crop['mistrack_pct']['01'] == [[0, 1]]
    # A crop past the grid edge is an error.
    with pytest.raises(AssertionError, match='outside grid'):
        gt.figure_crop(profile, (0, 2), (0, 1))
    with pytest.raises(AssertionError, match='outside grid'):
        gt.figure_crop(profile, (0, 1), (2, 2))


def test_assemble_marks_figure_7_and_sums_validated_configs(con):
    profiles = {('jnc7', 'bytetrackcython'): make_profile(tile_counts(NEVER, EASY, HARD)),
                ('jnc7', 'sortcython'): make_profile(tile_counts(NEVER, EASY, HARD))}
    gt.create_tables(con, profiles)
    # t = 0.8 at two sampling rates (rows and distinct settings), plus no pruning.
    configs = [dict(dataset='jnc7', tracker='bytetrackcython', threshold_pct=80, sample_rate=1, configs=2, pruning_settings=1),
               dict(dataset='jnc7', tracker='bytetrackcython', threshold_pct=80, sample_rate=4, configs=3, pruning_settings=2),
               dict(dataset='jnc7', tracker='bytetrackcython', threshold_pct=None, sample_rate=1, configs=10, pruning_settings=0),
               # A configuration of a tracker outside the measured profiles is ignored.
               dict(dataset='jnc7', tracker='ocsortcython', threshold_pct=80, sample_rate=1, configs=5, pruning_settings=1)]
    # Planted mismatches at two thresholds of one profile.
    mismatches = [r | dict(mismatches={80: 1, 40: 2}.get(r['threshold_pct'], 0)) if r['tracker'] == 'bytetrackcython' else r
                  for r in gt.recompute_mismatches(con)]
    kept_rates = [dict(dataset='jnc7', tracker='bytetrackcython', threshold_pct=80, sample_rate=4, settings=2, configs=3,
                       kept_fraction=0.5, kept_fraction_min=0.4, kept_fraction_max=0.6, bound_test_weighted=0.3,
                       covered_share=0.1, kept_outside=0)]
    kept_profiles = [dict(dataset='jnc7', tracker='bytetrackcython', earliest_pruned_output='2026-04-19T00:00:00+00:00',
                          latest_relevancy_input='2026-04-10T00:00:00+00:00', relevancy_newer_than_pruned=0,
                          kept_outside=0, covered_current_rule_share=0.99)]
    profiles[('jnc7', 'bytetrackcython')]['inputs'] = dict(max_rate_table=dict(mtime='2026-04-07T00:00:00+00:00'))
    report, rows = gt.assemble(profiles, gt.gap_summary(con), gt.gap_distribution(con),
                               gt.sample_rate_effect(con, [1, 2, 4]), configs, mismatches, kept_rates, kept_profiles)
    byte = report['jnc7']['trackers']['bytetrackcython']
    assert report['jnc7']['label'] == 'B3D4'
    assert byte['figure_7'] and not report['jnc7']['trackers']['sortcython']['figure_7']
    # The t = 0.8 rows land on tolerance 0.20, summed over sampling rates; no-pruning ones are kept apart.
    level = byte['tolerances']['0.20']
    assert level['validated_configs'] == 5 and level['validated_pruning_settings'] == 3
    assert level['validated_sample_rates'] == {'01': 2, '04': 3}
    assert byte['validated_configs'] == {'0.20': {'01': 2, '04': 3}, 'none': {'01': 10}}
    assert byte['tolerances']['0.60']['validated_configs'] == 0
    # Mismatches are summed per profile and listed per threshold.
    assert byte['recompute_mismatches'] == 3
    assert byte['tolerances']['0.20']['recompute_mismatches'] == 1 and byte['tolerances']['0.60']['recompute_mismatches'] == 2
    assert report['jnc7']['trackers']['sortcython']['recompute_mismatches'] == 0
    # Every threshold is reported, with the gap shares and the sampling-rate effect.
    assert sorted(byte['tolerances']) == sorted(gt.tolerance_key(t) for t in THRESHOLDS)
    assert level['relevant']['gap_share'] == {'01': 0.0, '02': 0.5, '04': 0.0, '08': 0.0, '16': 0.5}
    assert set(level['with_sample_rate']) == {'01', '02', '04'}
    # The measured kept fraction sits next to the floor at the same rate.
    assert level['with_sample_rate']['04']['measured']['kept_fraction'] == 0.5
    assert 'measured' not in level['with_sample_rate']['01']
    # Input timing: the profile predates the ILP outputs.
    assert byte['inputs']['profile_predates_pruning'] is True
    # Flat rows: 2 trackers x 9 thresholds x 3 classes, with gap shares as zero-padded columns.
    assert len(rows) == 2 * len(THRESHOLDS) * 3
    row = next(r for r in rows if r['tracker'] == 'bytetrackcython' and r['mistrack_tolerance'] == 0.2
               and r['tile_class'] == 'relevant')
    assert row['share_gap_16'] == 0.5 and row['validated_configs'] == 5 and row['tracking_accuracy_threshold'] == 0.8
    assert row['validated_pruning_settings'] == 3


def smoke_profile_counts(relevant: bool):
    # A 12 x 14 grid (Fig. 7's crop needs rows 8-10 and columns 9-12): easy tiles on top, hard below,
    # and a never-relevant first column; or no associations anywhere.
    counts = np.zeros((5, 12, 14, 2), dtype=np.int64)
    if relevant:
        counts[:, :6] = np.array(EASY)[:, None, None, :]
        counts[:, 6:] = np.array(HARD)[:, None, None, :]
        counts[:, :, 0] = 0
    return counts


def write_smoke_cache(tmp_cache, monkeypatch, table=None):
    # Profiles for jnc7: ByteTrack (Fig. 7's) with relevant tiles, SORT with none.
    for tracker, relevant in (('bytetrackcython', True), ('sortcython', False)):
        counts = smoke_profile_counts(relevant)
        write_profile(cache.index('jnc7', 'track_rates', f'{tracker}_60'), counts,
                      table=table if relevant else None)
    # Validated rows: one pruning setting with a padding variant, and one no-pruning row.
    frame = pd.DataFrame([config('ShuffleNet05_60_1_080_r025_none_s100_bytetrackcython', tracker='bytetrackcython'),
                          config('ShuffleNet05_60_1_080_r025_tr_s100_bytetrackcython', tracker='bytetrackcython',
                                 padding='tr'),
                          config('ShuffleNet05_60_1_r025_none_s100_sortcython', threshold=np.nan)],
                         columns=gt.VALIDATED_COLUMNS).drop(columns='dataset')
    monkeypatch.setattr(gt, 'load_test_tradeoff', lambda dataset: (frame, None))
    tradeoff = cache.eval('jnc7', 'tradeoff', 'tradeoff.csv')
    tradeoff.parent.mkdir(parents=True)
    tradeoff.write_text('variant_id\n')
    # One test video with its classifier scores and the ILP's kept tiles (half of the input).
    (store.dataset('jnc7', 'test')).mkdir(parents=True)
    (store.dataset('jnc7', 'test') / 'te01.mp4').write_text('')
    scores = np.zeros((3, 12, 14), dtype=np.uint8)
    scores[:, 8, 9:13] = 200
    kept = scores.copy()
    kept[1] = 0
    write_scores('jnc7', 'relevancy', 'te01.mp4', 'ShuffleNet05_60_1', scores)
    write_scores('jnc7', 'pruned-polyominoes', 'te01.mp4', 'ShuffleNet05_60_1_080_r025_bytetrackcython', kept)


def smoke_args():
    # main's arguments for the smoke cache.
    return argparse.Namespace(datasets=['jnc7'], trackers=['bytetrackcython', 'sortcython'], tile_size=60)


def test_main_writes_reports(tmp_cache, monkeypatch, capsys):
    write_smoke_cache(tmp_cache, monkeypatch)
    gt.main(smoke_args())
    out = tmp_cache / 'results' / 'r1o5_gap_tolerance'
    report = json.loads((out / 'gap_tolerance.json').read_text())
    rows = json.loads((out / 'table_rows.json').read_text())
    settings = json.loads((out / 'pruning_settings.json').read_text())
    byte = report['datasets']['jnc7']['trackers']['bytetrackcython']
    level = byte['tolerances']['0.20']
    # Two rows share one pruning setting; 8 of 12 input tile-frames were kept.
    assert level['validated_configs'] == 2 and level['validated_pruning_settings'] == 1
    assert level['with_sample_rate']['01']['measured']['kept_fraction'] == pytest.approx(8 / 12, abs=1e-4)
    assert [s['configs'] for s in settings] == [2] and settings[0]['mistrack_tolerance'] == 0.2
    assert settings[0]['label'] == 'B3D4' and settings[0]['kept_outside'] == 0
    # Input records and timing are in the report.
    assert byte['inputs']['counts_equal_partial_sum'] is None and byte['inputs']['kept_outside'] == 0
    assert report['inputs']['tradeoff_tables']['jnc7']['path'].endswith('tradeoff.csv')
    assert report['validated_totals'] == [dict(kind='none', sample_rate=1, configs=1, pruning_settings=0),
                                          dict(kind='pruned', sample_rate=1, configs=2, pruning_settings=1)]
    # The SORT profile has no relevant tiles; its rows and the printout skip that class.
    sort = report['datasets']['jnc7']['trackers']['sortcython']
    assert all('relevant' not in level for level in sort['tolerances'].values())
    assert sort['validated_configs'] == {'none': {'01': 1}}
    assert {r['tile_class'] for r in rows if r['tracker'] == 'sortcython'} == {'never_relevant', 'all'}
    # The Fig. 7 crop is the ByteTrack profile's.
    assert report['figure_7']['crop']['gaps_by_tolerance']['0.20'][0] == [2, 2, 2, 2]
    printed = capsys.readouterr().out
    assert '(Fig. 7)' in printed and 's= 1: kept 0.667' in printed


def test_main_fails_on_cached_gap_mismatch(tmp_cache, monkeypatch):
    # A cached table that disagrees with the counts stops the run before anything is written.
    table = pipeline_table(smoke_profile_counts(True))
    table[8, 9, THRESHOLDS.index(80)] = 16
    write_smoke_cache(tmp_cache, monkeypatch, table=table)
    with pytest.raises(AssertionError, match='cached gaps differ'):
        gt.main(smoke_args())
    assert not (tmp_cache / 'results' / 'r1o5_gap_tolerance' / 'gap_tolerance.json').exists()


def test_main_rejects_unknown_dataset(tmp_cache):
    # Only the paper's datasets have labels and tradeoff tables.
    with pytest.raises(AssertionError, match='unknown datasets'):
        gt.main(argparse.Namespace(datasets=['nope'], trackers=['sortcython'], tile_size=60))
