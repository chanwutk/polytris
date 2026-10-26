import json
import math
import sys

import duckdb
import pandas as pd
import pytest

from evaluation import p203_compare_stats as p203
from evaluation import p204_ablation_stats as p204
from measurements import common
from measurements import r7o1_hota_matched_speedup as ms


# Header of LEAP's raw stat CSV.
LEAP_HEADER = 'video_name,inference_total,detector,differencer,reid,match,decode,frames_sampled,frames_filtered'
# One raw SOTA box.
BOX = {'left': 1.0, 'top': 2.0, 'right': 3.0, 'bottom': 4.0, 'class': 'car', 'score': 1.0, 'track_id': 1}


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def make_points(con, rows, table='hota_points'):
    # A points table from (dataset, system, config, fps, hota) rows, HOTA as accuracy, plus an empty tie order.
    con.execute(f'CREATE OR REPLACE TABLE {table} (dataset VARCHAR, system VARCHAR, config VARCHAR, '
                'fps DOUBLE, hota DOUBLE, accuracy DOUBLE)')
    con.executemany(f'INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)', [(*r, r[-1]) for r in rows])
    ms.create_dataset_order(con, [])


def run_bounds(con, losses, fit='leap', table='hota_points', out='chosen'):
    # Choose every system's point at each bound, as main does.
    specs = ms.bound_specs(con, table, losses, fit)
    ms.choose(con, table, specs, ms.SYSTEMS, out)
    return specs


def configs(rows):
    # system_configs-style dicts from (system, config, fps, hota) rows.
    return [dict(system=s, config=c, fps=f, hota=h) for s, c, f, h in rows]


# Reference HOTA 0.8 and 10 fps on both datasets; 5% bound 0.76, 10% bound 0.72.
TWO_DATASETS = [
    ('A', 'reference', 'naive', 10.0, 0.8),
    ('A', 'tetris', 't1', 300.0, 0.70), ('A', 'tetris', 't2', 200.0, 0.77), ('A', 'tetris', 't3', 50.0, 0.85),
    ('A', 'otif', 'param 0', 40.0, 0.78), ('A', 'otif', 'param 1', 100.0, 0.73),
    ('A', 'leap', 'param 0', 150.0, 0.05),
    ('B', 'reference', 'naive', 10.0, 0.8),
    ('B', 'tetris', 't1', 120.0, 0.79),
    ('B', 'otif', 'param 0', 500.0, 0.60),
    ('B', 'leap', 'param 0', 20.0, 0.02),
]
# A complete, valid system_configs result.
VALID_CONFIGS = configs([('reference', 'naive', 1.0, 0.5), ('tetris', 'c', 9.0, 0.4),
                         ('otif', 'param 0', 3.0, 0.3), ('leap', 'param 0', 2.0, 0.01)])


def test_load_points_uses_manuscript_labels(con, monkeypatch):
    # Every dataset gets the same configurations; HOTA is copied into accuracy.
    monkeypatch.setattr(ms, 'system_configs', lambda d: VALID_CONFIGS)
    ms.load_points(con, ['jnc0', 'caldot1-y05'])
    rows = con.execute('SELECT DISTINCT dataset FROM p142_points ORDER BY dataset').fetchall()
    assert rows == [('B3D1',), ('CalDoT1',)]
    assert con.execute('SELECT count(*) FROM p142_points WHERE hota <> accuracy').fetchone()[0] == 0
    # The tie order follows p204's dataset order, CalDoT1 before B3D1.
    assert con.execute('SELECT dataset FROM dataset_order ORDER BY ord').fetchall() == [('CalDoT1',), ('B3D1',)]


def test_load_points_rejects_unknown_dataset(con):
    # A dataset outside the manuscript's seven is a caller error.
    with pytest.raises(AssertionError, match='unknown datasets'):
        ms.load_points(con, ['caldot1'])


@pytest.mark.parametrize('bad', [('tetris', 'c', 9.0, math.nan), ('leap', 'param 0', 2.0, math.nan),
                                 ('otif', 'param 0', math.nan, 0.3), ('otif', 'param 0', 0.0, 0.3),
                                 ('tetris', 'c', 9.0, 1.5)])
def test_load_points_rejects_invalid_values(con, monkeypatch, bad):
    # A NaN HOTA would win every bound and a NaN LEAP HOTA would make leap_fit NaN; both, and bad throughput, fail.
    rows = [r for r in VALID_CONFIGS if r['system'] != bad[0]] + configs([bad])
    monkeypatch.setattr(ms, 'system_configs', lambda d: rows)
    with pytest.raises(AssertionError, match='invalid HOTA or throughput'):
        ms.load_points(con, ['jnc0'])


def test_load_points_rejects_missing_system(con, monkeypatch):
    # A dataset without OTIF test rows (for example only validation rows were exported) is an error, not a silent gap.
    monkeypatch.setattr(ms, 'system_configs', lambda d: [r for r in VALID_CONFIGS if r['system'] != 'otif'])
    with pytest.raises(AssertionError, match='missing or repeated'):
        ms.load_points(con, ['jnc0'])


def test_load_points_rejects_duplicate_configs(con, monkeypatch):
    # Two rows with one configuration name would make the report ambiguous.
    monkeypatch.setattr(ms, 'system_configs', lambda d: VALID_CONFIGS + configs([('tetris', 'c', 5.0, 0.4)]))
    with pytest.raises(AssertionError, match='duplicate configurations'):
        ms.load_points(con, ['jnc0'])


def fake_tradeoff(rows, ref_fps=10.0, ref_hota=0.8, classifier='ShuffleNet05', frames=100):
    # A (tetris, naive) pair like load_test_tradeoff from (config, sample_rate, threshold, fps, hota) rows.
    tetris = pd.DataFrame({'variant_id': [r[0] for r in rows], 'classifier': classifier,
                           'sample_rate': [float(r[1]) for r in rows],
                           'tracking_accuracy_threshold': [math.nan if r[2] is None else r[2] for r in rows],
                           'throughput_fps': [r[3] for r in rows], 'HOTA_HOTA': [r[4] for r in rows]})
    naive = pd.Series({'throughput_fps': ref_fps, 'HOTA_HOTA': ref_hota, 'frame_count': frames})
    return tetris, naive


def test_load_tetris_knobs_rejects_perfect_classifier(con, monkeypatch):
    # The oracle classifier is dropped by p203 and p204 but not by system_configs, so it is an error.
    make_points(con, [('B3D1', 'tetris', 'c', 20.0, 0.8)], table='p142_points')
    monkeypatch.setattr(ms, 'load_test_tradeoff', lambda d: fake_tradeoff([('c', 1, None, 20.0, 0.8)], classifier='Perfect'))
    with pytest.raises(AssertionError, match='Perfect'):
        ms.load_tetris_knobs(con, ['jnc0'])


def test_load_tetris_knobs_rejects_disagreement(con, monkeypatch):
    # The knob rows and system_configs' rows come from one file; a difference means the loaders drifted.
    make_points(con, [('B3D1', 'tetris', 'c', 20.0, 0.8)], table='p142_points')
    monkeypatch.setattr(ms, 'load_test_tradeoff', lambda d: fake_tradeoff([('c', 1, None, 21.0, 0.8)]))
    with pytest.raises(AssertionError, match='differ'):
        ms.load_tetris_knobs(con, ['jnc0'])


def test_load_tetris_knobs_keeps_null_threshold_and_frames(con, monkeypatch):
    # A NaN threshold (no pruning) arrives as NULL, and the frame count is recorded.
    make_points(con, [('B3D1', 'tetris', 'c', 20.0, 0.8)], table='p142_points')
    monkeypatch.setattr(ms, 'load_test_tradeoff', lambda d: fake_tradeoff([('c', 1, None, 20.0, 0.8)], frames=250))
    ms.load_tetris_knobs(con, ['jnc0'])
    assert con.execute('SELECT threshold IS NULL FROM tetris_knobs').fetchone()[0]
    assert con.execute('SELECT frames FROM frame_counts').fetchone()[0] == 250.0


def test_load_clips_must_add_up_to_frame_count(con, monkeypatch):
    # Clip frames that do not sum to the tradeoff frame count mean the clips and the table disagree.
    con.execute("CREATE TABLE frame_counts AS SELECT 'B3D1' AS dataset, 100.0::DOUBLE AS frames")
    monkeypatch.setattr(ms, 'clip_manifest', lambda d: [dict(video_id=1, video='te01.mp4', frames=60)])
    with pytest.raises(AssertionError, match='do not add up'):
        ms.load_clips(con, ['jnc0'])
    # Matching clips load with the SOTA export's dataset name.
    monkeypatch.setattr(ms, 'clip_manifest', lambda d: [dict(video_id=1, video='te01.mp4', frames=60),
                                                        dict(video_id=4, video='te04.mp4', frames=40)])
    ms.load_clips(con, ['jnc0'])
    assert con.execute('SELECT DISTINCT raw FROM clips').fetchall() == [('jnc0',)]


def setup_leap(con, tmp_path, lines, leap_rows=(('B3D1', 'leap', 'param 0', 1.0, 0.05),)):
    # One B3D1 dataset with two clips (60 and 40 frames) and the given LEAP stat CSV lines.
    make_points(con, [('B3D1', 'reference', 'naive', 10.0, 0.8), ('B3D1', 'tetris', 't', 50.0, 0.8), *leap_rows],
                table='p142_points')
    con.execute("CREATE OR REPLACE TABLE frame_counts AS SELECT 'B3D1' AS dataset, 100.0::DOUBLE AS frames")
    con.execute("""CREATE OR REPLACE TABLE clips AS SELECT * FROM (VALUES ('B3D1', 'jnc0', 1, 'te01.mp4', 60.0::DOUBLE),
                   ('B3D1', 'jnc0', 4, 'te04.mp4', 40.0::DOUBLE)) t(dataset, raw, video_id, video, frames)""")
    (tmp_path / 'jnc0').mkdir(exist_ok=True)
    (tmp_path / 'jnc0' / 'leap_jnc0.csv').write_text('\n'.join([LEAP_HEADER, *lines]) + '\n')


def test_leap_runtime_averages_repeated_runs(con, tmp_path):
    # Video 1 ran twice (runtimes 2 and 3 after decode), video 4 twice (1 and 1); the total row sums all four.
    setup_leap(con, tmp_path, ['1,3.0,0,0,0,0,1.0,9,7', '4,2.0,0,0,0,0,1.0,9,7', '1,4.0,0,0,0,0,1.0,9,7',
                               '4,2.0,0,0,0,0,1.0,9,7', 'total,11.0,0,0,0,0,4.0,36,28'])
    out = ms.correct_leap_runtime(con, str(tmp_path))['B3D1']
    # One run per video: 2.5 + 1.0 = 3.5 s, so 100 / 3.5 fps instead of the total row's 7 s.
    assert (out['videos'], out['stat_rows'], out['max_runs_per_video']) == (2, 4, 2)
    assert (out['deduplicated_runtime_s'], out['total_row_runtime_s'], out['max_run_spread']) == (3.5, 7.0, 0.5)
    assert out['same_frames_across_runs'] and out['deduplicated_fps'] == round(100 / 3.5, 1)
    # p142's runtime is the frame count over its throughput (1 fps here).
    assert out['p142_runtime_s'] == 100.0
    # hota_points carries the corrected LEAP throughput and leaves every other row alone.
    assert con.execute("SELECT fps FROM hota_points WHERE system = 'leap'").fetchone()[0] == pytest.approx(100 / 3.5)
    assert con.execute("SELECT fps FROM hota_points WHERE system = 'tetris'").fetchone()[0] == 50.0


def test_leap_runtime_flags_runs_with_different_frames(con, tmp_path):
    # Two runs that sampled different frames are not repeats of one run; the JSON says so.
    setup_leap(con, tmp_path, ['1,3.0,0,0,0,0,1.0,9,7', '1,3.0,0,0,0,0,1.0,8,7', '4,2.0,0,0,0,0,1.0,9,7'])
    assert ms.correct_leap_runtime(con, str(tmp_path))['B3D1']['same_frames_across_runs'] is False


@pytest.mark.parametrize('lines, message', [
    (['1,3.0,0,0,0,0,1.0,9,7', 'total,3.0,0,0,0,0,1.0,9,7'], 'do not match the test videos'),
    (['1,3.0,0,0,0,0,1.0,9,7', '4,2.0,0,0,0,0,1.0,9,7', '7,2.0,0,0,0,0,1.0,9,7'], 'do not match the test videos'),
    (['1,3.0,0,0,0,0,1.0,9,7', '4,2.0,0,0,0,0,1.0,9,7', 'abc,2.0,0,0,0,0,1.0,9,7'], 'without a video id'),
    (['1,3.0,0,0,0,0,1.0,9,7', '4,1.0,0,0,0,0,1.0,9,7'], 'non-positive runtime'),
])
def test_leap_runtime_rejects_bad_stat_rows(con, tmp_path, lines, message):
    # A missing test video, an extra video, a non-numeric name, and a zero runtime are all errors.
    setup_leap(con, tmp_path, lines)
    with pytest.raises(AssertionError, match=message):
        ms.correct_leap_runtime(con, str(tmp_path))


def test_leap_runtime_rejects_missing_csv_and_several_configs(con, tmp_path):
    # No raw stat CSV for the dataset.
    setup_leap(con, tmp_path, ['1,3.0,0,0,0,0,1.0,9,7', '4,2.0,0,0,0,0,1.0,9,7'])
    with pytest.raises(AssertionError, match='missing raw LEAP stat CSV'):
        ms.correct_leap_runtime(con, str(tmp_path / 'elsewhere'))
    # The stat CSV describes one configuration, so two LEAP configurations are an error.
    setup_leap(con, tmp_path, ['1,3.0,0,0,0,0,1.0,9,7', '4,2.0,0,0,0,0,1.0,9,7'],
               leap_rows=(('B3D1', 'leap', 'param 0', 1.0, 0.05), ('B3D1', 'leap', 'param 1', 2.0, 0.04)))
    with pytest.raises(AssertionError, match='more than one configuration'):
        ms.correct_leap_runtime(con, str(tmp_path))


def setup_outputs(con, tmp_path, leap, otif):
    # One B3D1 clip of 3 frames, with LEAP's and OTIF param 0's raw outputs given as JSON-ready values.
    make_points(con, [('B3D1', 'reference', 'naive', 10.0, 0.8), ('B3D1', 'otif', 'param 0', 1.0, 0.5),
                      ('B3D1', 'leap', 'param 0', 1.0, 0.05)], table='p142_points')
    con.execute("CREATE OR REPLACE TABLE clips AS SELECT 'B3D1' AS dataset, 'jnc0' AS raw, 1 AS video_id, 'te01.mp4' AS video, 3.0::DOUBLE AS frames")
    for sub, data in (('leap', leap), ('0', otif)):
        (tmp_path / 'jnc0' / sub).mkdir(parents=True, exist_ok=True)
        (tmp_path / 'jnc0' / sub / '1.json').write_text(json.dumps(data))


def test_output_alignment_counts_length_nulls_and_duplicates(con, tmp_path):
    # LEAP: 4 entries for 3 frames, the first frame holding the same box twice. OTIF: 3 entries, one null (skipped).
    other = dict(BOX, left=9.0)
    setup_outputs(con, tmp_path, leap=[[BOX, BOX], [BOX], [BOX], [BOX]], otif=[[BOX], None, [other, BOX]])
    out = ms.output_alignment(con, str(tmp_path) + '/')['B3D1']
    leap, otif = out['leap']['param 0'], out['otif']['param 0']
    assert (leap['videos_longer_than_clip'], leap['max_extra_entries'], leap['null_entries']) == (1, 1, 0)
    assert (leap['detections'], leap['duplicate_boxes'], leap['duplicate_share']) == (5, 1, 0.2)
    # Distinct boxes in one frame are not duplicates, and a null entry is a skipped frame.
    assert (otif['videos_longer_than_clip'], otif['videos_shorter_than_clip'], otif['null_entries']) == (0, 0, 1)
    assert (otif['detections'], otif['duplicate_boxes']) == (3, 0)


def test_output_alignment_null_file_and_missing_file(con, tmp_path):
    # A null output counts as empty: shorter than the clip, no detections, no share.
    setup_outputs(con, tmp_path, leap=None, otif=[[], [], []])
    out = ms.output_alignment(con, str(tmp_path))['B3D1']
    assert (out['leap']['param 0']['entries'], out['leap']['param 0']['videos_shorter_than_clip']) == (0, 1)
    assert out['leap']['param 0']['duplicate_share'] is None and out['otif']['param 0']['detections'] == 0
    # A missing output file is an error naming it.
    (tmp_path / 'jnc0' / '0' / '1.json').unlink()
    with pytest.raises(AssertionError, match='missing raw SOTA outputs'):
        ms.output_alignment(con, str(tmp_path))


def test_bound_specs_names_relative_and_fit_bounds(con):
    make_points(con, TWO_DATASETS)
    specs = ms.bound_specs(con, 'hota_points', [0.05, 0.1], 'leap')
    # Relative bounds are named by their loss; the fit bound is LEAP's best HOTA per dataset.
    assert list(specs) == ['0.05', '0.10', 'leap_fit']
    assert specs['0.05'] == dict(loss=0.05, bounds=None)
    assert specs['leap_fit'] == dict(loss=None, bounds={'A': 0.05, 'B': 0.02})
    # Without a fit system only the relative bounds remain; a zero loss is valid.
    assert list(ms.bound_specs(con, 'hota_points', [0.0, 0.05], None)) == ['0.00', '0.05']


@pytest.mark.parametrize('losses, message', [([0.05, 0.054], 'collide'), ([0.05, 0.046], 'collide'),
                                             ([-0.1], r'\[0, 1\)'), ([1.0], r'\[0, 1\)'), ([1.5], r'\[0, 1\)')])
def test_bound_specs_rejects_colliding_or_invalid_losses(con, losses, message):
    # A bound silently replaced by another, or one above the reference or at zero, is a caller error.
    make_points(con, TWO_DATASETS)
    with pytest.raises(AssertionError, match=message):
        ms.bound_specs(con, 'hota_points', losses, None)


def test_macro_suffix_names_whole_percentages_only():
    # p203 names 1% to 10%; other losses and fit bounds have no macro.
    assert (ms.macro_suffix(0.05), ms.macro_suffix(0.1), ms.macro_suffix(0.01)) == ('FivePct', 'TenPct', 'OnePct')
    assert (ms.macro_suffix(0.075), ms.macro_suffix(0.2), ms.macro_suffix(None)) == (None, None, None)


def test_speedup_stats_per_bound_and_system(con):
    make_points(con, TWO_DATASETS)
    run_bounds(con, [0.05])
    stats = {(s['bound'], s['system']): s for s in ms.speedup_stats(con, 'chosen')}
    # Tetris meets 5% on both: t2 on A (20x) and t1 on B (12x).
    tetris = stats[('0.05', 'tetris')]
    assert (tetris['datasets'], tetris['meets_bound']) == (2, 2)
    assert (tetris['min'], tetris['max'], tetris['mean'], tetris['max_dataset']) == (12.0, 20.0, 16.0, 'A')
    # OTIF meets only on A, with its slower configuration (param 1 misses at 0.73).
    otif = stats[('0.05', 'otif')]
    assert (otif['meets_bound'], otif['min'], otif['max']) == (1, 4.0, 4.0)
    # LEAP misses everywhere: counted but without speedups.
    leap = stats[('0.05', 'leap')]
    assert (leap['datasets'], leap['meets_bound'], leap['min'], leap['mean'], leap['max_dataset']) == (2, 0, None, None, None)


def test_speedup_stats_breaks_ties_by_p204_dataset_order(con):
    # Amsterdam, B3D2, and B3D1 tie at 10x; p204's order (CONFIG) puts jnc0 (B3D1) before jnc2 and ams-y05.
    make_points(con, [(d, s, c, f, h) for d in ('Amsterdam', 'B3D2', 'B3D1')
                      for s, c, f, h in (('reference', 'naive', 10.0, 0.8), ('tetris', 't', 100.0, 0.8))])
    ms.create_dataset_order(con, ['ams-y05', 'jnc2', 'jnc0'])
    ms.choose(con, 'hota_points', ms.bound_specs(con, 'hota_points', [0.05], None), ['tetris'], 'chosen')
    assert ms.speedup_stats(con, 'chosen')[0]['max_dataset'] == 'B3D1'
    # Without an order, ties fall back to the dataset name.
    ms.create_dataset_order(con, [])
    assert ms.speedup_stats(con, 'chosen')[0]['max_dataset'] == 'Amsterdam'


def test_ratio_stats_fastest_prior_and_fail_count(con):
    make_points(con, TWO_DATASETS)
    run_bounds(con, [0.05, 0.10])
    ranges = ms.arrange_ranges(ms.speedup_stats(con, 'chosen'), ms.ratio_stats(con, 'chosen'))
    # At 5% only OTIF on A qualifies: Tetris 200 / 40 = 5x, and B has no qualifying prior.
    assert ranges['0.05']['tetris_vs'] == {'otif': dict(datasets=1, min=5.0, max=5.0),
                                           'fastest prior': dict(datasets=1, min=5.0, max=5.0)}
    assert (ranges['0.05']['prior_meets'], ranges['0.05']['prior_fail']) == (1, 1)
    # At 10% OTIF's 100 fps configuration qualifies on A (0.73 >= 0.72) and Tetris keeps t2 (t1's 0.70 misses): 2x.
    assert ranges['0.10']['tetris_vs']['otif'] == dict(datasets=1, min=2.0, max=2.0)
    # At LEAP's bound everything qualifies; on A the fastest prior is LEAP (150 fps), on B it is OTIF (500 fps).
    assert ranges['leap_fit']['tetris_vs']['fastest prior'] == dict(datasets=2, min=0.2, max=2.0)
    assert ranges['leap_fit']['tetris_vs']['leap'] == dict(datasets=2, min=2.0, max=6.0)
    assert (ranges['leap_fit']['prior_meets'], ranges['leap_fit']['prior_fail']) == (2, 0)


def test_ratio_stats_counts_prior_where_tetris_misses(con):
    # On B Tetris misses 5% but OTIF meets it: B counts toward prior_meets and is left out of the ratios.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 50.0, 0.8),
                      ('A', 'otif', 'param 0', 25.0, 0.8), ('A', 'leap', 'param 0', 9.0, 0.1),
                      ('B', 'reference', 'naive', 10.0, 0.8), ('B', 'tetris', 't', 50.0, 0.7),
                      ('B', 'otif', 'param 0', 25.0, 0.79), ('B', 'leap', 'param 0', 9.0, 0.1)])
    run_bounds(con, [0.05], fit=None)
    ranges = ms.arrange_ranges(ms.speedup_stats(con, 'chosen'), ms.ratio_stats(con, 'chosen'))['0.05']
    assert (ranges['prior_meets'], ranges['prior_fail']) == (2, 0)
    assert ranges['tetris_vs']['fastest prior'] == dict(datasets=1, min=2.0, max=2.0)
    assert ranges['systems']['tetris']['meets_bound'] == 1


def test_ratio_stats_without_any_qualifying_prior(con):
    # 5% has no qualifying prior while 50% has one: the empty bound gets no ratios and full prior_fail.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 50.0, 0.8),
                      ('A', 'otif', 'param 0', 90.0, 0.5), ('A', 'leap', 'param 0', 90.0, 0.1)])
    run_bounds(con, [0.05, 0.5], fit=None)
    ranges = ms.arrange_ranges(ms.speedup_stats(con, 'chosen'), ms.ratio_stats(con, 'chosen'))
    assert ranges['0.05']['tetris_vs'] == {}
    assert (ranges['0.05']['prior_meets'], ranges['0.05']['prior_fail']) == (0, 1)
    assert ranges['0.50']['tetris_vs']['otif'] == dict(datasets=1, min=0.6, max=0.6)


def test_choose_with_no_rows_leaves_an_empty_table(con):
    # A table holding only reference rows has no candidate for any system.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8)])
    ms.choose(con, 'hota_points', ms.bound_specs(con, 'hota_points', [0.05], None), ms.SYSTEMS, 'chosen')
    assert con.execute('SELECT count(*) FROM chosen').fetchone()[0] == 0
    assert ms.speedup_stats(con, 'chosen') == []


def test_fix_ratios_uses_unrounded_throughput(con):
    # B3D4-like throughputs where rounding to 0.1 fps before dividing moves the ratio by 0.1.
    make_points(con, [('A', 'reference', 'naive', 14.5, 0.8), ('A', 'tetris', 't', 663.94, 0.8),
                      ('A', 'otif', 'param 0', 25.66, 0.8), ('A', 'leap', 'param 0', 1.0, 0.0)])
    specs = ms.bound_specs(con, 'hota_points', [0.05], None)
    points = {n: common.operating_points(con, 'hota_points', ms.SYSTEMS, **s) for n, s in specs.items()}
    ms.choose(con, 'hota_points', specs, ms.SYSTEMS, 'chosen')
    # The shared helper divides 663.9 by 25.7 (25.83 -> 25.8); the true ratio is 25.87 -> 25.9.
    assert points['0.05']['A']['tetris_speedup_vs_otif'] == 25.8
    fixed = ms.fix_ratios(points, con, 'chosen')
    assert fixed['0.05']['A']['tetris_speedup_vs_otif'] == round(663.94 / 25.66, 1) == 25.9
    # A prior system that misses the bound still gets no ratio.
    assert 'tetris_speedup_vs_leap' not in fixed['0.05']['A']


def test_fix_ratios_corrects_both_priors_and_checks_agreement(con):
    # At LEAP's bound both priors qualify; both ratios are recomputed from unrounded throughputs.
    make_points(con, [('A', 'reference', 'naive', 14.5, 0.8), ('A', 'tetris', 't', 663.94, 0.8),
                      ('A', 'otif', 'param 0', 25.66, 0.8), ('A', 'leap', 'param 0', 16.44, 0.05)])
    specs = ms.bound_specs(con, 'hota_points', [], 'leap')
    points = {n: common.operating_points(con, 'hota_points', ms.SYSTEMS, **s) for n, s in specs.items()}
    ms.choose(con, 'hota_points', specs, ms.SYSTEMS, 'chosen')
    fixed = ms.fix_ratios(points, con, 'chosen')['leap_fit']['A']
    assert (fixed['tetris_speedup_vs_otif'], fixed['tetris_speedup_vs_leap']) == (25.9, round(663.94 / 16.44, 1))
    # A report whose configuration differs from the chosen table is an error.
    points['leap_fit']['A']['systems']['otif']['config'] = 'param 9'
    with pytest.raises(AssertionError, match='configurations'):
        ms.fix_ratios(points, con, 'chosen')
    # So is a ratio the chosen table does not have.
    points['leap_fit']['A']['systems']['otif']['config'] = 'param 0'
    points['leap_fit']['A']['tetris_speedup_vs_other'] = 1.0
    with pytest.raises(AssertionError, match='ratio pairs'):
        ms.fix_ratios(points, con, 'chosen')


def matched_rows(con, losses, fit=None):
    # The accuracy-matched table per bound and dataset, as dicts.
    run_bounds(con, losses, fit=fit)
    ms.accuracy_matched(con, 'hota_points', 'chosen', 'matched')
    return {(r['bound'], r['dataset']): r for r in common.query_dicts(con, 'SELECT * FROM matched')}


def test_accuracy_matched_uses_fastest_prior_when_not_more_accurate(con):
    # 10%: Tetris 300 fps at 0.73; OTIF's fastest point within the bound is 100 fps at 0.73, not more accurate.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 300.0, 0.73),
                      ('A', 'otif', 'param 0', 40.0, 0.78), ('A', 'otif', 'param 1', 100.0, 0.73),
                      ('A', 'leap', 'param 0', 150.0, 0.05)])
    row = matched_rows(con, [0.10])[('0.10', 'A')]
    assert (row['rule'], row['prior'], row['prior_config'], row['tetris_fps'] / row['prior_fps']) == (
        'threshold_feasible_not_more_accurate', 'otif', 'param 1', 3.0)


def test_accuracy_matched_falls_back_to_pareto_point_outside_bound(con):
    # OTIF's in-bound point (500 fps, 0.79) is more accurate than Tetris (0.761), so p203 takes the most accurate
    # Pareto point not above 0.761: param 1 (0.70); param 2 (0.75) is dominated by param 0 and is not on the front.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 120.0, 0.761),
                      ('A', 'otif', 'param 0', 500.0, 0.79), ('A', 'otif', 'param 1', 600.0, 0.70),
                      ('A', 'otif', 'param 2', 300.0, 0.75), ('A', 'leap', 'param 0', 20.0, 0.02)])
    row = matched_rows(con, [0.05])[('0.05', 'A')]
    assert (row['rule'], row['prior_config'], row['prior_hota']) == ('nearest_not_more_accurate', 'param 1', 0.70)
    # The matched point misses the 5% bound (0.70 < 0.76), which the summary counts.
    stats = ms.matched_stats(con, 'chosen', 'matched')[0]
    assert (stats['matched'], stats['min'], stats['matched_outside_bound'], stats['prior_systems']) == (1, 0.2, 1, ['otif'])


def test_accuracy_matched_without_a_comparable_prior(con):
    # Every prior point is more accurate than Tetris: no match, and the bound still has a summary row.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 120.0, 0.77),
                      ('A', 'otif', 'param 0', 50.0, 0.79), ('A', 'leap', 'param 0', 20.0, 0.78)])
    row = matched_rows(con, [0.05])[('0.05', 'A')]
    assert row['prior'] is None and row['rule'] is None
    assert ms.matched_stats(con, 'chosen', 'matched')[0] == dict(bound='0.05', matched=0, min=None, max=None,
                                                                 matched_outside_bound=0, prior_systems=[])


def test_accuracy_matched_ties_go_to_otif(con):
    # OTIF and LEAP share HOTA and throughput: p203 keeps the earlier system, OTIF.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 120.0, 0.77),
                      ('A', 'otif', 'param 0', 50.0, 0.5), ('A', 'leap', 'param 0', 50.0, 0.5)])
    assert matched_rows(con, [0.05])[('0.05', 'A')]['prior'] == 'otif'
    # The same holds for the fastest point within the bound.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 120.0, 0.79),
                      ('A', 'otif', 'param 0', 50.0, 0.77), ('A', 'leap', 'param 0', 50.0, 0.77)])
    row = matched_rows(con, [0.05])[('0.05', 'A')]
    assert (row['prior'], row['rule']) == ('otif', 'threshold_feasible_not_more_accurate')


def test_accuracy_matched_skips_datasets_where_tetris_misses(con):
    # Tetris misses 5% on A, so A has no accuracy-matched row.
    make_points(con, [('A', 'reference', 'naive', 10.0, 0.8), ('A', 'tetris', 't', 120.0, 0.70),
                      ('A', 'otif', 'param 0', 50.0, 0.5), ('A', 'leap', 'param 0', 20.0, 0.1)])
    assert matched_rows(con, [0.05]) == {}


def test_dominance_fastest_tetris_at_least_as_accurate(con):
    make_points(con, [
        ('A', 'reference', 'naive', 10.0, 0.8),
        # OTIF's 5% point is param 0 at HOTA 0.78.
        ('A', 'otif', 'param 0', 40.0, 0.78),
        # Tetris: t1 is faster but less accurate; t2 and t3 tie on speed at or above 0.78, t3 is more accurate.
        ('A', 'tetris', 't1', 300.0, 0.77), ('A', 'tetris', 't2', 80.0, 0.78), ('A', 'tetris', 't3', 80.0, 0.79),
        ('A', 'leap', 'param 0', 5.0, 0.01),
        ('B', 'reference', 'naive', 10.0, 0.8),
        # On B OTIF qualifies but no Tetris configuration is as accurate.
        ('B', 'otif', 'param 0', 20.0, 0.80), ('B', 'tetris', 't1', 100.0, 0.79),
        ('B', 'leap', 'param 0', 5.0, 0.01),
    ])
    run_bounds(con, [0.05], fit=None)
    out = ms.arrange_dominance(ms.dominance(con, 'hota_points', 'chosen'))
    # Equal throughput goes to the more accurate configuration; >= is inclusive.
    assert out['0.05']['A']['otif']['tetris'] == dict(config='t3', hota=0.79, fps=80.0, speedup_vs_reference=8.0,
                                                      speedup_vs_prior=2.0)
    assert out['0.05']['A']['otif']['prior'] == dict(config='param 0', hota=0.78, fps=40.0, speedup_vs_reference=4.0)
    # No Tetris configuration matches OTIF on B.
    assert out['0.05']['B']['otif']['tetris'] is None
    # LEAP misses the bound, so it has no dominance entry.
    assert 'leap' not in out['0.05']['A']


def test_check_same_mismatches_and_rounded():
    # Floats match up to summation noise, None matches only None, and other values match exactly.
    out = ms.check({'a': 1.0, 'b': None, 'c': 7, 'd': 'B3D1', 'e': 2.0}, {'a': 1.0 + 1e-12, 'b': None, 'c': 7, 'd': 'B3D2'}, 'p203')
    assert out['a'] == {'p203': 1.0, 'measured': 1.0 + 1e-12, 'match': True}
    assert (out['b']['match'], out['c']['match'], out['d']['match'], out['e']['match']) == (True, True, False, False)
    assert ms.mismatches({'0.05': out}) == ['0.05/d', '0.05/e']
    # Rounding for the manuscript touches floats only.
    assert ms.rounded({'x': 4.66, 'n': 7, 's': 'A', 'z': None}) == {'x': 4.7, 'n': 7, 's': 'A', 'z': None}


def test_r1_rounds_and_keeps_missing():
    # One decimal, as the macros print, and None stays None.
    assert ms.r1(4.66) == 4.7
    assert ms.r1(None) is None


def make_knobs(con, rows, ref=(('B3D1', 10.0, 0.8),)):
    # tetris_knobs from (dataset, config, sample_rate, threshold, fps, hota) rows, and reference rows in hota_points.
    make_points(con, [(d, 'reference', 'naive', f, h) for d, f, h in ref])
    con.execute('CREATE OR REPLACE TABLE tetris_knobs (dataset VARCHAR, config VARCHAR, classifier VARCHAR, '
                'sample_rate DOUBLE, threshold DOUBLE, fps DOUBLE, hota DOUBLE)')
    con.executemany('INSERT INTO tetris_knobs VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(d, c, 'ShuffleNet05', float(s), t, f, h) for d, c, s, t, f, h in rows])


def test_ablation_conditions_follow_sample_rate_and_pruning(con):
    # No pruning at s = 1, pruning at s = 1, and pruning with whole-frame sampling.
    make_knobs(con, [('B3D1', 'plain', 1, None, 20.0, 0.8), ('B3D1', 'pruned', 1, 0.6, 40.0, 0.8),
                     ('B3D1', 'sampled', 4, 0.6, 90.0, 0.8), ('B3D1', 'sampled_plain', 2, None, 60.0, 0.8)])
    ms.ablation_points(con)
    members = {(r[0], r[1]) for r in con.execute("SELECT system, config FROM ablation_points WHERE system <> 'reference'").fetchall()}
    # no_both keeps only s = 1 without pruning (the NULL threshold).
    assert {c for s, c in members if s == 'no_both'} == {'plain'}
    # no_sampling keeps every s = 1 configuration.
    assert {c for s, c in members if s == 'no_sampling'} == {'plain', 'pruned'}
    # full keeps everything.
    assert {c for s, c in members if s == 'full'} == {'plain', 'pruned', 'sampled', 'sampled_plain'}
    # One reference row per dataset, from hota_points.
    assert con.execute("SELECT count(*) FROM ablation_points WHERE system = 'reference'").fetchone()[0] == 1


def test_ablation_report_mean_max_and_missing(con):
    # B3D1: every condition qualifies; B3D2: the no-pruning configuration misses the 5% bound (0.70 < 0.76).
    make_knobs(con, [('B3D1', 'plain', 1, None, 30.0, 0.8), ('B3D1', 'pruned', 1, 0.6, 50.0, 0.79),
                     ('B3D1', 'sampled', 4, 0.6, 200.0, 0.77),
                     ('B3D2', 'plain', 1, None, 90.0, 0.70), ('B3D2', 'pruned', 1, 0.6, 70.0, 0.78),
                     ('B3D2', 'sampled', 4, 0.6, 100.0, 0.76)],
               ref=(('B3D1', 10.0, 0.8), ('B3D2', 10.0, 0.8)))
    ms.ablation_points(con)
    out, macros = ms.ablation_report(con, [0.05])
    five = out['0.05']
    # no_both qualifies on B3D1 only (3x); the mean covers that dataset alone, and B3D2 is None.
    assert (five['no_both']['meets_bound'], five['no_both']['mean_speedup'], five['no_both']['max_dataset']) == (1, 3.0, 'B3D1')
    assert five['no_both']['per_dataset']['B3D2'] is None and five['no_both']['datasets'] == 2
    # no_sampling: 5x and 7x; full: 20x and 10x (the boundary 0.76 is inclusive).
    assert (five['no_sampling']['mean_speedup'], five['no_sampling']['max_speedup'], five['no_sampling']['max_dataset']) == (6.0, 7.0, 'B3D2')
    assert (five['full']['mean_speedup'], five['full']['max_speedup']) == (15.0, 20.0)
    assert five['full']['per_dataset']['B3D2'] == dict(config='sampled', hota=0.76, fps=100.0, speedup_vs_reference=10.0)
    # Candidate pools per condition and dataset.
    assert five['no_both']['candidates'] == {'B3D1': 1, 'B3D2': 1} and five['full']['candidates'] == {'B3D1': 3, 'B3D2': 3}
    # Unrounded macro values under p204's names.
    assert macros['0.05']['ablationOneSpeedupAvg'] == 3.0 and macros['0.05']['ablationThreeSpeedupMaxDataset'] == 'B3D1'


def test_ablation_report_condition_without_candidates(con):
    # B3D2 has no s = 1 configuration, so no_both and no_sampling have no candidate there.
    make_knobs(con, [('B3D1', 'plain', 1, None, 30.0, 0.8), ('B3D2', 'sampled', 4, 0.6, 100.0, 0.8)],
               ref=(('B3D1', 10.0, 0.8), ('B3D2', 10.0, 0.8)))
    ms.ablation_points(con)
    out, macros = ms.ablation_report(con, [0.05])
    # Every dataset still appears, as None, with zero candidates; the mean covers B3D1 only.
    no_both = out['0.05']['no_both']
    assert (no_both['datasets'], no_both['meets_bound'], no_both['per_dataset']['B3D2'], no_both['candidates']['B3D2']) == (2, 1, None, 0)
    assert no_both['mean_speedup'] == 3.0
    # A condition with no candidate anywhere has None macro values.
    make_knobs(con, [('B3D2', 'sampled', 4, 0.6, 100.0, 0.8)], ref=(('B3D2', 10.0, 0.8),))
    ms.ablation_points(con)
    out, macros = ms.ablation_report(con, [0.05])
    assert out['0.05']['no_both']['per_dataset'] == {'B3D2': None}
    assert (macros['0.05']['ablationOneSpeedupAvg'], macros['0.05']['ablationOneSpeedupMaxDataset']) == (None, None)


# Synthetic raw inputs for main(). Per dataset: reference (fps, HOTA); Tetris (config, sample rate, threshold, fps, HOTA);
# OTIF (param, fps, HOTA); LEAP (deduplicated fps, HOTA); clips (video id, frames); LEAP's runs per video in its stat CSV.
DIFFERENTIAL = {
    # Tetris's 5% point sits exactly on the bound (0.8 x 0.95); OTIF param 2 is dominated (off its Pareto front).
    'jnc0': dict(ref=(10.0, 0.8), clips=((1, 60), (4, 40)), leap=(150.0, 0.05),
                 tetris=(('plain', 1, None, 30.0, 0.80), ('pruned', 1, 0.6, 50.0, 0.79), ('boundary', 2, 0.6, 120.0, 0.76),
                         ('fast', 4, 0.4, 300.0, 0.73), ('dominated', 2, 0.8, 40.0, 0.75)),
                 otif=((0, 40.0, 0.78), (1, 100.0, 0.73), (2, 20.0, 0.70))),
    # OTIF's in-bound point is more accurate than Tetris, so the accuracy-matched rule falls back to param 1;
    # without the Pareto step it would take the dominated param 2. LEAP's stat CSV lists every video twice.
    'jnc2': dict(ref=(10.0, 0.8), clips=((1, 50), (4, 50)), leap=(20.0, 0.02), leap_runs=2,
                 tetris=(('t1', 1, None, 120.0, 0.761), ('t2', 1, 0.6, 200.0, 0.70)),
                 otif=((0, 500.0, 0.79), (1, 600.0, 0.70), (2, 300.0, 0.75))),
    # Tetris misses 5% while OTIF meets it; at 10% the matched prior point is LEAP's.
    'jnc6': dict(ref=(10.0, 0.8), clips=((1, 100),), leap=(10.0, 0.01),
                 tetris=(('t1', 1, None, 50.0, 0.74),), otif=((0, 30.0, 0.77),)),
}
# One simple dataset shape, used for all seven datasets.
SIMPLE = dict(ref=(10.0, 0.8), clips=((1, 100),), leap=(60.0, 0.03), otif=((0, 40.0, 0.78),),
              tetris=(('plain', 1, None, 30.0, 0.8), ('pruned', 1, 0.6, 50.0, 0.79), ('sampled', 4, 0.6, 200.0, 0.77)))


def install_world(tmp_path, monkeypatch, world, args=()):
    # Fake p130/p142 tables, clip manifests, and a raw SOTA directory for every dataset, then point main() at them.
    frames = {d: sum(f for _, f in w['clips']) for d, w in world.items()}

    def tradeoff(d):
        # The canonical tradeoff.csv shape, with a validation-split row that would win every bound if not filtered.
        w = world[d]
        rows = [dict(dataset=d, videoset='test', variant='naive', variant_id='naive', classifier=None, sample_rate=1.0,
                     tracking_accuracy_threshold=math.nan, throughput_fps=w['ref'][0], HOTA_HOTA=w['ref'][1], frame_count=frames[d])]
        rows += [dict(dataset=d, videoset='test', variant='polytris', variant_id=c, classifier=w.get('classifier', 'ShuffleNet05'),
                      sample_rate=float(s), tracking_accuracy_threshold=math.nan if t is None else float(t),
                      throughput_fps=f, HOTA_HOTA=h, frame_count=frames[d]) for c, s, t, f, h in w['tetris']]
        rows.append(dict(dataset=d, videoset='valid', variant='polytris', variant_id='valid_only', classifier='ShuffleNet05',
                         sample_rate=1.0, tracking_accuracy_threshold=math.nan, throughput_fps=1e6, HOTA_HOTA=0.99,
                         frame_count=frames[d]))
        return pd.DataFrame(rows)

    def sota(datasets, system):
        # load_sota_tradeoff_data's shape; LEAP's p142 runtime counts every repeated run, as p140 does.
        rows = []
        for d in datasets:
            w = world[d]
            params = ([(p, frames[d] / f, h) for p, f, h in w['otif']] if system == 'otif'
                      else [(0, w.get('leap_runs', 1) * frames[d] / w['leap'][0], w['leap'][1])])
            rows += [dict(system=system.upper(), param_id=p, dataset=d, videoset='test', time=t, HOTA_HOTA=h,
                          throughput_fps=math.nan, sample_rate=1) for p, t, h in params]
            rows.append(dict(system=system.upper(), param_id=99, dataset=d, videoset='valid', time=1e-6, HOTA_HOTA=0.99,
                             throughput_fps=math.nan, sample_rate=1))
        return pd.DataFrame(rows)

    # The raw SOTA directory: LEAP's stat CSV and one-box-per-frame outputs for LEAP and every OTIF param.
    sota_dir = tmp_path / 'sota'
    for d, w in world.items():
        root = sota_dir / ms.dataset_name_in_sota(d)
        (root / 'leap').mkdir(parents=True)
        runtime, runs = frames[d] / w['leap'][0], w.get('leap_runs', 1)
        lines = [LEAP_HEADER] + [f'{i},{runtime * f / frames[d] + 1.0},0,0,0,0,1.0,{f},{f // 2}'
                                 for _ in range(runs) for i, f in w['clips']]
        lines.append(f"total,{runs * (runtime + len(w['clips']))},0,0,0,0,{runs * len(w['clips'])},0,0")
        (root / f'leap_{root.name}.csv').write_text('\n'.join(lines) + '\n')
        for i, f in w['clips']:
            (root / 'leap' / f'{i}.json').write_text(json.dumps([[BOX, BOX]] * (f + 3)))
            for p, _, _ in w['otif']:
                (root / str(p)).mkdir(exist_ok=True)
                (root / str(p) / f'{i}.json').write_text(json.dumps([[BOX], None] * (f // 2)))
    # Route every loader, in this script, common, p203, and p204, to the fakes.
    for module in (common, p203):
        monkeypatch.setattr(module, 'load_tradeoff_data', tradeoff)
        monkeypatch.setattr(module, 'load_sota_tradeoff_data', sota)
    monkeypatch.setattr(p204, 'load_tradeoff_data', tradeoff)
    monkeypatch.setattr(ms, 'clip_manifest', lambda d: [dict(video_id=i, video=f'te{i:02d}.mp4', frames=f) for i, f in world[d]['clips']])
    monkeypatch.setattr(common, 'RESULTS_ROOT', tmp_path / 'results')
    monkeypatch.setattr(sys, 'argv', ['r7o1_hota_matched_speedup.py', '--sota-dir', str(sota_dir), *args])
    return tmp_path / 'results' / 'r7o1_hota_matched_speedup'


def read_reports(out):
    # Both JSON reports of a run.
    return (json.loads((out / 'matched_speedup.json').read_text()), json.loads((out / 'ablation_speedup.json').read_text()))


def test_main_agrees_with_p203_and_p204(tmp_path, monkeypatch):
    # Datasets given out of p204's order; the port check runs p203's and p204's own functions on the same raw inputs.
    out = install_world(tmp_path, monkeypatch, DIFFERENTIAL, ['--datasets', 'jnc6', 'jnc2', 'jnc0'])
    ms.main()
    report, ablation = read_reports(out)
    # Every p203 and p204 value agrees (main would have failed otherwise).
    assert ms.mismatches(report['port_check']) == [] and ms.mismatches(ablation['port_check']) == []
    five, ten = report['port_check']['0.05'], report['port_check']['0.10']
    # 5%: Tetris meets on B3D1 (exactly on the bound) and B3D2; OTIF meets on all three; matched ratios 1.2 and 0.2.
    assert {k: v['measured'] for k, v in five.items()} == pytest.approx({
        'comparePolytrisMeetFivePct': 2, 'comparePriorMeetFivePct': 3, 'comparePriorFailDatasetsFivePct': 0,
        'compareAccuracyMatchedPriorCountFivePct': 2, 'compareSpeedupMinFivePct': 0.24, 'compareSpeedupMaxFivePct': 3.0,
        'compareAccuracyMatchedSpeedupMinFivePct': 0.2, 'compareAccuracyMatchedSpeedupMaxFivePct': 1.2,
        'compareNaiveSpeedupMinFivePct': 12.0, 'compareNaiveSpeedupMaxFivePct': 12.0})
    # 10%: B3D3's matched prior is LEAP (5x), B3D1's uses the in-bound rule (3x).
    assert (ten['compareAccuracyMatchedSpeedupMinTenPct']['measured'], ten['compareAccuracyMatchedSpeedupMaxTenPct']['measured']) == pytest.approx((0.2, 5.0))
    matched = report['accuracy_matched']
    assert matched['0.05']['per_dataset']['B3D2']['prior'] == dict(system='otif', config='param 1', hota=0.7, fps=600.0,
                                                                   meets_bound=False, rule='nearest_not_more_accurate')
    assert matched['0.10']['per_dataset']['B3D3']['prior']['system'] == 'leap'
    assert matched['0.10']['per_dataset']['B3D1']['prior']['rule'] == 'threshold_feasible_not_more_accurate'
    # p204: full ties at 12x on B3D1 and B3D2 and p204 names B3D1; no_both covers B3D1 (3x) and B3D2 (12x).
    abl = {k: v['measured'] for k, v in ablation['port_check']['0.05'].items()}
    assert (abl['ablationThreeSpeedupMax'], abl['ablationThreeSpeedupMaxDataset']) == (12.0, 'B3D1')
    assert (abl['ablationOneSpeedupAvg'], abl['ablationOneSpeedupMaxDataset']) == (7.5, 'B3D2')
    # The validation-split rows never win.
    assert report['operating_points']['0.05']['B3D1']['systems']['tetris']['config'] == 'boundary'
    # LEAP's doubled stat rows on B3D2 are deduplicated: 20 fps instead of p142's 10 fps.
    assert (report['leap_checks']['runtime']['B3D2']['p142_fps'], report['leap_checks']['runtime']['B3D2']['deduplicated_fps']) == (10.0, 20.0)
    assert report['operating_points']['leap_fit']['B3D2']['systems']['leap']['fps'] == 20.0
    # LEAP's outputs are 3 entries longer than every clip and half their boxes are duplicates; OTIF's end early.
    alignment = report['leap_checks']['output_alignment']['B3D1']
    assert (alignment['leap']['param 0']['videos_longer_than_clip'], alignment['leap']['param 0']['duplicate_share']) == (2, 0.5)
    assert alignment['otif']['param 0']['videos_longer_than_clip'] == 0
    # A subset run skips the manuscript checks.
    assert report['manuscript_check'].startswith('skipped') and ablation['manuscript_check'].startswith('skipped')


def test_main_seven_datasets_writes_checks_and_caveats(tmp_path, monkeypatch):
    # The same simple shape on all seven datasets, with LEAP's CalDoT stat rows doubled as in the real export.
    world = {d: dict(SIMPLE, leap_runs=2 if d.startswith('caldot') else 1) for d in common.DATASET_LABELS}
    out = install_world(tmp_path, monkeypatch, world)
    ms.main()
    report, ablation = read_reports(out)
    # Three bounds over seven datasets, and both manuscript checks with the accuracy-matched macros.
    assert sorted(report['operating_points']) == ['0.05', '0.10', 'leap_fit'] and len(report['operating_points']['0.05']) == 7
    assert sorted(report['manuscript_check']) == ['0.05', '0.10']
    assert report['manuscript_check']['0.05']['compareAccuracyMatchedSpeedupMaxFivePct']['manuscript'] == 17.4
    assert sorted(ablation['manuscript_check']) == sorted(ms.MANUSCRIPT_ABLATION)
    # Tetris's 5% point is 'sampled' (20x) and OTIF's is 4x, so Tetris is 5x OTIF under the shared bound.
    assert report['operating_points']['0.05']['B3D1']['tetris_speedup_vs_otif'] == 5.0
    # OTIF's in-bound point (0.78) is more accurate than Tetris's (0.77), so p203's accuracy-matched rule falls back
    # to the most accurate prior point not above 0.77, LEAP's (0.03 HOTA, 60 fps): 200 / 60 = 3.3x, outside the bound.
    b3d1 = report['accuracy_matched']['0.05']['per_dataset']['B3D1']
    assert (b3d1['prior']['system'], b3d1['prior']['rule'], b3d1['prior']['meets_bound']) == ('leap', 'nearest_not_more_accurate', False)
    assert b3d1['tetris_speedup_vs_prior'] == 3.3
    # Matching OTIF's 0.78 HOTA instead, the fastest Tetris configuration is 'pruned' (50 / 40 = 1.25 -> 1.2).
    assert report['dominance']['0.05']['B3D1']['otif']['tetris']['config'] == 'pruned'
    assert report['dominance']['0.05']['B3D1']['otif']['tetris']['speedup_vs_prior'] == 1.2
    # LEAP is marked provisional, and CalDoT's doubled LEAP runtime is halved.
    assert report['provisional']['systems'] == ['leap'] and 'leap_fit' in report['provisional']['bounds']
    assert 'matched comparison' not in json.dumps(report['about'])
    assert report['leap_checks']['runtime']['CalDoT1']['max_runs_per_video'] == 2
    # The matched maximum is CalDoT's LEAP point: 200 / 30 = 6.7x with p142's doubled runtime, 200 / 60 = 3.3x deduplicated.
    effect = report['leap_runtime_effect']
    assert sorted(effect) == ['0.05', '0.10'] and list(effect['0.05']) == ['compareAccuracyMatchedSpeedupMaxFivePct']
    assert effect['0.05']['compareAccuracyMatchedSpeedupMaxFivePct'] == pytest.approx(dict(p142=200 / 30, deduplicated=200 / 60))
    # The B3D caveat and the input provenance are in both reports.
    assert 'R1.O1' in report['about']['ground_truth'] and 'R1.O1' in ablation['about']['ground_truth']
    assert report['about']['inputs']['run_at'] and report['about']['inputs']['files']['B3D1']['leap_stat']['exists']
    assert ablation['bounds']['0.05']['no_both']['mean_speedup'] == 3.0


def test_main_writes_nothing_on_perfect_classifier(tmp_path, monkeypatch):
    # Oracle-classifier rows stop the run before either report is written.
    out = install_world(tmp_path, monkeypatch, {'jnc0': dict(SIMPLE, classifier='Perfect')}, ['--datasets', 'jnc0'])
    with pytest.raises(AssertionError, match='Perfect'):
        ms.main()
    assert not list(out.glob('*.json'))


def test_main_stops_when_the_port_disagrees_with_p203(tmp_path, monkeypatch):
    # HOTA 1e-15 below the 5% bound: p203's round(loss, 12) admits it, HOTA >= bound does not. The run stops unwritten.
    world = {'jnc0': dict(SIMPLE, tetris=(('plain', 1, None, 30.0, 0.8), ('noise', 2, 0.6, 120.0, 0.76 - 1e-15)))}
    out = install_world(tmp_path, monkeypatch, world, ['--datasets', 'jnc0'])
    with pytest.raises(AssertionError, match='compareNaiveSpeedupMaxFivePct'):
        ms.main()
    assert not list(out.glob('*.json'))
