import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from measurements import r9o3_ilp_timings as it


# A valid p022 parameter string and the throughput variant id that shares its pruning run.
PARAM = 'ShuffleNet05_60_4_060_r050_sortcython'
VARIANT = 'ShuffleNet05_60_4_060_r050_none_s100_sortcython'
# A second pruning configuration and its variant id.
PARAM8 = 'ShuffleNet05_60_8_080_r050_sortcython'
VARIANT8 = 'ShuffleNet05_60_8_080_r050_none_s100_sortcython'
# The five operations p022 logs, in ms.
OPS = dict(prepare_bitmaps=1.0, group_tiles=2.0, build_ilp=3.0, solve_ilp=4.0, extract_grids=10.0)
# A file modification time (seconds since the epoch) that the tests offset from.
T0 = 1_700_000_000
# The two test videos of the extraction fixtures, and the split listing that goes with them.
TEST_VIDEOS = ('te00.mp4', 'te01.mp4')
LISTING = [('d', 'valid', 'va00.mp4')] + [('d', 'test', v) for v in TEST_VIDEOS]


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def write_run(root, dataset, video, param=PARAM, ops=None, time_limit=None, text=None, mtime=None, stage=None):
    # Write one runtime.jsonl as p022 does, or the given raw text, optionally set its mtime, and return its path.
    path = Path(root) / dataset / 'execution' / video / (stage or it.PRUNE_DIR) / param / 'score' / 'runtime.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'runtime': [{'op': k, 'time': v} for k, v in (ops or OPS).items()], 'time_limit': time_limit}
    path.write_text(text if text is not None else json.dumps(record) + '\n')
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


def listing(dataset, *videos):
    # (dataset, split, video) rows for the given video names, with the split taken from the name prefix.
    return [(dataset, it.SPLITS[v[:2]], v) for v in videos]


def clip_positions(con):
    # (dataset, video, param, invocation, launch_rank, clip_position) of every run, in a stable order.
    return con.execute('SELECT dataset, video, param, invocation, launch_rank, clip_position FROM runs '
                       'ORDER BY dataset, param, video').fetchall()


def set_runs(con, rows):
    # A runs table from (split, sample_rate, build_ms, solve_ms, stage_ms[, budget_s]) rows; the budget defaults to 0.1 s.
    con.execute('CREATE OR REPLACE TABLE runs (dataset VARCHAR, split VARCHAR, sample_rate INTEGER, budget_s DOUBLE, '
                'build_ms DOUBLE, solve_ms DOUBLE, build_plus_solve_ms DOUBLE, stage_ms DOUBLE)')
    con.executemany('INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    [('d', r[0], r[1], r[5] if len(r) > 5 else 0.1, r[2], r[3], r[2] + r[3], r[4]) for r in rows])


def test_default_time_limit_from_p022_is_one_tenth_second():
    # The real pipeline falls back to 0.1 s when --time-limit is not given.
    assert it.default_time_limit() == 0.1


def test_default_time_limit_parses_spacing_and_rejects_ambiguity(tmp_path):
    # Spacing around the keyword is tolerated.
    one = tmp_path / 'one.py'
    one.write_text('solve_ilp(x, time_limit_seconds = time_limit or 0.25)\n')
    assert it.default_time_limit(one) == 0.25
    # No call site, or two, cannot define the default.
    for text in ('solve_ilp(x)\n', 'a(time_limit_seconds=time_limit or 0.1)\nb(time_limit_seconds=time_limit or 0.5)\n'):
        path = tmp_path / 'bad.py'
        path.write_text(text)
        with pytest.raises(AssertionError, match='expected one'):
            it.default_time_limit(path)


def test_runtime_files_lists_only_the_pruning_stage_and_fails_on_empty_dataset(tmp_path):
    # One saved run, plus a time-limit experiment and another stage that must not be read.
    run = write_run(tmp_path, 'd1', 'va00.mp4')
    other = tmp_path / 'd1' / 'execution' / 'va00.mp4' / '022t_pruned_polyominoes_timelimit' / PARAM / 'tl0.5' / 'score'
    other.mkdir(parents=True)
    (other / 'runtime.jsonl').write_text('{}\n')
    assert it.runtime_files(tmp_path, ['d1']) == [run]
    # A dataset with no saved runs is an error, not an empty contribution.
    with pytest.raises(AssertionError, match='d2'):
        it.runtime_files(tmp_path, ['d1', 'd2'])


def test_split_videos_lists_sorted_video_files_and_fails_on_empty_or_missing_split(tmp_path, monkeypatch):
    # Dataset directories under tmp_path instead of /polyis-data.
    monkeypatch.setattr(it.store, 'dataset', lambda d, *a: tmp_path / d / Path(*a))
    # Valid and test videos, written out of order, with a non-video file that p022 ignores.
    for name in ('valid/va01.mp4', 'valid/va00.avi', 'valid/notes.txt', 'test/te00.mkv'):
        (tmp_path / 'd' / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'd' / name).write_text('')
    assert it.split_videos(['d']) == [('d', 'valid', 'va00.avi'), ('d', 'valid', 'va01.mp4'), ('d', 'test', 'te00.mkv')]
    # A split directory holding no video.
    (tmp_path / 'e' / 'valid').mkdir(parents=True)
    (tmp_path / 'e' / 'test').mkdir(parents=True)
    (tmp_path / 'e' / 'valid' / 'va00.mp4').write_text('')
    with pytest.raises(AssertionError, match='no videos listed'):
        it.split_videos(['e'])
    # A dataset without split directories at all.
    with pytest.raises(FileNotFoundError):
        it.split_videos(['missing'])


def test_load_runs_parses_path_params_budget_times_and_first_clip(con, tmp_path):
    files = [
        # Null and zero time limits both fall back to the default, as `time_limit or 0.1` does.
        write_run(tmp_path, 'd1', 'va00.mp4', time_limit=None),
        write_run(tmp_path, 'd1', 'te03.mp4', param='Some_Clf_60_16_080_r025_bytetrackcython', time_limit=0),
        # PARAM's only saved d1 test run is on te05, but its task started at te03, the first listed video.
        write_run(tmp_path, 'd1', 'te05.mp4'),
        # An explicit limit is kept.
        write_run(tmp_path, 'd2', 'te00.mp4', time_limit=0.5),
    ]
    it.load_runs(con, files, 0.1, listing('d1', 'va00.mp4', 'te03.mp4', 'te05.mp4') + listing('d2', 'te00.mp4'))
    rows = con.execute('SELECT dataset, video, split, sample_rate, tilesize, budget_s, build_ms, solve_ms, '
                       'build_plus_solve_ms, stage_ms, clip_position FROM runs ORDER BY dataset, video').fetchall()
    # All four tasks are among the first 20 an invocation starts, so first clips are launch clips.
    assert rows == [
        ('d1', 'te03.mp4', 'test', 16, 60, 0.1, 3.0, 4.0, 7.0, 20.0, 'launch_first_clip'),
        ('d1', 'te05.mp4', 'test', 4, 60, 0.1, 3.0, 4.0, 7.0, 20.0, 'later_clip'),
        ('d1', 'va00.mp4', 'valid', 4, 60, 0.1, 3.0, 4.0, 7.0, 20.0, 'launch_first_clip'),
        ('d2', 'te00.mp4', 'test', 4, 60, 0.5, 3.0, 4.0, 7.0, 20.0, 'launch_first_clip'),
    ]


def test_load_runs_first_clip_comes_from_the_listing_not_the_saved_runs(con, tmp_path):
    # Configuration A kept va00 and va01; configuration B lost va00 to a later invocation and kept only va01.
    a, b = 'A_60_4_060_r050_sortcython', 'B_60_4_060_r050_sortcython'
    files = [write_run(tmp_path, 'd', 'va00.mp4', param=a, mtime=T0), write_run(tmp_path, 'd', 'va01.mp4', param=a, mtime=T0),
             write_run(tmp_path, 'd', 'va01.mp4', param=b, mtime=T0)]
    it.load_runs(con, files, 0.1, listing('d', 'va00.mp4', 'va01.mp4'))
    # B's va01 was its task's second clip, so it is not a first clip.
    assert [r[-1] for r in clip_positions(con)] == ['launch_first_clip', 'later_clip', 'later_clip']


def test_load_runs_launch_batch_by_first_write_ties_and_invocations(con, tmp_path):
    # Five tasks of one dataset split, with two workers: written at T0, T0+1 (two tied), T0+5, and in a later invocation.
    p = {k: f'{k}_60_4_060_r050_sortcython' for k in 'ABCDE'}
    files = [
        write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['A'], mtime=T0),
        write_run(tmp_path, 'jnc0', 'va01.mp4', param=p['A'], mtime=T0 + 1),
        write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['B'], mtime=T0 + 1),
        write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['C'], mtime=T0 + 1),
        write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['D'], mtime=T0 + 5),
        # Further from the last write than the gap, so a new invocation that starts its own launch batch.
        write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['E'], mtime=T0 + 5 + it.INVOCATION_GAP_S + 1),
    ]
    it.load_runs(con, files, 0.1, listing('jnc0', 'va00.mp4', 'va01.mp4'), workers=2)
    # B and C tie at rank 2, so both are in the launch batch; D at rank 4 is not.
    assert clip_positions(con) == [
        ('jnc0', 'va00.mp4', p['A'], 1, 1, 'launch_first_clip'),
        ('jnc0', 'va01.mp4', p['A'], 1, 1, 'later_clip'),
        ('jnc0', 'va00.mp4', p['B'], 1, 2, 'launch_first_clip'),
        ('jnc0', 'va00.mp4', p['C'], 1, 2, 'launch_first_clip'),
        ('jnc0', 'va00.mp4', p['D'], 1, 4, 'other_first_clip'),
        ('jnc0', 'va00.mp4', p['E'], 2, 1, 'launch_first_clip'),
    ]


def test_load_runs_rejects_run_on_unlisted_video(con, tmp_path):
    # A saved run on a video the split listing does not hold.
    with pytest.raises(AssertionError, match='missing from the split directory listing'):
        it.load_runs(con, [write_run(tmp_path, 'd', 'va07.mp4')], 0.1, listing('d', 'va00.mp4'))


@pytest.mark.parametrize('runtime', [
    # No solve_ilp entry.
    [{'op': 'build_ilp', 'time': 1.0}, {'op': 'group_tiles', 'time': 1.0}],
    # A negative time.
    [{'op': 'build_ilp', 'time': 1.0}, {'op': 'solve_ilp', 'time': -1.0}],
    # A null time.
    [{'op': 'build_ilp', 'time': None}, {'op': 'solve_ilp', 'time': 1.0}],
    # build_ilp logged twice would double-count construction.
    [{'op': 'build_ilp', 'time': 1.0}, {'op': 'build_ilp', 'time': 2.0}, {'op': 'solve_ilp', 'time': 3.0}],
    # No operation at all.
    [],
    # A null operation list.
    None,
])
def test_load_runs_rejects_missing_negative_null_or_repeated_ops(con, tmp_path, runtime):
    text = json.dumps({'runtime': runtime, 'time_limit': None})
    with pytest.raises(AssertionError, match='exactly one valid build_ilp and solve_ilp'):
        it.load_runs(con, [write_run(tmp_path, 'd', 'va00.mp4', text=text)], 0.1, listing('d', 'va00.mp4'))


@pytest.mark.parametrize('text', [
    # Two records in one file.
    json.dumps({'runtime': [], 'time_limit': None}) + '\n' + json.dumps({'runtime': [], 'time_limit': None}) + '\n',
    # No time_limit key, so the budget is unknown.
    json.dumps({'runtime': [{'op': 'build_ilp', 'time': 1}, {'op': 'solve_ilp', 'time': 1}]}),
    # A time that is not a number.
    json.dumps({'runtime': [{'op': 'build_ilp', 'time': 'x'}, {'op': 'solve_ilp', 'time': 1}], 'time_limit': None}),
    # Not JSON at all.
    'nope',
])
def test_load_runs_rejects_malformed_files(con, tmp_path, text):
    with pytest.raises(duckdb.Error):
        it.load_runs(con, [write_run(tmp_path, 'd', 'va00.mp4', text=text)], 0.1, listing('d', 'va00.mp4'))


@pytest.mark.parametrize('video,param', [
    # Training videos are not a pruning split.
    ('tr00.mp4', PARAM),
    # A parameter string without the sampling-rate and threshold tokens.
    ('va00.mp4', 'ShuffleNet05_60_sortcython'),
])
def test_load_runs_rejects_unknown_split_or_param(con, tmp_path, video, param):
    with pytest.raises(AssertionError, match='unknown split or unparseable'):
        it.load_runs(con, [write_run(tmp_path, 'd', video, param=param)], 0.1, listing('d', 'va00.mp4'))


def test_load_runs_rejects_unrecognized_path_and_empty_input(con, tmp_path):
    # A runtime log outside a pruning stage directory.
    path = write_run(tmp_path, 'd', 'va00.mp4', stage='021_other_stage')
    with pytest.raises(AssertionError, match='unrecognized runtime paths'):
        it.load_runs(con, [path], 0.1, listing('d', 'va00.mp4'))
    # No file at all.
    with pytest.raises(AssertionError, match='no runtime files'):
        it.load_runs(con, [], 0.1, listing('d', 'va00.mp4'))


def test_timing_stats_budget_bins_are_inclusive_at_lower_edges(con):
    # Solve times at and just below each bin edge of a 0.1 s budget: 0.9x, 1.0x, 1.05x, 1.1x.
    solves = [89.99, 90.0, 99.99, 100.0, 104.99, 105.0, 109.99, 110.0]
    set_runs(con, [('valid', 1, 1.0, s, 2 * s) for s in solves])
    b = it.timing_stats(con, [])[0]['solve_reaches_budget']
    assert b == dict(runs=5, percent=62.5, lower_bound_percent=12.5, median_solve_over_budget=pytest.approx(1.025, abs=1e-3),
                     runs_0_90_to_1_00x=2, runs_1_00_to_1_05x=2, runs_1_05_to_1_10x=2, runs_from_1_10x=1)


def test_timing_stats_uses_each_runs_own_budget(con):
    # 120 ms reaches a 0.1 s budget but not a 0.5 s one, and 500 ms reaches the 0.5 s budget exactly.
    set_runs(con, [('test', 1, 1.0, 120.0, 200.0, 0.1), ('test', 1, 1.0, 120.0, 200.0, 0.5),
                   ('test', 1, 1.0, 500.0, 600.0, 0.5)])
    b = it.timing_stats(con, [])[0]['solve_reaches_budget']
    assert (b['runs'], b['runs_from_1_10x'], b['runs_0_90_to_1_00x'], b['runs_1_00_to_1_05x']) == (2, 1, 0, 1)
    # Solve over budget: 1.2, 0.24, 1.0, whose median is 1.0.
    assert b['median_solve_over_budget'] == 1.0


def test_timing_stats_percentiles_shares_and_groups(con):
    set_runs(con, [
        # Solve times straddle the 100 ms budget; exactly 100 ms counts as reaching it.
        ('valid', 1, 2.0, 10.0, 24.0),
        ('valid', 1, 4.0, 99.99, 207.98),
        ('test', 2, 6.0, 100.0, 424.0),
        ('test', 2, 8.0, 150.0, 316.0),
    ])
    a = it.timing_stats(con, [])[0]
    assert a['runs'] == 4
    # Even count: the median averages the two middle values; p95 interpolates between the top two.
    assert a['build_ms'] == dict(median=5.0, mean=5.0, p95=pytest.approx(7.7), max=8.0)
    assert a['solve_ms']['median'] == pytest.approx(99.995, abs=0.01)
    # Two runs at or above the budget, one of them at 1.5x.
    assert (a['solve_reaches_budget']['runs'], a['solve_reaches_budget']['percent']) == (2, 50.0)
    assert a['solve_reaches_budget']['lower_bound_percent'] == 25.0
    # Per-run shares 50, 50, 25, 50 percent: the median is 50; the aggregate weights runs by stage time.
    assert a['ilp_share_of_pruning_stage']['median_per_run_percent'] == 50.0
    assert a['ilp_share_of_pruning_stage']['aggregate_percent'] == pytest.approx(round(100 * 379.99 / 971.98, 2))
    # Grouped statistics keep groups apart and carry their key.
    by_split = {r['split']: r for r in it.timing_stats(con, ['split'])}
    assert by_split['valid']['solve_reaches_budget']['runs'] == 0
    assert by_split['test']['solve_reaches_budget']['runs'] == 2
    assert by_split['test']['build_ms']['max'] == 8.0


def test_timing_stats_zero_length_stage_has_no_share(con):
    # A run whose every operation took 0 ms has no defined share, rather than a division error.
    set_runs(con, [('valid', 1, 0.0, 0.0, 0.0)])
    a = it.timing_stats(con, [])[0]
    assert a['ilp_share_of_pruning_stage'] == dict(median_per_run_percent=None, aggregate_percent=None)
    assert (a['solve_reaches_budget']['runs'], a['solve_reaches_budget']['percent']) == (0, 0.0)


def test_timing_report_coverage_invocations_labels_and_groupings(con, tmp_path):
    # Three configurations of jnc0 (B3D1) over two listed validation videos; C has only va00, D runs later.
    p = {k: f'{k}_60_{r}_060_r050_sortcython' for k, r in (('A', 1), ('C', 4), ('D', 4))}
    files = [write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['A'], mtime=T0),
             write_run(tmp_path, 'jnc0', 'va01.mp4', param=p['A'], mtime=T0),
             write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['C'], mtime=T0 + 1),
             write_run(tmp_path, 'jnc0', 'va00.mp4', param=p['D'], mtime=T0 + it.INVOCATION_GAP_S + 2)]
    it.load_runs(con, files, 0.1, listing('jnc0', 'va00.mp4', 'va01.mp4'), workers=1)
    report = it.timing_report(con, 0.1, workers=1)
    # Coverage: 3 configurations x 2 listed videos, 4 saved, and per-configuration video counts.
    cov = report['inputs']['coverage'][0]
    assert {k: cov[k] for k in ('dataset', 'split', 'runs', 'listed_videos', 'saved_videos', 'configs', 'expected_runs',
                                'missing_runs', 'min_videos_per_config', 'max_videos_per_config', 'tile_sizes')} == dict(
        dataset='jnc0', split='valid', runs=4, listed_videos=2, saved_videos=2, configs=3, expected_runs=6,
        missing_runs=2, min_videos_per_config=1, max_videos_per_config=2, tile_sizes=[60])
    assert report['inputs']['totals'] == dict(runs=4, expected_runs=6, missing_runs=2)
    # Two invocations; with one worker only A (first) and D (alone in its invocation) are launch tasks.
    inv = [{k: r[k] for k in ('invocation', 'runs', 'tasks', 'launch_tasks', 'launch_first_clips', 'datasets', 'splits')}
           for r in report['inputs']['invocations']]
    assert inv == [dict(invocation=1, runs=3, tasks=2, launch_tasks=1, launch_first_clips=1, datasets=['jnc0'], splits=['valid']),
                   dict(invocation=2, runs=1, tasks=1, launch_tasks=1, launch_first_clips=1, datasets=['jnc0'], splits=['valid'])]
    # Every run logged a null time limit, so all used the default.
    assert report['solve_budget']['logged_time_limits'] == [dict(logged_time_limit=None, budget_s=0.1, runs=4)]
    # Groupings are keyed by label, rate, invocation, and clip position.
    assert list(report['by_dataset']) == ['B3D1']
    assert list(report['by_sample_rate']) == ['1', '4']
    assert list(report['by_invocation']) == ['1', '2']
    assert {k: v['runs'] for k, v in report['by_clip_position'].items()} == dict(
        launch_first_clip=2, other_first_clip=1, later_clip=1)
    assert list(report['by_dataset_and_clip_position']) == [
        'B3D1 later_clip', 'B3D1 launch_first_clip', 'B3D1 other_first_clip']
    assert report['all']['runs'] == 4


def tradeoff(rows):
    # Tetris rows (variant_id, fps, hota, time_s, threshold) and a reference row at 10 fps, HOTA 0.8, 100 s.
    tetris = pd.DataFrame(rows, columns=['variant_id', 'throughput_fps', 'HOTA_HOTA', 'time', 'tracking_accuracy_threshold'])
    naive = pd.Series(dict(throughput_fps=10.0, HOTA_HOTA=0.8, time=100.0))
    return tetris, naive


def write_ops(path, rows, videoset='test'):
    # A per-operation runtime table with (dataset, variant_id, video, stage, op, seconds) rows.
    pd.DataFrame([dict(op=o, time=t, stage=s, dataset=d, videoset=videoset, video=v, variant_id=c)
                  for d, c, v, s, o, t in rows]).to_csv(path, index=False)
    return path


def extraction_rows(dataset, config, prune_ms=None, other_s=1.0):
    # Each test video of one configuration: classify and detect time, plus pruning ops ({video: {op: ms}}) where given.
    rows = []
    for video in TEST_VIDEOS:
        rows += [(dataset, config, video, '020_exec_classify', 'inference', other_s / 2),
                 (dataset, config, video, '040_exec_detect', 'detect', other_s / 2)]
        rows += [(dataset, config, video, it.PRUNE_STAGE, op, ms / 1000) for op, ms in (prune_ms or {}).get(video, {}).items()]
    return rows


# Pruning ops per test video in ms. VARIANT: te00 is a slow launch clip (0.4 s ILP), te01 solves at the budget.
PRUNE_4 = {'te00.mp4': dict(build_ilp=340.0, solve_ilp=60.0, group_tiles=100.0),
           'te01.mp4': dict(build_ilp=40.0, solve_ilp=100.0, group_tiles=60.0)}
# VARIANT8: 0.3 s of ILP inside a 0.4 s stage on each video.
PRUNE_8 = {v: dict(build_ilp=100.0, solve_ilp=200.0, group_tiles=100.0) for v in TEST_VIDEOS}


def saved_runs(param, prune_ms, mtime):
    # The p022 runs behind a pruning variant's throughput rows: (param, video, ops, mtime).
    return [(param, v, ops, mtime) for v, ops in prune_ms.items()]


def setup_extraction(con, tmp_path, tetris_rows, ops_rows, runs=(), workers=it.PRUNE_WORKERS):
    # A validation run in an earlier invocation, so the runs table exists and takes no test launch slot.
    files = [write_run(tmp_path, 'd', 'va00.mp4', mtime=T0 - 10 * it.INVOCATION_GAP_S)]
    # The test runs behind the pruning variants.
    files += [write_run(tmp_path, 'd', v, param=p, ops=o, mtime=m) for p, v, o, m in runs]
    it.load_runs(con, files, 0.1, LISTING, workers=workers)
    # Throughput ops, their match with the runs, the tradeoff points, and per-configuration runtimes.
    it.load_throughput_ops(con, {'d': write_ops(tmp_path / 'ops.csv', ops_rows)})
    check = it.variant_videos(con)
    it.load_points(con, {'d': tradeoff(tetris_rows)})
    it.variant_runtimes(con)
    return check


def test_extraction_report_operating_and_pruning_points(con, tmp_path):
    # 1 s of other work per video; VARIANT adds 0.7 s of stage (0.54 s ILP), VARIANT8 0.8 s (0.6 s ILP).
    check = setup_extraction(con, tmp_path, [
        # Fastest within the bound (0.76), but does not prune.
        ('fast', 50.0, 0.77, 2.0, None),
        # Prunes, within the bound, slower.
        (VARIANT, 40.0, 0.79, 2.7, 0.6),
        # Prunes and is fastest of all, but misses the bound.
        (VARIANT8, 90.0, 0.70, 2.8, 0.8),
    ], extraction_rows('d', 'fast') + extraction_rows('d', VARIANT, PRUNE_4) + extraction_rows('d', VARIANT8, PRUNE_8),
        # With one worker, only VARIANT's task (written first) is a launch task.
        saved_runs(PARAM, PRUNE_4, T0) + saved_runs(PARAM8, PRUNE_8, T0 + 5), workers=1)
    # Every pruning row found its run with identical ILP times; four test runs saved, four used.
    assert check == dict(throughput_rows=4, matched_runs=4, max_abs_diff_ms=pytest.approx(0.0, abs=1e-9),
                         test_runs_used=4, test_runs_saved=4)
    d = it.extraction_report(con, 0.05)['d']
    assert (d['hota_bound'], d['reference_hota']) == (pytest.approx(0.76), 0.8)
    # The operating point has no pruning stage, so its ILP share is zero and it has no clip timings.
    assert d['operating_point'] == dict(config='fast', hota=0.77, fps=50.0, prunes=False, videos=2, launch_clips=0,
                                        extraction_s=2.0, ilp_s=0.0, ilp_percent=0.0, pruning_stage_percent=0.0,
                                        ilp_percent_excluding_launch_clips=0.0, clip_timings=None)
    # The fastest pruning point inside the bound: 0.54 s ILP and 0.7 s stage out of 2.7 s.
    p = d['fastest_pruning_point']
    assert (p['config'], p['prunes'], p['videos'], p['launch_clips'], p['fps'], p['hota']) == (VARIANT, True, 2, 1, 40.0, 0.79)
    assert (p['extraction_s'], p['ilp_s']) == (2.7, 0.54)
    assert (p['ilp_percent'], p['pruning_stage_percent']) == (20.0, 25.926)
    # Without the launch clip (te00): 0.14 s ILP out of te01's 1.2 s.
    assert p['ilp_percent_excluding_launch_clips'] == 11.667
    # Per-clip timings of its two test runs; te01's 100 ms solve reaches the budget.
    c = p['clip_timings']
    assert c['runs'] == 2
    assert (c['build_ms']['median'], c['build_ms']['max'], c['solve_ms']['median'], c['solve_ms']['max']) == (190.0, 340.0, 80.0, 100.0)
    assert (c['build_plus_solve_ms']['median'], c['build_plus_solve_ms']['max']) == (270.0, 400.0)
    assert (c['solve_reaches_budget']['runs'], c['solve_reaches_budget']['percent']) == (1, 50.0)
    # The spread covers both pruning configurations, inside or outside the bound: 20.0% and 21.429% ILP.
    assert d['all_pruning_configs'] == dict(dataset='d', configs=2, min_ilp_percent=20.0, median_ilp_percent=20.714,
                                            max_ilp_percent=21.429, median_pruning_stage_percent=27.249,
                                            max_pruning_stage_percent=28.571)


def test_extraction_report_without_pruning_configs(con, tmp_path):
    # A dataset whose validated set never prunes has no pruning point and no spread.
    check = setup_extraction(con, tmp_path, [('fast', 50.0, 0.79, 2.0, None)], extraction_rows('d', 'fast'))
    assert (check['throughput_rows'], check['matched_runs'], check['max_abs_diff_ms']) == (0, 0, None)
    d = it.extraction_report(con, 0.05)['d']
    assert d['fastest_pruning_point'] == dict(meets_bound=False, best_hota=None)
    assert d['all_pruning_configs'] is None
    assert d['operating_point']['config'] == 'fast'


def test_extraction_report_pruning_point_misses_bound(con, tmp_path):
    # The only pruning configuration misses the bound, so the pruning point reports its best HOTA instead.
    setup_extraction(con, tmp_path, [('fast', 50.0, 0.79, 2.0, None), (VARIANT8, 90.0, 0.70, 2.8, 0.8)],
                     extraction_rows('d', 'fast') + extraction_rows('d', VARIANT8, PRUNE_8), saved_runs(PARAM8, PRUNE_8, T0))
    d = it.extraction_report(con, 0.05)['d']
    assert d['fastest_pruning_point'] == dict(meets_bound=False, best_hota=0.70)
    assert d['all_pruning_configs']['configs'] == 1


def test_extraction_report_rejects_dataset_without_tetris_rows(con, tmp_path):
    # A dataset whose tradeoff table has a reference row but no Tetris configuration.
    setup_extraction(con, tmp_path, [('fast', 50.0, 0.79, 2.0, None)], extraction_rows('d', 'fast'))
    it.load_points(con, {'d': tradeoff([('fast', 50.0, 0.79, 2.0, None)]), 'e': tradeoff([])})
    with pytest.raises(AssertionError, match='no Tetris configuration'):
        it.extraction_report(con, 0.05)


# VARIANT pruning both videos with 0.2 s of stage each.
PRUNE_FLAT = {v: dict(build_ilp=40.0, solve_ilp=60.0, group_tiles=100.0) for v in TEST_VIDEOS}


@pytest.mark.parametrize('tetris_rows,ops_rows,runs', [
    # Runtime differs from the tradeoff table's.
    ([('fast', 50.0, 0.79, 3.0, None)], extraction_rows('d', 'fast'), []),
    # A validated configuration with no runtime rows.
    ([('fast', 50.0, 0.79, 2.0, None), ('lost', 40.0, 0.79, 2.0, None)], extraction_rows('d', 'fast'), []),
    # A pruning configuration whose runtime has no pruning stage.
    ([(VARIANT, 50.0, 0.79, 2.0, 0.6)], extraction_rows('d', VARIANT), []),
    # A configuration without a threshold whose runtime has a pruning stage.
    ([(VARIANT, 50.0, 0.79, 2.4, None)], extraction_rows('d', VARIANT, PRUNE_FLAT), saved_runs(PARAM, PRUNE_FLAT, T0)),
    # A pruning configuration that prunes only one of its two videos.
    ([(VARIANT, 50.0, 0.79, 2.2, 0.6)], extraction_rows('d', VARIANT, {'te00.mp4': PRUNE_FLAT['te00.mp4']}),
     saved_runs(PARAM, {'te00.mp4': PRUNE_FLAT['te00.mp4']}, T0)),
])
def test_variant_runtimes_rejects_disagreement_with_tradeoff(con, tmp_path, tetris_rows, ops_rows, runs):
    with pytest.raises(AssertionError, match='disagrees with the tradeoff table'):
        setup_extraction(con, tmp_path, tetris_rows, ops_rows, runs)


def test_variant_videos_rejects_missing_run_unmapped_id_and_stale_times(con, tmp_path):
    # A pruning variant whose te01 run is not saved.
    with pytest.raises(AssertionError, match='without a saved p022 run'):
        setup_extraction(con, tmp_path, [(VARIANT, 50.0, 0.79, 2.4, 0.6)], extraction_rows('d', VARIANT, PRUNE_FLAT),
                         saved_runs(PARAM, {'te00.mp4': PRUNE_FLAT['te00.mp4']}, T0))
    # A pruning variant id that VARIANT_PATTERN cannot map back to p022's parameter string.
    with pytest.raises(AssertionError, match='without a saved p022 run'):
        setup_extraction(con, tmp_path / 'b', [('odd', 50.0, 0.79, 2.4, 0.6)], extraction_rows('d', 'odd', PRUNE_FLAT),
                         saved_runs(PARAM, PRUNE_FLAT, T0))
    # A stale throughput table: te01's solve time differs from the saved run by 1 ms.
    stale = dict(PRUNE_FLAT, **{'te01.mp4': dict(PRUNE_FLAT['te01.mp4'], solve_ilp=61.0)})
    with pytest.raises(AssertionError, match='differ from the p022 logs'):
        setup_extraction(con, tmp_path / 'c', [(VARIANT, 50.0, 0.79, 2.401, 0.6)], extraction_rows('d', VARIANT, stale),
                         saved_runs(PARAM, PRUNE_FLAT, T0))


def test_load_throughput_ops_rejects_wrong_split_missing_file_and_other_dataset(con, tmp_path):
    # A validation-split table cannot be compared with the test tradeoff.
    with pytest.raises(AssertionError, match='not the test split'):
        it.load_throughput_ops(con, {'d': write_ops(tmp_path / 'va.csv', extraction_rows('d', 'fast'), 'valid')})
    # A missing table is reported rather than dropping the dataset.
    with pytest.raises(AssertionError, match='missing throughput tables'):
        it.load_throughput_ops(con, {'d': tmp_path / 'missing.csv'})
    # A table holding another dataset's rows.
    with pytest.raises(AssertionError, match='throughput tables cover'):
        it.load_throughput_ops(con, {'d': write_ops(tmp_path / 'other.csv', extraction_rows('e', 'fast'))})


def test_main_end_to_end_on_a_small_cache(tmp_path, monkeypatch):
    # One dataset (jnc0, B3D1): split listings, saved p022 runs, a throughput table, and a tradeoff table.
    data, cache_root, out = tmp_path / 'data', tmp_path / 'cache', tmp_path / 'out'
    for name in ('valid/va00.mp4', 'test/te00.mp4', 'test/te01.mp4', 'test/readme.txt'):
        (data / 'jnc0' / name).parent.mkdir(parents=True, exist_ok=True)
        (data / 'jnc0' / name).write_text('')
    write_run(cache_root, 'jnc0', 'va00.mp4', mtime=T0)
    for video, ops in PRUNE_4.items():
        write_run(cache_root, 'jnc0', video, ops=ops, mtime=T0 + 10 * it.INVOCATION_GAP_S)
    ops_path = write_ops(tmp_path / 'ops.csv', extraction_rows('jnc0', 'fast') + extraction_rows('jnc0', VARIANT, PRUNE_4))
    out.mkdir()
    # Point the script at the small cache instead of /polyis-cache and /polyis-data.
    monkeypatch.setattr(it.cache, 'CACHE_DIR', str(cache_root))
    monkeypatch.setattr(it.store, 'dataset', lambda d, *a: data / d / Path(*a))
    monkeypatch.setattr(it.cache, 'eval', lambda d, *a: ops_path)
    monkeypatch.setattr(it, 'load_test_tradeoff',
                        lambda d: tradeoff([('fast', 50.0, 0.77, 2.0, None), (VARIANT, 40.0, 0.79, 2.7, 0.6)]))
    monkeypatch.setattr(it, 'results_dir', lambda f: out)
    monkeypatch.setattr(sys, 'argv', ['r9o3_ilp_timings.py', '--datasets', 'jnc0'])
    it.main()
    timings = json.loads((out / 'ilp_timings.json').read_text())
    extraction = json.loads((out / 'extraction_share.json').read_text())
    # Three saved runs in two invocations, all with the 0.1 s default budget; te01's 100 ms solve reaches it.
    assert timings['all']['runs'] == 3
    assert timings['inputs']['totals'] == dict(runs=3, expected_runs=3, missing_runs=0)
    assert [r['runs'] for r in timings['inputs']['invocations']] == [1, 2]
    assert timings['solve_budget']['default_seconds'] == 0.1
    assert timings['all']['solve_reaches_budget']['runs'] == 1
    assert list(timings['by_dataset']) == ['B3D1']
    # The extraction shares, with the cross-check against the p022 logs.
    assert extraction['throughput_matches_p022_logs']['matched_runs'] == 2
    b3d1 = extraction['datasets']['B3D1']
    assert b3d1['operating_point']['config'] == 'fast'
    assert (b3d1['fastest_pruning_point']['config'], b3d1['fastest_pruning_point']['ilp_percent']) == (VARIANT, 20.0)
    assert b3d1['fastest_pruning_point']['ilp_percent_excluding_launch_clips'] == 11.667
