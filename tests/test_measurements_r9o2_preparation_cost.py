import ast
import gzip
import importlib.util
import inspect
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation import p121_throughput_compute as p121
from measurements import common
from measurements import r9o2_preparation_cost as prep
from polyis.io import cache


# The GPU plan lives in a directory whose name is not a package name, so load it from its path.
GPU_PLAN_PATH = Path(__file__).resolve().parent.parent / 'measurements-gpu' / 'r9o2_preparation_cost.py'
# The shell driver that runs the plan on bombe and collects its records.
DRIVER_PATH = GPU_PLAN_PATH.with_name('r9o2_preparation_retiming.sh')
spec = importlib.util.spec_from_file_location('r9o2_gpu_plan', GPU_PLAN_PATH)
gpu_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_plan)


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def write_lines(path, lines):
    # Write one JSON value per line, creating parent directories.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    return str(path)


def ops(con):
    # {(file name, op): (seconds, records, counted)} from runtime_ops.
    return {(Path(f).name, op): (s, n, c) for f, op, s, n, c in
            con.execute('SELECT runtime_file, op, seconds, records, counted FROM runtime_ops').fetchall()}


def fake_file_times(monkeypatch, times):
    # Replace stat with a table of (creation, modification) times; paths not in it are unknown, as missing files are.
    monkeypatch.setattr(prep, 'file_times', lambda paths: {p: times[p] for p in paths if p in times})


# ---------------------------------------------------------------- parsing, against p121


def test_op_list_paths_cover_every_p121_stage():
    # A stage p121 parses but this script cannot locate would silently drop its time.
    assert set(prep.OP_LIST_PATHS) == set(p121.INDEX_DATA_ACCESSORS) | set(p121.QUERY_DATA_ACCESSORS)


def test_ignored_ops_match_p121_source():
    # p121 keeps its ignored ops in a local list inside parse_runtime_file; read the literal so a change there fails here.
    (literal,) = re.findall(r'ignored_ops = (\[[^\]]*\])', inspect.getsource(p121.parse_runtime_file))
    assert tuple(ast.literal_eval(literal)) == prep.IGNORED_OPS


# How each JSON path wraps a list of op items into one line of that stage's layout.
LAYOUTS = {'$[#-1]': lambda items: [0, [[1, 2, 3, 4, 0.9]], items], '$': lambda items: items,
           '$.runtime': lambda items: {'step': 's', 'runtime': items}}


@pytest.mark.parametrize('stage', sorted(prep.OP_LIST_PATHS))
def test_load_runtime_ops_matches_p121_parser(con, tmp_path, stage):
    accessor = {**p121.INDEX_DATA_ACCESSORS, **p121.QUERY_DATA_ACCESSORS}[stage]
    wrap = LAYOUTS[prep.OP_LIST_PATHS[stage]]
    # Every op p121 drops for this stage, among ops it counts.
    dropped = sorted({*prep.IGNORED_OPS, *p121.excluded_runtime_ops(stage)})
    lines = [
        # Integer and float times, a non-numeric "time" (p040's image file name), and an item without a time.
        wrap([{'op': 'work', 'time': 1500}, {'op': 'work', 'time': 2.25}, {'op': 'image_file', 'time': 'x.jpg'},
              {'op': 'note'}]),
        wrap([{'op': op, 'time': 1000 + i} for i, op in enumerate(dropped)] + [{'op': 'other', 'time': 7}]),
        # A single op object where a list is expected; p121 treats it as a one-item list.
        wrap({'op': 'single', 'time': 5}),
    ]
    path = tmp_path / f'{stage}.jsonl'
    # A blank line between records, which both parsers skip.
    path.write_text(json.dumps(lines[0]) + '\n\n' + '\n'.join(json.dumps(line) for line in lines[1:]) + '\n')
    parsed = p121.filter_excluded_runtime_ops(p121.parse_runtime_file(str(path), stage, accessor), stage)
    expected = parsed.groupby('op')['time'].sum().to_dict()
    prep.load_runtime_ops(con, [(stage, str(path))])
    got = dict(con.execute('SELECT op, seconds FROM runtime_ops WHERE counted').fetchall())
    # The same ops, with the same seconds.
    assert got.keys() == expected.keys()
    assert all(got[op] == pytest.approx(expected[op], abs=1e-12) for op in expected)


def test_load_runtime_ops_reads_every_layout(con, tmp_path):
    files = [
        # p011: [frame, boxes, ops]; times are milliseconds.
        ('011_tune_detect', write_lines(tmp_path / 'det.jsonl', [
            [0, [[1, 2, 3, 4, 0.9]], [{'op': 'read', 'time': 30}, {'op': 'detect', 'time': 40}]],
            [1, [], [{'op': 'read', 'time': 10}, {'op': 'detect', 'time': 40}]]])),
        # p012: one op object per line.
        ('012_tune_create_training_data', write_lines(tmp_path / 'p012.jsonl', [
            {'op': 'split', 'time': 1000, 'frame': 0}, {'op': 'save', 'time': 500, 'frame': 0}])),
        # 002 tracking and 060: ops under 'runtime'.
        ('002_preprocess_groundtruth_tracking', write_lines(tmp_path / 'trk.jsonl', [
            {'frame_idx': 0, 'runtime': [{'op': 'tracker_update', 'time': 2}]}])),
        # p020: a list per line; p121 drops its video decoding ('read') and 'retrieve'.
        ('020_exec_classify', write_lines(tmp_path / 'cls.jsonl', [
            [{'op': 'inference', 'time': 4}], [{'op': 'read', 'time': 3000}, {'op': 'retrieve', 'time': 1}]])),
        # p030: 'save_collage' is excluded for p030 and 'save_canvas' is ignored everywhere.
        ('030_exec_compress', write_lines(tmp_path / 'cmp.jsonl', [
            {'step': 'group_tiles', 'runtime': [{'op': 'group_tiles', 'time': 6}]},
            {'step': 'process_collage', 'runtime': [{'op': 'save_collage', 'time': 9}, {'op': 'save_canvas', 'time': 8}]}])),
        # p040: the image file name rides along as an op with a string "time"; it is not a timing.
        ('040_exec_detect', write_lines(tmp_path / 'cdet.jsonl', [
            [{'op': 'image_file', 'time': '0000_0001.jpg'}, {'op': 'read', 'time': 1}, {'op': 'detect', 'time': 20}]])),
        # An empty p040 file (no canvases) has no ops at all.
        ('040_exec_detect', write_lines(tmp_path / 'empty.jsonl', [])),
    ]
    prep.load_runtime_ops(con, files)
    got = ops(con)
    assert got[('det.jsonl', 'read')] == (pytest.approx(0.04), 2, True)
    assert got[('det.jsonl', 'detect')] == (pytest.approx(0.08), 2, True)
    assert got[('p012.jsonl', 'split')] == (pytest.approx(1.0), 1, True)
    assert got[('trk.jsonl', 'tracker_update')][2] is True
    assert got[('cls.jsonl', 'inference')][2] is True
    assert got[('cls.jsonl', 'read')] == (pytest.approx(3.0), 1, False)
    assert got[('cls.jsonl', 'retrieve')][2] is False
    assert got[('cmp.jsonl', 'group_tiles')][2] is True
    assert got[('cmp.jsonl', 'save_collage')][2] is False and got[('cmp.jsonl', 'save_canvas')][2] is False
    # Non-numeric "times" and empty files leave no rows.
    assert ('cdet.jsonl', 'image_file') not in got
    assert got[('cdet.jsonl', 'detect')] == (pytest.approx(0.02), 1, True)
    assert not any(name == 'empty.jsonl' for name, _ in got)


def test_load_runtime_ops_marks_video_decoding(con, tmp_path):
    # The reference pipeline's and p011's per-frame 'read' is counted by p121 and flagged as decoding.
    ref = write_lines(tmp_path / 'ref.jsonl', [[{'op': 'read', 'time': 10}, {'op': 'detect', 'time': 90}]])
    det = write_lines(tmp_path / 'det.jsonl', [[0, [], [{'op': 'read', 'time': 10}, {'op': 'detect', 'time': 90}]]])
    # p040's 'read' (reading a canvas image) is not video decoding.
    canvas = write_lines(tmp_path / 'canvas.jsonl', [[{'op': 'read', 'time': 10}]])
    prep.load_runtime_ops(con, [('001_preprocess_groundtruth_detection', ref), ('011_tune_detect', det),
                                ('040_exec_detect', canvas)])
    decode = {(Path(f).name, op): d for f, op, d in con.execute('SELECT runtime_file, op, decode FROM runtime_ops').fetchall()}
    assert decode == {('ref.jsonl', 'read'): True, ('ref.jsonl', 'detect'): False, ('det.jsonl', 'read'): True,
                      ('det.jsonl', 'detect'): False, ('canvas.jsonl', 'read'): False}


def test_load_runtime_ops_accepts_no_files(con):
    # No files is an empty table, not a DuckDB error.
    prep.load_runtime_ops(con, [])
    assert con.execute('SELECT count(*) FROM runtime_ops').fetchone() == (0,)


def test_load_runtime_ops_rejects_json_arrays(con, tmp_path):
    # p121 reads .json files as one array, which the JSONL queries would misread.
    with pytest.raises(AssertionError, match='JSONL'):
        prep.load_runtime_ops(con, [('060_exec_track', str(tmp_path / 'runtime.json'))])


def test_load_runtime_ops_rejects_lines_without_an_op_list(con, tmp_path):
    # A p022 line without 'runtime' would otherwise vanish from the sum.
    path = write_lines(tmp_path / 'prune.jsonl', [{'time_limit': None}])
    with pytest.raises(AssertionError, match='without an op list'):
        prep.load_runtime_ops(con, [('022_exec_prune_polyominoes', path)])


def test_load_runtime_ops_rejects_timed_items_without_an_op(con, tmp_path):
    # A time with no op cannot be attributed (p121 fails on it too).
    path = write_lines(tmp_path / 'trk.jsonl', [{'runtime': [{'time': 5}]}])
    with pytest.raises(AssertionError, match='without an op'):
        prep.load_runtime_ops(con, [('060_exec_track', path)])


def test_load_runtime_ops_rejects_boolean_times(con, tmp_path):
    # JSON true is not a duration, although Python's isinstance(True, int) would let p121 count it.
    path = write_lines(tmp_path / 'trk.jsonl', [{'runtime': [{'op': 'a', 'time': True}]}])
    with pytest.raises(AssertionError, match='boolean'):
        prep.load_runtime_ops(con, [('060_exec_track', path)])


@pytest.mark.parametrize('stage, op, raises', [
    # A counted op.
    ('060_exec_track', 'tracker_update', True),
    # An op p121 excludes for this stage is still checked by p121, so it is here too.
    ('020_exec_classify', 'read', True),
    # An ignored op is skipped by p121 before its check.
    ('060_exec_track', 'save_canvas', False),
])
def test_load_runtime_ops_negative_times(con, tmp_path, stage, op, raises):
    wrap = LAYOUTS[prep.OP_LIST_PATHS[stage]]
    path = write_lines(tmp_path / 'x.jsonl', [wrap([{'op': op, 'time': -5}])])
    # Raise exactly where p121 would.
    if raises:
        with pytest.raises(AssertionError, match='negative'):
            prep.load_runtime_ops(con, [(stage, path)])
    else:
        prep.load_runtime_ops(con, [(stage, path)])


def test_load_runtime_ops_integer_and_float_times(con, tmp_path):
    # JSON integers (DuckDB types them UBIGINT) and floats both count.
    path = write_lines(tmp_path / 'trk.jsonl', [{'runtime': [{'op': 'a', 'time': 1000}, {'op': 'a', 'time': 500.5}]}])
    prep.load_runtime_ops(con, [('060_exec_track', path)])
    assert ops(con)[('trk.jsonl', 'a')] == (pytest.approx(1.5005), 2, True)


def test_first_frame_seconds_takes_each_files_earliest_split(con, tmp_path):
    # Frames out of order: the earliest frame's split is the start-up call, whatever its line position.
    a = write_lines(tmp_path / 'a.jsonl', [{'op': 'split', 'time': 50, 'frame': 3}, {'op': 'save', 'time': 900, 'frame': 0},
                                            {'op': 'split', 'time': 8000, 'frame': 0}])
    b = write_lines(tmp_path / 'b.jsonl', [{'op': 'split', 'time': 2000, 'frame': 0}])
    assert prep.first_frame_seconds(con, [a, b]) == pytest.approx(10.0)
    # No files, no start-up time.
    assert prep.first_frame_seconds(con, []) == 0.0


def test_first_frame_seconds_rejects_a_frame_split_twice(con, tmp_path):
    # Two splits of one frame make "the first call" ambiguous.
    a = write_lines(tmp_path / 'a.jsonl', [{'op': 'split', 'time': 50, 'frame': 0}, {'op': 'split', 'time': 60, 'frame': 0}])
    with pytest.raises(AssertionError, match='more than one split'):
        prep.first_frame_seconds(con, [a])


def touch(path, mtime):
    # Create a file and set its modification time.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    os.utime(path, (mtime, mtime))


def test_file_times_matches_os_stat(tmp_path):
    path = tmp_path / 'a b.txt'
    touch(path, 1_700_000_000)
    birth, mtime = common.file_times([str(path)])[str(path)]
    assert mtime == 1_700_000_000
    # Creation time is unknown on some file systems; when known it is now, after the older forced modification time.
    assert birth is None or birth >= mtime
    assert common.file_times([]) == {}
    # A missing path is left out, so callers see it as unknown.
    assert common.file_times([str(tmp_path / 'missing'), str(path)]).keys() == {str(path)}


def test_file_times_is_shared_by_the_measurement_and_the_gpu_plan():
    # One implementation in measurements/common.py; the plan no longer imports the measurement module.
    assert prep.file_times is common.file_times and gpu_plan.file_times is common.file_times
    assert 'measurements.r9o2_preparation_cost' not in GPU_PLAN_PATH.read_text()


# ---------------------------------------------------------------- work units and their process time


def unit_file(tmp_path, video, stage_dir, name, lines, level=1, params='cfg'):
    # A runtime file at <tmp>/execution/<video>/<stage_dir>/<params>/[score/]<name>.
    parent = tmp_path / 'execution' / video / stage_dir / params / ('score' if level == 2 else '')
    return write_lines(parent / name, lines)


def build_units(con, monkeypatch, files, times, ranks):
    """Load runtime ops and units for (stage, path, videoset, video) rows with fake timestamps and split ranks."""
    fake_file_times(monkeypatch, times)
    prep.load_runtime_ops(con, [(stage, path) for stage, path, _, _ in files])
    con.execute('CREATE OR REPLACE TABLE wu (stage VARCHAR, runtime_file VARCHAR, videoset VARCHAR, video VARCHAR)')
    con.executemany('INSERT INTO wu VALUES (?, ?, ?, ?)', files)
    con.execute('CREATE OR REPLACE TABLE sv (videoset VARCHAR, video VARCHAR, rank BIGINT)')
    con.executemany('INSERT INTO sv VALUES (?, ?, ?)', ranks)
    prep.load_units(con, 'wu', 'sv')
    return {r['runtime_file']: r for r in prep.query_dicts(con, 'SELECT * FROM units')}


def detect_task(tmp_path, videos, seconds):
    # One p040 task (one configuration) over videos, each with its recorded detect seconds.
    return [('040_exec_detect', unit_file(tmp_path, v, '040_compressed_detections', 'runtimes.jsonl',
                                          [[{'op': 'detect', 'time': s * 1000}]]), 'valid', v)
            for v, s in zip(videos, seconds)]


def span(path, start, end):
    # Timestamps of a level-1 unit: its directory's creation and its detections file's last write.
    directory = str(Path(path).parent)
    return {directory: (start, 0.0), path: (None, start), str(Path(path).with_name('detections.jsonl')): (None, end)}


def test_load_units_chains_consecutive_videos_of_one_task(con, tmp_path, monkeypatch):
    files = detect_task(tmp_path, ['v0', 'v1', 'v2', 'v3'], [5, 6, 4, 3])
    paths = [f[1] for f in files]
    times = {**span(paths[0], 100, 110),
             # Starts 0.5 s after v0 ended: chained, and costs the time since v0's last write.
             **span(paths[1], 110.5, 120),
             # Starts 80 s after v1 ended, beyond CHAIN_GAP_S: its own span.
             **span(paths[2], 200, 205),
             # Starts before v2 ended, so it ran concurrently: its own span.
             **span(paths[3], 204, 230)}
    units = build_units(con, monkeypatch, files, times, [('valid', v, i + 1) for i, v in enumerate(['v0', 'v1', 'v2', 'v3'])])
    got = [(units[p]['chained'], units[p]['process_s']) for p in paths]
    # The first video of a task gets max(own span, recorded).
    assert got == [(False, 10.0), (True, 10.0), (False, 5.0), (False, 26.0)]
    assert [units[p]['recorded_s'] for p in paths] == [5, 6, 4, 3]


def test_load_units_gap_boundary_is_inclusive(con, tmp_path, monkeypatch):
    files = detect_task(tmp_path, ['v0', 'v1'], [1, 1])
    paths = [f[1] for f in files]
    # Exactly CHAIN_GAP_S after the previous unit's last write still chains.
    times = {**span(paths[0], 0, 10), **span(paths[1], 10 + prep.CHAIN_GAP_S, 75)}
    units = build_units(con, monkeypatch, files, times, [('valid', 'v0', 1), ('valid', 'v1', 2)])
    assert units[paths[1]]['chained'] and units[paths[1]]['process_s'] == pytest.approx(65)


def test_load_units_does_not_chain_across_a_missing_video(con, tmp_path, monkeypatch):
    files = detect_task(tmp_path, ['v0', 'v2'], [1, 1])
    paths = [f[1] for f in files]
    times = {**span(paths[0], 0, 10), **span(paths[1], 10.5, 20)}
    # v1 sits between them in the split, so v2's predecessor in the task is unknown.
    units = build_units(con, monkeypatch, files, times, [('valid', 'v0', 1), ('valid', 'v1', 2), ('valid', 'v2', 3)])
    assert not units[paths[1]]['chained'] and units[paths[1]]['process_s'] == pytest.approx(9.5)


def test_load_units_rejects_a_chain_shorter_than_the_recorded_ops(con, tmp_path, monkeypatch):
    files = detect_task(tmp_path, ['v0', 'v1'], [1, 6])
    paths = [f[1] for f in files]
    # 2 s since v0's last write cannot hold 6 s of recorded detection: the timestamps do not describe this unit.
    times = {**span(paths[0], 0, 10), **span(paths[1], 10.5, 12)}
    units = build_units(con, monkeypatch, files, times, [('valid', 'v0', 1), ('valid', 'v1', 2)])
    assert not units[paths[1]]['chained'] and units[paths[1]]['process_s'] == pytest.approx(6)


def test_load_units_stale_and_unknown_creation_times_fall_back_to_recorded(con, tmp_path, monkeypatch):
    files = detect_task(tmp_path, ['v0', 'v1'], [2, 3])
    paths = [f[1] for f in files]
    # v0's directory was created long before its last write (outputs rewritten in place); v1 has no creation time.
    times = {**span(paths[0], 0, prep.STALE_SPAN_S + 1), paths[1]: (None, 5000.0)}
    units = build_units(con, monkeypatch, files, times, [('valid', 'v0', 1), ('valid', 'v1', 2)])
    assert units[paths[0]]['stale_start'] and units[paths[0]]['process_s'] == 2
    assert not units[paths[1]]['chained'] and units[paths[1]]['process_s'] == 3


def test_load_units_never_chains_one_task_per_video_stages(con, tmp_path, monkeypatch):
    # The reference pipeline runs one task per video; consecutive files that look chained are still separate.
    files = [('001_preprocess_groundtruth_detection',
              unit_file(tmp_path, v, '002_naive', 'detection_runtime.jsonl', [[{'op': 'detect', 'time': 1000}]], level=0,
                        params=''), 'test', v) for v in ('v0', 'v1')]
    paths = [f[1] for f in files]
    times = {paths[0]: (0.0, 1.0), str(Path(paths[0]).with_name('detection.jsonl')): (None, 4.0),
             paths[1]: (4.5, 5.0), str(Path(paths[1]).with_name('detection.jsonl')): (None, 7.0)}
    units = build_units(con, monkeypatch, files, times, [('test', 'v0', 1), ('test', 'v1', 2)])
    # Own spans: 4 s and 2.5 s, each above its 1 s of recorded detection.
    assert [(units[p]['chained'], units[p]['process_s']) for p in paths] == [(False, 4.0), (False, 2.5)]


def test_load_units_reads_creation_two_levels_up_and_the_latest_sibling(con, tmp_path, monkeypatch):
    # p020's runtime file is <params>/score/runtime.jsonl; its unit starts at <params>/ and ends at score.jsonl.
    path = unit_file(tmp_path, 'v0', '020_relevancy', 'runtime.jsonl', [[{'op': 'inference', 'time': 1000}]], level=2)
    params = str(Path(path).parent.parent)
    times = {params: (100.0, 100.0), path: (None, 103.0), str(Path(path).with_name('score.jsonl')): (None, 104.0)}
    units = build_units(con, monkeypatch, [('020_exec_classify', path, 'valid', 'v0')], times, [('valid', 'v0', 1)])
    assert units[path]['start_s'] == 100 and units[path]['end_s'] == 104 and units[path]['process_s'] == 4


def test_load_units_keeps_recorded_time_for_untimed_stages(con, tmp_path, monkeypatch):
    # p011 has no usable creation times, so its process time is its recorded read and detect, decoding flagged.
    path = write_lines(tmp_path / 'tr0.mp4.detections.jsonl', [[0, [], [{'op': 'read', 'time': 100}, {'op': 'detect', 'time': 900}]]])
    units = build_units(con, monkeypatch, [('011_tune_detect', path, 'train', 'tr0.mp4')], {path: (0.0, 9999.0)},
                        [('train', 'tr0.mp4', 1)])
    assert units[path]['process_s'] == pytest.approx(1.0) and units[path]['decode_s'] == pytest.approx(0.1)
    assert units[path]['start_s'] is None


def make_manifest(con, rows):
    # A manifest table from (stage, runtime_file, video, variant_id) rows.
    con.execute('CREATE OR REPLACE TABLE m (stage VARCHAR, runtime_file VARCHAR, video VARCHAR, variant_id VARCHAR)')
    con.executemany('INSERT INTO m VALUES (?, ?, ?, ?)', rows)


def test_stage_summary_counts_shared_work_once(con, tmp_path, monkeypatch):
    # One classification shared by two configurations, and one tracking run each.
    shared = write_lines(tmp_path / 'cls.jsonl', [[{'op': 'inference', 'time': 10000}]])
    t1 = write_lines(tmp_path / 't1.jsonl', [{'runtime': [{'op': 'tracker_update', 'time': 1000}]}])
    t2 = write_lines(tmp_path / 't2.jsonl', [{'runtime': [{'op': 'tracker_update', 'time': 2000}]}])
    build_units(con, monkeypatch, [('020_exec_classify', shared, 'valid', 'v'), ('060_exec_track', t1, 'valid', 'v'),
                                   ('060_exec_track', t2, 'valid', 'v')], {}, [('valid', 'v', 1)])
    make_manifest(con, [('020_exec_classify', shared, 'v', 'a'), ('020_exec_classify', shared, 'v', 'b'),
                        ('060_exec_track', t1, 'v', 'a'), ('060_exec_track', t2, 'v', 'b')])
    rows = {r['stage']: r for r in prep.stage_summary(con, 'm')}
    # The shared file is one work unit; charging every configuration would count it twice.
    assert rows['020_exec_classify']['work_units'] == 1 and rows['020_exec_classify']['instrumented_s'] == pytest.approx(10)
    assert rows['020_exec_classify']['manifest_rows'] == 2
    assert rows['020_exec_classify']['without_sharing_instrumented_s'] == pytest.approx(20)
    assert rows['020_exec_classify']['without_sharing_process_s'] == pytest.approx(20)
    assert rows['060_exec_track']['process_s'] == pytest.approx(3) == rows['060_exec_track']['without_sharing_process_s']


def test_stage_summary_excludes_dropped_and_decoding_ops_and_keeps_empty_files(con, tmp_path, monkeypatch):
    # p020's decoding is outside p121's boundary; the reference's decoding is inside it but outside the instrumented one.
    cls = write_lines(tmp_path / 'cls.jsonl', [[{'op': 'inference', 'time': 1000}, {'op': 'read', 'time': 9000}]])
    ref = write_lines(tmp_path / 'ref.jsonl', [[{'op': 'read', 'time': 500}, {'op': 'detect', 'time': 1500}]])
    # An empty detection file (no canvases) is a zero-second unit.
    empty = write_lines(tmp_path / 'empty.jsonl', [])
    build_units(con, monkeypatch, [('020_exec_classify', cls, 'valid', 'v'), ('040_exec_detect', empty, 'valid', 'v'),
                                   ('001_preprocess_groundtruth_detection', ref, 'valid', 'v')], {}, [('valid', 'v', 1)])
    make_manifest(con, [('020_exec_classify', cls, 'v', 'a'), ('040_exec_detect', empty, 'v', 'a'),
                        ('001_preprocess_groundtruth_detection', ref, 'v', 'naive')])
    rows = {r['stage']: r for r in prep.stage_summary(con, 'm')}
    assert rows['020_exec_classify']['recorded_s'] == pytest.approx(1)
    assert rows['040_exec_detect']['work_units'] == 1 and rows['040_exec_detect']['recorded_s'] == 0
    reference = rows['001_preprocess_groundtruth_detection']
    assert (reference['recorded_s'], reference['decode_s'], reference['instrumented_s']) == (pytest.approx(2), pytest.approx(0.5),
                                                                                              pytest.approx(1.5))
    # The dropped op is still reported, per stage and op.
    assert prep.dropped_ops(con, 'm') == {'020_exec_classify': {'read': 9.0}}


def test_stage_summary_counts_a_file_without_units_as_zero(con, tmp_path, monkeypatch):
    # A manifest file that never became a unit (e.g. not loaded) costs nothing rather than vanishing from the counts.
    t1 = write_lines(tmp_path / 't1.jsonl', [{'runtime': [{'op': 'a', 'time': 1000}]}])
    build_units(con, monkeypatch, [('060_exec_track', t1, 'valid', 'v')], {}, [('valid', 'v', 1)])
    make_manifest(con, [('060_exec_track', t1, 'v', 'a'), ('060_exec_track', str(tmp_path / 'other.jsonl'), 'v', 'b')])
    (row,) = prep.stage_summary(con, 'm')
    assert row['work_units'] == 2 and row['process_s'] == pytest.approx(1)


# ---------------------------------------------------------------- classifier training, profiles, p015, scoring


def training_file(tmp_path, epochs):
    # A throughput file holding `epochs` fine-tuning epochs of 100 s each (60 s train step, 40 s validation).
    return write_lines(tmp_path / 'tp.jsonl', [[{'op': 'train_step', 'time': 60000},
                                                 {'op': 'val_inference', 'time': 40000}]] * epochs)


def test_training_runs_uses_consistent_timestamps(con, tmp_path):
    path = training_file(tmp_path, 2)
    prep.load_runtime_ops(con, [('013_tune_train_classifier', path)])
    # Started at 0, frozen stage ended at 300, fine-tuning ended at 510: 210 s against 200 s recorded.
    (run,) = prep.training_runs(con, [('M_60', path, 0, 300, 510)])
    assert run['finetune_epochs'] == 2 and run['finetune_recorded_s'] == pytest.approx(200)
    assert run['timestamps_consistent'] and run['seconds'] == 510 and run['frozen_wall_s'] == 300


@pytest.mark.parametrize('frozen_end, end, consistent', [
    # Fine-tuning at its upper bound: recorded x FINETUNE_WALL_MAX_RATIO + slack = 200 x 1.5 + 300 = 600 s.
    (100, 700, True),
    (100, 700.5, False),
    # The frozen stage at its upper bound: 7 epochs x 100 s x 1.5 + 300 = 1350 s.
    (1350, 1950, True),
    (1350.5, 1950, False),
    # Fine-tuning shorter than its recorded epochs, beyond the 1 s timestamp tolerance.
    (100, 298.9, False),
    (100, 299, True),
])
def test_training_runs_boundaries(con, tmp_path, frozen_end, end, consistent):
    path = training_file(tmp_path, 2)
    prep.load_runtime_ops(con, [('013_tune_train_classifier', path)])
    (run,) = prep.training_runs(con, [('M', path, 0, frozen_end, end)])
    assert run['timestamps_consistent'] is consistent
    # Inconsistent timestamps fall back to the recorded epochs, a lower bound.
    assert run['seconds'] == pytest.approx(end if consistent else 200)


def test_training_runs_falls_back_on_stale_unknown_or_empty_runs(con, tmp_path):
    path = training_file(tmp_path, 1)
    # A throughput file without epochs proves nothing about its timestamps.
    empty = write_lines(tmp_path / 'empty.jsonl', [])
    prep.load_runtime_ops(con, [('013_tune_train_classifier', path), ('013_tune_train_classifier', empty)])
    runs = {r['model']: r for r in prep.training_runs(con, [
        # A rerun without --clear keeps the first run's creation times: the fine-tuning span is days.
        ('stale', path, 0, 100, 100 + 5 * 86400),
        # File systems without creation times give no start.
        ('unknown', path, None, None, 1000),
        # A frozen stage that started before its directory existed.
        ('negative', path, 100, 50, 200),
        ('empty', empty, 0, 100, 200),
    ])}
    for model in ('stale', 'unknown', 'negative', 'empty'):
        assert not runs[model]['timestamps_consistent'], model
    # The recorded epochs are the cost, as a lower bound (none for the empty run).
    assert [runs[m]['seconds'] for m in ('stale', 'unknown', 'negative', 'empty')] == [pytest.approx(100)] * 3 + [0]


def test_training_runs_without_runs(con):
    # No trained classifier gives no rows.
    prep.load_runtime_ops(con, [])
    assert prep.training_runs(con, []) == []


def test_profile_estimates_chain_consecutive_profiles(con):
    rows = [
        # One p016 run: three profiles back to back, 4 s between one's last output and the next's first partial.
        ('d1', 'sort_60', 60, 100.0, 107.0),
        ('d1', 'byte_60', 60, 111.0, 117.0),
        ('d2', 'sort_60', 8, 121.0, 122.0),
        # A later run, far apart: its first profile is not chained.
        ('d2', 'byte_60', 8, 5000.0, 5002.0),
    ]
    got = {(r['dataset'], r['profile']): r for r in prep.profile_estimates(con, rows, gap=60)}
    # The first profile of a run gets its own span, flagged as a lower bound.
    assert got[('d1', 'sort_60')]['seconds'] == pytest.approx(7) and got[('d1', 'sort_60')]['lower_bound']
    # Chained profiles cost the difference of consecutive last outputs.
    assert got[('d1', 'byte_60')]['seconds'] == pytest.approx(10) and not got[('d1', 'byte_60')]['lower_bound']
    assert got[('d2', 'sort_60')]['seconds'] == pytest.approx(5)
    assert got[('d2', 'byte_60')]['seconds'] == pytest.approx(2) and got[('d2', 'byte_60')]['lower_bound']


def test_profile_estimates_overlapping_profiles_are_not_chained(con):
    # A profile whose partial files predate the previous one's last output ran concurrently, not after it.
    rows = [('d', 'a', 8, 100.0, 110.0), ('d', 'b', 8, 105.0, 112.0)]
    got = {r['profile']: r for r in prep.profile_estimates(con, rows, gap=60)}
    assert got['b']['lower_bound'] and got['b']['seconds'] == pytest.approx(7)


def test_profile_estimates_gap_boundary_and_ties(con):
    # Exactly `gap` seconds apart still chains.
    rows = [('d', 'a', 8, 0.0, 10.0), ('d', 'b', 8, 70.0, 75.0)]
    got = {r['profile']: r for r in prep.profile_estimates(con, rows, gap=60)}
    assert got['b']['chained'] and got['b']['seconds'] == pytest.approx(65)
    # Equal last outputs are ordered by dataset and profile, so the result does not depend on input order.
    tied = [('d', 'y', 8, 10.0, 20.0), ('d', 'x', 8, 10.0, 20.0)]
    first = prep.profile_estimates(con, tied, gap=60)
    assert first == prep.profile_estimates(con, tied[::-1], gap=60)
    assert {r['profile']: r['chained'] for r in first} == {'x': False, 'y': False}
    # No profiles, no rows.
    assert prep.profile_estimates(con, [], gap=60) == []


def test_gather_profiles_reports_complete_and_incomplete_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.cache, 'index', lambda dataset, stage, *a: tmp_path / dataset / stage)
    profile = tmp_path / 'd' / 'track_rates' / 'sortcython_60'
    touch(profile / 'partial' / 'tr00.mp4.npy', 100)
    touch(profile / 'partial' / 'tr01.mp4.npy', 103)
    touch(profile / 'accuracy.npy', 104)
    touch(profile / 'max_rate_060.txt', 105)
    # A visualization directory and a stray file are not profiles or outputs.
    touch(profile / 'visualization' / 'x.png', 999)
    touch(tmp_path / 'd' / 'track_rates' / 'notes.txt', 999)
    # Partial counts without aggregate outputs: p016 did not finish this one.
    touch(tmp_path / 'd' / 'track_rates' / 'bytetrackcython_60' / 'partial' / 'tr00.mp4.npy', 100)
    complete, incomplete = prep.gather_profiles(['d', 'empty'])
    assert complete == [('d', 'sortcython_60', 2, 100.0, 105.0)]
    assert incomplete == [dict(dataset='d', profile='bytetrackcython_60', partials=1, outputs=0)]


def test_optimization_estimates_time_each_resolution_group(con):
    rows = [
        # First group of the run: two datasets written within P015_GROUP_GAP_S; nothing to measure it from.
        ('a', 'M1', 100.0), ('b', 'M1', 100.0 + prep.P015_GROUP_GAP_S),
        # Second group: 29.5 s after the first finished.
        ('a', 'M2', 130.5),
        # After a long pause: a new run, not timed.
        ('a', 'M3', 1000.0),
    ]
    got = {(r['dataset'], r['model']): r for r in prep.optimization_estimates(con, rows, gap=60)}
    assert got[('a', 'M1')]['group_index'] == got[('b', 'M1')]['group_index']
    assert got[('a', 'M1')]['seconds'] is None and got[('b', 'M1')]['seconds'] is None
    assert got[('a', 'M2')]['seconds'] == pytest.approx(29.5)
    assert got[('a', 'M3')]['seconds'] is None
    # No benchmarks, no rows.
    assert prep.optimization_estimates(con, [], gap=60) == []


def test_scoring_estimate_scales_usable_test_tasks(con):
    rows = [
        # Usable: 10 s over 2 videos each.
        ('A', 2, 0.0, 10.0), ('B', 2, 5.0, 15.0),
        # Unusable: no creation time, no videos, a stale creation time.
        ('C', 2, None, 20.0), ('D', 0, 0.0, 5.0), ('E', 2, 0.0, prep.STALE_SPAN_S + 1),
    ]
    got = prep.scoring_estimate(con, rows, configurations=3, videos=4)
    assert (got['tasks'], got['pairs'], got['unusable_tasks']) == (2, 4, 3)
    assert got['seconds_per_pair'] == pytest.approx(5) and got['estimate_s'] == pytest.approx(5 * 3 * 4)
    # Without a usable task there is no estimate.
    assert prep.scoring_estimate(con, rows[2:], 3, 4)['estimate_s'] is None
    assert prep.scoring_estimate(con, [], 3, 4)['estimate_s'] is None


def test_gather_scoring_reads_test_task_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.cache, 'eval', lambda dataset, stage, *a: tmp_path.joinpath(dataset, stage, *a))
    root = tmp_path / 'd' / 'acc' / 'raw' / 'test'
    # One scored configuration (two videos plus the combined file), the reference's task, and an empty directory.
    for name in ('v0.json', 'v1.json', 'DATASET.json'):
        touch(root / 'A' / name, 0)
    touch(root / 'naive' / 'v0.json', 0)
    (root / 'empty').mkdir()
    fake_file_times(monkeypatch, {str(root / 'A'): (100.0, 0.0), str(root / 'A' / 'v0.json'): (None, 103.0),
                                  str(root / 'A' / 'v1.json'): (None, 104.0), str(root / 'A' / 'DATASET.json'): (None, 105.0)})
    assert prep.gather_scoring('d') == [('A', 2, 100.0, 105.0)]


# ---------------------------------------------------------------- splits, operating point, break-even


def test_video_split_counts_frames_and_minutes(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.store, 'dataset', lambda dataset, split, *name: tmp_path.joinpath(dataset, split, *name))
    (tmp_path / 'd' / 'train').mkdir(parents=True)
    # Two 15 FPS clips of 30 and 45 frames: 75 frames, 5 seconds; a non-video file is ignored.
    for name, frames in (('a.mp4', 30), ('b.mp4', 45)):
        writer = cv2.VideoWriter(str(tmp_path / 'd' / 'train' / name), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 48))
        for _ in range(frames):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
    (tmp_path / 'd' / 'train' / 'notes.txt').write_text('x')
    got = prep.video_split('d', 'train')
    assert got['videos'] == 2 and got['frames'] == 75 and got['names'] == ['a.mp4', 'b.mp4']
    assert got['seconds'] == pytest.approx(5) and got['minutes'] == pytest.approx(0.08)
    # A split without videos is an error.
    (tmp_path / 'd' / 'valid').mkdir()
    with pytest.raises(AssertionError, match='no videos'):
        prep.video_split('d', 'valid')
    # So is an unreadable video file.
    (tmp_path / 'd' / 'test').mkdir()
    (tmp_path / 'd' / 'test' / 'bad.mp4').write_text('not a video')
    with pytest.raises(AssertionError, match='unreadable'):
        prep.video_split('d', 'test')


def fake_tradeoff(rows, reference_fps=10.0, reference_hota=0.8, frames=30):
    # A load_test_tradeoff stand-in from (variant_id, fps, hota) rows; each time is frames / fps.
    tetris = pd.DataFrame([(c, f, h, frames / f, frames) for c, f, h in rows],
                          columns=['variant_id', 'throughput_fps', 'HOTA_HOTA', 'time', 'frame_count'])
    naive = pd.Series({'throughput_fps': reference_fps, 'HOTA_HOTA': reference_hota, 'time': frames / reference_fps,
                       'frame_count': frames})
    return lambda dataset: (tetris, naive)


POINT = 'ShuffleNet05_60_4_040_r025_bl_s100_bytetrackcython'
OTHER = 'ShuffleNet05_60_4_040_r025_bl_s100_sortcython'
TRADEOFF = [
    # Fastest, but below 0.8 x 0.95 = 0.76.
    ('ShuffleNet05_60_16_r050_none_s100_sortcython', 900.0, 0.70),
    # Fastest within the bound.
    (POINT, 300.0, 0.77),
    (OTHER, 100.0, 0.79),
]


def test_operating_point_is_fastest_within_the_bound(con, monkeypatch):
    monkeypatch.setattr(prep, 'load_test_tradeoff', fake_tradeoff(TRADEOFF))
    point = prep.operating_point(con, 'd', 0.05)
    assert point['config'] == POINT
    assert (point['classifier'], point['tilesize'], point['tracker'], point['sample_rate']) == ('ShuffleNet05', 60, 'bytetrackcython', 4)
    assert point['fps'] == 300 and point['reference_fps'] == 10 and point['bound'] == pytest.approx(0.76)
    assert point['time_s'] == pytest.approx(0.1) and point['frames'] == 30 == point['test_frames']
    assert point['provisional'] is None
    # B3D's operating points wait for R1.O1's ground truth.
    assert prep.operating_point(con, 'jnc0', 0.05)['provisional']


def test_operating_point_fails_when_no_configuration_qualifies(con, monkeypatch):
    monkeypatch.setattr(prep, 'load_test_tradeoff', fake_tradeoff([('ShuffleNet05_60_1_r025_none_s100_sortcython', 50.0, 0.5)]))
    # Without an operating point there is nothing to break even at.
    with pytest.raises(AssertionError, match='no Tetris configuration'):
        prep.operating_point(con, 'd', 0.05)


def test_break_even_repays_preparation_with_saved_extraction_time():
    # 900 frames per minute: the reference takes 90 s, Tetris 9 s, saving 81 s per video minute.
    got = prep.break_even(8100, reference_fps=10, tetris_fps=100, frames_per_minute=900)
    assert got['saving_s_per_video_minute'] == pytest.approx(81)
    assert got['break_even_video_minutes'] == pytest.approx(100) and got['break_even_video_hours'] == pytest.approx(1.67)
    # No preparation breaks even immediately.
    assert prep.break_even(0, 10, 100, 900)['break_even_video_minutes'] == 0


def test_break_even_never_without_a_saving():
    # Equal or slower extraction never repays preparation.
    for tetris_fps in (10, 5):
        got = prep.break_even(100, reference_fps=10, tetris_fps=tetris_fps, frames_per_minute=900)
        assert got['break_even_video_minutes'] is None and got['break_even_video_hours'] is None
        assert got['saving_s_per_video_minute'] <= 0


@pytest.mark.parametrize('args', [(100, 0, 10, 900), (100, 10, 0, 900), (100, 10, 20, 0), (100, -1, 10, 900), (-1, 10, 20, 900)])
def test_break_even_rejects_bad_inputs(args):
    # Rates must be positive and preparation non-negative.
    with pytest.raises(AssertionError):
        prep.break_even(*args)


def test_sweep_coverage_scales_by_complete_videos(con):
    con.execute('CREATE TABLE valid_manifest (variant VARCHAR, video VARCHAR, "exists" BOOLEAN)')
    con.executemany('INSERT INTO valid_manifest VALUES (?, ?, ?)', [
        # Two complete videos and one with a missing sweep file; the reference row does not count.
        ('polytris', 'v0', True), ('polytris', 'v0', True), ('polytris', 'v1', True),
        ('polytris', 'v2', True), ('polytris', 'v2', False), ('naive', 'v2', True), ('naive', 'v3', False)])
    got = prep.sweep_coverage(con)
    assert got == dict(videos=3, complete_videos=2, scale=1.5, incomplete={'v2': 1})
    # The complete videos are a table the sweep manifest is filtered by.
    assert sorted(v for (v,) in con.execute('SELECT video FROM complete_videos').fetchall()) == ['v0', 'v1']
    # Without a complete video there is nothing to scale.
    con.execute("UPDATE valid_manifest SET \"exists\" = false WHERE variant = 'polytris'")
    with pytest.raises(AssertionError, match='no validation video'):
        prep.sweep_coverage(con)


def test_component_rounds_once_and_reports_hours():
    got = prep.component('basis', 1.04, 3600.06, 7200.04, extra=1)
    assert (got['recorded_s'], got['instrumented_s'], got['process_s']) == (1.0, 3600.1, 7200.0)
    assert (got['instrumented_hours'], got['process_hours'], got['unit'], got['extra']) == (1.0, 2.0, prep.PROCESS_UNIT, 1)


# ---------------------------------------------------------------- measure_dataset, end to end on a fake cache


DS = 'ds'
# The world's test split: the operating point extracts its 30 frames in 3 s (10 fps), as its recorded ops add up to.
WORLD_TRADEOFF = [('ShuffleNet05_60_16_r050_none_s100_sortcython', 900.0, 0.70), (POINT, 10.0, 0.77), (OTHER, 5.0, 0.79)]


@pytest.fixture
def world(tmp_path, monkeypatch, con):
    """A two-video training, validation, and one-video test split with every input measure_dataset reads."""
    root = tmp_path / 'cache' / DS
    execution = root / 'execution'
    training = root / 'indexing' / 'training'
    files = {}

    def add(key, path, lines):
        # Write a runtime file and remember its path.
        files[key] = write_lines(path, lines)

    # Training split: p011 times 10 frames per video (0.01 s read + 0.1 s detect each).
    for v in ('tr0.mp4', 'tr1.mp4'):
        add(('011', v), root / 'indexing' / 'segment' / 'detection' / f'{v}.detections.jsonl',
            [[i, [], [{'op': 'read', 'time': 10}, {'op': 'detect', 'time': 100}]] for i in range(10)])
        # p012 at both tile sizes; only 60 is the operating point's (0.3 s per frame).
        for tile, ms in ((60, 200), (30, 999)):
            add(('012', tile, v), training / 'runtime' / f'tilesize_{tile}' / f'{v}_creating_training_data.jsonl',
                [line for i in range(10) for line in ({'op': 'split', 'time': ms, 'frame': i}, {'op': 'save', 'time': 100, 'frame': i})])
    # Two trained classifiers with 100 s fine-tuning epochs: the operating point's (two epochs) and p017's baseline (one).
    for model, epochs in (('ShuffleNet05_60', 2), ('Baseline_60', 1)):
        add(('013', model), training / 'results' / model / 'throughput_per_epoch.jsonl',
            [[{'op': 'train_step', 'time': 60000}, {'op': 'val_inference', 'time': 40000}]] * epochs)

    def reference(videoset, video, frames):
        # The reference pipeline on one video: read 0.01 s, detect 0.1 s, and track 0.005 s per frame.
        add(('001', video), execution / video / '002_naive' / 'detection_runtime.jsonl',
            [[{'op': 'read', 'time': 10}, {'op': 'detect', 'time': 100}] for _ in range(frames)])
        add(('002', video), execution / video / '002_naive' / 'tracking_runtime.jsonl',
            [{'frame_idx': i, 'runtime': [{'op': 'tracker_update', 'time': 5}]} for i in range(frames)])
        return [('001_preprocess_groundtruth_detection', files[('001', video)], videoset, video, 'naive', 'naive', None, None),
                ('002_preprocess_groundtruth_tracking', files[('002', video)], videoset, video, 'naive', 'naive', None, None)]

    def sweep(videoset, video, config, write_060=True):
        # Classification (2 s, shared by every configuration of ShuffleNet05_60_4; decoding dropped) and tracking (1 s).
        add(('020', video), execution / video / '020_relevancy' / 'ShuffleNet05_60_4' / 'score' / 'runtime.jsonl',
            [[{'op': 'inference', 'time': 2000}, {'op': 'read', 'time': 1000}]])
        track = execution / video / '060_uncompressed_tracks' / config / 'runtimes.jsonl'
        # A cleared tracking output is listed but missing.
        if write_060:
            add(('060', video, config), track, [{'runtime': [{'op': 'track', 'time': 1000}]}])
        return [('020_exec_classify', files[('020', video)], videoset, video, 'polytris', config, 'ShuffleNet05', '60'),
                ('060_exec_track', str(track), videoset, video, 'polytris', config, 'ShuffleNet05', '60')]

    columns = ['stage', 'runtime_file', 'videoset', 'video', 'variant', 'variant_id', 'classifier', 'tilesize']
    # p120's index manifest: p011 and p012 per video, p013 per dataset.
    index = pd.DataFrame(
        [('011_tune_detect', files[('011', v)], 'train', v, 'index', 'index', pd.NA, pd.NA) for v in ('tr0.mp4', 'tr1.mp4')]
        + [('012_tune_create_training_data', files[('012', t, v)], 'train', v, 'index', 'index', pd.NA, t)
           for t in (60, 30) for v in ('tr0.mp4', 'tr1.mp4')]
        + [('013_tune_train_classifier', files[('013', m)], 'train', 'dataset_level', 'index', 'index', m.split('_')[0], 60)
           for m in ('ShuffleNet05_60', 'Baseline_60')], columns=columns)
    # Validation: both configurations on va0; OTHER's tracking on va1 was cleared, so va1 is incomplete.
    valid = pd.DataFrame(reference('valid', 'va0.mp4', 20) + reference('valid', 'va1.mp4', 20)
                         + sweep('valid', 'va0.mp4', POINT) + sweep('valid', 'va0.mp4', OTHER)
                         + sweep('valid', 'va1.mp4', POINT) + sweep('valid', 'va1.mp4', OTHER, write_060=False), columns=columns)
    # Test: the reference and the operating point on one 30-frame video.
    test = pd.DataFrame(reference('test', 'te0.mp4', 30) + sweep('test', 'te0.mp4', POINT), columns=columns)

    def split_manifest(dataset, grid=None, videoset='valid'):
        # The test manifest is built for the operating point's configuration only.
        assert grid is None or (videoset == 'test' and grid['variant_id'].tolist() == [POINT])
        df = (valid if videoset == 'valid' else test).copy()
        df['exists'] = df['runtime_file'].map(os.path.exists)
        return df

    # The fake dataset's label, and the inputs measure_dataset reads, all from this world.
    monkeypatch.setitem(prep.DATASET_LABELS, DS, 'Fake')
    monkeypatch.setattr(prep, 'build_index_construction_manifest', lambda datasets: index)
    monkeypatch.setattr(prep, 'split_manifest', split_manifest)
    monkeypatch.setattr(prep, 'build_polytris_variant_manifest', lambda: pd.DataFrame({'variant_id': [POINT, OTHER]}))
    monkeypatch.setattr(prep.cache, 'index', lambda dataset, stage, *a: tmp_path.joinpath('cache', dataset, 'indexing',
                                                                                         cache.INDEX_STAGE_MAP[stage], *a))
    monkeypatch.setattr(prep, 'load_test_tradeoff', fake_tradeoff(WORLD_TRADEOFF, reference_fps=30 / 3.45))
    monkeypatch.setattr(prep, 'TRACKERS', ['sortcython', 'bytetrackcython'])
    monkeypatch.setattr(prep, 'gather_scoring', lambda dataset: [(POINT, 1, 0.0, 10.0)])
    splits = dict(train=dict(videos=2, frames=20, seconds=20 / 15), valid=dict(videos=2, frames=40, seconds=40 / 15),
                  test=dict(videos=1, frames=30, seconds=2.0))
    monkeypatch.setattr(prep, 'video_split', lambda dataset, split: {**splits[split], 'minutes': 0, 'names': []})
    # Timestamps: both training runs consistent (510 s and 410 s), p012 10 s per video, the test reference 5 s, the
    # test classification 4 s; every other unit has none and keeps its recorded time.
    times = {}
    for model, end in (('ShuffleNet05_60', 510.0), ('Baseline_60', 410.0)):
        times[str(Path(files[('013', model)]).parent)] = (0.0, 0.0)
        times[files[('013', model)]] = (300.0, end)
    for v in ('tr0.mp4', 'tr1.mp4'):
        times[files[('012', 60, v)]] = (1000.0, 1010.0)
    times[files[('001', 'te0.mp4')]] = (0.0, 0.1)
    times[str(Path(files[('001', 'te0.mp4')]).with_name('detection.jsonl'))] = (None, 5.0)
    times[str(Path(files[('020', 'te0.mp4')]).parent.parent)] = (0.0, 0.0)
    times[str(Path(files[('020', 'te0.mp4')]).with_name('score.jsonl'))] = (None, 4.0)
    fake_file_times(monkeypatch, times)
    # Three profiles of one p016 run: the configured trackers' and an unneeded one.
    prep.profile_estimates(con, [(DS, 'sortcython_60', 2, 100.0, 105.0), (DS, 'bytetrackcython_60', 2, 106.0, 110.0),
                                 (DS, 'ocsortcython_60', 2, 111.0, 115.0)])
    return dict(splits=splits, compilations=[dict(dataset=DS, model='ShuffleNet05_60', seconds=30.0, finish_s=5000.0)])


def test_measure_dataset_end_to_end(con, world):
    got = prep.measure_dataset(con, DS, 0.05, world['compilations'])
    c = got['components']
    seconds = {k: (v['recorded_s'], v['instrumented_s'], v['process_s']) for k, v in c.items()}
    assert seconds == {
        # p011: 20 frames x 0.11 s; the instrumented boundary drops 0.01 s of decoding per frame.
        'reference_detection_train': (2.2, 2.0, 2.2),
        # p012 at tile size 60 only: 20 frames x 0.3 s recorded, 10 s per video from timestamps.
        'training_data_creation': (6.0, 6.0, 20.0),
        # The operating point's classifier: fine-tuning recorded, both stages from timestamps.
        'classifier_training': (200.0, 510.0, 510.0),
        'classifier_optimization': (0.0, 30.0, 30.0),
        # Two needed profiles of 5 s elapsed each, x min(40 workers, 2 videos).
        'mistrack_profiling': (0.0, 20.0, 20.0),
        # The validation reference: 40 frames x (0.11 + 0.005) s; decoding 0.4 s.
        'reference_outputs_valid': (4.6, 4.2, 4.6),
        # va0 only: one shared classification (2 s) and two tracking runs (1 s each), scaled by 2 videos / 1.
        'validation_sweep': (4.0, 8.0, 8.0),
        # 10 s per (configuration, video) x 2 configurations x 2 videos.
        'validation_scoring': (0.0, 40.0, 40.0),
    }
    sweep = c['validation_sweep']
    assert sweep['coverage'] == dict(videos=2, complete_videos=1, scale=2.0, incomplete={'va1.mp4': 1})
    # Charging the shared classification to both configurations: (2 + 2 + 1 + 1) x 2.
    assert sweep['without_sharing_instrumented_s'] == 12.0 and sweep['configurations'] == 2
    assert sweep['dropped_ops_s'] == {'020_exec_classify': {'read': 1.0}}
    # Only the operating point's classifier and trackers are charged; the rest is reported apart.
    assert c['classifier_training']['model'] == 'ShuffleNet05_60' and c['classifier_training']['timestamps_consistent']
    assert {k: v['seconds'] for k, v in got['not_needed_by_reported_configuration']['classifier_training'].items()} == \
        {'Baseline_60': 410.0}
    assert list(got['not_needed_by_reported_configuration']['mistrack_profiling']) == ['ocsortcython_60']
    assert c['reference_detection_train']['frames'] == 20
    assert c['reference_detection_train']['estimate_from_test_reference_detect_s'] == pytest.approx(2.0)
    assert c['training_data_creation']['first_frame_split_s'] == pytest.approx(0.4)
    # Totals are sums of the components the report shows, and the shares of each boundary add up to one.
    totals = got['totals']
    assert totals['recorded_lower_bound_s'] == pytest.approx(216.8)
    assert totals['instrumented_s'] == pytest.approx(620.2) and totals['process_s'] == pytest.approx(634.8)
    assert 'sweep on 1 incomplete videos' in totals['recorded_lower_bound_omits']
    for boundary in ('instrumented', 'process'):
        assert sum(totals['shares'][boundary].values()) == pytest.approx(1, abs=1e-3)
    # Test-split extraction: the reference 3.45 s as the paper counts it, 3.15 s without decoding, 5.15 s by timestamps;
    # Tetris 3 s, 3 s, and 5 s.
    extraction = got['extraction']
    assert extraction['seconds'] == dict(reference=dict(paper=3.45, instrumented=3.15, process=5.15),
                                         tetris=dict(paper=3.0, instrumented=3.0, process=5.0))
    assert extraction['paper_time_ratio_to_tradeoff_table'] == dict(reference=1.0, tetris=pytest.approx(1.0, abs=1e-3))
    # Break-even at 900 frames per minute: the same 4.5 s saving per video minute on both boundaries here.
    even = got['break_even']
    assert even['frames_per_video_minute'] == 900
    assert even['instrumented']['saving_s_per_video_minute'] == pytest.approx(4.5)
    assert even['instrumented']['break_even_video_minutes'] == pytest.approx(620.2 / 4.5, abs=0.1)
    assert even['process']['break_even_video_minutes'] == pytest.approx(634.8 / 4.5, abs=0.1)


def test_measure_dataset_requires_every_training_frame_timed(con, world, monkeypatch):
    # One training frame more than p011 timed means p011's records are incomplete.
    splits = {**world['splits'], 'train': {**world['splits']['train'], 'frames': 21}}
    monkeypatch.setattr(prep, 'video_split', lambda dataset, split: {**splits[split], 'minutes': 0, 'names': []})
    with pytest.raises(AssertionError, match='p011 timed 20 of 21'):
        prep.measure_dataset(con, DS, 0.05, world['compilations'])


def test_measure_dataset_requires_the_configured_profiles(con, world):
    # A configured tracker without a finished profile cannot be charged.
    prep.profile_estimates(con, [(DS, 'sortcython_60', 2, 100.0, 105.0)])
    with pytest.raises(AssertionError, match='missing mistrack profiles'):
        prep.measure_dataset(con, DS, 0.05, world['compilations'])


def test_measure_dataset_rejects_profiles_of_another_video_set(con, world):
    # A profile built from 3 partial files when the split has 2 videos is not this split's.
    prep.profile_estimates(con, [(DS, 'sortcython_60', 3, 100.0, 105.0), (DS, 'bytetrackcython_60', 2, 106.0, 110.0)])
    with pytest.raises(AssertionError, match='one partial file per training video'):
        prep.measure_dataset(con, DS, 0.05, world['compilations'])


def test_measure_dataset_without_a_p015_estimate(con, world):
    # The first resolution group of a p015 run has no estimate: zero seconds, flagged missing.
    got = prep.measure_dataset(con, DS, 0.05, [])
    optimization = got['components']['classifier_optimization']
    assert optimization['process_s'] == 0 and not optimization['estimated'] and optimization['basis'].startswith('missing')


# ---------------------------------------------------------------- the GPU re-timing plan


def covered(monkeypatch, videos, excluded=()):
    # Validation coverage as covered_validation_videos would find it in the source cache.
    monkeypatch.setattr(gpu_plan, 'covered_validation_videos', lambda dataset: (list(videos), list(excluded)))


def test_gpu_plan_dry_run_executes_and_writes_nothing(tmp_path, monkeypatch, capsys):
    # Any subprocess call in a dry run is a bug.
    monkeypatch.setattr(gpu_plan.subprocess, 'run', lambda *a, **k: pytest.fail(f'dry run executed {a}'))
    covered(monkeypatch, ['va00.mp4', 'va01.mp4'])
    monkeypatch.setattr('sys.argv', ['plan', '--scratch', str(tmp_path / 'scratch'), '--datasets', 'caldot1-y05', 'jnc0'])
    gpu_plan.main()
    out = capsys.readouterr().out
    assert out.startswith('DRY RUN')
    config = tmp_path / 'scratch' / 'configs' / 'caldot1-y05.yaml'
    # CPU steps first, with no GPU visible and p110 at P110_CPU_COUNT CPUs; GPU steps on the launcher's only GPU, 0.
    assert f"CONFIG={config} CUDA_VISIBLE_DEVICES='' PYTHON_CPU_COUNT=20 python evaluation/p110_accuracy_compute.py --valid" in out
    assert f'CONFIG={config} CUDA_VISIBLE_DEVICES=0 python scripts/p012_tune_create_training_data.py' in out
    assert out.index('p110_accuracy_compute') < out.index('p012_tune_create_training_data')
    # p011 and p016 are optional.
    assert 'p011_tune_detect' not in out and 'p016_tune_track_rate' not in out
    # Baseline steps are printed as manual, and B3D has no OTIF-format data.
    assert 'MANUAL' in out and '30_speed_accuracy.go' in out and 'no OTIF-format copy of jnc0' in out
    assert not (tmp_path / 'scratch').exists()


def test_gpu_plan_steps_and_copies(tmp_path, monkeypatch):
    covered(monkeypatch, ['va00.mp4'])
    scratch = tmp_path / 'scratch'
    steps = gpu_plan.polyis_steps('caldot1-y05', list(gpu_plan.DEFAULT_STEPS), scratch, 4)
    assert [s['step'] for s in steps] == ['copy validation tracking outputs', 'p110', 'copy cached p011 detections',
                                          'p012', 'p013', 'detector_load']
    # Validation tracking outputs are cloned with their relative paths, and the ground truth's directory is created.
    copy = steps[0]
    assert copy['cwd'] == str(cache.execution('caldot1-y05'))
    assert copy['argv'][:2] == ['find', 'va00.mp4'] and str(scratch / 'cache' / 'caldot1-y05' / 'execution') in copy['argv']
    assert '--reflink=auto' in copy['argv'] and str(scratch / 'cache' / 'caldot1' / 'execution') in copy['mkdir']
    # With every video covered, p110 reads the dataset's own config and records what it scores.
    p110 = steps[1]
    assert p110['env']['CONFIG'] == str(scratch / 'configs' / 'caldot1-y05.yaml') and p110['config'] == {}
    assert p110['env']['PYTHON_CPU_COUNT'] == str(gpu_plan.P110_CPU_COUNT)
    assert p110['meta'] == dict(videos=['va00.mp4'], excluded_videos=[])
    # p011's cached detections are copied from the cache to the same place under scratch.
    detections = steps[2]
    assert detections['argv'] == ['cp', '-r', str(cache.index('caldot1-y05', 'det')),
                                  str(scratch / 'cache' / 'caldot1-y05' / 'indexing' / 'segment')]
    # CPU steps see no GPU; GPU steps see only the chosen one.
    assert {s['step']: s['env'].get('CUDA_VISIBLE_DEVICES') for s in steps if s['env']} == \
        dict(p110='', p012='4', p013='4', detector_load='4')
    # With p011 in the run nothing is copied, and p011 comes before p016, which runs with one worker.
    steps = gpu_plan.polyis_steps('d', ['p016', 'p011'], scratch, 4)
    assert [s['step'] for s in steps] == ['p011', 'p016'] and steps[1]['argv'][-2:] == ['--num_workers', '1']
    # p013 prunes p012's output, so it cannot run alone.
    with pytest.raises(AssertionError, match='p013 needs p012'):
        gpu_plan.polyis_steps('d', ['p013'], scratch, 4)


def test_gpu_plan_p110_density_scores_at_the_deployed_density(tmp_path, monkeypatch):
    covered(monkeypatch, ['va00.mp4'], ['va01.mp4'])
    scratch = tmp_path / 'scratch'
    # The CPUs this process may use, as the launcher pins it to a 40-CPU NUMA node.
    monkeypatch.setattr(gpu_plan.os, 'sched_getaffinity', lambda pid: set(range(40)))
    steps = gpu_plan.polyis_steps('caldot2-y05', ['p110', 'p110_density'], scratch, 0)
    # One copy and one directory of links serve both scorings, the re-timing's first.
    assert [s['step'] for s in steps] == ['copy validation tracking outputs', 'link covered validation videos', 'p110',
                                          'p110_density']
    p110, dense = steps[2:]
    # The same command, config, and videos; only the CPU count differs (16 and 32 TrackEval processes).
    assert p110['argv'] == dense['argv'] and p110['meta'] == dense['meta'] and p110['config'] == dense['config']
    assert (p110['env']['PYTHON_CPU_COUNT'], dense['env']['PYTHON_CPU_COUNT']) == (str(gpu_plan.P110_CPU_COUNT), '40')
    assert {k: v for k, v in p110['env'].items() if k != 'PYTHON_CPU_COUNT'} == \
        {k: v for k, v in dense['env'].items() if k != 'PYTHON_CPU_COUNT'}
    # Its outputs are recorded like p110's, and it is not run by default.
    assert gpu_plan.STEP_OUTPUTS['p110_density'] == gpu_plan.STEP_OUTPUTS['p110']
    assert 'p110_density' not in gpu_plan.DEFAULT_STEPS and gpu_plan.STEPS.index('p110_density') == 1
    # Alone it still copies its inputs.
    assert [s['step'] for s in gpu_plan.polyis_steps('caldot2-y05', ['p110_density'], scratch, 0)] == \
        ['copy validation tracking outputs', 'link covered validation videos', 'p110_density']


def test_gpu_plan_and_measurement_share_the_p016_worker_count():
    # The measurement checks the record's worker count against its own constant; the plan runs p016 with its own.
    assert gpu_plan.P016_RETIMING_WORKERS == prep.P016_RETIMING_WORKERS == 1
    # The cache's p016 ran with p016's default.
    assert prep.P016_CACHE_WORKERS == 40


def test_gpu_plan_refuses_scratch_inside_real_data(tmp_path):
    for bad in ('/polyis-cache', '/polyis-cache/scratch', '/polyis-data/x', '/otif-dataset/x', 'relative/scratch'):
        with pytest.raises(AssertionError):
            gpu_plan.check_scratch(bad, '/polyis-cache/')
    # The configured cache directory is protected even when it is not one of the usual roots.
    with pytest.raises(AssertionError):
        gpu_plan.check_scratch(str(tmp_path / 'cache' / 'x'), str(tmp_path / 'cache'))
    assert gpu_plan.check_scratch(str(tmp_path / 'scratch'), '/polyis-cache/') == tmp_path / 'scratch'


def test_gpu_plan_inside_resolves_links(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    # A link inside the scratch directory that points outside it does not count as inside.
    (scratch / 'escape').symlink_to(tmp_path)
    assert gpu_plan.inside(scratch / 'a' / 'b', scratch) and gpu_plan.inside(scratch, scratch)
    assert not gpu_plan.inside(scratch / 'escape' / 'x', scratch) and not gpu_plan.inside(scratch / '..' / 'x', scratch)


def test_gpu_plan_scratch_config_moves_only_the_cache(tmp_path):
    config = gpu_plan.scratch_config('jnc0', tmp_path)
    assert config['DATA']['CACHE_DIR'] == f'{tmp_path}/cache/'
    assert config['EXEC']['DATASETS'] == ['jnc0']
    # Videos are still read from the real dataset directory.
    assert config['DATA']['DATASETS_DIR'] == prep.CONFIG['DATA']['DATASETS_DIR']


@pytest.mark.parametrize('utilization, memory, busy', [(0, 99, False), (1, 0, True), (0, 100, True)])
def test_gpu_plan_busy_thresholds(utilization, memory, busy):
    # Idle is 0% and under 100 MiB, as the launcher requires.
    assert gpu_plan.busy(dict(utilization_percent=utilization, memory_used_mib=memory)) is busy


def test_gpu_plan_wait_idle_waits_for_a_lingering_utilization(monkeypatch):
    # Utilization is a trailing average: busy twice, then idle.
    snapshots = iter([dict(utilization_percent=u, memory_used_mib=0) for u in (40, 3, 0, 0)])
    monkeypatch.setattr(gpu_plan, 'gpu_snapshot', lambda gpu: next(snapshots))
    monkeypatch.setattr(gpu_plan.time, 'sleep', lambda s: None)
    assert gpu_plan.wait_idle(0) == dict(utilization_percent=0, memory_used_mib=0)
    # Past the deadline the last busy snapshot comes back, so the caller refuses.
    monkeypatch.setattr(gpu_plan, 'IDLE_WAIT_S', 0)
    monkeypatch.setattr(gpu_plan, 'gpu_snapshot', lambda gpu: dict(utilization_percent=50, memory_used_mib=0))
    assert gpu_plan.busy(gpu_plan.wait_idle(0))


def fake_run(monkeypatch, returncodes=None, utilization=0):
    # Record step commands instead of running them (stat calls still run), and report the GPU as idle or busy.
    calls = []
    codes = iter(returncodes or [])
    real = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == 'xargs':
            return real(argv, **kwargs)
        calls.append(dict(argv=argv, cwd=kwargs['cwd'], env=kwargs['env']))
        return subprocess.CompletedProcess(argv, next(codes, 0))
    monkeypatch.setattr(gpu_plan.subprocess, 'run', run)
    monkeypatch.setattr(gpu_plan, 'gpu_snapshot', lambda gpu: dict(utilization_percent=utilization, memory_used_mib=0))
    # No waiting for a busy GPU, and no launcher samples (outside the launcher).
    monkeypatch.setattr(gpu_plan, 'IDLE_WAIT_S', 0)
    monkeypatch.delenv('BOMBE_RUN', raising=False)
    return calls


def test_gpu_plan_run_writes_config_timings_and_logs_under_scratch(tmp_path, monkeypatch):
    calls = fake_run(monkeypatch)
    scratch = tmp_path / 'scratch'
    steps = gpu_plan.polyis_steps('jnc0', ['p016'], scratch, 4)
    gpu_plan.run(steps, scratch, 4)
    # The copy, then p016 with the scratch config.
    assert [c['argv'][0] for c in calls] == ['cp', 'python'] and calls[1]['env']['CONFIG'] == str(scratch / 'configs' / 'jnc0.yaml')
    config = yaml.safe_load((scratch / 'configs' / 'jnc0.yaml').read_text())
    assert config['DATA']['CACHE_DIR'] == f'{scratch}/cache/' and config['EXEC']['DATASETS'] == ['jnc0']
    # One timing record per step, each with its log, CPU time, interference, and placement.
    records = [json.loads(line) for line in (scratch / 'timings.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == ['copy cached p011 detections', 'p016']
    assert all(r['returncode'] == 0 and Path(r['log']).exists() for r in records)
    assert len({r['record_id'] for r in records}) == 2 and all(r['cpu_s'] >= 0 for r in records)
    assert records[1]['interference']['gpu_samples'] is None and records[1]['interference']['cpu_set_size'] > 0
    assert records[1]['placement']['cpus'] == sorted(os.sched_getaffinity(0))
    # Outside the launcher the record id says so.
    assert all(r['record_id'].startswith(f'{gpu_plan.socket.gethostname()}-local-') for r in records)
    # The copy's target directory was created; nothing was written outside the test's directory.
    assert (scratch / 'cache' / 'jnc0' / 'indexing' / 'segment').is_dir()
    assert {p.name for p in tmp_path.iterdir()} == {'scratch'}


def test_gpu_plan_record_ids_name_the_launcher_run(tmp_path, monkeypatch):
    fake_run(monkeypatch)
    # Two containers have the same host name and pid (1); the launcher's run id tells their records apart.
    monkeypatch.setenv('BOMBE_RUN', 'g4-20261027T061546')
    monkeypatch.setattr(gpu_plan, 'LAUNCHER_SAMPLES', tmp_path / 'samples')
    scratch = tmp_path / 'scratch'
    gpu_plan.run(gpu_plan.polyis_steps('jnc0', ['p016'], scratch, 0), scratch, 0)
    records = [json.loads(line) for line in (scratch / 'timings.jsonl').read_text().splitlines()]
    assert all(r['record_id'].startswith(f'{gpu_plan.socket.gethostname()}-g4-20261027T061546-') for r in records)
    assert all(r['placement']['launcher_run'] == 'g4-20261027T061546' for r in records)


def test_gpu_plan_run_stops_on_a_failed_step_and_keeps_its_record(tmp_path, monkeypatch):
    calls = fake_run(monkeypatch, returncodes=[0, 1, 0])
    scratch = tmp_path / 'scratch'
    with pytest.raises(AssertionError, match='p012 failed'):
        gpu_plan.run(gpu_plan.polyis_steps('jnc0', ['p012', 'p013'], scratch, 4), scratch, 4)
    # p013 never ran; the failure is recorded.
    assert len(calls) == 2
    records = [json.loads(line) for line in (scratch / 'timings.jsonl').read_text().splitlines()]
    assert [(r['step'], r['returncode']) for r in records] == [('copy cached p011 detections', 0), ('p012', 1)]


def test_gpu_plan_run_refuses_a_busy_gpu(tmp_path, monkeypatch):
    calls = fake_run(monkeypatch, utilization=90)
    scratch = tmp_path / 'scratch'
    steps = gpu_plan.polyis_steps('jnc0', ['p012'], scratch, 4)
    # The CPU copy runs; the GPU step does not start.
    with pytest.raises(AssertionError, match='in use'):
        gpu_plan.run(steps, scratch, 4)
    assert [c['argv'][0] for c in calls] == ['cp']
    # --allow-busy times it anyway.
    gpu_plan.run(steps, scratch, 4, allow_busy=True)
    assert [c['argv'][0] for c in calls] == ['cp', 'cp', 'python']


def test_gpu_plan_run_refuses_writes_outside_scratch_before_running_anything(tmp_path, monkeypatch):
    calls = fake_run(monkeypatch)
    scratch = tmp_path / 'scratch'
    bad = gpu_plan.executable('bad', 'jnc0', ['true'], writes=[Path('/polyis-cache/jnc0')], gpu=False)
    with pytest.raises(AssertionError, match='outside'):
        gpu_plan.run(gpu_plan.polyis_steps('jnc0', ['p016'], scratch, 4) + [bad], scratch, 4)
    assert calls == [] and not scratch.exists()


def test_gpu_plan_covered_validation_videos_needs_every_tracking_output(tmp_path, monkeypatch):
    # Two configurations and the reference on two videos; va1 lost one configuration's tracking output.
    manifest = pd.DataFrame([dict(video=v, variant=variant, variant_id=variant_id) for v in ('va0.mp4', 'va1.mp4')
                             for variant, variant_id in (('naive', 'naive'), ('polytris', 'A'), ('polytris', 'B'))])
    monkeypatch.setattr(gpu_plan, 'build_query_task_manifest', lambda **kwargs: manifest)
    monkeypatch.setattr(gpu_plan.cache, 'execution', lambda dataset, *a: tmp_path.joinpath(dataset, *a))
    for video, relative in (('va0.mp4', '002_naive'), ('va0.mp4', '060_uncompressed_tracks/A'),
                            ('va0.mp4', '060_uncompressed_tracks/B'), ('va1.mp4', '002_naive'),
                            ('va1.mp4', '060_uncompressed_tracks/A')):
        touch(tmp_path / 'd' / video / relative / 'tracking.jsonl', 0)
    assert gpu_plan.covered_validation_videos('d') == (['va0.mp4'], ['va1.mp4'])


def test_gpu_plan_p110_scores_only_covered_videos(tmp_path, monkeypatch, capsys):
    # CalDoT2's case: va00 lacks sweep outputs, so p110 reads a directory of links to the covered videos.
    covered(monkeypatch, ['va10.mp4', 'va11.mp4'], ['va00.mp4'])
    scratch = tmp_path / 'scratch'
    steps = gpu_plan.polyis_steps('caldot2-y05', ['p110'], scratch, 0)
    assert [s['step'] for s in steps] == ['copy validation tracking outputs', 'link covered validation videos', 'p110']
    copy, link, p110 = steps
    # Only covered videos are copied and linked.
    assert copy['argv'][1:3] == ['va10.mp4', 'va11.mp4'] and 'va00.mp4' not in ' '.join(copy['argv'])
    videos_dir = scratch / 'datasets' / 'caldot2-y05' / 'valid'
    assert link['argv'][:4] == ['ln', '-sf', '-t', str(videos_dir)]
    assert link['argv'][4:] == [str(gpu_plan.store.dataset('caldot2-y05', 'valid', v)) for v in ('va10.mp4', 'va11.mp4')]
    # p110 gets its own config, whose DATASETS_DIR is the directory of links.
    assert p110['env']['CONFIG'] == str(scratch / 'configs' / 'caldot2-y05.p110.yaml')
    assert p110['config'] == dict(DATASETS_DIR=str(scratch / 'datasets'))
    assert p110['meta'] == dict(videos=['va10.mp4', 'va11.mp4'], excluded_videos=['va00.mp4'])
    # The plan names what p110 leaves out.
    assert 'excluded (incomplete outputs): va00.mp4' in gpu_plan.describe(p110)
    # Running writes p110's config with the override, and the dataset's other settings as usual.
    fake_run(monkeypatch)
    gpu_plan.run(steps, scratch, 0)
    config = yaml.safe_load((scratch / 'configs' / 'caldot2-y05.p110.yaml').read_text())
    assert config['DATA']['DATASETS_DIR'] == str(scratch / 'datasets') and config['EXEC']['DATASETS'] == ['caldot2-y05']
    assert config['DATA']['CACHE_DIR'] == f'{scratch}/cache/' and videos_dir.is_dir()
    # A dataset without any covered video cannot be scored.
    covered(monkeypatch, [], ['va00.mp4'])
    with pytest.raises(AssertionError, match='no validation video'):
        gpu_plan.polyis_steps('caldot2-y05', ['p110'], scratch, 0)


def test_gpu_plan_scratch_config_overrides_data_entries(tmp_path):
    config = gpu_plan.scratch_config('jnc0', tmp_path, dict(DATASETS_DIR='/x'))
    assert config['DATA']['DATASETS_DIR'] == '/x' and config['DATA']['CACHE_DIR'] == f'{tmp_path}/cache/'


def test_gpu_plan_run_records_outputs_and_summaries(tmp_path, monkeypatch):
    fake_run(monkeypatch)
    scratch = tmp_path / 'scratch'
    # What p012 and p013 leave in the scratch cache (the fake commands write nothing).
    relevant = scratch / 'cache' / 'jnc0' / 'indexing' / 'always_relevant'
    touch(relevant / '60_tr0.mp4.npy', 1000)
    (relevant / '60_optimization_stats.json').write_text(json.dumps(dict(files_removed=5)))
    touch(scratch / 'cache' / 'jnc0' / 'indexing' / 'training' / 'runtime' / 'tilesize_60' / 'tr0.mp4_creating_training_data.jsonl', 2000)
    gpu_plan.run(gpu_plan.polyis_steps('jnc0', ['p012', 'p013'], scratch, 0), scratch, 0)
    records = {r['step']: r for r in map(json.loads, (scratch / 'timings.jsonl').read_text().splitlines())}
    files = [json.loads(line) for line in (scratch / 'files.jsonl').read_text().splitlines()]
    # Each step's outputs are recorded under its record id, with paths relative to the scratch cache.
    by_step = {}
    for f in files:
        by_step.setdefault(f['step'], set()).add(f['path'])
        assert f['record_id'] == records[f['step']]['record_id'] and f['dataset'] == 'jnc0'
    assert by_step == {'p012': {'jnc0/indexing/training/runtime/tilesize_60/tr0.mp4_creating_training_data.jsonl'},
                       'p013': {'jnc0/indexing/always_relevant/60_tr0.mp4.npy',
                                'jnc0/indexing/always_relevant/60_optimization_stats.json'}}
    assert {f['mtime_s'] for f in files if f['path'].endswith('.npy')} == {1000}
    # p013's statistics are kept in its record.
    assert records['p013']['outputs'] == {'jnc0/indexing/always_relevant/60_optimization_stats.json': dict(files_removed=5)}
    assert records['p012']['outputs'] == {}


def test_gpu_plan_launcher_samples_in_the_step_window(tmp_path, monkeypatch):
    # Outside the launcher nothing was sampled.
    monkeypatch.delenv('BOMBE_RUN', raising=False)
    assert gpu_plan.launcher_samples(0, 1)['gpu_samples'] is None
    # The launcher's samples, stamped to the second; the last line is still being written.
    monkeypatch.setenv('BOMBE_RUN', 'g4-x')
    monkeypatch.setattr(gpu_plan, 'LAUNCHER_SAMPLES', tmp_path)
    lines = [dict(t='2026-10-27T06:00:0%d+00:00' % s, util=u, mem_mib=0, pids=[], foreign=f, load_1m=l)
             for s, u, f, l in ((0, 0, 0, 1.0), (2, 90, 0, 2.0), (3, 80, 1, 3.0), (9, 10, 1, 9.0))]
    (tmp_path / 'g4-x.jsonl').write_text('\n'.join(map(json.dumps, lines)) + '\n{"t": "2026-10-27T06:00:04')
    start = datetime(2026, 10, 27, 6, 0, 2, tzinfo=timezone.utc).timestamp()
    got = gpu_plan.launcher_samples(start, start + 1.5)
    # Seconds 1 to 4.5 (one second of slack each side): the samples at 2 and 3 s.
    assert got == dict(gpu_samples=2, gpu_foreign_samples=1, gpu_util_max=90, load_1m_max=3.0)


def test_gpu_plan_interference_subtracts_the_steps_own_cpu(monkeypatch):
    # CPUs 0 and 1 are this process's set; CPU 2 is elsewhere on the host. 100 ticks per second.
    monkeypatch.setattr(gpu_plan.os, 'sched_getaffinity', lambda pid: {0, 1})
    monkeypatch.setattr(gpu_plan.os, 'sysconf', lambda name: 100)
    monkeypatch.delenv('BOMBE_RUN', raising=False)
    before = {0: (0, 0), 1: (0, 0), 2: (0, 0)}
    # Over 10 s: CPU 0 busy all the time, CPU 1 half, CPU 2 a quarter.
    after = {0: (1000, 1000), 1: (500, 1000), 2: (250, 1000)}
    # The step used 5 CPU seconds, half a CPU.
    got = gpu_plan.interference(before, after, 10.0, 5.0, 0, 10)
    assert (got['cpu_set_size'], got['cpu_set_busy_cpus'], got['own_cpus'], got['other_busy_cpus'],
            got['host_busy_cpus']) == (2, 1.5, 0.5, 1.0, 1.75)


def test_gpu_plan_cpu_jiffies_reads_every_cpu():
    # One entry per CPU of the host, busy never above total.
    jiffies = gpu_plan.cpu_jiffies()
    assert len(jiffies) == os.cpu_count() and all(0 <= busy <= total for busy, total in jiffies.values())


# ---------------------------------------------------------------- re-timing records (measurements-gpu --run)


# The launcher's summary of the run every test record belongs to.
LAUNCHER_RUN = dict(run='g4-x', host='bombe', gpu=4, numa_node=1, cpus='20-39,60-79', image='sha256:x', exit=0,
                    samples=10, foreign_samples=0, contaminated=False)


def retiming_dir(tmp_path, records, files, loads, launchers=(LAUNCHER_RUN,)):
    """The four re-timing files measurements-gpu/r9o2_preparation_retiming.sh collects."""
    directory = tmp_path / 'retiming'
    directory.mkdir(parents=True)
    write_lines(directory / 'timings.jsonl', records)
    with gzip.open(directory / 'files.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(line) + '\n' for line in files)
    write_lines(directory / 'detector_load.jsonl', loads)
    write_lines(directory / 'launcher_runs.jsonl', launchers)
    return directory


def record(step, start, seconds, record_id=None, returncode=0, foreign=0, **extra):
    # One timing record as the plan writes it (the fields the measurement reads); p110 runs see 20 CPUs by default.
    scoring = step.startswith('p110')
    return dict(record_id=record_id or f'{step}-{start}', step=step, dataset=DS, start=start, end=start + seconds,
                seconds=seconds, cpu_s=seconds / 2, returncode=returncode, host='bombe', uses_gpu=not scoring,
                argv=extra.pop('argv', ['python', step]), env=extra.pop('env', dict(PYTHON_CPU_COUNT='20') if scoring else {}),
                meta=extra.pop('meta', {}), outputs=extra.pop('outputs', {}),
                interference=dict(own_cpus=1.0, other_busy_cpus=2.5, host_busy_cpus=9.0, gpu_samples=5,
                                  gpu_foreign_samples=foreign, cpu_set_size=40),
                placement=dict(physical_gpu='4', launcher_cpus='20-39,60-79', launcher_run=extra.pop('launcher_run', 'g4-x')))


def output(step, path, birth, mtime, record_id=None):
    # One output's timestamps under its record.
    return dict(record_id=record_id, step=step, dataset=DS, path=f'{DS}/{path}', birth_s=birth, mtime_s=mtime)


def p110_outputs(record_id, tasks, step='p110'):
    # p110's task directories and result files: {task: (start, end, videos)}.
    raw = 'evaluation/070_accuracy/raw/valid'
    return [line for task, (start, end, videos) in tasks.items() for line in (
        [output(step, f'{raw}/{task}', start, start, record_id),
         output(step, f'{raw}/{task}/DATASET.json', None, end, record_id)]
        + [output(step, f'{raw}/{task}/{v}.json', None, end - 1, record_id) for v in videos])]


def world_retiming(tmp_path, p016_workers='1', p012_videos=('tr0.mp4', 'tr1.mp4'), tasks=None, foreign=0, skip=(),
                   p013_tile_sizes=(60,), p011_birth=5.0, p016_profiles=('sortcython_60', 'bytetrackcython_60'),
                   p016_partials=('tr0.mp4', 'tr1.mp4'), loads=None, density=None, density_meta=None):
    """Re-timing records of every step for the world's dataset (spans in the comments of the end-to-end test);
    `density` adds a p110_density run with those tasks."""
    tasks = tasks or {POINT: (100.0, 110.0, ['va0.mp4']), OTHER: (200.0, 220.0, ['va0.mp4']), 'naive': (300.0, 305.0, ['va0.mp4'])}
    meta = dict(videos=['va0.mp4'], excluded_videos=['va1.mp4'])
    # One statistics file per p013 task (tile size).
    stats = {f'{DS}/indexing/always_relevant/{t}_optimization_stats.json': dict(files_removed=5) for t in p013_tile_sizes}
    records = [
        # An older successful p110 run whose outputs must not be mixed in, the chosen one, and a later failure.
        record('p110', 10, 50, 'old', meta=meta), record('p110', 95, 230, meta=meta),
        record('p110', 400, 5, returncode=1, meta=meta),
        record('p011', 1000, 11), record('p012', 1100, 20), record('p013', 1200, 3, outputs=stats),
        record('detector_load', 1300, 20, foreign=foreign),
        record('p016', 1400, 12, argv=['python', 'scripts/p016_tune_track_rate.py']
               + (['--num_workers', p016_workers] if p016_workers else [])),
    ] + ([record('p110_density', 500, 140, meta=density_meta or meta, env=dict(PYTHON_CPU_COUNT='40'))] if density else [])
    records = [r for r in records if r['step'] not in skip]
    files = (p110_outputs('old', {'stray': (0.0, 1.0, ['va0.mp4'])}) + p110_outputs('p110-95', tasks)
             + (p110_outputs('p110_density-500', density, 'p110_density') if density else [])
             + [output('p011', f'indexing/segment/detection/{v}.detections.jsonl', b, m, 'p011-1000')
                for v, b, m in (('tr0.mp4', 0.0, 4.0), ('tr1.mp4', p011_birth, 10.0))]
             + [output('p012', f'indexing/training/runtime/tilesize_60/{v}_creating_training_data.jsonl', b, m, 'p012-1100')
                for v, b, m in zip(p012_videos, (0.0, 10.0, 20.0), (7.0, 18.0, 29.0))]
             + [output('p013', f'indexing/always_relevant/{name}', 0.0, 1.0, 'p013-1200')
                for name in ('60_tr1.mp4.npy', '60_all.npy', '60_optimization_stats.json')]
             + [output('p016', f'indexing/track_rate/{profile}/{name}', 0.0, 1.0, 'p016-1400')
                for profile in p016_profiles
                for name in [*(f'partial/{v}.npy' for v in p016_partials), 'accuracy.npy']])
    loads = loads or [
        dict(dataset=DS, start=1301.0, end=1310.0, device='Quadro RTX 6000', cuda_init_s=2.0, load_s=[3.0, 1.0, 2.0, 1.5]),
        # A detector_load record from outside the chosen run's window is ignored.
        dict(dataset=DS, start=5.0, end=6.0, device='x', cuda_init_s=9.0, load_s=[9.0, 9.0])]
    return retiming_dir(tmp_path, records, files, loads)


def test_load_retiming_requires_the_records(con, tmp_path):
    with pytest.raises(AssertionError, match='missing'):
        prep.load_retiming(con, tmp_path / 'nothing')


def test_load_retiming_takes_the_latest_successful_run(con, tmp_path):
    prep.load_retiming(con, world_retiming(tmp_path))
    runs = {r['step']: r for r in prep.query_dicts(con, 'SELECT step, record_id, attempts FROM retimed_runs')}
    # The later failure is counted but not chosen; the older success is superseded.
    assert runs['p110'] == dict(step='p110', record_id='p110-95', attempts=3)
    # Only the chosen runs' outputs and detector constructions remain.
    assert con.execute("SELECT count(*) FROM retimed_files WHERE record_id = 'old'").fetchone() == (0,)
    assert con.execute('SELECT count(*), any_value(cuda_init_s) FROM detector_loads').fetchone() == (1, 2.0)


def test_load_retiming_rejects_repeated_record_ids(con, tmp_path):
    directory = retiming_dir(tmp_path, [record('p011', 0, 1, 'x'), record('p012', 5, 1, 'x')], [], [])
    with pytest.raises(AssertionError, match='repeated record ids'):
        prep.load_retiming(con, directory)


def test_measure_dataset_with_retiming(con, world, tmp_path):
    prep.load_retiming(con, world_retiming(tmp_path))
    got = prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)
    retiming = got['retiming']
    # The cache's figures are unchanged.
    assert got['totals']['instrumented_s'] == pytest.approx(620.2) and got['totals']['process_s'] == pytest.approx(634.8)
    c = retiming['components']
    seconds = {k: (v['instrumented_s'], v['process_s']) for k, v in c.items()}
    assert seconds == {
        # Task spans 10 s and 20 s over one of two videos: 15 s per pair x 2 configurations x 2 videos.
        'validation_scoring': (60.0, 60.0),
        # p016's wall clock with one worker.
        'mistrack_profiling': (12.0, 12.0),
        # p013's wall clock.
        'training_data_pruning': (3.0, 3.0),
        # p012's spans (7 + 8 s) are a cross-check, so it keeps the cache's figures; p011 keeps its recorded ops at the
        # instrumented boundary and takes its spans (4 + 5 s) as process time.
        'training_data_creation': (6.0, 20.0),
        'reference_detection_train': (2.0, 9.0),
    }
    scoring = c['validation_scoring']
    assert (scoring['spans_s'], scoring['scored_videos'], scoring['valid_videos'], scoring['scale']) == (30.0, 1, 2, 2.0)
    assert scoring['excluded_videos'] == ['va1.mp4'] and scoring['reference_task_s'] == 5.0 and 'scaled' in scoring['basis']
    assert scoring['cache']['process_s'] == 40.0 and scoring['replaces'] == ['instrumented', 'process']
    # Its density is named (16 TrackEval processes on 40 CPUs), and the ratio to the cache's estimate is kept.
    assert (scoring['processes'], scoring['processes_per_cpu'], scoring['measured_over_cache_process']) == (16, 0.4, 1.5)
    assert '16 TrackEval processes on 40 CPUs' in scoring['basis'] and scoring['density_cross_check'] is None
    assert scoring['run']['launcher_exit'] == 0 and scoring['run']['launcher_contaminated'] is False
    assert scoring['run']['attempts'] == 3 and scoring['run']['other_busy_cpus'] == 2.5 and scoring['run']['cpus'] == '20-39,60-79'
    assert c['training_data_pruning']['cache'] is None and c['training_data_pruning']['relevance_bitmaps'] == 1
    assert c['training_data_pruning']['stats'] == {'60_optimization_stats.json': dict(files_removed=5)}
    # Without the cache's statistics there is nothing to compare p013's work with.
    assert c['training_data_pruning']['cache_run']['stats'] is None and c['training_data_pruning']['files_removed_over_cache'] is None
    assert 'finding 10' in c['training_data_pruning']['basis']
    assert c['training_data_creation']['replaces'] == [] and c['training_data_creation']['measured_s'] == 15.0
    assert c['training_data_creation']['cache']['process_s'] == 20.0 and 'cross-check' in c['training_data_creation']['basis']
    # The run's CPU set size is kept, so a saturated set is visible.
    assert c['training_data_creation']['run']['cpu_set_size'] == 40 and c['training_data_creation']['run']['own_cpus'] == 1.0
    assert c['mistrack_profiling']['profiles'] == {'bytetrackcython_60': 2, 'sortcython_60': 2}
    # Detector loading: CUDA start-up 2 s, first construction 3 s, warm median 1.5 s; reported, not added.
    loading = retiming['detector_loading']
    assert (loading['cuda_init_s'], loading['first_load_s'], loading['warm_load_s'], loading['cold_process_s'],
            loading['p011_cold_loads_s']) == (2.0, 3.0, 1.5, 5.0, 10.0)
    assert retiming['missing_steps'] == [] and retiming['gpu_contaminated_steps'] == []
    # Totals: the cache's components with the measured figures in place, plus p013.
    totals = retiming['totals_with_retiming']
    assert totals['instrumented_s'] == pytest.approx(620.2 - 40 + 60 - 20 + 12 + 3)
    assert totals['process_s'] == pytest.approx(634.8 - 40 + 60 - 20 + 12 + 3 - 2.2 + 9)
    # Replaced, cross-checked, and cache-only components are named apart.
    assert totals['measured_components'] == ['mistrack_profiling', 'reference_detection_train', 'training_data_pruning',
                                             'validation_scoring']
    assert totals['cross_checks'] == ['training_data_creation']
    assert totals['cache_components'] == ['classifier_optimization', 'classifier_training', 'reference_outputs_valid',
                                          'training_data_creation', 'validation_sweep']
    # Same 4.5 s saving per video minute as the cache's break-even.
    even = retiming['break_even_with_retiming']
    assert even['instrumented']['break_even_video_minutes'] == pytest.approx(totals['instrumented_s'] / 4.5, abs=0.1)
    assert even['process']['break_even_video_minutes'] == pytest.approx(totals['process_s'] / 4.5, abs=0.1)


def test_measure_dataset_without_retiming_has_none(con, world):
    assert prep.measure_dataset(con, DS, 0.05, world['compilations'])['retiming'] is None


def test_measure_dataset_retiming_reports_missing_and_contaminated_steps(con, world, tmp_path):
    prep.load_retiming(con, world_retiming(tmp_path, foreign=3, skip=('p011',)))
    retiming = prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)['retiming']
    # Without a p011 run its component keeps the cache's figures.
    assert retiming['missing_steps'] == ['p011'] and 'reference_detection_train' not in retiming['components']
    assert 'reference_detection_train' in retiming['totals_with_retiming']['cache_components']
    # A foreign process on the GPU during a step flags it.
    assert retiming['gpu_contaminated_steps'] == ['detector_load']


@pytest.mark.parametrize('kwargs, message', [
    # p016 with 40 workers: its wall clock is not process seconds.
    (dict(p016_workers='40'), 'not 1 worker'),
    # p016 without a worker count runs p016's default (40).
    (dict(p016_workers=None), 'not 1 worker'),
    # p012 timed one of two training videos.
    (dict(p012_videos=('tr0.mp4',)), 'p012 re-run has 1 files for 2'),
    # p110 scored one of two configurations.
    (dict(tasks={POINT: (100.0, 110.0, ['va0.mp4'])}), 'scored 1 of 2 configurations'),
    # A configuration scored on fewer videos than the run covered.
    (dict(tasks={POINT: (100.0, 110.0, ['va0.mp4']), OTHER: (200.0, 220.0, [])}), '1 on fewer'),
    # p013 ran two tasks (tile sizes) in parallel: its wall clock is not one process's.
    (dict(p013_tile_sizes=(30, 60)), '2 statistics files, not one task'),
])
def test_measure_dataset_retiming_checks_the_runs(con, world, tmp_path, kwargs, message):
    prep.load_retiming(con, world_retiming(tmp_path, **kwargs))
    with pytest.raises(AssertionError, match=message):
        prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)


def test_load_retiming_prefers_runs_without_foreign_gpu_use(con, tmp_path):
    # A clean early run and a contaminated later one of the same step: the clean one counts.
    clean, dirty = record('p011', 0, 1, 'clean'), record('p011', 10, 1, 'dirty', foreign=2)
    prep.load_retiming(con, retiming_dir(tmp_path / 'a', [clean, dirty], [], []))
    assert con.execute('SELECT record_id, attempts, foreign_gpu_use FROM retimed_runs').fetchall() == [('clean', 2, False)]
    # With only contaminated runs, the latest is taken (and flagged by the caller).
    prep.load_retiming(con, retiming_dir(tmp_path / 'b', [dirty, record('p011', 20, 1, 'later', foreign=1)], [], []))
    assert con.execute('SELECT record_id, foreign_gpu_use FROM retimed_runs').fetchall() == [('later', True)]
    # An unsampled run (outside the launcher) counts as clean.
    unsampled = record('p011', 30, 1, 'unsampled')
    unsampled['interference']['gpu_foreign_samples'] = None
    prep.load_retiming(con, retiming_dir(tmp_path / 'c', [clean, unsampled], [], []))
    assert con.execute('SELECT record_id FROM retimed_runs').fetchall() == [('unsampled',)]


def test_load_retiming_requires_each_launcher_runs_summary(con, tmp_path):
    # A record of a launcher run without a summary (killed, or not collected) is refused.
    with pytest.raises(AssertionError, match="without a summary.*g4-x"):
        prep.load_retiming(con, retiming_dir(tmp_path / 'a', [record('p011', 0, 1)], [], [], launchers=()))
    # Records made outside the launcher need none.
    local = record('p011', 0, 1, launcher_run=None)
    prep.load_retiming(con, retiming_dir(tmp_path / 'b', [local], [], [], launchers=()))
    assert con.execute('SELECT record_id, launcher_exit FROM retimed_runs').fetchall() == [('p011-0', None)]
    # Two summaries with one run id cannot be told apart.
    with pytest.raises(AssertionError, match='repeated run ids'):
        prep.load_retiming(con, retiming_dir(tmp_path / 'c', [record('p011', 0, 1)], [], [],
                                             launchers=(LAUNCHER_RUN, LAUNCHER_RUN)))


def test_load_retiming_rejects_a_repeated_id_among_unchosen_records(con, tmp_path):
    # An older p011 attempt shares its id with the chosen p012 run, so their outputs would mix.
    directory = retiming_dir(tmp_path, [record('p011', 0, 1, 'x'), record('p011', 5, 1, 'y'), record('p012', 9, 1, 'x')],
                             [], [])
    with pytest.raises(AssertionError, match='repeated record ids'):
        prep.load_retiming(con, directory)


def test_measure_dataset_retiming_density_cross_check(con, world, tmp_path):
    # The deployed-density re-run: spans 20 s and 40 s over one video, so 30 s per pair x 2 x 2 = 120 s.
    density = {POINT: (500.0, 520.0, ['va0.mp4']), OTHER: (600.0, 640.0, ['va0.mp4']), 'naive': (650.0, 655.0, ['va0.mp4'])}
    prep.load_retiming(con, world_retiming(tmp_path, density=density))
    retiming = prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)['retiming']
    scoring = retiming['components']['validation_scoring']
    cross = scoring['density_cross_check']
    # 32 TrackEval processes on 40 CPUs, twice the substituted run's seconds; reported only.
    assert (cross['measured_s'], cross['processes'], cross['processes_per_cpu'], cross['over_substituted']) == \
        (120.0, 32, 0.8, 2.0)
    assert cross['seconds_per_pair'] == 30.0 and cross['run']['record_id'] == 'p110_density-500'
    assert 'not substituted' in cross['basis']
    # The substituted figure and the totals are those without the cross-check.
    assert (scoring['instrumented_s'], scoring['process_s']) == (60.0, 60.0)
    assert retiming['totals_with_retiming']['instrumented_s'] == pytest.approx(620.2 - 40 + 60 - 20 + 12 + 3)
    assert retiming['missing_steps'] == []


def test_measure_dataset_retiming_density_is_optional_and_never_alone(con, world, tmp_path):
    # Without a density re-run nothing is missing; without a p110 re-run it cannot be reported.
    prep.load_retiming(con, world_retiming(tmp_path / 'a'))
    assert 'p110_density' not in prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)['retiming']['missing_steps']
    density = {POINT: (500.0, 520.0, ['va0.mp4']), OTHER: (600.0, 640.0, ['va0.mp4'])}
    prep.load_retiming(con, world_retiming(tmp_path / 'b', density=density, skip=('p110',)))
    with pytest.raises(AssertionError, match='p110_density re-run without a p110 re-run'):
        prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)


def test_measure_dataset_retiming_compares_p013_with_the_cache(con, world, tmp_path):
    # The cache's p013 removed 4 files from two bitmaps; the re-run removed 5 from one.
    cached = tmp_path / 'cache' / DS / 'indexing' / 'always_relevant'
    for name in ('60_tr0.mp4.npy', '60_tr1.mp4.npy', '60_all.npy', '30_tr0.mp4.npy'):
        touch(cached / name, 0)
    (cached / '60_optimization_stats.json').write_text(json.dumps(dict(files_removed=4, never_relevant_tiles=3)))
    prep.load_retiming(con, world_retiming(tmp_path / 'r'))
    pruning = prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)['retiming']['components'][
        'training_data_pruning']
    assert pruning['cache_run']['stats'] == dict(files_removed=4, never_relevant_tiles=3)
    assert (pruning['cache_run']['relevance_bitmaps'], pruning['relevance_bitmaps']) == (2, 1)
    assert pruning['files_removed_over_cache'] == 1.25


@pytest.mark.parametrize('kwargs, message', [
    # A p011 detections file without a creation time has no span.
    (dict(p011_birth=None), r'p011 re-run has 2 files for 2 training videos \(1 without a creation time\)'),
    # p016 profiled another tracker than the configured ones.
    (dict(p016_profiles=('sortcython_60', 'ocsortcython_60')), 'p016 re-run profiles'),
    # p016 profiled one of two training videos.
    (dict(p016_partials=('tr0.mp4',)), 'p016 re-run profiles'),
    # Two detector_load records inside the chosen run's window.
    (dict(loads=[dict(dataset=DS, start=s, end=s + 1, device='d', cuda_init_s=1.0, load_s=[2.0, 1.0]) for s in (1301.0, 1305.0)]),
     'detector_load records'),
    # A single construction has no warm one.
    (dict(loads=[dict(dataset=DS, start=1301.0, end=1302.0, device='d', cuda_init_s=1.0, load_s=[2.0])]), 'detector_load records'),
    # The density re-run scored other videos than the substituted one.
    (dict(density={POINT: (500.0, 520.0, ['va0.mp4']), OTHER: (600.0, 640.0, ['va0.mp4'])},
          density_meta=dict(videos=['va0.mp4', 'va1.mp4'], excluded_videos=[])), 'scored'),
])
def test_measure_dataset_retiming_rejects_inconsistent_runs(con, world, tmp_path, kwargs, message):
    prep.load_retiming(con, world_retiming(tmp_path, **kwargs))
    with pytest.raises(AssertionError, match=message):
        prep.measure_dataset(con, DS, 0.05, world['compilations'], retiming=True)


def test_driver_collect_merges_every_gpus_records(con, tmp_path):
    # A fake track with two GPUs' scratch directories, as the run subcommand leaves them.
    track, tetris = tmp_path / 'track', tmp_path / 'tetris'
    for g, start in (('g3', 0), ('g4', 100)):
        retime = track / g / 'retime'
        launcher = dict(LAUNCHER_RUN, run=f'{g}-x')
        write_lines(retime / 'timings.jsonl', [record('p011', start, 1, f'{g}-p011', launcher_run=f'{g}-x')])
        write_lines(retime / 'detector_load.jsonl', [dict(dataset=DS, start=start, end=start + 1, device='d',
                                                          cuda_init_s=1.0, load_s=[1.0, 1.0])])
        write_lines(retime / 'files.jsonl', [output('p011', 'x.detections.jsonl', 0.0, 1.0, f'{g}-p011')])
        write_lines(track / g / 'samples' / f'{g}-x.json', [launcher])
        (retime / 'logs').mkdir()
        (retime / 'logs' / 'ds_p011.log').write_text(g)
        (track / g / 'run.log').write_text(g)
    env = dict(os.environ, R9O2_TRACK=str(track), R9O2_TETRIS=str(tetris))
    result = subprocess.run(['bash', str(DRIVER_PATH), 'collect'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = tetris / 'measurements-results' / 'r9o2_preparation_retiming'
    # Both GPUs' records, logs prefixed with their GPU, and the launcher summaries, in the files the measurement reads.
    assert sorted(p.name for p in (out / 'logs').iterdir()) == ['g3_ds_p011.log', 'g3_run.log', 'g4_ds_p011.log', 'g4_run.log']
    with gzip.open(out / 'files.jsonl.gz', 'rt') as f:
        assert [json.loads(line)['record_id'] for line in f] == ['g3-p011', 'g4-p011']
    prep.load_retiming(con, out)
    # Both records were read; the later one (GPU 4) is chosen, with its own launcher run's summary.
    assert con.execute('SELECT count(*) FROM retiming_records').fetchone() == (2,)
    assert con.execute("SELECT record_id, placement->>'launcher_run', launcher_exit FROM retimed_runs").fetchall() == \
        [('g4-p011', 'g4-x', 0)]
    assert con.execute('SELECT count(*) FROM launcher_runs').fetchone() == (2,)
    # An unknown subcommand prints the usage and exits 2.
    bad = subprocess.run(['bash', str(DRIVER_PATH), 'nothing'], env=env, capture_output=True, text=True)
    assert bad.returncode == 2 and 'usage' in bad.stderr
