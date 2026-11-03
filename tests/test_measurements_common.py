import ast
import gzip
import inspect
import json
import math
import os
import re
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from evaluation import p121_throughput_compute as p121
from evaluation.p140_otif_transform import unwrap_leap_frames
from measurements import common


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def make_points(con, rows):
    # A points table for fastest_within from (dataset, system, config, fps, accuracy) rows.
    con.execute('CREATE OR REPLACE TABLE pts (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, accuracy DOUBLE)')
    con.executemany('INSERT INTO pts VALUES (?, ?, ?, ?, ?)', rows)


def test_fastest_within_symmetric_bound(con):
    make_points(con, [
        ('d', 'reference', 'naive', 10.0, 0.9),
        # Tetris: the fastest config misses the bound; the next one meets it.
        ('d', 'tetris', 'a', 500.0, 0.80),
        ('d', 'tetris', 'b', 200.0, 0.86),
        # OTIF meets the bound exactly (0.855 = 0.9 * 0.95, inclusive).
        ('d', 'otif', 'param 1', 100.0, 0.855),
        # LEAP never meets it.
        ('d', 'leap', 'param 0', 300.0, 0.5),
    ])
    rows = {r['system']: r for r in common.fastest_within(con, 'pts', 0.05, ['tetris', 'otif', 'leap'])}
    assert rows['tetris']['config'] == 'b' and rows['tetris']['meets_bound']
    assert rows['otif']['config'] == 'param 1'
    assert rows['tetris']['bound'] == pytest.approx(0.855)
    # A system that misses the bound reports its best accuracy and no configuration.
    assert not rows['leap']['meets_bound'] and rows['leap']['config'] is None
    assert rows['leap']['best_accuracy'] == 0.5


def test_fastest_within_ties_go_to_accuracy_then_name(con):
    make_points(con, [
        ('d', 'reference', 'naive', 1.0, 1.0),
        # Equal throughput: the more accurate configuration wins.
        ('d', 'tetris', 'b', 100.0, 0.97),
        ('d', 'tetris', 'a', 100.0, 0.99),
        # Equal throughput and accuracy: the smaller name wins, for determinism.
        ('d', 'otif', 'z', 50.0, 0.98),
        ('d', 'otif', 'y', 50.0, 0.98),
    ])
    rows = {r['system']: r['config'] for r in common.fastest_within(con, 'pts', 0.05, ['tetris', 'otif'])}
    assert rows == {'tetris': 'a', 'otif': 'y'}


def test_fastest_within_keeps_datasets_apart(con):
    # Each dataset is bounded by its own reference.
    make_points(con, [
        ('d1', 'reference', 'naive', 1.0, 0.5), ('d1', 'tetris', 'a', 9.0, 0.5),
        ('d2', 'reference', 'naive', 1.0, 0.9), ('d2', 'tetris', 'a', 9.0, 0.5),
    ])
    rows = {r['dataset']: r['meets_bound'] for r in common.fastest_within(con, 'pts', 0.05, ['tetris'])}
    assert rows == {'d1': True, 'd2': False}


def test_fastest_within_rejects_bad_reference(con):
    # A non-positive reference accuracy (possible with OTIF's metric) cannot set a relative bound.
    make_points(con, [('d', 'reference', 'naive', 1.0, -0.2), ('d', 'tetris', 'a', 9.0, 0.5)])
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', 0.05, ['tetris'])
    # A dataset without a reference row has no bound.
    make_points(con, [('d', 'tetris', 'a', 9.0, 0.5)])
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', 0.05, ['tetris'])
    # Two reference rows make the bound ambiguous.
    make_points(con, [('d', 'reference', 'n1', 1.0, 0.9), ('d', 'reference', 'n2', 1.0, 0.8)])
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', 0.05, ['tetris'])


def test_results_dir_matches_script_name(tmp_path, monkeypatch):
    # Redirect the results root so the test does not touch the repository.
    monkeypatch.setattr(common, 'RESULTS_ROOT', tmp_path)
    path = common.results_dir('/polyis/measurements/r1o2_count_query.py')
    assert path == tmp_path / 'r1o2_count_query'
    assert path.is_dir()


def make_hota_points(con, rows):
    # A points table for operating_points from (dataset, system, config, fps, hota, accuracy) rows.
    con.execute('CREATE OR REPLACE TABLE hpts (dataset VARCHAR, system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE, accuracy DOUBLE)')
    con.executemany('INSERT INTO hpts VALUES (?, ?, ?, ?, ?, ?)', rows)


def test_fastest_within_explicit_bounds_let_the_weakest_system_qualify(con):
    make_points(con, [
        ('d', 'reference', 'naive', 10.0, 0.9),
        ('d', 'tetris', 'a', 500.0, 0.6), ('d', 'tetris', 'b', 100.0, 0.85),
        ('d', 'leap', 'param 0', 300.0, 0.5),
    ])
    # LEAP's own accuracy as the bound: LEAP qualifies exactly, and Tetris's fastest config now qualifies too.
    rows = {r['system']: r for r in common.fastest_within(con, 'pts', None, ['tetris', 'leap'], bounds={'d': 0.5})}
    assert rows['leap']['meets_bound'] and rows['leap']['config'] == 'param 0'
    assert rows['tetris']['config'] == 'a'
    # The bound is reported as a relative loss from the reference: 1 - 0.5 / 0.9.
    assert rows['leap']['loss'] == pytest.approx(1 - 0.5 / 0.9)


def test_fastest_within_accepts_negative_bounds(con):
    # OTIF's counting accuracy can be negative; a negative bound admits every configuration.
    make_points(con, [('d', 'reference', 'naive', 1.0, 0.9), ('d', 'leap', 'p', 5.0, -0.4), ('d', 'tetris', 'a', 9.0, 0.1)])
    rows = {r['system']: r['meets_bound'] for r in common.fastest_within(con, 'pts', None, ['tetris', 'leap'], bounds={'d': -0.4})}
    assert rows == {'tetris': True, 'leap': True}


def test_fastest_within_requires_exactly_one_bound_kind(con):
    make_points(con, [('d', 'reference', 'naive', 1.0, 0.9), ('d', 'tetris', 'a', 9.0, 0.5)])
    # Neither or both is a caller error.
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', None, ['tetris'])
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', 0.05, ['tetris'], bounds={'d': 0.5})
    # Explicit bounds must cover every dataset.
    with pytest.raises(AssertionError):
        common.fastest_within(con, 'pts', None, ['tetris'], bounds={'other': 0.5})


def test_best_accuracy_bounds_per_dataset(con):
    make_points(con, [
        ('d1', 'reference', 'naive', 1.0, 0.9), ('d1', 'leap', 'p0', 5.0, 0.5), ('d1', 'leap', 'p1', 2.0, 0.6),
        ('d2', 'reference', 'naive', 1.0, 0.9), ('d2', 'leap', 'p0', 5.0, -0.3),
    ])
    assert common.best_accuracy_bounds(con, 'pts', 'leap') == {'d1': 0.6, 'd2': -0.3}
    # A system missing from a dataset has no bound there.
    make_points(con, [('d1', 'reference', 'naive', 1.0, 0.9), ('d2', 'reference', 'naive', 1.0, 0.9), ('d1', 'leap', 'p0', 5.0, 0.5)])
    with pytest.raises(AssertionError):
        common.best_accuracy_bounds(con, 'pts', 'leap')


def test_operating_points_report(con):
    make_hota_points(con, [
        ('d', 'reference', 'naive', 10.0, 0.8, 0.9),
        ('d', 'tetris', 'b', 200.0, 0.7, 0.86),
        ('d', 'otif', 'param 1', 100.0, 0.6, 0.855),
        ('d', 'leap', 'param 0', 300.0, 0.3, 0.5),
    ])
    out = common.operating_points(con, 'hpts', ['tetris', 'otif', 'leap'], loss=0.05)['d']
    assert out['systems']['tetris'] == dict(meets_bound=True, config='b', accuracy=0.86, fps=200.0, hota=0.7,
                                            speedup_vs_reference=20.0)
    assert out['tetris_speedup_vs_otif'] == 2.0
    # A system missing the bound gets no ratio.
    assert out['systems']['leap'] == dict(meets_bound=False, best_accuracy=0.5)
    assert 'tetris_speedup_vs_leap' not in out
    assert out['loss'] == pytest.approx(0.05)
    # At LEAP's bound, LEAP qualifies and Tetris's ratio over it appears (200 / 300).
    out = common.operating_points(con, 'hpts', ['tetris', 'otif', 'leap'], bounds={'d': 0.5})['d']
    assert out['tetris_speedup_vs_leap'] == round(200 / 300, 1)


def test_output_path_maps_each_system_to_its_stage():
    # Ground truth, reference, and Tetris tracks and detections come from the execution cache stages.
    assert common.output_path('d', 'ground truth', '003_groundtruth', 'te01.mp4').parts[-3:] == ('te01.mp4', '003_groundtruth', 'tracking.jsonl')
    # Detector variants share their root dataset's ground truth, as in p110 and p141.
    assert common.output_path('caldot1-y05', 'ground truth', '003_groundtruth', 'te01.mp4').parts[-5:-3] == ('caldot1', 'execution')
    assert common.output_path('ams-y05', 'ground truth', '003_groundtruth', 'te01.mp4').parts[-5:-3] == ('ams', 'execution')
    assert common.output_path('jnc0', 'ground truth', '003_groundtruth', 'te01.mp4').parts[-5:-3] == ('jnc0', 'execution')
    assert common.output_path('d', 'reference', 'naive', 'te01.mp4', 'detection').parts[-2:] == ('002_naive', 'detection.jsonl')
    assert common.output_path('d', 'tetris', 'cfg', 'te01.mp4').parts[-3:] == ('060_uncompressed_tracks', 'cfg', 'tracking.jsonl')
    # Tetris detections live under detection-configuration names, which output_path does not build.
    with pytest.raises(AssertionError):
        common.output_path('d', 'tetris', 'cfg', 'te01.mp4', 'detection')
    # Prior systems use zero-padded parameter directories under the SOTA cache.
    path = common.output_path('d', 'otif', 'param 7', 'te01.mp4')
    assert path.parts[-7:] == ('SOTA', 'otif', 'd', 'te01.mp4', 'tracking_results', '007', 'tracking.jsonl')


def test_output_path_rejects_unknown_outputs():
    # Prior systems have no detection output, and unknown systems have no path.
    with pytest.raises(AssertionError):
        common.output_path('d', 'otif', 'param 1', 'te01.mp4', 'detection')
    with pytest.raises(AssertionError):
        common.output_path('d', 'mystery', 'x', 'te01.mp4')


def test_system_configs_derives_prior_throughput_from_runtime(monkeypatch):
    # One Tetris configuration and a reference row over 1,000 test frames.
    tetris = pd.DataFrame({'variant_id': ['cfg'], 'throughput_fps': [50.0], 'HOTA_HOTA': [0.7]})
    naive = pd.Series({'frame_count': 1000, 'throughput_fps': 10.0, 'HOTA_HOTA': 0.8})
    monkeypatch.setattr(common, 'load_test_tradeoff', lambda dataset: (tetris, naive))
    # Each prior system has one configuration that took 4 s, so 250 fps.
    prior = pd.DataFrame({'param_id': [3], 'time': [4.0], 'HOTA_HOTA': [0.5]})
    monkeypatch.setattr(common, 'prior_system_test_rows', lambda dataset, system: prior)
    rows = common.system_configs('d')
    assert rows[0] == dict(system='reference', config='naive', fps=10.0, hota=0.8)
    assert rows[1] == dict(system='tetris', config='cfg', fps=50.0, hota=0.7)
    assert dict(system='otif', config='param 3', fps=250.0, hota=0.5) in rows
    assert dict(system='leap', config='param 3', fps=250.0, hota=0.5) in rows


def test_operating_points_ratios_use_unrounded_throughput(con):
    # 663.94 / 25.66 = 25.87x; dividing the rounded 663.9 / 25.7 would give 25.8x.
    make_hota_points(con, [
        ('d', 'reference', 'naive', 10.0, 0.8, 0.8),
        ('d', 'tetris', 'a', 663.94, 0.8, 0.8),
        ('d', 'otif', 'p', 25.66, 0.8, 0.8),
    ])
    out = common.operating_points(con, 'hpts', ['tetris', 'otif'], loss=0.05)['d']
    assert out['tetris_speedup_vs_otif'] == 25.9
    assert out['systems']['otif']['fps'] == 25.7


def test_prior_system_test_rows_rejects_tables_without_test_rows(monkeypatch):
    # A table with only validation rows would silently drop the system from a test comparison.
    monkeypatch.setattr(common, 'load_sota_tradeoff_data',
                        lambda datasets, system: pd.DataFrame({'videoset': ['valid'], 'time': [1.0]}))
    with pytest.raises(AssertionError):
        common.prior_system_test_rows('d', 'otif')


# Shared helpers moved here from the measurement scripts.

def imported_measurement_modules(source: str) -> set[str]:
    # Every measurements.* module a source file imports, by its dotted name.
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith('measurements.')}
        elif isinstance(node, ast.ImportFrom) and node.module == 'measurements':
            found |= {f'measurements.{a.name}' for a in node.names}
        elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith('measurements.'):
            found.add(node.module)
    return found


def test_imported_measurement_modules_sees_every_import_form():
    # The three spellings that would couple one script to another.
    source = ('import measurements.r9o3_ilp_timings\nfrom measurements import common, r1o5_gap_tolerance as gt\n'
              'from measurements.r9o3_ilp_instance_sizes import decode_grids\nimport os\n')
    assert imported_measurement_modules(source) == {'measurements.r9o3_ilp_timings', 'measurements.common',
                                                    'measurements.r1o5_gap_tolerance', 'measurements.r9o3_ilp_instance_sizes'}


def test_measurement_scripts_import_shared_code_only_from_common():
    # A script importing another script would tie its semantics to that script's; shared code belongs in common.py.
    scripts = sorted(Path(common.__file__).resolve().parent.glob('*.py'))
    assert len(scripts) > 1
    offenders = {p.name: sorted(imported_measurement_modules(p.read_text()) - {'measurements.common'}) for p in scripts}
    assert not {name: mods for name, mods in offenders.items() if mods}


def test_rounded_rounds_floats_only():
    # Floats (numpy floats included) are rounded through dicts and lists; ints, bools, strings, and None are kept.
    value = dict(a=0.123456, b=[np.float64(1.000049), 3, True, None], c=dict(d='x', e=(2.5, 0.00001)))
    assert common.rounded(value) == dict(a=0.1235, b=[1.0, 3, True, None], c=dict(d='x', e=[2.5, 0.0]))
    # One decimal, as the manuscript macros print, on a scalar and a flat dict; None stays None.
    assert common.rounded(4.66, 1) == 4.7 and common.rounded(None, 1) is None
    assert common.rounded({'x': 4.66, 'n': 7, 's': 'A', 'z': None}, 1) == {'x': 4.7, 'n': 7, 's': 'A', 'z': None}


def test_plain_unwraps_numpy_and_missing_values():
    # NumPy scalars become Python values, and NaN, NA, and None become None.
    assert common.plain(np.int64(3)) == 3 and type(common.plain(np.int64(3))) is int
    assert common.plain(np.float64(0.5)) == 0.5 and common.plain('x') == 'x'
    assert common.plain(float('nan')) is None and common.plain(None) is None


def test_assert_files_exist_counts_and_names_missing_paths(tmp_path):
    present = tmp_path / 'a.jsonl'
    present.write_text('')
    # Existing paths pass, str or Path.
    common.assert_files_exist([present, str(present)], 'logs')
    # Missing ones are counted and the first few named.
    with pytest.raises(AssertionError, match=r"2 logs missing, e.g. \['.*b.jsonl', '.*c.jsonl'\]"):
        common.assert_files_exist([present, tmp_path / 'b.jsonl', tmp_path / 'c.jsonl'], 'logs')


def test_iso_utc_and_file_record(tmp_path):
    # Epoch seconds as a UTC ISO string to the second; None stays None.
    assert common.iso_utc(0) == '1970-01-01T00:00:00+00:00' and common.iso_utc(None) is None
    path = tmp_path / 'table.csv'
    path.write_bytes(b'abc')
    os.utime(path, (86400, 86400))
    # Size, modification time, and SHA-256 identify the version read.
    assert common.file_record(path) == dict(
        path=str(path), exists=True, bytes=3, mtime='1970-01-02T00:00:00+00:00',
        sha256='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    # A missing file is recorded, not fatal.
    assert common.file_record(tmp_path / 'nope') == dict(path=str(tmp_path / 'nope'), exists=False)


@pytest.mark.parametrize('result, gpus', [
    (FileNotFoundError(), {}),
    ('nvidia-smi failed', {}),
    ('Quadro RTX 6000\nQuadro RTX 6000\nA100\n\n', {'A100': 1, 'Quadro RTX 6000': 2}),
])
def test_host_info(monkeypatch, result, gpus):
    # nvidia-smi missing, failing, or listing two models.
    def fake_run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        if result == 'nvidia-smi failed':
            raise subprocess.CalledProcessError(9, 'nvidia-smi')
        return subprocess.CompletedProcess(args, 0, stdout=result)
    monkeypatch.setattr(common.subprocess, 'run', fake_run)
    info = common.host_info()
    assert info['gpus'] == gpus and info['hostname'] and info['cpus'] == os.cpu_count()


def score_entries(grids, idx=None):
    # score.jsonl lines for a [frames, rows, cols] uint8 array, as 020 and 022 write them.
    grids = np.asarray(grids, dtype=np.uint8)
    idx = list(range(len(grids))) if idx is None else idx
    return [{'classification_size': list(g.shape), 'classification_hex': g.tobytes().hex(), 'idx': i}
            for g, i in zip(grids, idx)]


def test_decode_grids_roundtrip():
    # Two 2x3 frames with their frame indices.
    grids = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    decoded, idx = common.decode_grids(score_entries(grids, [0, 4]))
    # Values, shape, and indices survive the hex round trip.
    assert decoded.shape == (2, 2, 3) and (decoded == grids).all() and idx == [0, 4]


def test_decode_grids_rejects_bad_files():
    # A file without frames cannot be an instance.
    with pytest.raises(AssertionError, match='no frames'):
        common.decode_grids([])
    # Frames with different grids.
    with pytest.raises(AssertionError, match='inconsistent grid sizes'):
        common.decode_grids(score_entries(np.zeros((1, 2, 2))) + score_entries(np.zeros((1, 3, 2))))
    # A hex string shorter than its declared grid.
    short = score_entries(np.zeros((1, 2, 2)))
    short[0]['classification_hex'] = '0000'
    with pytest.raises(AssertionError, match='hex length'):
        common.decode_grids(short)


def test_relevance_rules_differ_only_at_the_floor_score():
    # At r = 0.5 the floor score is 127: the floor rule keeps it, the current rule (> 127.5) does not.
    scores = np.array([126, 127, 128], dtype=np.uint8)
    assert common.RELEVANCE_RULES['current'](scores, 0.5).tolist() == [False, False, True]
    assert common.RELEVANCE_RULES['floor'](scores, 0.5).tolist() == [False, True, True]


def write_text(path, text):
    # Write a runtime log and return its path as a string.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def runtime_items(con, paths, op_path='$'):
    # common.runtime_items_sql's rows, ordered by file and line (the timestamp as a flag; Python needs pytz for it).
    return query(con, f'SELECT * EXCLUDE (last_modified), last_modified IS NOT NULL AS has_mtime '
                      f'FROM ({common.runtime_items_sql(op_path)}) ORDER BY filename, line, op NULLS LAST', {'paths': paths})


def query(con, sql, params=None):
    # Rows as dicts.
    return common.query_dicts(con, sql, params)


def test_runtime_items_numbers_lines_and_flattens_op_lists(con, tmp_path):
    # Two list lines, a blank line, and a trailing newline; p020's layout.
    path = write_text(tmp_path / 'a.jsonl', '[{"op": "send", "time": 1.5}, {"op": "inference", "time": 2}]\n\n'
                                            '[{"op": "read", "time": 900}]\n')
    rows = runtime_items(con, [path])
    assert [(r['line'], r['op'], r['time'], r['list_type'], r['items']) for r in rows] == [
        (1, 'inference', 2.0, 'ARRAY', 2), (1, 'send', 1.5, 'ARRAY', 2), (3, 'read', 900.0, 'ARRAY', 1)]
    # The file's modification time and raw line ride along.
    assert rows[0]['has_mtime'] and rows[2]['text'] == '[{"op": "read", "time": 900}]'


def test_runtime_items_follows_the_json_path_and_keeps_bad_lines(con, tmp_path):
    # An object op list counts as one item; empty, null, and absent op lists leave one row without an op.
    path = write_text(tmp_path / 'b.jsonl', '\n'.join([
        '{"runtime": [{"op": "solve_ilp", "time": 3}], "time_limit": null}',
        '{"runtime": {"op": "track", "time": 1}}',
        '{"runtime": []}', '{"runtime": null}', '{"time_limit": 0.1}']) + '\n')
    rows = {r['line']: r for r in runtime_items(con, [path], '$.runtime')}
    assert (rows[1]['op'], rows[1]['time'], rows[1]['list_type']) == ('solve_ilp', 3.0, 'ARRAY')
    assert (rows[2]['op'], rows[2]['list_type'], rows[2]['items']) == ('track', 'OBJECT', 1)
    assert [(rows[i]['op'], rows[i]['list_type'], rows[i]['items']) for i in (3, 4, 5)] == [
        (None, 'ARRAY', 0), (None, 'NULL', 0), (None, None, 0)]


def test_runtime_items_types_times(con, tmp_path):
    # Integers and floats are times; strings, booleans, null, and a missing time are not, and say which type they were.
    path = write_text(tmp_path / 'c.jsonl', '[{"op": "a", "time": 1}, {"op": "b", "time": 2.5}, {"op": "c", "time": "x"}, '
                                            '{"op": "d", "time": true}, {"op": "e", "time": null}, {"op": "f"}, {"time": 4}]\n')
    rows = {r['op']: (r['time'], r['time_type']) for r in runtime_items(con, [path])}
    assert rows == {'a': (1.0, 'UBIGINT'), 'b': (2.5, 'DOUBLE'), 'c': (None, 'VARCHAR'), 'd': (None, 'BOOLEAN'),
                    'e': (None, 'NULL'), 'f': (None, None), None: (4.0, 'UBIGINT')}


def test_runtime_items_sql_rejects_a_quote_in_the_json_path():
    # The path is embedded as a SQL string literal, so a quote would end it early.
    with pytest.raises(AssertionError, match='bad JSON path'):
        common.runtime_items_sql("$.runtime' OR true --")


def test_runtime_items_empty_file_and_malformed_json(con, tmp_path):
    # An empty file has no lines; a line that is not JSON stops the read.
    assert runtime_items(con, [write_text(tmp_path / 'empty.jsonl', '')]) == []
    with pytest.raises(duckdb.InvalidInputException, match='Malformed JSON'):
        runtime_items(con, [write_text(tmp_path / 'bad.jsonl', 'not json\n')])


def test_default_time_limit_from_p022_is_one_tenth_second():
    # The real pipeline falls back to 0.1 s when --time-limit is not given.
    assert common.default_time_limit() == 0.1


def test_default_time_limit_parses_spacing_and_rejects_ambiguity(tmp_path):
    # Spacing around the keyword is tolerated.
    one = tmp_path / 'one.py'
    one.write_text('solve_ilp(x, time_limit_seconds = time_limit or 0.25)\n')
    assert common.default_time_limit(one) == 0.25
    # No call site, or two, cannot define the default.
    for text in ('solve_ilp(x)\n', 'a(time_limit_seconds=time_limit or 0.1)\nb(time_limit_seconds=time_limit or 0.5)\n'):
        path = tmp_path / 'bad.py'
        path.write_text(text)
        with pytest.raises(AssertionError, match='expected one'):
            common.default_time_limit(path)


PRUNE_PARAM = 'ShuffleNet05_60_4_060_r050_sortcython'


def prune_run(root, video='va00.mp4', ops=None, time_limit=None, text=None, param=PRUNE_PARAM):
    # One p022 runtime.jsonl where p022 writes it, or the given raw text.
    path = root / 'd' / 'execution' / video / common.PRUNE_DIR / param / 'score' / 'runtime.jsonl'
    ops = ops if ops is not None else [{'op': 'group_tiles', 'time': 1.0}, {'op': 'build_ilp', 'time': 2.0},
                                       {'op': 'solve_ilp', 'time': 3.0}]
    return write_text(path, text if text is not None else json.dumps({'runtime': ops, 'time_limit': time_limit}) + '\n')


def test_load_prune_runs_parses_paths_times_and_budgets(con, tmp_path):
    files = [prune_run(tmp_path), prune_run(tmp_path, 'te01.mp4', time_limit=0.5), prune_run(tmp_path, 'te02.mp4', time_limit=0)]
    common.load_prune_runs(con, files, 0.1)
    rows = query(con, 'SELECT video, split, sample_rate, tilesize, time_limit, budget_s, build_ms, solve_ms, '
                      'build_plus_solve_ms, stage_ms FROM runs ORDER BY video')
    # The budget is p022's `time_limit or default`: null and 0 use the default.
    assert [(r['video'], r['split'], r['time_limit'], r['budget_s']) for r in rows] == [
        ('te01.mp4', 'test', 0.5, 0.5), ('te02.mp4', 'test', 0.0, 0.1), ('va00.mp4', 'valid', None, 0.1)]
    assert (rows[0]['sample_rate'], rows[0]['tilesize'], rows[0]['build_ms'], rows[0]['solve_ms'],
            rows[0]['build_plus_solve_ms'], rows[0]['stage_ms']) == (4, 60, 2.0, 3.0, 5.0, 6.0)


@pytest.mark.parametrize('text, message', [
    # Two records in one file, and none.
    ('{"runtime": [], "time_limit": null}\n{"runtime": [], "time_limit": null}\n', 'exactly one record'),
    ('\n', 'exactly one record'),
    # No time_limit key, or a non-numeric one, so the budget is unknown.
    ('{"runtime": [{"op": "build_ilp", "time": 1}, {"op": "solve_ilp", "time": 1}]}', 'numeric or null time_limit'),
    ('{"runtime": [{"op": "build_ilp", "time": 1}, {"op": "solve_ilp", "time": 1}], "time_limit": "x"}',
     'numeric or null time_limit'),
    # A non-numeric, a negative, or a missing time; a missing op list; a repeated build.
    ('{"runtime": [{"op": "build_ilp", "time": "x"}, {"op": "solve_ilp", "time": 1}], "time_limit": null}', 'exactly one valid'),
    ('{"runtime": [{"op": "build_ilp", "time": -1}, {"op": "solve_ilp", "time": 1}], "time_limit": null}', 'exactly one valid'),
    ('{"runtime": [{"op": "build_ilp"}, {"op": "solve_ilp", "time": 1}], "time_limit": null}', 'exactly one valid'),
    ('{"time_limit": null}', 'exactly one valid'),
    ('{"runtime": {"op": "build_ilp", "time": 1}, "time_limit": null}', 'exactly one valid'),
    ('{"runtime": [{"op": "build_ilp", "time": 1}, {"op": "build_ilp", "time": 1}, {"op": "solve_ilp", "time": 1}], '
     '"time_limit": null}', 'exactly one valid'),
])
def test_load_prune_runs_rejects_malformed_runs(con, tmp_path, text, message):
    with pytest.raises(AssertionError, match=message):
        common.load_prune_runs(con, [prune_run(tmp_path, text=text)], 0.1)


def test_load_prune_runs_rejects_bad_paths_and_no_files(con, tmp_path):
    # A path outside a pruning stage directory, a training video, and an unparseable parameter string.
    other = write_text(tmp_path / 'd' / 'execution' / 'va00.mp4' / '021_x' / PRUNE_PARAM / 'score' / 'runtime.jsonl',
                       '{"runtime": [{"op": "build_ilp", "time": 1}, {"op": "solve_ilp", "time": 1}], "time_limit": null}')
    with pytest.raises(AssertionError, match='unrecognized runtime paths'):
        common.load_prune_runs(con, [other], 0.1)
    with pytest.raises(AssertionError, match='unknown split'):
        common.load_prune_runs(con, [prune_run(tmp_path, 'tr00.mp4')], 0.1)
    with pytest.raises(AssertionError, match='unparseable'):
        common.load_prune_runs(con, [prune_run(tmp_path, param='ShuffleNet05_60_sortcython')], 0.1)
    with pytest.raises(AssertionError, match='no runtime files'):
        common.load_prune_runs(con, [], 0.1)


def tradeoff_pair(rows, reference_fps=10.0, reference_hota=0.8, **reference):
    # A load_test_tradeoff stand-in from (variant_id, fps, HOTA, time, threshold) rows.
    tetris = pd.DataFrame([dict(variant_id=c, throughput_fps=f, HOTA_HOTA=h, time=t,
                                tracking_accuracy_threshold=math.nan if a is None else a) for c, f, h, t, a in rows],
                          columns=['variant_id', 'throughput_fps', 'HOTA_HOTA', 'time', 'tracking_accuracy_threshold'])
    return tetris, pd.Series(dict(throughput_fps=reference_fps, HOTA_HOTA=reference_hota, time=100.0, **reference))


def test_load_tradeoff_points_rows_and_extra_columns(con):
    pair = tradeoff_pair([('a', 500.0, 0.7, 2.0, None), ('b', 300.0, 0.77, 3.0, 0.4)], frame_count=np.int64(1000))
    pair[0]['frame_count'] = [1000.0, 1000.0]
    common.load_tradeoff_points(con, 'pts', {'d': pair}, {'time': 'DOUBLE', 'frame_count': 'BIGINT',
                                                          'tracking_accuracy_threshold': 'DOUBLE'})
    rows = con.execute('SELECT * FROM pts ORDER BY system, config').fetchall()
    # The reference row takes its own values (NULL where it has none); NaN thresholds arrive as NULL.
    assert rows == [('d', 'reference', 'naive', 10.0, 0.8, 100.0, 1000, None),
                    ('d', 'tetris', 'a', 500.0, 0.7, 2.0, 1000, None), ('d', 'tetris', 'b', 300.0, 0.77, 3.0, 1000, 0.4)]
    # No tradeoff tables is an error rather than an empty table.
    with pytest.raises(AssertionError, match='no tradeoff tables'):
        common.load_tradeoff_points(con, 'pts', {})


def test_load_pruning_points_adds_the_pruning_subset(con):
    common.load_pruning_points(con, {'d': tradeoff_pair([('a', 500.0, 0.7, 2.0, None), ('b', 300.0, 0.77, 3.0, 0.4)],
                                                        tracking_accuracy_threshold=0.9)})
    rows = con.execute('SELECT system, config, time_s, prunes FROM points ORDER BY system, config').fetchall()
    # The reference never prunes, even when its row carries a threshold; only 'b' prunes, so it appears twice.
    assert rows == [('reference', 'naive', 100.0, False), ('tetris', 'a', 2.0, False), ('tetris', 'b', 3.0, True),
                    ('tetris_pruned', 'b', 3.0, True)]


def test_fastest_tetris_picks_the_fastest_within_the_bound(con):
    # Bound 0.76 at 5%: 'a' misses it, 'b' meets it; the bound is inclusive.
    rows = [('a', 500.0, 0.70, 2.0, None), ('b', 300.0, 0.77, 3.0, None), ('c', 100.0, 0.79, 9.0, None)]
    common.load_tradeoff_points(con, 'pts', {'d': tradeoff_pair(rows)})
    chosen = common.fastest_tetris(con, 'pts', 0.05)
    assert list(chosen) == ['d'] and chosen['d']['config'] == 'b' and chosen['d']['bound'] == pytest.approx(0.76)
    common.load_tradeoff_points(con, 'pts', {'d': tradeoff_pair([('a', 500.0, 0.76, 2.0, None), *rows[1:]])})
    assert common.fastest_tetris(con, 'pts', 0.05)['d']['config'] == 'a'
    # No configuration within 1% of the reference, or a dataset without any Tetris row, is an error naming it.
    with pytest.raises(AssertionError, match=r"no Tetris configuration within 1% HOTA loss on \['d'\]"):
        common.fastest_tetris(con, 'pts', 0.01)
    common.load_tradeoff_points(con, 'pts', {'d': tradeoff_pair(rows), 'e': tradeoff_pair([])})
    with pytest.raises(AssertionError, match=r"on \['e'\]"):
        common.fastest_tetris(con, 'pts', 0.05)


def test_pruning_operating_points_names_both_selections(con):
    # 'a' (no pruning) is fastest within 5%; 'b' is the fastest pruning one; 'c' prunes but misses the bound.
    common.load_pruning_points(con, {'d': tradeoff_pair([('a', 500.0, 0.79, 2.0, None), ('b', 300.0, 0.77, 3.0, 0.4),
                                                        ('c', 900.0, 0.5, 1.0, 0.8)])})
    chosen = {r['point']: r for r in common.pruning_operating_points(con, 0.05)}
    assert (chosen['operating_point']['config'], chosen['fastest_pruning_point']['config']) == ('a', 'b')
    assert con.execute('SELECT * FROM chosen_points ORDER BY point').fetchall() == [
        ('d', 'fastest_pruning_point', 'b'), ('d', 'operating_point', 'a')]
    # Nothing meets 1%: both rows report their best accuracy and chosen_points is empty.
    chosen = common.pruning_operating_points(con, 0.01)
    assert not any(r['meets_bound'] for r in chosen) and con.execute('SELECT count(*) FROM chosen_points').fetchone() == (0,)


def test_runtime_items_unnumbered_reader_matches_the_numbered_one(con, tmp_path):
    # The parallel reader yields the same items, without line numbers or modification times.
    path = write_text(tmp_path / 'd.jsonl', '[{"op": "send", "time": 1.5}, {"op": "x", "time": true}]\n\n'
                                            '{"op": "read", "time": 900}\n[]\nnull\n')
    keys = ('op', 'time', 'time_type', 'list_type', 'items')
    numbered = [tuple(r[k] for k in keys) for r in runtime_items(con, [path])]
    unnumbered = query(con, f'SELECT * EXCLUDE (last_modified), last_modified IS NULL AS no_mtime FROM '
                            f'({common.runtime_items_sql("$", numbered=False)}) ORDER BY op NULLS LAST, list_type',
                       {'paths': [path]})
    assert sorted(numbered, key=str) == sorted((tuple(r[k] for k in keys) for r in unnumbered), key=str)
    assert all(r['line'] is None and r['no_mtime'] for r in unnumbered)
    with pytest.raises(duckdb.InvalidInputException, match='Malformed JSON'):
        query(con, f'SELECT * FROM ({common.runtime_items_sql("$", numbered=False)})',
              {'paths': [write_text(tmp_path / 'bad.jsonl', 'not json\n')]})


# ---------------------------------------------------------------- p121's runtime-log parser and per-op table


def test_p121_ignored_ops_match_p121_source_and_parser(tmp_path):
    # p121 keeps its ignored ops in a local list inside parse_runtime_file; read the literal so a change there fails here.
    (literal,) = re.findall(r'ignored_ops = (\[[^\]]*\])', inspect.getsource(p121.parse_runtime_file))
    assert tuple(ast.literal_eval(literal)) == common.P121_IGNORED_OPS
    # p121's parser drops exactly those ops (not a name that only starts like one).
    path = write_text(tmp_path / 'runtime.jsonl',
                      json.dumps([{'op': op, 'time': 1.0} for op in [*common.P121_IGNORED_OPS, 'kept', 'read_framex']]) + '\n')
    assert set(p121.parse_runtime_file(path, '040_exec_detect', p121.QUERY_DATA_ACCESSORS['040_exec_detect'])['op']) == \
        {'kept', 'read_framex'}


# How each JSON path wraps a list of op items into one line of that stage's layout.
P121_LAYOUTS = {'$[#-1]': lambda items: [0, [[1, 2, 3, 4, 0.9]], items], '$': lambda items: items,
                '$.runtime': lambda items: {'step': 's', 'runtime': items}}


def write_runtime(path: Path, stage: str, lines: list, compress: bool = False) -> str:
    # One JSONL runtime log in the stage's layout, a blank line after the first record, optionally gzip-compressed.
    wrap = P121_LAYOUTS[common.OP_LIST_PATHS[stage]]
    text = json.dumps(wrap(lines[0])) + '\n\n' + ''.join(json.dumps(wrap(x)) + '\n' for x in lines[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with gzip.open(path, 'wt') as f:
            f.write(text)
    else:
        path.write_text(text)
    return str(path)


@pytest.mark.parametrize('stage', sorted(common.OP_LIST_PATHS))
def test_load_p121_runtime_ops_matches_p121_parser(con, tmp_path, stage):
    # Every op p121 drops for this stage, among ops it counts.
    dropped = sorted({*common.P121_IGNORED_OPS, *p121.excluded_runtime_ops(stage)})
    lines = [
        # Integer and float times, a non-numeric "time" (p040's image file name), and an item without a time.
        [{'op': 'work', 'time': 1500}, {'op': 'work', 'time': 2.25}, {'op': 'image_file', 'time': 'x.jpg'}, {'op': 'note'}],
        [{'op': op, 'time': 1000 + i} for i, op in enumerate(dropped)] + [{'op': 'other', 'time': 7}],
        # A single op object where a list is expected; p121 treats it as a one-item list.
        {'op': 'single', 'time': 5},
    ]
    path = write_runtime(tmp_path / f'{stage}.jsonl', stage, lines)
    accessor = {**p121.INDEX_DATA_ACCESSORS, **p121.QUERY_DATA_ACCESSORS}[stage]
    parsed = p121.aggregate_per_op(p121.filter_excluded_runtime_ops(p121.parse_runtime_file(path, stage, accessor), stage))
    seconds, items = ({r.op: r.time for r in parsed.itertuples()}, {r.op: r.count for r in parsed.itertuples()})
    # The same file gzip-compressed (as the GPU jobs harvest logs) reads the same.
    gz = write_runtime(tmp_path / 'gz' / f'{stage}.jsonl.gz', stage, lines, compress=True)
    for f in (path, gz):
        common.load_p121_runtime_ops(con, 'ops', [(stage, f)])
        rows = query(con, 'SELECT * FROM ops')
        # The same counted ops with the same seconds and items, every row of this file and stage.
        assert {r['op']: r['seconds'] for r in rows if r['counted']} == pytest.approx(seconds, abs=1e-12)
        assert {r['op']: r['records'] for r in rows if r['counted']} == items
        assert all(r['stage'] == stage and r['runtime_file'] == f for r in rows)
        # Dropped ops are kept but not counted; ignored marks p121's global list only; untimed ops are gone.
        assert {r['op'] for r in rows if not r['counted']} == set(dropped)
        assert {r['op'] for r in rows if r['ignored']} == set(common.P121_IGNORED_OPS)
        assert not {'image_file', 'note'} & {r['op'] for r in rows}


def test_load_p121_runtime_ops_flags_and_shared_paths(con, tmp_path):
    det = write_runtime(tmp_path / 'det.jsonl', '001_preprocess_groundtruth_detection',
                        [[{'op': 'read', 'time': 2.0}, {'op': 'detect', 'time': 40.0}]])
    cls = write_runtime(tmp_path / 'cls.jsonl', '020_exec_classify', [[{'op': 'read', 'time': 50.0}]])
    # A file named twice (two configurations sharing it) is read once; flags mark only their own stage's ops.
    common.load_p121_runtime_ops(con, 'ops', [('001_preprocess_groundtruth_detection', det)] * 2 +
                                 [('020_exec_classify', cls)],
                                 {'decode': {'001_preprocess_groundtruth_detection': ('read',)}, 'canvas_read': {}})
    rows = query(con, 'SELECT stage, op, seconds, records, counted, decode, canvas_read FROM ops ORDER BY stage, op')
    assert rows == [dict(stage='001_preprocess_groundtruth_detection', op='detect', seconds=0.04, records=1, counted=True,
                         decode=False, canvas_read=False),
                    dict(stage='001_preprocess_groundtruth_detection', op='read', seconds=0.002, records=1, counted=True,
                         decode=True, canvas_read=False),
                    dict(stage='020_exec_classify', op='read', seconds=0.05, records=1, counted=False, decode=False,
                         canvas_read=False)]
    # No files give an empty table.
    common.load_p121_runtime_ops(con, 'empty', [])
    assert con.execute('SELECT count(*) FROM empty').fetchone() == (0,)


@pytest.mark.parametrize('stage, line, message', [
    ('060_exec_track', {'frame_idx': 0}, 'runtime lines without an op list'),
    ('060_exec_track', {'frame_idx': 0, 'runtime': 3}, 'runtime lines without an op list'),
    ('040_exec_detect', [{'time': 1.0}], 'timed items without an op'),
    ('040_exec_detect', [{'op': 'detect', 'time': True}], 'boolean op times'),
    ('040_exec_detect', [{'op': 'detect', 'time': -1.0}], 'negative op times'),
    # p121 checks the sign before it drops a stage's excluded ops.
    ('020_exec_classify', [{'op': 'read', 'time': -1.0}], 'negative op times'),
])
def test_load_p121_runtime_ops_rejects_what_p121_rejects(con, tmp_path, stage, line, message):
    path = write_text(tmp_path / 'r.jsonl', json.dumps(line) + '\n')
    with pytest.raises(AssertionError, match=f'^{message}: '):
        common.load_p121_runtime_ops(con, 'ops', [(stage, path)])


def test_load_p121_runtime_ops_allows_negative_ignored_ops_and_checks_its_arguments(con, tmp_path):
    # p121 skips its ignored ops before the sign check.
    path = write_text(tmp_path / 'r.jsonl', json.dumps({'runtime': [{'op': 'save_canvas', 'time': -1.0}]}) + '\n')
    common.load_p121_runtime_ops(con, 'ops', [('030_exec_compress', path)])
    assert query(con, 'SELECT op, counted, ignored FROM ops') == [dict(op='save_canvas', counted=False, ignored=True)]
    # JSON array files (which p121 also parses) and flag names that are not plain identifiers are refused.
    with pytest.raises(AssertionError, match='must be JSONL'):
        common.load_p121_runtime_ops(con, 'ops', [('060_exec_track', str(tmp_path / 'runtime.json'))])
    with pytest.raises(AssertionError, match='bad flag names'):
        common.load_p121_runtime_ops(con, 'ops', [], {'decode; DROP TABLE ops': {}})


def test_load_p121_per_op_reads_typed_rows_and_needs_every_file(con, tmp_path):
    # p121's per-op tables with every column it writes, for two datasets.
    header = ['op', 'time', 'count', 'stage', 'dataset', 'videoset', 'video', 'variant', 'variant_id', 'classifier',
              'tilesize', 'tilepadding', 'sample_rate', 'tracking_accuracy_threshold', 'canvas_scale', 'tracker']
    paths = {}
    for d, rows in (('a', [['detect', 1.5, 3, '040_exec_detect', 'a', 'test', 'te00.mp4', 'polytris', 'X', 'Net', 60, 'bl',
                            4, '', 1.0, 'sort']]),
                    ('b', [['read', 2, 10, '001_preprocess_groundtruth_detection', 'b', 'valid', 'va00.mp4', 'naive',
                            'naive', '', '', '', '', '', '', '']])):
        paths[d] = tmp_path / d / 'query_execution_per_op.csv'
        paths[d].parent.mkdir()
        pd.DataFrame(rows, columns=header).to_csv(paths[d], index=False)
    common.load_p121_per_op(con, 'per_op', paths)
    assert query(con, 'SELECT * FROM per_op ORDER BY dataset') == [
        dict(dataset='a', videoset='test', variant_id='X', video='te00.mp4', stage='040_exec_detect', op='detect',
             seconds=1.5, records=3),
        dict(dataset='b', videoset='valid', variant_id='naive', video='va00.mp4', stage='001_preprocess_groundtruth_detection',
             op='read', seconds=2.0, records=10)]
    # Typed as p121 writes them, even where a value looks like another type.
    assert dict(con.execute('SELECT column_name, column_type FROM (DESCRIBE per_op)').fetchall()) == dict(
        dataset='VARCHAR', videoset='VARCHAR', variant_id='VARCHAR', video='VARCHAR', stage='VARCHAR', op='VARCHAR',
        seconds='DOUBLE', records='BIGINT')
    with pytest.raises(AssertionError, match='1 per-op throughput tables .* missing'):
        common.load_p121_per_op(con, 'per_op', {'a': paths['a'], 'c': tmp_path / 'c.csv'})


# ---------------------------------------------------------------- p140's unwrap.json and LEAP's raw exports


def write_unwrap(path, clip_frames, export_frames, objects=0, wrapped=(), ambiguous=(), wrapped_boxes=0, past=0, drop=()):
    # One p140 unwrap.json, without the fields named in drop.
    record = dict(export_frames=export_frames, clip_frames=clip_frames, objects=objects, wrapped_objects=list(wrapped),
                  ambiguous_full_span_objects=list(ambiguous), wrapped_boxes_dropped=wrapped_boxes,
                  boxes_past_clip_dropped=past)
    return write_text(path, json.dumps({k: v for k, v in record.items() if k not in drop}))


def test_load_unwrap_reads_every_field_and_names_the_clip_length_n(con, tmp_path):
    # Two clips, one of them with every correction.
    common.load_unwrap(con, [('te01.mp4', write_unwrap(tmp_path / 'b.json', 7, 9, objects=4, wrapped=[3],
                                                       ambiguous=[5, 4], wrapped_boxes=2, past=6)),
                             ('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 0))])
    rows = query(con, 'SELECT * FROM unwrap ORDER BY video')
    # unwrap.json's fields, its clip_frames under the name n, and lists as written.
    assert rows == [dict(video='te00.mp4', export_frames=0, n=5, objects=0, wrapped_objects=[],
                         ambiguous_full_span_objects=[], wrapped_boxes_dropped=0, boxes_past_clip_dropped=0),
                    dict(video='te01.mp4', export_frames=9, n=7, objects=4, wrapped_objects=[3],
                         ambiguous_full_span_objects=[5, 4], wrapped_boxes_dropped=2, boxes_past_clip_dropped=6)]


@pytest.mark.parametrize('text', [
    # An empty file has no record; a list or two lines hold two.
    '',
    json.dumps([dict(export_frames=5, clip_frames=5, objects=0, wrapped_objects=[], ambiguous_full_span_objects=[],
                     wrapped_boxes_dropped=0, boxes_past_clip_dropped=0)] * 2),
    '\n'.join([json.dumps(dict(export_frames=5, clip_frames=5, objects=0, wrapped_objects=[],
                                ambiguous_full_span_objects=[], wrapped_boxes_dropped=0, boxes_past_clip_dropped=0))] * 2),
])
def test_load_unwrap_rejects_files_without_exactly_one_record(con, tmp_path, text):
    with pytest.raises(AssertionError, match='records missing, repeated, or incomplete'):
        common.load_unwrap(con, [('te00.mp4', write_text(tmp_path / 'u.json', text))])


@pytest.mark.parametrize('field', ['export_frames', 'clip_frames', 'objects', 'wrapped_objects',
                                   'ambiguous_full_span_objects', 'wrapped_boxes_dropped', 'boxes_past_clip_dropped'])
def test_load_unwrap_rejects_a_missing_field(con, tmp_path, field):
    # Every field is needed by one of the scripts.
    with pytest.raises(AssertionError, match='incomplete'):
        common.load_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'u.json', 5, 5, drop=[field]))])


@pytest.mark.parametrize('clip_frames, export_frames', [(0, 0), (-3, 0), (5, -1)])
def test_load_unwrap_rejects_impossible_lengths(con, tmp_path, clip_frames, export_frames):
    # A clip has frames, and an export cannot be shorter than empty.
    with pytest.raises(AssertionError, match='incomplete'):
        common.load_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'u.json', clip_frames, export_frames))])


def test_load_unwrap_rejects_missing_files_repeated_videos_and_no_files(con, tmp_path):
    # A missing file means a LEAP cache older than the fixed p140.
    with pytest.raises(AssertionError, match='unwrap.json files .rerun the fixed p140. missing'):
        common.load_unwrap(con, [('te00.mp4', str(tmp_path / 'missing.json'))])
    # One clip with two files, and no file at all.
    path = write_unwrap(tmp_path / 'u.json', 5, 5)
    with pytest.raises(AssertionError, match='more than once'):
        common.load_unwrap(con, [('te00.mp4', path), ('te00.mp4', write_unwrap(tmp_path / 'v.json', 5, 5))])
    with pytest.raises(AssertionError, match='no LEAP unwrap.json files'):
        common.load_unwrap(con, [])


def test_load_unwrap_rejects_malformed_json(con, tmp_path):
    # DuckDB's JSON reader fails on a file that is not JSON.
    with pytest.raises(duckdb.Error):
        common.load_unwrap(con, [('te00.mp4', write_text(tmp_path / 'u.json', 'not json'))])


def test_unwrap_totals_and_empty_exports(con, tmp_path):
    # te00's export is empty; te01 has one wrapped and two ambiguous objects; te02 one ambiguous object.
    common.load_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 0)),
                             ('te01.mp4', write_unwrap(tmp_path / 'b.json', 7, 9, objects=4, wrapped=[3],
                                                       ambiguous=[4, 5], wrapped_boxes=2, past=6)),
                             ('te02.mp4', write_unwrap(tmp_path / 'c.json', 6, 6, objects=1, ambiguous=[1]))])
    assert common.unwrap_totals(con) == dict(clips=3, empty_exports=1, objects=5, wrapped_objects=1,
                                             ambiguous_full_span_objects=3, clips_with_ambiguous=2,
                                             wrapped_boxes_dropped=2, boxes_past_clip_dropped=6,
                                             empty_export_videos=['te00.mp4'])


def test_unwrap_totals_without_empty_exports_lists_none(con, tmp_path):
    # Regression: no empty export gives an empty list, not null.
    common.load_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 5, objects=1))])
    assert common.unwrap_totals(con)['empty_export_videos'] == []
    assert common.unwrap_totals(con)['empty_exports'] == 0


def leap_export(intervals, length=None):
    # What export-leap-outputs.py writes: each object's box at list index start + idx (a negative index wraps), a
    # list as long as the largest end, and one box per object and frame.
    length = max(e for _, e in intervals.values()) if length is None else length
    frames = [[] for _ in range(length)]
    for track_id, (s, e) in intervals.items():
        for f in range(s, e):
            if all(d['track_id'] != track_id for d in frames[f]):
                frames[f].append(dict(track_id=track_id, left=0.0, top=0.0, right=1.0, bottom=1.0, score=1.0))
    return frames


def test_load_leap_exports_handles_null_files_frames_and_empty_lists(con, tmp_path):
    # A null file, an empty list, and an export with a null frame.
    files = [('te00.mp4', write_text(tmp_path / 'a.json', 'null')), ('te01.mp4', write_text(tmp_path / 'b.json', '[]')),
             ('te02.mp4', write_text(tmp_path / 'c.json', json.dumps([None, [{'track_id': 4}], []])))]
    common.load_leap_exports(con, files)
    # Lengths 0, 0, 3, and one box on frame 1.
    assert con.execute('SELECT video, L FROM export_len ORDER BY video').fetchall() == \
        [('te00.mp4', 0), ('te01.mp4', 0), ('te02.mp4', 3)]
    assert con.execute('SELECT video, frame, track_id FROM export_obs').fetchall() == [('te02.mp4', 1, 4)]


def test_load_leap_exports_keeps_clips_apart(con, tmp_path):
    # The same track id in two clips' exports stays with its clip.
    common.load_leap_exports(con, [('te00.mp4', write_text(tmp_path / 'a.json', json.dumps(leap_export({1: (0, 2)})))),
                                   ('te01.mp4', write_text(tmp_path / 'b.json', json.dumps(leap_export({1: (1, 3)}))))])
    assert con.execute('SELECT video, frame, track_id FROM export_obs ORDER BY ALL').fetchall() == \
        [('te00.mp4', 0, 1), ('te00.mp4', 1, 1), ('te01.mp4', 1, 1), ('te01.mp4', 2, 1)]
    assert con.execute('SELECT video, L FROM export_len ORDER BY video').fetchall() == [('te00.mp4', 2), ('te01.mp4', 3)]


@pytest.mark.parametrize('frames', [[[{'track_id': 4}, {'track_id': 4}]], [[{'left': 0.0}]], [[{'track_id': None}]]])
def test_load_leap_exports_rejects_repeated_and_missing_ids(con, tmp_path, frames):
    # Two boxes of one object on one frame, and boxes without an id.
    with pytest.raises(AssertionError, match='missing or repeated id'):
        common.load_leap_exports(con, [('te00.mp4', write_text(tmp_path / 'a.json', json.dumps(frames)))])


def test_load_leap_exports_rejects_missing_files_repeated_videos_and_no_files(con, tmp_path):
    # The message names the exports, so a missing one is not a bare I/O error.
    with pytest.raises(AssertionError, match='raw LEAP exports missing'):
        common.load_leap_exports(con, [('te00.mp4', str(tmp_path / 'missing.json'))])
    path = write_text(tmp_path / 'a.json', '[]')
    with pytest.raises(AssertionError, match='more than once'):
        common.load_leap_exports(con, [('te00.mp4', path), ('te00.mp4', write_text(tmp_path / 'b.json', '[]'))])
    with pytest.raises(AssertionError, match='no raw LEAP exports'):
        common.load_leap_exports(con, [])


def recover(con, tmp_path, exports):
    # Every object's recovered interval, by (video, track_id), from {video: export frames}.
    common.load_leap_exports(con, [(v, write_text(tmp_path / f'{v}.json', json.dumps(f))) for v, f in exports.items()])
    common.leap_intervals(con)
    return {(r.pop('video'), r.pop('track_id')): r for r in query(con, 'SELECT * FROM leap_intervals')}


def test_leap_intervals_recover_every_kind(con, tmp_path):
    rows = recover(con, tmp_path, {
        # L = 12: contiguous, wrapped (start -3), ending on the last frame without starting at 0, longer than the
        # export (wraps onto itself), starting at 0, exactly the whole export, and a single frame.
        'a': leap_export({1: (2, 5), 2: (-3, 4), 3: (7, 12), 4: (-12, 12), 5: (0, 3), 6: (0, 12), 7: (5, 6)}),
        # L = 4: the same id 1 wraps in this clip (start -1), and id 2 ends on the last frame.
        'b': leap_export({1: (-1, 2), 2: (1, 4)}),
    })
    keys = ('runs', 'L', 'ambiguous', 'wrapped', 'interval_start', 'interval_end')
    assert {k: tuple(r[c] for c in keys) for k, r in rows.items()} == {
        ('a', 1): (1, 12, False, False, 2, 5),
        ('a', 2): (2, 12, False, True, -3, 4),
        ('a', 3): (1, 12, False, False, 7, 12),
        # Both ambiguous objects are kept as [0, L), whatever their true ends.
        ('a', 4): (1, 12, True, False, 0, 12),
        ('a', 5): (1, 12, False, False, 0, 3),
        ('a', 6): (1, 12, True, False, 0, 12),
        ('a', 7): (1, 12, False, False, 5, 6),
        ('b', 1): (2, 4, False, True, -1, 2),
        ('b', 2): (1, 4, False, False, 1, 4),
    }
    # The wrapped object's head run [0, 3] and tail run [9, 11].
    assert tuple(rows[('a', 2)][c] for c in ('head_start', 'head_end', 'tail_start', 'tail_end')) == (0, 3, 9, 11)


def test_leap_intervals_of_empty_exports(con, tmp_path):
    # No object, no interval, and no error.
    assert recover(con, tmp_path, {'a': [], 'b': [[], []]}) == {}


@pytest.mark.parametrize('ids', [
    # A hole that touches only the start, only the end, or neither; and three runs touching both ends.
    [[1], [], [1], []], [[], [1], [], [1]], [[], [1], [], [1], []], [[1], [], [1], [], [1]],
])
def test_leap_intervals_reject_other_shapes(con, tmp_path, ids):
    # None of these is one interval or one wrapped interval.
    frames = [[dict(track_id=t) for t in f] for f in ids]
    with pytest.raises(AssertionError, match='neither one interval nor one wrapped interval'):
        recover(con, tmp_path, {'a': frames})


def test_leap_intervals_match_p140_and_the_true_intervals_on_random_exports(con, tmp_path):
    # Twenty random 40-frame exports of 25 objects each, every one ending inside the export and some starting long
    # before it; the first object sets the export's length.
    rng = np.random.default_rng(0)
    length, clip_frames = 40, 30
    truth = {}
    for trial in range(20):
        ends = rng.integers(1, length + 1, 25)
        ends[0] = length
        truth[f'v{trial:02d}'] = {t + 1: (int(rng.integers(-length, e)), int(e)) for t, e in enumerate(ends)}
    exports = {v: leap_export(intervals) for v, intervals in truth.items()}
    rows = recover(con, tmp_path, exports)
    for video, intervals in truth.items():
        # The same wrapped and ambiguous objects as p140's unwrap.
        _, stats = unwrap_leap_frames(exports[video], clip_frames)
        mine = {t: r for (v, t), r in rows.items() if v == video}
        assert sorted(t for t, r in mine.items() if r['wrapped']) == stats['wrapped_objects']
        assert sorted(t for t, r in mine.items() if r['ambiguous']) == stats['ambiguous_full_span_objects']
        # Every true interval, except where it covers the whole export and is kept as [0, L).
        expected = {t: (0, length) if e - s >= length else (s, e) for t, (s, e) in intervals.items()}
        assert {t: (r['interval_start'], r['interval_end']) for t, r in mine.items()} == expected
    # The trials cover every kind.
    assert {(r['wrapped'], r['ambiguous']) for r in rows.values()} == {(False, False), (True, False), (False, True)}


# ---------------------------------------------------------------- TrackEval reruns


def test_metric_summary_means_loosest_alpha_and_scalars():
    # Two alphas only, for readability; counts are not averaged.
    metrics = {'HOTA': {'HOTA': [0.2, 0.4], 'HOTA_TP': np.array([3, 1]), 'HOTA(0)': np.float64(0.5)},
               'Count': {'Dets': np.int64(7)}}
    out = common.metric_summary(metrics)
    assert out['HOTA']['HOTA'] == pytest.approx(0.3) and 'HOTA_TP' not in out['HOTA']
    assert out['HOTA']['HOTA(0)'] == 0.5 and type(out['HOTA']['HOTA(0)']) is float
    assert out['HOTA']['alpha_0.05'] == {'HOTA': 0.2, 'HOTA_TP': 3.0}
    assert out['Count'] == {'Dets': 7} and type(out['Count']['Dets']) is int


def write_track_lines(path: Path, rows):
    # A tracking.jsonl with one line per (frame_idx, tracks) pair.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': f, 'tracks': t}) + '\n' for f, t in rows))


def test_trackeval_rerun_and_unscored_boxes(tmp_path):
    # Ground truth on frames 0-3, tracks on frames 0-1 that also hold a second object the ground truth lacks.
    gt = [(f, [[1, 0.0, 0.0, 8.0, 8.0]]) for f in range(4)]
    tracks = [(f, [[1, 0.0, 0.0, 8.0, 8.0], [2, 50.0, 50.0, 58.0, 58.0]]) for f in range(2)]
    write_track_lines(tmp_path / 'gt' / 'v' / 'gt.jsonl', gt)
    write_track_lines(tmp_path / 'tr' / 'v' / 'tr.jsonl', tracks)
    combined = common.trackeval_rerun(str(tmp_path / 'gt'), 'gt.jsonl', str(tmp_path / 'tr'), 'tr.jsonl', ['v'],
                                      str(tmp_path / 'out'), ('HOTA', 'CLEAR', 'Identity'), pad_scoring_window=False)
    summary = common.metric_summary(combined)
    # Every requested family and TrackEval's own Count.
    assert set(summary) == {'HOTA', 'CLEAR', 'Identity', 'Count'}
    # Two hits and two false positives on the frames both files reach; the ground truth's frames 2-3 are counted only.
    assert (summary['Count']['GT_Dets'], summary['Count']['Dets']) == (4, 4)
    loosest = summary['HOTA']['alpha_0.05']
    assert (loosest['HOTA_TP'], loosest['HOTA_FN'], loosest['HOTA_FP']) == (2.0, 0.0, 2.0)
    assert common.unscored_boxes(summary) == dict(gt=2, tracker=0)
    # The canonical (padded) window scores the ground truth on frames 2-3 as misses.
    padded = common.metric_summary(common.trackeval_rerun(str(tmp_path / 'gt'), 'gt.jsonl', str(tmp_path / 'tr'), 'tr.jsonl',
                                                          ['v'], str(tmp_path / 'outp')))
    assert (padded['HOTA']['alpha_0.05']['HOTA_FN'], common.unscored_boxes(padded)) == (2.0, dict(gt=0, tracker=0))
    # The HOTA metric alone is the default, as p110 and p141 run it.
    assert set(common.metric_summary(common.trackeval_rerun(
        str(tmp_path / 'gt'), 'gt.jsonl', str(tmp_path / 'tr'), 'tr.jsonl', ['v'], str(tmp_path / 'out2')))) == {'HOTA', 'Count'}
    # An unknown family is rejected before TrackEval runs.
    with pytest.raises(AssertionError, match='unknown TrackEval metrics'):
        common.trackeval_rerun(str(tmp_path / 'gt'), 'gt.jsonl', str(tmp_path / 'tr'), 'tr.jsonl', ['v'], str(tmp_path / 'o'),
                               ('MOTA',))


def test_unscored_boxes_on_both_sides():
    # Counts from a summary: GT_Dets - (TP + FN) and Dets - (TP + FP) at the loosest alpha, as integers.
    summary = {'HOTA': {'alpha_0.05': {'HOTA_TP': 5.0, 'HOTA_FN': 2.0, 'HOTA_FP': 1.0}}, 'Count': {'GT_Dets': 10, 'Dets': 9}}
    assert common.unscored_boxes(summary) == dict(gt=3, tracker=3)
    assert all(type(v) is int for v in common.unscored_boxes(summary).values())
