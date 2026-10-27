import ast
import json
import math
import os
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

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
