import duckdb
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
