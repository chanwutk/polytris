import json

import duckdb
import pytest

from measurements import common
from measurements import r1o2_count_query as cq


# A unit square and a concave "L" polygon for point-in-polygon checks.
SQUARE = ((0, 0), (10, 0), (10, 10), (0, 10))
L_SHAPE = ((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10))
# Two disjoint regions for the polygon rule: top (y < 10) and bottom (20 < y < 30).
TOP = ((0, 0), (100, 0), (100, 10), (0, 10))
BOTTOM = ((0, 20), (100, 20), (100, 30), (0, 30))


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def write_tracks(path, frames):
    # Write a tracking.jsonl with one line per (frame_idx, [[id, x1, y1, x2, y2], ...]), in the given order.
    with open(path, 'w') as f:
        for frame_idx, tracks in frames:
            f.write(json.dumps({'frame_idx': frame_idx, 'tracks': tracks}) + '\n')


def set_endpoints(con, rows):
    # Create the endpoints table directly from (filename, track_id, f0, x0, y0, f1, x1, y1) rows.
    con.execute('CREATE OR REPLACE TABLE endpoints (filename VARCHAR, track_id BIGINT, f0 BIGINT, x0 DOUBLE, '
                'y0 DOUBLE, f1 BIGINT, x1 DOUBLE, y1 DOUBLE)')
    con.executemany('INSERT INTO endpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)


def inside(con, polygon, points):
    # Regions containing each point, via the same SQL the measurement uses.
    create_points = [('f', i, 0, x, y, 0, x, y) for i, (x, y) in enumerate(points)]
    set_endpoints(con, create_points)
    cq.create_edges(con, {'shape': polygon})
    cq.locate_endpoints(con)
    found = {r[0] for r in con.execute("SELECT track_id FROM located WHERE endpoint = 'start'").fetchall()}
    return [i in found for i in range(len(points))]


def counted(con):
    # Segment of every counted track, as {track_id: segment}.
    return dict(con.execute('SELECT track_id, segment FROM counted').fetchall())


def test_locate_square_inside_and_outside(con):
    # The interior is inside; points beyond an edge are outside.
    assert inside(con, SQUARE, [(5, 5), (15, 5), (5, -1)]) == [True, False, False]


def test_locate_concave_notch_is_outside(con):
    # Both arms of the L are inside; the notch between them is not, which a bounding box would get wrong.
    assert inside(con, L_SHAPE, [(8, 2), (2, 8), (8, 8)]) == [True, True, False]


def test_locate_otif_caldot1_regions(con):
    # A point deep in CalDoT1's bottom-left region and one in its top region, checked against both regions.
    assert inside(con, cq.CALDOT1_BOTTOM, [(100, 300), (450, 200)]) == [True, False]
    assert inside(con, cq.CALDOT1_TOP, [(100, 300), (450, 200)]) == [False, True]


def test_read_ground_truth_skips_header_and_blank_lines(tmp_path):
    # A CalDoT-style table with a '#' header, two segments, and a trailing blank line.
    path = tmp_path / 'gt.txt'
    path.write_text('#\tleft\tright\n0\t0\t8\n1\t19\t0\n\n')
    assert cq.read_ground_truth(path) == {0: [0, 8], 1: [19, 0]}


def test_read_ground_truth_single_segment_without_header(tmp_path):
    # Amsterdam's table has one count per video and no header.
    path = tmp_path / 'gt.txt'
    path.write_text('0\t5\n1\t7\n')
    assert cq.read_ground_truth(path) == {0: [5], 1: [7]}


def test_read_ground_truth_rejects_empty_and_ragged_tables(tmp_path):
    # An empty table has no videos to average over.
    empty = tmp_path / 'empty.txt'
    empty.write_text('#\tleft\tright\n')
    with pytest.raises(AssertionError):
        cq.read_ground_truth(empty)
    # Rows with different segment counts cannot be scored consistently.
    ragged = tmp_path / 'ragged.txt'
    ragged.write_text('0\t1\t2\n1\t3\n')
    with pytest.raises(AssertionError):
        cq.read_ground_truth(ragged)


def test_load_endpoints_uses_earliest_and_latest_frames(con, tmp_path):
    # Track 1 appears at frames 5, 2, 9 written out of order; track 2 appears once; one file is empty.
    path, empty = tmp_path / 'tracking.jsonl', tmp_path / 'empty.jsonl'
    write_tracks(path, [(5, [[1, 0, 0, 2, 2]]), (2, [[1, 10, 10, 12, 12], [2, 4, 4, 6, 6]]), (9, [[1, 20, 20, 22, 22]])])
    empty.write_text('')
    cq.load_endpoints(con, [str(path), str(empty)])
    rows = con.execute('SELECT track_id, f0, x0, y0, f1, x1, y1 FROM endpoints ORDER BY track_id').fetchall()
    # First is the frame-2 center and last the frame-9 center, regardless of line order.
    assert rows[0] == (1, 2, 11.0, 11.0, 9, 21.0, 21.0)
    # A single observation is both first and last.
    assert rows[1] == (2, 2, 5.0, 5.0, 2, 5.0, 5.0)
    # The empty file contributes no tracks.
    assert len(rows) == 2


def test_load_endpoints_keeps_files_apart(con, tmp_path):
    # The same track id in two files is two different tracks.
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    write_tracks(a, [(0, [[1, 0, 0, 2, 2]])])
    write_tracks(b, [(3, [[1, 8, 8, 10, 10]])])
    cq.load_endpoints(con, [str(a), str(b)])
    assert con.execute('SELECT count(*) FROM endpoints').fetchone()[0] == 2


def test_load_endpoints_rejects_no_files(con):
    # Reading zero files is a caller error.
    with pytest.raises(AssertionError):
        cq.load_endpoints(con, [])


def test_count_polygon_directions(con):
    set_endpoints(con, [
        # Top to bottom: segment 0.
        ('f', 1, 0, 50, 5, 10, 50, 25),
        # Bottom to top: segment 1.
        ('f', 2, 0, 50, 25, 10, 50, 5),
        # Starts and ends in the top region: not counted.
        ('f', 3, 0, 10, 5, 10, 90, 5),
        # Ends between the regions: not counted.
        ('f', 4, 0, 50, 5, 10, 50, 15),
        # A one-box track cannot start and end in disjoint regions.
        ('f', 5, 3, 50, 5, 3, 50, 5),
    ])
    cq.create_edges(con, {'top': TOP, 'bottom': BOTTOM})
    cq.count_tracks(con, 'polygon')
    assert counted(con) == {1: 0, 2: 1}


def test_count_polygon_overlap_prefers_segment_zero(con):
    # With overlapping regions a track can satisfy both directions; eval.go checks top->bottom first.
    set_endpoints(con, [('f', 1, 0, 5, 5, 10, 5, 5)])
    cq.create_edges(con, {'top': SQUARE, 'bottom': SQUARE})
    cq.count_tracks(con, 'polygon')
    assert counted(con) == {1: 0}


def set_video_rules(con, files, rules):
    # Map filenames to videos and give each video its minimum duration, as score_dataset does.
    con.execute('CREATE OR REPLACE TABLE files (system VARCHAR, config VARCHAR, video INTEGER, filename VARCHAR)')
    con.executemany('INSERT INTO files VALUES (?, ?, ?, ?)', [('tetris', 'a', v, f) for f, v in files])
    con.execute('CREATE OR REPLACE TABLE video_rules (video INTEGER, min_duration INTEGER)')
    con.executemany('INSERT INTO video_rules VALUES (?, ?)', rules)


def test_count_travel_thresholds_are_inclusive(con):
    set_endpoints(con, [
        # Exactly the minimum duration and distance: counted (eval.go skips only below).
        ('f', 1, 0, 0, 0, 5, cq.OTIF_MIN_DISTANCE, 0),
        # One frame too short: dropped.
        ('f', 2, 0, 0, 0, 4, 200, 0),
        # Long enough but just under the minimum distance: dropped.
        ('f', 3, 0, 0, 0, 50, 30, 39.9),
        # A (30, 40) diagonal is exactly 50: counted.
        ('f', 4, 0, 0, 0, 50, 30, 40),
    ])
    set_video_rules(con, [('f', 0)], [(0, 5)])
    cq.count_tracks(con, 'travel')
    assert counted(con) == {1: 0, 4: 0}


def test_count_travel_uses_each_videos_threshold(con):
    # The same 7-frame track passes in a video needing 5 frames and fails in one needing 10.
    set_endpoints(con, [('fast', 1, 0, 0, 0, 7, 100, 0), ('slow', 2, 0, 0, 0, 7, 100, 0)])
    set_video_rules(con, [('fast', 0), ('slow', 1)], [(0, 5), (1, 10)])
    cq.count_tracks(con, 'travel')
    assert counted(con) == {1: 0}


def setup_scoring(con, files, truth, configs, counted_rows):
    # Create the tables score_configs reads.
    con.execute('CREATE OR REPLACE TABLE files (system VARCHAR, config VARCHAR, video INTEGER, filename VARCHAR)')
    con.executemany('INSERT INTO files VALUES (?, ?, ?, ?)', files)
    con.execute('CREATE OR REPLACE TABLE truth (video INTEGER, segment INTEGER, count INTEGER)')
    con.executemany('INSERT INTO truth VALUES (?, ?, ?)', truth)
    con.execute('CREATE OR REPLACE TABLE configs (system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE)')
    con.executemany('INSERT INTO configs VALUES (?, ?, ?, ?)', configs)
    con.execute('CREATE OR REPLACE TABLE counted (filename VARCHAR, track_id BIGINT, segment INTEGER)')
    if counted_rows:
        con.executemany('INSERT INTO counted VALUES (?, ?, ?)', counted_rows)


def test_score_configs_applies_otif_accuracy_rules(con):
    # One configuration over three videos with two segments each.
    setup_scoring(
        con,
        files=[('tetris', 'a', 0, 'v0'), ('tetris', 'a', 1, 'v1'), ('tetris', 'a', 2, 'v2')],
        truth=[(0, 0, 3), (0, 1, 0), (1, 0, 4), (1, 1, 2), (2, 0, 3), (2, 1, 5)],
        configs=[('tetris', 'a', 100.0, 0.7)],
        counted_rows=[
            # Video 0: 3 and 0 match exactly (both-zero segment scores 1) -> 1.0.
            ('v0', 1, 0), ('v0', 2, 0), ('v0', 3, 0),
            # Video 1: segment 0 predicts 0 of 4 -> 0; segment 1 predicts 3 of 2 -> 0.5; mean 0.25.
            ('v1', 1, 1), ('v1', 2, 1), ('v1', 3, 1),
            # Video 2: segment 0 predicts 9 of 3 -> -1 (negative, as OTIF allows); segment 1 predicts 0 of 5 -> 0.
            *[('v2', i, 0) for i in range(9)],
        ])
    (row,) = cq.score_configs(con)
    # Mean over videos: (1.0 + 0.25 - 0.5) / 3.
    assert row['accuracy'] == pytest.approx(round(0.75 / 3, 4))
    assert row['videos'] == 3
    assert row['predicted_per_segment'] == [12, 3]
    assert (row['fps'], row['hota']) == (100.0, 0.7)


def test_score_configs_separates_configurations(con):
    # Two configurations share a video index but not a file.
    setup_scoring(
        con,
        files=[('tetris', 'a', 0, 'a0'), ('otif', 'param 1', 0, 'b0')],
        truth=[(0, 0, 2)],
        configs=[('tetris', 'a', 10.0, 0.5), ('otif', 'param 1', 20.0, 0.4)],
        counted_rows=[('a0', 1, 0), ('a0', 2, 0), ('b0', 1, 0)])
    rows = {r['system']: r['accuracy'] for r in cq.score_configs(con)}
    assert rows == {'tetris': 1.0, 'otif': 0.5}


def test_scaled_min_duration_converts_whole_factors():
    # Same frame rate keeps OTIF's threshold.
    assert cq.scaled_min_duration(15.0, 15.0) == 10
    # Amsterdam: 30 FPS OTIF video, 15 FPS clips.
    assert cq.scaled_min_duration(30.0, 15.0) == 5
    # NTSC rates round to the same factor.
    assert cq.scaled_min_duration(29.97, 14.985) == 5


def test_scaled_min_duration_rejects_inexact_conversions():
    # A 1.5x factor does not map whole frames to whole frames.
    with pytest.raises(AssertionError):
        cq.scaled_min_duration(30.0, 20.0)
    # Clips faster than OTIF's video are not a subsampling.
    with pytest.raises(AssertionError):
        cq.scaled_min_duration(15.0, 30.0)
    # Factor 3 does not divide 10 OTIF frames evenly.
    with pytest.raises(AssertionError):
        cq.scaled_min_duration(45.0, 15.0)


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
