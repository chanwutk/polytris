import argparse
import gzip
import importlib.util
import json
import random
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd
import pytest

from measurements import r1o4_packing_detection_accuracy as pa
from polyis.io import cache, store


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def set_boxes(con, gt_rows, det_rows):
    # gt(video, frame, gid, x1, y1, x2, y2) and dets(source, video, frame, did, x1, y1, x2, y2, score) from rows.
    con.execute('CREATE OR REPLACE TABLE gt (video VARCHAR, frame BIGINT, gid BIGINT, x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE)')
    con.execute('CREATE OR REPLACE TABLE dets (source VARCHAR, video VARCHAR, frame BIGINT, did BIGINT, '
                'x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE, score DOUBLE)')
    if gt_rows:
        con.executemany('INSERT INTO gt VALUES (?, ?, ?, ?, ?, ?, ?)', gt_rows)
    if det_rows:
        con.executemany('INSERT INTO dets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', det_rows)


def set_tiles(con, rows):
    # tiles(cfg, video, frame, ty, tx) from rows.
    con.execute('CREATE OR REPLACE TABLE tiles (cfg VARCHAR, video VARCHAR, frame BIGINT, ty BIGINT, tx BIGINT)')
    if rows:
        con.executemany('INSERT INTO tiles VALUES (?, ?, ?, ?, ?)', rows)


def set_configs(con, rows):
    # configs(cfg, sample_rate, tilepadding) from (cfg, sample_rate[, tilepadding]) rows; padding defaults to 'none'.
    con.execute('CREATE OR REPLACE TABLE configs (cfg VARCHAR, sample_rate BIGINT, tilepadding VARCHAR)')
    con.executemany('INSERT INTO configs VALUES (?, ?, ?)', [(*r, 'none')[:3] for r in rows])


def matched(con):
    # {(source, frame, did): gid} of the matching.
    return {(s, f, d): g for s, f, d, g in con.execute('SELECT source, frame, did, gid FROM matches').fetchall()}


def iou(a, b):
    # IoU of two (x1, y1, x2, y2) boxes, as the SQL computes it.
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def sequential_greedy(gt_rows, det_rows, threshold=0.5):
    # COCO's sequential greedy matching, written plainly, as the oracle for the set-based version.
    out = {}
    keys = {(s, v, f) for s, v, f, *_ in det_rows}
    for s, v, f in keys:
        dets = sorted([d for d in det_rows if d[:3] == (s, v, f)], key=lambda d: (-d[8], d[3]))
        gts = sorted([g for g in gt_rows if g[:2] == (v, f)], key=lambda g: g[2])
        taken = set()
        for d in dets:
            best, best_iou = None, -1.0
            for g in gts:
                o = iou(d[4:8], g[3:7])
                if g[2] in taken or o < threshold:
                    continue
                if o > best_iou:
                    best, best_iou = g[2], o
            if best is not None:
                taken.add(best)
                out[(s, f, d[3])] = best
    return out


def row(**kw):
    # A tradeoff row with defaults for an unpruned configuration.
    base = dict(classifier='ShuffleNet05', tilesize=60.0, sample_rate=1.0, tracking_accuracy_threshold=float('nan'),
                relevance_threshold=0.25, tilepadding='none', canvas_scale=1.0, tracker='sortcython',
                variant_id='ShuffleNet05_60_1_r025_none_s100_sortcython')
    base.update(kw)
    return base


def test_accuracy_threshold_none_nan_and_value():
    # NaN and None both mean the configuration skipped pruning; a value comes back as a float.
    assert pa.accuracy_threshold(row()) is None
    assert pa.accuracy_threshold(row(tracking_accuracy_threshold=None)) is None
    assert pa.accuracy_threshold(row(tracking_accuracy_threshold=np.float64(0.4))) == 0.4


def test_detection_config_unpruned_drops_tracker():
    # Unpruned configurations share one 050 directory across trackers.
    assert pa.detection_config(row()) == 'ShuffleNet05_60_1_r025_none_s100'
    # None works like NaN for a missing threshold.
    assert pa.detection_config(row(tracking_accuracy_threshold=None)) == 'ShuffleNet05_60_1_r025_none_s100'


def test_detection_config_pruned_keeps_tracker():
    # Pruned configurations carry threshold and tracker through every stage.
    r = row(sample_rate=8.0, tracking_accuracy_threshold=0.4, relevance_threshold=0.5, tilepadding='bl',
            tracker='bytetrackcython', variant_id='ShuffleNet05_60_8_040_r050_bl_s100_bytetrackcython')
    assert pa.detection_config(r) == 'ShuffleNet05_60_8_040_r050_bl_s100_bytetrackcython'


def test_detection_config_rejects_mismatched_variant():
    # A variant id that the parameters do not produce means the naming convention changed.
    with pytest.raises(AssertionError):
        pa.detection_config(row(variant_id='ShuffleNet05_60_2_r025_none_s100_sortcython'))


def test_metadata_table_zips_columns_and_keeps_none(con):
    # Columns are zipped row by row; None becomes NULL.
    pa.metadata_table(con, 't', a=('VARCHAR', ['x', 'y']), b=('DOUBLE', [None, 0.5]))
    assert con.execute('SELECT a, b FROM t ORDER BY a').fetchall() == [('x', None), ('y', 0.5)]
    # Empty lists give an empty, typed table.
    pa.metadata_table(con, 'e', a=('BIGINT', []))
    assert con.execute('SELECT count(*) FROM e').fetchone()[0] == 0
    # Columns of different lengths would be padded with NULL.
    with pytest.raises(AssertionError):
        pa.metadata_table(con, 'bad', a=('BIGINT', [1, 2]), b=('BIGINT', [1]))


def write_canvas(directory, name, index_map, lookup):
    # One canvas's index map and offset lookup, in p030's formats.
    (directory / 'index_maps').mkdir(parents=True, exist_ok=True)
    (directory / 'offset_lookups').mkdir(parents=True, exist_ok=True)
    np.save(directory / 'index_maps' / f'{name}.npy', np.array(index_map, dtype=np.uint16))
    (directory / 'offset_lookups' / f'{name}.jsonl').write_text(''.join(json.dumps(l) + '\n' for l in lookup))


def test_canvas_tiles_inverts_placement(tmp_path):
    # Group 1 (two cells at canvas (0, 0) and (0, 1)) came from frame 4 at source (2, 3);
    # group 2 (one cell at canvas (1, 2)) came from frame 8 at source (0, 0).
    write_canvas(tmp_path, '0000_0004_0008', [[1, 1, 0], [0, 0, 2]], [[[0, 0], [2, 3], 4], [[1, 2], [0, 0], 8]])
    # A second canvas of the same configuration.
    write_canvas(tmp_path, '0001_0012_0012', [[0, 1]], [[[0, 1], [5, 5], 12]])
    tiles = {tuple(t) for t in pa.canvas_tiles(tmp_path).tolist()}
    assert tiles == {(4, 2, 3), (4, 2, 4), (8, 0, 0), (12, 5, 5)}


def test_canvas_tiles_empty_and_missing(tmp_path):
    # A video where the configuration retained nothing has mapping directories but no canvases.
    (tmp_path / 'index_maps').mkdir()
    assert pa.canvas_tiles(tmp_path).shape == (0, 3)
    # Without mapping files the retained region is unknown.
    with pytest.raises(AssertionError):
        pa.canvas_tiles(tmp_path / 'absent')


def test_canvas_tiles_rejects_missing_offset_lookup(tmp_path):
    # An index map whose offset lookup is gone cannot be placed.
    write_canvas(tmp_path, '0000_0000_0000', [[1]], [[[0, 0], [0, 0], 0]])
    (tmp_path / 'offset_lookups' / '0000_0000_0000.jsonl').unlink()
    with pytest.raises(AssertionError):
        pa.canvas_tiles(tmp_path)


def test_canvas_tiles_rejects_group_beyond_lookup(tmp_path):
    # Group 2 has no offset line.
    write_canvas(tmp_path, '0000_0000_0000', [[1, 2]], [[[0, 0], [0, 0], 0]])
    with pytest.raises(AssertionError):
        pa.canvas_tiles(tmp_path)


def write_jsonl(path, key, frames):
    # A box file with one line per (frame_idx, boxes).
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': f, key: b}) + '\n' for f, b in frames))
    return str(path)


def test_read_boxes_ground_truth_and_detections(con, tmp_path):
    gt = write_jsonl(tmp_path / 'gt.jsonl', 'tracks', [(0, [[7, 1, 2, 3, 4], [9, 5, 6, 7, 8]]), (2, [[7, 1, 1, 2, 2]])])
    pa.read_boxes(con, 'gt', [('gt', 'v', gt)], 'gt')
    # Track ids become box ids; coordinates follow the id.
    assert con.execute('SELECT frame, gid, x1, y2 FROM gt ORDER BY frame, gid').fetchall() == \
        [(0, 7, 1.0, 4.0), (0, 9, 5.0, 8.0), (2, 7, 1.0, 2.0)]
    # Detections get their 1-based position in the frame; empty frames add nothing; an empty file is fine.
    a = write_jsonl(tmp_path / 'a.jsonl', 'bboxes', [(0, []), (1, [[1, 2, 3, 4, 0.9], [5, 6, 7, 8, 0.3]])])
    b = write_jsonl(tmp_path / 'b.jsonl', 'bboxes', [])
    pa.read_boxes(con, 'dets', [('c', 'v', a), ('c', 'w', b)], 'packed')
    assert con.execute('SELECT source, video, frame, did, x1, score FROM dets ORDER BY did').fetchall() == \
        [('c', 'v', 1, 1, 1.0, 0.9), ('c', 'v', 1, 2, 5.0, 0.3)]
    # Every line, including empty frames, is kept as a frame line.
    assert con.execute('SELECT source, video, frame FROM dets_lines ORDER BY frame').fetchall() == [('c', 'v', 0), ('c', 'v', 1)]


def test_read_boxes_same_frame_in_two_files_is_fine(con, tmp_path):
    # Frame 0 of two videos (or two sources) are different frames.
    a = write_jsonl(tmp_path / 'a.jsonl', 'detections', [(0, [[0, 0, 1, 1, 0.5]])])
    b = write_jsonl(tmp_path / 'b.jsonl', 'detections', [(0, [[0, 0, 1, 1, 0.5]])])
    pa.read_boxes(con, 'dets', [('reference', 'v', a), ('reference', 'w', b)], 'reference')
    assert con.execute('SELECT count(*) FROM dets').fetchone()[0] == 2


@pytest.mark.parametrize('kind, key, frames', [
    # A detection without a confidence.
    ('reference', 'detections', [(0, [[1, 2, 3, 4]])]),
    # A ground-truth box without its last coordinate.
    ('gt', 'tracks', [(0, [[1, 0, 0, 1]])]),
    # Two ground-truth boxes with the same track id on one frame.
    ('gt', 'tracks', [(0, [[1, 0, 0, 1, 1], [1, 2, 2, 3, 3]])]),
    # A frame written twice, which would reuse detection ids.
    ('packed', 'bboxes', [(0, [[0, 0, 1, 1, 0.5]]), (0, [[2, 2, 3, 3, 0.5]])]),
    # A frame written twice in ground truth, with different track ids.
    ('gt', 'tracks', [(0, [[1, 0, 0, 1, 1]]), (0, [[2, 2, 2, 3, 3]])]),
])
def test_read_boxes_rejects_bad_files(con, tmp_path, kind, key, frames):
    bad = write_jsonl(tmp_path / 'bad.jsonl', key, frames)
    with pytest.raises(AssertionError):
        pa.read_boxes(con, 'boxes', [(kind, 'v', bad)], kind)


def test_read_boxes_rejects_missing_and_no_files(con, tmp_path):
    # A missing file would silently drop a video.
    with pytest.raises(AssertionError):
        pa.read_boxes(con, 'dets', [('reference', 'v', str(tmp_path / 'missing.jsonl'))], 'reference')
    # No files at all.
    with pytest.raises(AssertionError):
        pa.read_boxes(con, 'dets', [], 'reference')


def test_greedy_higher_score_takes_best_and_is_not_optimal(con):
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10), ('v', 0, 2, 2, 0, 12, 10)], [
        # d1 overlaps g1 at IoU 0.90 and g2 at 0.74; it takes g1.
        ('s', 'v', 0, 1, 0.5, 0, 10.5, 10, 0.9),
        # d2 clears the threshold only on g1 (0.54; 0.33 on g2), which d1 took: greedy leaves it unmatched
        # although d1 -> g2, d2 -> g1 would match both.
        ('s', 'v', 0, 2, -3, 0, 7, 10, 0.8),
    ])
    pa.greedy_match(con)
    assert matched(con) == {('s', 0, 1): 1}


def test_greedy_score_tie_goes_to_lower_did_and_iou_tie_to_lower_gid(con):
    # Two identical detections on two identical ground-truth boxes.
    set_boxes(con, [('v', 0, 5, 0, 0, 10, 10), ('v', 0, 3, 0, 0, 10, 10)],
              [('s', 'v', 0, 2, 0, 0, 10, 10, 0.5), ('s', 'v', 0, 1, 0, 0, 10, 10, 0.5)])
    pa.greedy_match(con)
    assert matched(con) == {('s', 0, 1): 3, ('s', 0, 2): 5}


def test_greedy_threshold_is_inclusive(con):
    # IoU exactly 0.5 (half of a 10x10 box) matches; 0.499 does not.
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10), ('v', 1, 1, 0, 0, 10, 10)],
              [('s', 'v', 0, 1, 0, 0, 10, 5, 0.9), ('s', 'v', 1, 1, 0, 0, 10, 4.99, 0.9)])
    pa.greedy_match(con)
    assert matched(con) == {('s', 0, 1): 1}


def test_greedy_chain_needs_several_rounds(con):
    # d1 competes with d2 for g1, d2 with d3 for g2: each is final only after the one above it.
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10), ('v', 0, 2, 4, 0, 14, 10), ('v', 0, 3, 8, 0, 18, 10)], [
        ('s', 'v', 0, 1, 0, 0, 10, 10, 0.9),
        ('s', 'v', 0, 2, 2, 0, 12, 10, 0.8),
        ('s', 'v', 0, 3, 6, 0, 16, 10, 0.7),
    ])
    assert pa.greedy_match(con) == 3
    assert matched(con) == {('s', 0, 1): 1, ('s', 0, 2): 2, ('s', 0, 3): 3}


def test_greedy_keeps_sources_and_frames_apart(con):
    # The same boxes under two sources and on another frame are matched independently.
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10)], [
        ('a', 'v', 0, 1, 0, 0, 10, 10, 0.9), ('b', 'v', 0, 1, 0, 0, 10, 10, 0.9), ('a', 'v', 1, 1, 0, 0, 10, 10, 0.9),
    ])
    pa.greedy_match(con)
    assert matched(con) == {('a', 0, 1): 1, ('b', 0, 1): 1}


def test_greedy_empty_inputs(con):
    # No boxes, no matches, no rounds.
    set_boxes(con, [], [])
    assert pa.greedy_match(con) == 0
    assert matched(con) == {}


def test_greedy_equals_sequential_greedy_on_random_frames(con):
    # Crowded random frames with score ties, against the plain sequential algorithm.
    rng = random.Random(0)
    gt_rows, det_rows = [], []
    for frame in range(300):
        gts = [(rng.uniform(0, 40), rng.uniform(0, 40), rng.uniform(8, 20), rng.uniform(8, 20)) for _ in range(rng.randint(0, 6))]
        gt_rows += [('v', frame, gid, x, y, x + w, y + h) for gid, (x, y, w, h) in enumerate(gts, start=1)]
        for source in ('a', 'b'):
            n = rng.randint(0, 8)
            for did in range(1, n + 1):
                # Mostly jittered copies of ground-truth boxes, some random boxes.
                x, y, w, h = rng.choice(gts) if gts and rng.random() < 0.8 else (rng.uniform(0, 40), rng.uniform(0, 40), 12, 12)
                dx, dy = rng.uniform(-4, 4), rng.uniform(-4, 4)
                det_rows.append((source, 'v', frame, did, x + dx, y + dy, x + dx + w * rng.uniform(0.8, 1.2),
                                 y + dy + h * rng.uniform(0.8, 1.2), round(rng.random(), 1)))
    set_boxes(con, gt_rows, det_rows)
    pa.greedy_match(con)
    assert matched(con) == sequential_greedy(gt_rows, det_rows)


def test_box_regions_center_and_cover(con):
    # Tiles of size 10: configuration c retains (0, 0) and (0, 1) of frame 0.
    set_tiles(con, [('c', 'v', 0, 0, 0), ('c', 'v', 0, 0, 1)])
    set_boxes(con, [
        # Spans exactly the two retained tiles; the center (10, 5) falls on the boundary into tile (0, 1).
        ('v', 0, 1, 0, 0, 20, 10),
        # Spans 2x3 tiles; 75 of its 200 px are retained, and its center tile (1, 1) is not.
        ('v', 0, 2, 5, 5, 25, 15),
        # Ends exactly on a tile boundary, so it spans only tile (0, 0).
        ('v', 0, 3, 0, 0, 10, 10),
        # Entirely outside the retained tiles, and zero area: no rows.
        ('v', 0, 4, 100, 100, 110, 110),
        ('v', 0, 5, 3, 3, 3, 8),
        # Partly outside the frame: a quarter inside tile (0, 0), which holds its center.
        ('v', 0, 6, -5, -5, 5, 5),
        # Another frame without retained tiles.
        ('v', 1, 7, 0, 0, 10, 10),
        # Fractional coordinates across both tiles: fully covered, up to float error in the sum.
        ('v', 0, 8, 1.3, 0.7, 19.9, 9.1),
    ], [
        # The reference and c's own packed detection are scored against c; c2's packed detection is not.
        ('reference', 'v', 0, 1, 0, 0, 10, 10, 0.9),
        ('c', 'v', 0, 1, 0, 0, 10, 10, 0.9),
        ('c2', 'v', 0, 1, 0, 0, 10, 10, 0.9),
    ])
    pa.box_regions(con, 10)
    rows = {(r[0], r[1], r[2], r[3]): (r[4], round(r[5], 4)) for r in con.execute(
        'SELECT cfg, role, source, bid, center_in, cover FROM regions').fetchall()}
    assert rows == {
        ('c', 'gt', 'gt', 1): (True, 1.0),
        ('c', 'gt', 'gt', 2): (False, 0.375),
        ('c', 'gt', 'gt', 3): (True, 1.0),
        ('c', 'gt', 'gt', 6): (True, 0.25),
        ('c', 'gt', 'gt', 8): (True, 1.0),
        ('c', 'det', 'reference', 1): (True, 1.0),
        ('c', 'det', 'c', 1): (True, 1.0),
    }
    # The fractional box passes the 'inside' threshold despite float error in its area sum.
    cover = con.execute("SELECT cover FROM regions WHERE role = 'gt' AND bid = 8").fetchone()[0]
    assert cover >= pa.FILTERS['inside']


def scored(con, alternatives=False):
    # Match, locate, build config inputs, and arrange the scores of the sweep's sides.
    pa.greedy_match(con)
    pa.box_regions(con, 10)
    pa.build_config_inputs(con)
    rows = pa.score(con, pa.MAIN_SIDES)
    return pa.arrange(rows, pa.deltas(con, pa.MAIN_PAIRS), pa.retained_alternatives(con) if alternatives else None)


def entry(n_gt, tp, fp, precision, recall, ap50, matched_nonretained=0, unmatched_outside=0, fp_outside=0):
    # One side's expected report entry.
    return dict(n_gt=n_gt, tp=tp, fp=fp, precision=precision, recall=recall, ap50=ap50,
                ignored_matched_nonretained=matched_nonretained, ignored_unmatched_outside=unmatched_outside,
                fp_outside_filter=fp_outside)


def test_config_scores_isolate_retained_content(con):
    set_configs(con, [('c', 1), ('empty', 1)])
    # c packed tiles (0, 0) and (0, 1) of frame 0 only; 'empty' packed nothing.
    set_tiles(con, [('c', 'v', 0, 0, 0), ('c', 'v', 0, 0, 1)])
    set_boxes(con, [
        # g1 inside tile (0, 0), g2 far outside, g3 on frame 1 (no detection ran there).
        ('v', 0, 1, 1, 1, 9, 9), ('v', 0, 2, 50, 50, 58, 58), ('v', 1, 3, 1, 1, 9, 9),
    ], [
        # Reference: hits g1, hits g2 (outside: ignored), a false positive inside tile (0, 1), and a hit on frame 1.
        ('reference', 'v', 0, 1, 1, 1, 9, 9, 0.9), ('reference', 'v', 0, 2, 50, 50, 58, 58, 0.8),
        ('reference', 'v', 0, 3, 12, 1, 18, 9, 0.7), ('reference', 'v', 1, 1, 1, 1, 9, 9, 0.9),
        # Packed: a slightly shifted hit on g1.
        ('c', 'v', 0, 1, 1.5, 1, 9.5, 9, 0.6),
    ])
    out = scored(con)
    # Retained content: only g1 counts; the reference's in-region false positive costs it precision.
    for name in ('center', 'center_cover50', 'inside'):
        assert out['c'][name]['reference'] == entry(1, 1, 1, 0.5, 1.0, 1.0, matched_nonretained=1)
        assert out['c'][name]['packed'] == entry(1, 1, 0, 1.0, 1.0, 1.0)
        assert out['c'][name]['packed_ignore_outside'] == entry(1, 1, 0, 1.0, 1.0, 1.0)
        assert out['c'][name]['delta'] == {side: dict(precision=0.5, recall=0.0, ap50=0.0)
                                           for side in ('packed', 'packed_ignore_outside')}
    # No region filter: g2 counts too, which only the reference detects (region-selection loss).
    assert out['c']['frame']['reference'] == entry(2, 2, 1, 0.6667, 1.0, 1.0)
    assert out['c']['frame']['packed'] == entry(2, 1, 0, 1.0, 0.5, 0.5)
    # The delta is rounded after subtracting: 1 - 2/3, not 1 - 0.6667.
    assert out['c']['frame']['delta']['packed'] == dict(precision=0.3333, recall=-0.5, ap50=-0.5)
    # A configuration that ran no detection has nothing to score.
    for name in pa.FILTERS:
        for side in pa.MAIN_SIDES:
            assert out['empty'][name][side] == entry(0, 0, 0, None, None, None)
        assert out['empty'][name]['delta']['packed'] == dict(precision=None, recall=None, ap50=None)


def test_packed_detections_outside_the_region(con):
    set_configs(con, [('c', 1)])
    # c retains tiles (0, 0) and (0, 1) of frame 0: x in [0, 20), y in [0, 10).
    set_tiles(con, [('c', 'v', 0, 0, 0), ('c', 'v', 0, 0, 1)])
    set_boxes(con, [
        # g1 in tile (0, 0); g2 with its center (19.5, 4) in tile (0, 1); g3 outside.
        ('v', 0, 1, 1, 1, 9, 9), ('v', 0, 2, 16, 0, 23, 8), ('v', 0, 3, 50, 50, 58, 58),
    ], [
        # Reference: hits g1 and g2, and an unmatched box outside the region (ignored).
        ('reference', 'v', 0, 1, 1, 1, 9, 9, 0.9), ('reference', 'v', 0, 2, 16, 0, 23, 8, 0.8),
        ('reference', 'v', 0, 3, 40, 0, 48, 8, 0.6),
        # Packed p1 hits g1.
        ('c', 'v', 0, 1, 1, 1, 9, 9, 0.9),
        # p2 is unmatched, with its center (20, 4.5) outside the region, like a tile-sized box over black cells.
        ('c', 'v', 0, 2, 5, 0, 35, 9, 0.5),
        # p3 is shifted so its center (21, 4) leaves the region, yet it matches the retained g2 (IoU 0.65): a true positive.
        ('c', 'v', 0, 3, 17.5, 0, 24.5, 8, 0.7),
        # p4 is unmatched inside tile (0, 0) (IoU 0.25 with g1): a false positive under every rule.
        ('c', 'v', 0, 4, 2, 2, 6, 6, 0.8),
        # p5 matches the non-retained g3: ignored under every rule.
        ('c', 'v', 0, 5, 50, 50, 58, 58, 0.95),
        # p6 is unmatched (IoU 0.45 with g2), centered in tile (0, 1) but reaching x = 22: it fails only 'inside'.
        ('c', 'v', 0, 6, 12, 2, 22, 8, 0.3),
    ])
    out = scored(con)['c']
    # Ranked: p1 TP, p4 FP, p3 TP, p2 FP, p6 FP: precisions 1, 1/2, 2/3, 1/2, 2/5; AP = (1 + 2/3) / 2.
    assert out['center']['reference'] == entry(2, 2, 0, 1.0, 1.0, 1.0, unmatched_outside=1)
    assert out['center']['packed'] == entry(2, 2, 3, 0.4, 1.0, 0.8333, matched_nonretained=1, fp_outside=1)
    # The reference's rule on packed detections drops p2 from the false positives.
    assert out['center']['packed_ignore_outside'] == entry(2, 2, 2, 0.5, 1.0, 0.8333, matched_nonretained=1, unmatched_outside=1)
    assert out['center']['delta'] == dict(packed=dict(precision=-0.6, recall=0.0, ap50=-0.1667),
                                          packed_ignore_outside=dict(precision=-0.5, recall=0.0, ap50=-0.1667))
    # 'inside' drops g2 (it reaches x = 23 > 20), so p3 and the reference's hit on g2 are ignored. p2, centered
    # outside, stays a false positive; p6, centered inside but crossing the boundary, is ignored like the reference's r3.
    assert out['inside']['reference'] == entry(1, 1, 0, 1.0, 1.0, 1.0, matched_nonretained=1, unmatched_outside=1)
    assert out['inside']['packed'] == entry(1, 1, 2, 0.3333, 1.0, 1.0, matched_nonretained=2, unmatched_outside=1, fp_outside=1)
    assert out['inside']['packed_ignore_outside'] == entry(1, 1, 1, 0.5, 1.0, 1.0, matched_nonretained=2, unmatched_outside=2)


def test_center_cover50_and_inside_are_stricter_than_center(con):
    set_configs(con, [('c', 1)])
    # Tile (0, 0), and tiles (0, 2) and (0, 3) (x in [20, 40), y in [0, 10)).
    set_tiles(con, [('c', 'v', 0, 0, 0), ('c', 'v', 0, 0, 2), ('c', 'v', 0, 0, 3)])
    set_boxes(con, [
        # Center (9.5, 9.5) in tile (0, 0), but only 16 of its 49 px inside retained tiles.
        ('v', 0, 1, 6, 6, 13, 13),
        # Center (27, 9) in tile (0, 2) and 60 of its 100 px retained: passes center_cover50, not inside.
        ('v', 0, 2, 22, 4, 32, 14),
    ], [('c', 'v', 0, 1, 6, 6, 13, 13, 0.9), ('c', 'v', 0, 2, 22, 4, 32, 14, 0.8)])
    out = scored(con)['c']
    assert out['center']['packed']['n_gt'] == 2 and out['center']['packed']['recall'] == 1.0
    assert out['center_cover50']['packed']['n_gt'] == 1
    assert out['inside']['packed']['n_gt'] == 0


def test_retained_alternatives_counts_detections_that_could_switch(con):
    set_configs(con, [('c', 1)])
    # Only tile (0, 0) (x < 10) is retained.
    set_tiles(con, [('c', 'v', 0, 0, 0)])
    set_boxes(con, [
        # gB (gid 1) has its center (10, 4) outside, gA (gid 2) its center (6, 4) inside.
        ('v', 0, 1, 6, 0, 14, 8), ('v', 0, 2, 2, 0, 10, 8),
    ], [
        # Equal IoU 0.6 with both: the tie goes to the lower gid, the non-retained gB, while gA was a candidate.
        ('c', 'v', 0, 1, 4, 0, 12, 8, 0.9),
    ])
    out = scored(con, alternatives=True)['c']
    # Under 'center' the detection could have taken the retained gA; under 'frame' both boxes are retained.
    assert out['center']['packed']['matched_nonretained_with_retained_candidate'] == 1
    assert out['center']['packed_ignore_outside']['matched_nonretained_with_retained_candidate'] == 1
    assert out['frame']['packed']['matched_nonretained_with_retained_candidate'] == 0
    assert out['center']['reference']['matched_nonretained_with_retained_candidate'] == 0


def set_score_inputs(con, scopes, gt_rows, det_rows):
    # scopes(cfg, filter), gt_in(cfg, filter, video, frame, gid, retained), and
    # det_in(cfg, filter, side, video, frame, did, score, passes, center_in, gid); center_in defaults to passes.
    con.execute('CREATE OR REPLACE TABLE scopes (cfg VARCHAR, filter VARCHAR)')
    con.executemany('INSERT INTO scopes VALUES (?, ?)', scopes)
    con.execute('CREATE OR REPLACE TABLE gt_in (cfg VARCHAR, filter VARCHAR, video VARCHAR, frame BIGINT, gid BIGINT, retained BOOLEAN)')
    if gt_rows:
        con.executemany('INSERT INTO gt_in VALUES (?, ?, ?, ?, ?, ?)', gt_rows)
    con.execute('CREATE OR REPLACE TABLE det_in (cfg VARCHAR, filter VARCHAR, side VARCHAR, video VARCHAR, frame BIGINT, '
                'did BIGINT, score DOUBLE, passes BOOLEAN, center_in BOOLEAN, gid BIGINT)')
    if det_rows:
        con.executemany('INSERT INTO det_in VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        [r if len(r) == 10 else (*r[:8], r[7], r[8]) for r in det_rows])


def test_score_all_point_ap_and_ignore_rules(con):
    gt = [('k', 'frame', 'v', 0, g, True) for g in (1, 2, 3)] + [('k', 'frame', 'v', 0, 4, False)]
    dets = [
        # Ranked: FP, TP, TP, FP -> precisions 0, 1/2, 2/3, 1/2; interpolated 2/3 at both TPs; AP = (2/3 + 2/3) / 3.
        ('k', 'frame', 'reference', 'v', 0, 1, 0.9, True, None),
        ('k', 'frame', 'reference', 'v', 0, 2, 0.8, True, 1),
        ('k', 'frame', 'reference', 'v', 0, 3, 0.7, True, 2),
        ('k', 'frame', 'reference', 'v', 0, 4, 0.6, True, None),
        # Matched to a non-retained box, and unmatched outside the region: both ignored despite top scores.
        ('k', 'frame', 'reference', 'v', 0, 5, 0.95, True, 4),
        ('k', 'frame', 'reference', 'v', 0, 6, 0.99, False, None),
        # Outside the region but matched to a retained box: a true positive.
        ('k', 'frame', 'reference', 'v', 0, 7, 0.1, False, 3),
    ]
    set_score_inputs(con, [('k', 'frame')], gt, dets)
    rows = {r['side']: r for r in pa.score(con, ['reference', 'packed'])}
    assert (rows['reference']['n_gt'], rows['reference']['tp'], rows['reference']['fp']) == (3, 3, 2)
    assert (rows['reference']['ignored_matched_nonretained'], rows['reference']['ignored_unmatched_outside'],
            rows['reference']['fp_outside_filter']) == (1, 1, 0)
    assert rows['reference']['precision'] == pytest.approx(0.6)
    assert rows['reference']['recall'] == pytest.approx(1.0)
    # The late true positive at rank 5 (precision 3/5) adds 3/5 to the interpolated sum.
    assert rows['reference']['ap50'] == pytest.approx((2 / 3 + 2 / 3 + 3 / 5) / 3)
    # A side without detections: no precision, zero recall and AP.
    assert (rows['packed']['tp'], rows['packed']['fp'], rows['packed']['precision'], rows['packed']['recall'],
            rows['packed']['ap50']) == (0, 0, None, 0.0, 0.0)


def test_score_coco_recall_grid_includes_zero_and_unreached_recall(con):
    gt = [('c', 'frame', 'v', 0, g, True) for g in (1, 2, 3)]
    dets = [('c', 'frame', 'reference', 'v', 0, 1, .9, True, 1)]
    set_score_inputs(con, [('c', 'frame')], gt, dets)
    rows = {r['side']: r for r in pa.score(con, ['reference', 'packed'], coco_recall=True)}
    assert rows['reference']['ap50'] == pytest.approx(34 / 101)
    assert rows['packed']['ap50'] == 0


def test_mean_ap_rematches_at_each_threshold(con):
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10)], [
        ('reference', 'v', 0, 1, 0, 0, 6, 10, .9),
        ('reference', 'v', 0, 2, 0, 0, 10, 10, .8),
        ('c', 'v', 0, 1, 0, 0, 8, 10, .9)])
    set_configs(con, [('c', 1)])
    set_tiles(con, [('c', 'v', 0, 0, 0)])
    pa.box_regions(con, 10)
    rows = {(r['filter'], r['side']): r for r in pa.mean_ap(con)}
    assert rows['center', 'reference']['map'] == pytest.approx(.65)
    assert rows['center', 'packed']['map'] == pytest.approx(.7)
    assert rows['center', 'reference']['thresholds'][3] == dict(iou=.65, ap=.5)
    assert len(rows['center', 'packed']['thresholds']) == 10


def test_mean_ap_without_ground_truth(con):
    set_boxes(con, [], [('reference', 'v', 0, 1, 0, 0, 10, 10, .9)])
    set_configs(con, [('c', 1)])
    set_tiles(con, [('c', 'v', 0, 0, 0)])
    pa.box_regions(con, 10)
    assert all(r['map'] is None for r in pa.mean_ap(con))


def test_mean_ap_matches_official_coco_evaluator(con):
    coco = pytest.importorskip('pycocotools.coco')
    cocoeval = pytest.importorskip('pycocotools.cocoeval')
    gt = [('v', 0, 1, 0, 0, 10, 10), ('v', 0, 2, 20, 20, 30, 30), ('v', 1, 3, 0, 0, 10, 10)]
    detections = [('reference', 'v', 0, 1, 0, 0, 6, 10, .9),
                  ('reference', 'v', 0, 2, 0, 0, 10, 10, .8),
                  ('reference', 'v', 0, 3, 20, 20, 30, 30, .7)]
    set_boxes(con, gt, detections)
    set_configs(con, [('c', 1)])
    set_tiles(con, [('c', 'v', frame, tile, tile) for frame in (0, 1) for tile in (0, 2)])
    pa.box_regions(con, 10)
    actual = next(r['map'] for r in pa.mean_ap(con) if r['filter'] == 'center' and r['side'] == 'reference')
    truth = coco.COCO()
    truth.dataset = dict(info={}, images=[dict(id=frame + 1) for frame in (0, 1)], categories=[dict(id=1)],
        annotations=[dict(id=gid, image_id=frame + 1, category_id=1, bbox=[x1, y1, x2-x1, y2-y1],
                          area=(x2-x1)*(y2-y1), iscrowd=0) for _, frame, gid, x1, y1, x2, y2 in gt])
    truth.createIndex()
    predicted = truth.loadRes([dict(image_id=frame + 1, category_id=1, bbox=[x1, y1, x2-x1, y2-y1], score=confidence)
                               for _, _, frame, _, x1, y1, x2, y2, confidence in detections])
    evaluator = cocoeval.COCOeval(truth, predicted, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    assert actual == pytest.approx(evaluator.eval['precision'][:, :, 0, 0, 2].mean())


def test_score_without_ground_truth_and_with_score_ties(con):
    set_score_inputs(con, [('z', 'frame'), ('t', 'frame')], [('t', 'frame', 'v', 0, 1, True)], [
        # No ground truth: one false positive gives precision 0 and undefined recall and AP.
        ('z', 'frame', 'reference', 'v', 0, 1, 0.5, True, None),
        # Equal scores rank by position: the false positive (did 1) precedes the true positive (did 2).
        ('t', 'frame', 'reference', 'v', 0, 1, 0.5, True, None),
        ('t', 'frame', 'reference', 'v', 0, 2, 0.5, True, 1),
    ])
    rows = {r['cfg']: r for r in pa.score(con, ['reference'])}
    assert (rows['z']['precision'], rows['z']['recall'], rows['z']['ap50']) == (0.0, None, None)
    assert rows['t']['ap50'] == pytest.approx(0.5)


def oracle_score(scopes, gt_rows, det_rows, side):
    # Plain-Python scores of one SIDES entry per scope, as the oracle for score().
    source, center_outside_fp = pa.SIDES[side]
    out = {}
    for cfg, flt in scopes:
        retained = {(v, f, g): r for c, fl, v, f, g, r in gt_rows if (c, fl) == (cfg, flt)}
        n_gt = sum(retained.values())
        labeled, tallies = [], dict(ignored_matched_nonretained=0, ignored_unmatched_outside=0, fp_outside_filter=0)
        for c, fl, s, v, f, did, sc, passes, center_in, gid in det_rows:
            if (c, fl, s) != (cfg, flt, source):
                continue
            if gid is not None:
                if retained.get((v, f, gid)):
                    labeled.append((sc, v, f, did, 1))
                else:
                    tallies['ignored_matched_nonretained'] += 1
            elif passes or (center_outside_fp and not center_in):
                labeled.append((sc, v, f, did, 0))
                tallies['fp_outside_filter'] += not passes
            else:
                tallies['ignored_unmatched_outside'] += 1
        # Confidence order, ties by position.
        labeled.sort(key=lambda d: (-d[0], d[1], d[2], d[3]))
        flags = [d[4] for d in labeled]
        precisions = [sum(flags[:i + 1]) / (i + 1) for i in range(len(flags))]
        envelope = [max(precisions[i:]) for i in range(len(flags))]
        tp = sum(flags)
        out[(cfg, flt)] = dict(n_gt=n_gt, tp=tp, fp=len(flags) - tp, precision=tp / len(flags) if flags else None,
                               recall=tp / n_gt if n_gt else None,
                               ap50=sum(e for e, t in zip(envelope, flags) if t) / n_gt if n_gt else None, **tallies)
    return out


def test_score_equals_python_oracle_on_random_scopes(con):
    # Random scopes with ties, ignored boxes, and unmatched detections inside and outside the region.
    rng = random.Random(1)
    scopes = [(f'c{i}', flt) for i in range(6) for flt in ('frame', 'center')]
    gt_rows, det_rows = [], []
    for cfg, flt in scopes:
        boxes = [(v, f, g) for v in ('a', 'b') for f in range(rng.randint(0, 4)) for g in range(rng.randint(0, 3))]
        gt_rows += [(cfg, flt, v, f, g, rng.random() < 0.7) for v, f, g in boxes]
        for side in ('reference', 'packed', 'rerun', 'jpeg'):
            for did in range(rng.randint(0, 25)):
                # Matched to an existing box (sometimes one missing from gt_in), or unmatched.
                gid = rng.choice(boxes)[2] if boxes and rng.random() < 0.6 else (99 if rng.random() < 0.05 else None)
                v, f = rng.choice(boxes)[:2] if boxes else ('a', 0)
                # A passing box always has its center retained; a failing one may or may not.
                passes = rng.random() < 0.6
                det_rows.append((cfg, flt, side, v, f, did, round(rng.random(), 1), passes, passes or rng.random() < 0.5, gid))
    set_score_inputs(con, scopes, gt_rows, det_rows)
    rows = pa.score(con, list(pa.SIDES))
    for side in pa.SIDES:
        expected = oracle_score(scopes, gt_rows, det_rows, side)
        got = {(r['cfg'], r['filter']): r for r in rows if r['side'] == side}
        assert set(got) == set(expected)
        for key, e in expected.items():
            for k, v in e.items():
                assert got[key][k] == (None if v is None else pytest.approx(v)), (side, key, k)


def test_all_frame_inputs_drop_ground_truth_beyond_clip(con):
    # The clip has frames 0 and 1; a ground-truth box on frame 5 is not part of it.
    con.execute('CREATE TABLE frames AS SELECT * FROM (VALUES (\'v\', 0), (\'v\', 1)) t(video, frame)')
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10), ('v', 5, 1, 0, 0, 10, 10)],
              [('reference', 'v', 0, 1, 0, 0, 10, 10, 0.9), ('reference', 'v', 1, 1, 0, 0, 10, 10, 0.8)])
    pa.greedy_match(con)
    pa.build_all_frame_inputs(con)
    r = pa.score(con, ['reference'])[0]
    assert (r['cfg'], r['n_gt'], r['tp'], r['fp'], r['recall']) == (pa.ALL_FRAMES, 1, 1, 1, 1.0)


def test_arrange_reports_float_noise_as_zero():
    # Two equal metrics summed in different orders can differ by float noise; the delta reads 0.0, not -0.0.
    rows = [dict(cfg='c', filter='center', side='reference', n_gt=1, tp=1, fp=0, ignored_matched_nonretained=0,
                 ignored_unmatched_outside=0, fp_outside_filter=0, precision=1.0, recall=1.0, ap50=1.0)]
    out = pa.arrange(rows, [dict(cfg='c', filter='center', side='rerun', precision=0.0, recall=-1e-17, ap50=None)])
    delta = out['c']['center']['delta']['rerun']
    assert delta == dict(precision=0.0, recall=0.0, ap50=None)
    assert str(delta['recall']) == '0.0'


def test_sweep_summary_per_padding_and_overall(con):
    # Three detection configs: two unpadded, one padded.
    set_configs(con, [('a', 1, 'none'), ('b', 1, 'none'), ('c', 1, 'bl')])
    con.execute('CREATE TABLE deltas (cfg VARCHAR, filter VARCHAR, side VARCHAR, precision DOUBLE, recall DOUBLE, ap50 DOUBLE)')
    con.executemany('INSERT INTO deltas VALUES (?, ?, ?, ?, ?, ?)', [
        ('a', 'center', 'packed', -0.1, -0.2, -0.3), ('b', 'center', 'packed', -0.3, -0.4, None), ('c', 'center', 'packed', 0.1, 0.0, 0.0),
    ])
    rows = {(r['tilepadding'], r['metric']): r for r in pa.sweep_summary(con)}
    # Medians, minima, and maxima per padding mode and over all configs.
    assert rows[('none', 'recall')] == dict(tilepadding='none', filter='center', side='packed', metric='recall',
                                            detection_configs=2, median=-0.3, min=-0.4, max=-0.2)
    assert rows[('all', 'precision')]['median'] == -0.1 and rows[('all', 'precision')]['detection_configs'] == 3
    # An undefined delta is left out rather than counted.
    assert rows[('none', 'ap50')]['detection_configs'] == 1 and rows[('all', 'ap50')]['detection_configs'] == 2
    # A float-noise delta reads 0.0, not -0.0.
    con.execute("UPDATE deltas SET precision = -1e-17 WHERE cfg = 'c'")
    rows = {(r['tilepadding'], r['metric']): r for r in pa.sweep_summary(con)}
    assert str(rows[('bl', 'precision')]['median']) == '0.0'


def test_operating_config_picks_fastest_within_bound():
    # The GPU job's entry point to the shared selection rule (common.fastest_tetris).
    naive = pd.Series(dict(throughput_fps=10.0, HOTA_HOTA=0.8))
    tetris = pd.DataFrame(dict(variant_id=['a', 'b', 'c'], throughput_fps=[500.0, 300.0, 100.0], HOTA_HOTA=[0.70, 0.77, 0.79]))
    assert pa.operating_config(tetris, naive, 0.05)['config'] == 'b'
    # The bound is inclusive: exactly 0.76 qualifies.
    tetris.loc[0, 'HOTA_HOTA'] = 0.76
    assert pa.operating_config(tetris, naive, 0.05)['config'] == 'a'
    # No configuration within 1% of the reference.
    with pytest.raises(AssertionError, match='no Tetris configuration'):
        pa.operating_config(tetris, naive, 0.01)


def write_video(path, width, height, frames):
    # A small black mp4 clip.
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_video_grid_reads_size_and_rejects_unreadable(tmp_path, monkeypatch):
    # Datasets live under the patched dataset root.
    monkeypatch.setattr(store, 'DATASETS_DIR', str(tmp_path))
    write_video(tmp_path / 'd' / 'test' / 'te00.mp4', 34, 26, 2)
    # Whole tiles only: 26 // 10 rows, 34 // 10 columns.
    assert pa.video_grid('d', 'te00.mp4', 10) == (2, 3)
    # A file that is not a video has no size.
    (tmp_path / 'd' / 'test' / 'bad.mp4').write_text('not a video')
    with pytest.raises(AssertionError):
        pa.video_grid('d', 'bad.mp4', 10)


@pytest.fixture
def checked(con):
    # Tables check_inputs reads: a 4-frame clip, configuration c at sample rate 2 on an 8x12 grid.
    con.execute('CREATE TABLE frames AS SELECT \'v\' AS video, range AS frame FROM range(4)')
    con.execute('CREATE TABLE packed_dets_lines AS SELECT \'c\' AS source, \'v\' AS video, range AS frame FROM range(4)')
    set_configs(con, [('c', 2)])
    set_tiles(con, [('c', 'v', 0, 0, 0), ('c', 'v', 2, 7, 11)])
    set_boxes(con, [('v', 0, 1, 0, 0, 10, 10), ('v', 9, 1, 0, 0, 10, 10), ('v', 0, 2, 5, 5, 5, 9)],
              [('reference', 'v', 1, 1, 0, 0, 10, 10, 0.9), ('c', 'v', 2, 1, 0, 0, 10, 10, 0.9)])
    pa.sample_frames(con)
    return con


GRIDS = {'v': (8, 12)}


def test_check_inputs_counts(checked):
    counts = pa.check_inputs(checked, 'd', GRIDS, 4)
    assert counts == dict(gt_boxes_outside_clip_frames=1, gt_boxes_zero_area=1, detections_zero_area=0, gt_boxes=2,
                          tile_grids='8x12')


def test_sample_frames_adds_last_frame(checked):
    # Sample rate 2 on frames 0-3: 0 and 2, plus the last frame 3 that p020 always classifies.
    assert checked.execute("SELECT frame FROM sampled_frames ORDER BY frame").fetchall() == [(0,), (2,), (3,)]
    # So a retained tile on frame 3 passes the checks.
    checked.execute("INSERT INTO tiles VALUES ('c', 'v', 3, 0, 0)")
    pa.check_inputs(checked, 'd', GRIDS, 4)


def test_sample_frames_per_video_and_rate(con):
    # Two clips of 5 and 7 frames, two configurations at rates 3 and 4.
    con.execute("CREATE TABLE frames AS SELECT 'a' AS video, range AS frame FROM range(5) UNION ALL SELECT 'b', range FROM range(7)")
    set_configs(con, [('r3', 3), ('r4', 4)])
    pa.sample_frames(con)
    got = {}
    for cfg, video, frame in con.execute('SELECT cfg, video, frame FROM sampled_frames ORDER BY ALL').fetchall():
        got.setdefault((cfg, video), []).append(frame)
    # Each clip adds its own last frame (4 for a, 6 for b), once even when it is already a multiple.
    assert got == {('r3', 'a'): [0, 3, 4], ('r3', 'b'): [0, 3, 6], ('r4', 'a'): [0, 4], ('r4', 'b'): [0, 4, 6]}


def test_check_inputs_rejects_frame_count_mismatch(checked):
    # The tradeoff table counts a different number of frames.
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', GRIDS, 5)


@pytest.mark.parametrize('tile', [('c', 'v', 1, 0, 0), ('c', 'v', 2, 8, 0), ('c', 'v', 2, 0, -1), ('c', 'v', 6, 0, 0),
                                  ('c', 'w', 0, 0, 0)])
def test_check_inputs_rejects_bad_tiles(checked, tile):
    # Off the sampled frames, outside the grid, negative, beyond the clip, or in a video without a grid.
    checked.execute('INSERT INTO tiles VALUES (?, ?, ?, ?, ?)', list(tile))
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', GRIDS, 4)


def test_check_inputs_uses_each_videos_grid(checked):
    # A second video with a smaller grid: a tile valid in v's grid is outside w's.
    checked.execute("INSERT INTO frames SELECT 'w', range FROM range(2)")
    checked.execute("INSERT INTO packed_dets_lines SELECT 'c', 'w', range FROM range(2)")
    pa.sample_frames(checked)
    checked.execute("INSERT INTO tiles VALUES ('c', 'w', 0, 7, 11)")
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', {'v': (8, 12), 'w': (4, 6)}, 6)
    # With w's grid as large as v's, the same tile is fine.
    pa.check_inputs(checked, 'd', {'v': (8, 12), 'w': (8, 12)}, 6)


def test_check_inputs_rejects_stray_packed_detection(checked):
    # A packed detection on a frame its configuration never packed.
    checked.execute("INSERT INTO dets VALUES ('c', 'v', 3, 1, 0, 0, 10, 10, 0.9)")
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', GRIDS, 4)


@pytest.mark.parametrize('change', [
    # A missing frame in the middle of the clip.
    'DELETE FROM frames WHERE frame = 2',
    # A gap plus a repeated frame, which keeps the count at 4.
    "DELETE FROM frames WHERE frame = 2; INSERT INTO frames VALUES ('v', 1)",
    # A clip that does not start at frame 0.
    'UPDATE frames SET frame = frame + 1',
])
def test_check_inputs_rejects_bad_clip_frames(checked, change):
    checked.execute(change)
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', GRIDS, 4)


def test_check_inputs_rejects_video_without_frames(checked):
    # A test video whose reference file had no lines.
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', {'v': (8, 12), 'w': (8, 12)}, 4)


@pytest.mark.parametrize('change', [
    # A truncated packed file, and a packed line beyond the clip.
    'DELETE FROM packed_dets_lines WHERE frame = 3',
    "INSERT INTO packed_dets_lines VALUES ('c', 'v', 4)",
])
def test_check_inputs_rejects_bad_packed_lines(checked, change):
    checked.execute(change)
    with pytest.raises(AssertionError):
        pa.check_inputs(checked, 'd', GRIDS, 4)


# A fake cache for score_dataset and main: tile size 10, two clips of 30x20 px (2x3 tiles).
DC_A = 'ShuffleNet05_10_1_r025_none_s100'
DC_B = 'ShuffleNet05_10_2_040_r025_bl_s100_sortcython'


def fake_tradeoff():
    # Two unpruned tracker variants sharing DC_A (the bytetrack one is the operating point) and one pruned, padded DC_B.
    base = dict(classifier='ShuffleNet05', tilesize=10.0, canvas_scale=1.0, relevance_threshold=0.25)
    tetris = pd.DataFrame([
        dict(base, sample_rate=1.0, tracking_accuracy_threshold=float('nan'), tilepadding='none', tracker='sortcython',
             variant_id=f'{DC_A}_sortcython', throughput_fps=100.0, HOTA_HOTA=0.79),
        dict(base, sample_rate=1.0, tracking_accuracy_threshold=float('nan'), tilepadding='none', tracker='bytetrackcython',
             variant_id=f'{DC_A}_bytetrackcython', throughput_fps=200.0, HOTA_HOTA=0.78),
        dict(base, sample_rate=2.0, tracking_accuracy_threshold=0.4, tilepadding='bl', tracker='sortcython',
             variant_id=DC_B, throughput_fps=500.0, HOTA_HOTA=0.70),
    ])
    return tetris, pd.Series(dict(throughput_fps=10.0, HOTA_HOTA=0.8, frame_count=7))


def build_fake_cache(root, dataset, gt_dataset):
    # Videos, ground truth, reference detections, canvases, and unpacked detections of one dataset.
    for video, frames in (('te00.mp4', 4), ('te01.mp4', 3)):
        write_video(root / 'data' / dataset / 'test' / video, 30, 20, frames)
    ex, gx = root / 'cache' / dataset / 'execution', root / 'cache' / gt_dataset / 'execution'
    # Ground truth: te00 frames 0, 2, 3 and te01 frame 1.
    write_jsonl(gx / 'te00.mp4' / '003_groundtruth' / 'tracking.jsonl', 'tracks',
                [(0, [[1, 1, 1, 9, 9]]), (1, []), (2, [[1, 11, 1, 19, 9]]), (3, [[2, 21, 11, 29, 19]])])
    write_jsonl(gx / 'te01.mp4' / '003_groundtruth' / 'tracking.jsonl', 'tracks', [(1, [[5, 1, 1, 9, 9]])])
    # Reference: every ground-truth box, plus a false positive in tile (1, 0) of te00 frame 3.
    write_jsonl(ex / 'te00.mp4' / '002_naive' / 'detection.jsonl', 'detections',
                [(0, [[1, 1, 9, 9, 0.9]]), (1, []), (2, [[11, 1, 19, 9, 0.8]]), (3, [[21, 11, 29, 19, 0.7], [1, 11, 9, 19, 0.6]])])
    write_jsonl(ex / 'te01.mp4' / '002_naive' / 'detection.jsonl', 'detections', [(0, []), (1, [[1, 1, 9, 9, 0.9]]), (2, [])])
    # DC_A packs te00 frame 0 tile (0, 0), frame 2 tile (0, 1), frame 3 tile (1, 2), and te01 frame 1 tile (0, 0).
    write_canvas(ex / 'te00.mp4' / '033_compressed_frames' / DC_A, '0000_0000_0002', [[1, 2, 0], [0, 0, 0]],
                 [[[0, 0], [0, 0], 0], [[0, 1], [0, 1], 2]])
    write_canvas(ex / 'te00.mp4' / '033_compressed_frames' / DC_A, '0001_0003_0003', [[1, 0, 0], [0, 0, 0]],
                 [[[0, 0], [1, 2], 3]])
    write_canvas(ex / 'te01.mp4' / '033_compressed_frames' / DC_A, '0000_0001_0001', [[1, 0, 0], [0, 0, 0]],
                 [[[0, 0], [0, 0], 1]])
    # DC_A misses te00 frame 2 and adds a false positive centered in tile (1, 1), outside its region.
    write_jsonl(ex / 'te00.mp4' / '050_uncompressed_detections' / DC_A / 'detections.jsonl', 'bboxes',
                [(0, [[1, 1, 9, 9, 0.9], [10, 10, 20, 20, 0.4]]), (1, []), (2, []), (3, [[21, 11, 29, 19, 0.7]])])
    write_jsonl(ex / 'te01.mp4' / '050_uncompressed_detections' / DC_A / 'detections.jsonl', 'bboxes',
                [(0, []), (1, [[1, 1, 9, 9, 0.5]]), (2, [])])
    # DC_B (sample rate 2, padded) shows te00 frame 0 tile (0, 0) in two groups, and frame 2 tile (0, 1); nothing in te01.
    write_canvas(ex / 'te00.mp4' / '033_compressed_frames' / DC_B, '0000_0000_0002', [[1, 2, 3], [0, 0, 0]],
                 [[[0, 0], [0, 0], 0], [[0, 1], [0, 0], 0], [[0, 2], [0, 1], 2]])
    (ex / 'te01.mp4' / '033_compressed_frames' / DC_B / 'index_maps').mkdir(parents=True)
    write_jsonl(ex / 'te00.mp4' / '050_uncompressed_detections' / DC_B / 'detections.jsonl', 'bboxes',
                [(0, [[1, 1, 9, 9, 0.95]]), (1, []), (2, [[11, 1, 19, 9, 0.85]]), (3, [])])
    write_jsonl(ex / 'te01.mp4' / '050_uncompressed_detections' / DC_B / 'detections.jsonl', 'bboxes', [(0, []), (1, []), (2, [])])


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    # Cache and dataset roots under tmp_path, the fake tradeoff, and results in tmp_path/out.
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(store, 'DATASETS_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(pa, 'load_test_tradeoff', lambda dataset: fake_tradeoff())
    monkeypatch.setattr(pa, 'results_dir', lambda script: tmp_path / 'out')
    (tmp_path / 'out').mkdir()
    # caldot1-y05 reads caldot1's ground truth; jnc0 its own.
    build_fake_cache(tmp_path, 'caldot1-y05', 'caldot1')
    build_fake_cache(tmp_path, 'jnc0', 'jnc0')
    return tmp_path


def test_score_dataset_on_fake_cache(fake_cache):
    s = pa.score_dataset('caldot1-y05', 0.05)
    # Inputs and checks.
    assert (s['provisional'], s['ground_truth_dataset'], s['videos'], s['frames'], s['tile_size']) == (False, 'caldot1', 2, 7, 10)
    assert (s['gt_boxes'], s['tile_grids'], s['gt_boxes_outside_clip_frames']) == (4, '2x3', 0)
    # Reference on every clip frame: 4 hits and the false positive.
    assert s['reference']['all_frames'] == entry(4, 4, 1, 0.8, 1.0, 1.0)
    # The operating point is the faster tracker variant of DC_A.
    assert s['operating_point'] == dict(config=f'{DC_A}_bytetrackcython', detection_config=DC_A, hota_bound=0.76, loss=0.05)
    assert [(c['config'], c['detection_config'], c['operating_point']) for c in s['configs']] == [
        (f'{DC_A}_bytetrackcython', DC_A, True), (f'{DC_A}_sortcython', DC_A, False), (DC_B, DC_B, False)]
    # One score set per detection config, listing the variants that share it.
    a, b = s['detection_configs'][DC_A], s['detection_configs'][DC_B]
    assert set(s['detection_configs']) == {DC_A, DC_B}
    assert (a['variants'], a['tracker'], a['tracking_accuracy_threshold'], a['tilepadding']) == \
        ([f'{DC_A}_bytetrackcython', f'{DC_A}_sortcython'], None, None, 'none')
    assert (b['variants'], b['tracker'], b['tracking_accuracy_threshold'], b['tilepadding']) == ([DC_B], 'sortcython', 0.4, 'bl')
    # Frames: DC_A samples all 7 and packs 4; DC_B samples te00 0, 2, 3 and te01 0, 2, and packs 2 (the padded tile once).
    assert (a['frames_sampled'], a['frames_detected'], a['retained_tiles']) == (7, 4, 4)
    assert (b['frames_sampled'], b['frames_detected'], b['retained_tiles']) == (5, 2, 2)
    # DC_A under 'center': the reference's false positive in tile (1, 0) lies outside the region and is ignored;
    # the packed side misses te00 frame 2 and its outside false positive counts.
    center = a['scores']['center']
    assert center['reference'] == dict(entry(4, 4, 0, 1.0, 1.0, 1.0, unmatched_outside=1), matched_nonretained_with_retained_candidate=0)
    assert center['packed'] == dict(entry(4, 3, 1, 0.75, 0.75, 0.75, fp_outside=1), matched_nonretained_with_retained_candidate=0)
    assert center['packed_ignore_outside'] == dict(entry(4, 3, 0, 1.0, 0.75, 0.75, unmatched_outside=1),
                                                   matched_nonretained_with_retained_candidate=0)
    assert center['delta'] == dict(packed=dict(precision=-0.25, recall=-0.25, ap50=-0.25),
                                   packed_ignore_outside=dict(precision=0.0, recall=-0.25, ap50=-0.25))
    # No region filter: the reference's false positive counts too.
    assert a['scores']['frame']['reference'] == dict(entry(4, 4, 1, 0.8, 1.0, 1.0), matched_nonretained_with_retained_candidate=0)
    # DC_B finds both retained boxes on its two frames.
    assert b['scores']['center']['packed'] == dict(entry(2, 2, 0, 1.0, 1.0, 1.0), matched_nonretained_with_retained_candidate=0)
    # The sweep counts DC_A once although two variants share it.
    sweep = {(r['tilepadding'], r['filter'], r['side'], r['metric']): r for r in s['sweep']}
    assert sweep[('all', 'center', 'packed', 'recall')]['detection_configs'] == 2
    assert sweep[('none', 'center', 'packed', 'recall')] == dict(tilepadding='none', filter='center', side='packed', metric='recall',
                                                                 detection_configs=1, median=-0.25, min=-0.25, max=-0.25)
    # No JPEG control was asked for.
    assert s['jpeg_control'] is None


# The JPEG control at DC_A's frames (te00 0, 2, 3; te01 1): 'plain' repeats the reference, 'jpeg' misses te00 frame 2.
CONTROL = {'te00.mp4': [(0, [[1, 1, 9, 9, 0.9]], [[1, 1, 9, 9, 0.9]]), (2, [[11, 1, 19, 9, 0.8]], []),
                        (3, [[21, 11, 29, 19, 0.7], [1, 11, 9, 19, 0.6]], [[21, 11, 29, 19, 0.7], [1, 11, 9, 19, 0.6]])],
           'te01.mp4': [(1, [[1, 1, 9, 9, 0.9]], [[1, 1, 9, 9, 0.9]])]}


def write_control(root, dataset, detection_config=DC_A, control=CONTROL):
    # The GPU job's files for one dataset under root/out/jpeg_control, and its manifest record.
    directory = root / 'out' / pa.CONTROL_DIR
    (directory / dataset).mkdir(parents=True, exist_ok=True)
    for video, lines in control.items():
        with gzip.open(directory / dataset / f'{video}.jsonl.gz', 'wt') as f:
            f.write(''.join(json.dumps(dict(frame_idx=i, plain=p, jpeg=j)) + '\n' for i, p, j in lines))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(datasets={})
    manifest['datasets'][dataset] = dict(detection_config=detection_config, frames=sum(map(len, control.values())),
                                         environment=dict(host='fake'), seconds=1.0)
    manifest_path.write_text(json.dumps(manifest))
    return directory, manifest['datasets'][dataset]


def test_score_dataset_with_jpeg_control(fake_cache):
    directory, record = write_control(fake_cache, 'caldot1-y05')
    s = pa.score_dataset('caldot1-y05', 0.05, directory, record)
    control = s['jpeg_control']
    assert (control['detection_config'], control['frames'], control['rerun_detections'], control['jpeg_detections']) == (DC_A, 4, 5, 4)
    center = control['scores']['center']
    # The rerun repeats the reference exactly; JPEG loses te00 frame 2; packing as a whole loses the same box.
    assert center['rerun'] == center['reference'] == dict(entry(4, 4, 0, 1.0, 1.0, 1.0, unmatched_outside=1),
                                                          matched_nonretained_with_retained_candidate=0)
    assert center['jpeg'] == dict(entry(4, 3, 0, 1.0, 0.75, 0.75, unmatched_outside=1), matched_nonretained_with_retained_candidate=0)
    # The control's packed side is the sweep's packed side at the operating point.
    assert center['packed'] == s['detection_configs'][DC_A]['scores']['center']['packed']
    assert center['delta'] == dict(
        packed=dict(precision=-0.25, recall=-0.25, ap50=-0.25), packed_ignore_outside=dict(precision=0.0, recall=-0.25, ap50=-0.25),
        rerun=dict(precision=0.0, recall=0.0, ap50=0.0), jpeg=dict(precision=0.0, recall=-0.25, ap50=-0.25),
        jpeg_vs_rerun=dict(precision=0.0, recall=-0.25, ap50=-0.25), packed_vs_jpeg=dict(precision=-0.25, recall=0.0, ap50=0.0))


@pytest.mark.parametrize('change', ['other config', 'missing frame', 'extra frame', 'missing record', 'missing file'])
def test_score_dataset_rejects_mismatched_jpeg_control(fake_cache, change):
    # A control run on another configuration, on other frames, or not at all for this dataset or video.
    control = dict(CONTROL)
    if change == 'missing frame':
        control['te00.mp4'] = CONTROL['te00.mp4'][:2]
    if change == 'extra frame':
        control['te01.mp4'] = CONTROL['te01.mp4'] + [(2, [], [])]
    directory, record = write_control(fake_cache, 'caldot1-y05', DC_B if change == 'other config' else DC_A, control)
    if change == 'missing file':
        (directory / 'caldot1-y05' / 'te01.mp4.jsonl.gz').unlink()
    with pytest.raises(AssertionError):
        pa.score_dataset('caldot1-y05', 0.05, directory, None if change == 'missing record' else record)


def test_score_dataset_fails_on_missing_packed_file(fake_cache):
    # A configuration whose 050 output is gone must not silently drop out.
    (fake_cache / 'cache' / 'caldot1-y05' / 'execution' / 'te01.mp4' / '050_uncompressed_detections' / DC_B / 'detections.jsonl').unlink()
    with pytest.raises(AssertionError):
        pa.score_dataset('caldot1-y05', 0.05)


def test_score_dataset_fails_without_configurations(fake_cache, monkeypatch):
    # An empty validated set is reported as such.
    monkeypatch.setattr(pa, 'load_test_tradeoff', lambda dataset: (fake_tradeoff()[0].iloc[:0], fake_tradeoff()[1]))
    with pytest.raises(AssertionError, match='no validated'):
        pa.score_dataset('caldot1-y05', 0.05)


def test_main_writes_both_reports(fake_cache, monkeypatch):
    # The JPEG control for both datasets.
    write_control(fake_cache, 'caldot1-y05')
    write_control(fake_cache, 'jnc0')
    monkeypatch.setattr(pa, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05', 'jnc0'], loss=0.05,
                                                                     skip_jpeg_control=False))
    pa.main()
    configs = json.loads((fake_cache / 'out' / 'configs.json').read_text())
    points = json.loads((fake_cache / 'out' / 'operating_points.json').read_text())
    # Both datasets under their labels; only B3D is provisional.
    assert set(configs['datasets']) == set(points['datasets']) == {'CalDoT1', 'B3D1'}
    assert (points['datasets']['CalDoT1']['provisional'], points['datasets']['B3D1']['provisional']) == (False, True)
    # The operating point carries its detection config's scores and frame counts.
    op = points['datasets']['CalDoT1']
    assert (op['config'], op['detection_config'], op['fps'], op['hota'], op['hota_bound']) == (f'{DC_A}_bytetrackcython', DC_A, 200.0, 0.78, 0.76)
    assert op['scores'] == configs['datasets']['CalDoT1']['detection_configs'][DC_A]['scores']
    assert (op['frames_detected'], op['retained_tiles'], op['tilepadding']) == (4, 4, 'none')
    assert set(op['scores']) == set(pa.FILTERS)
    # The JPEG control appears in both reports.
    assert op['jpeg_control'] == configs['datasets']['CalDoT1']['jpeg_control']
    assert op['jpeg_control']['scores']['center']['delta']['jpeg_vs_rerun']['recall'] == -0.25
    # Both files record the method.
    assert configs['method'] == points['method'] and configs['method']['sampling'].startswith('none')


def test_main_without_control_fails_unless_skipped(fake_cache, monkeypatch):
    # No manifest: a clear error naming the GPU job.
    monkeypatch.setattr(pa, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05'], loss=0.05, skip_jpeg_control=False))
    with pytest.raises(AssertionError, match='measurements-gpu'):
        pa.main()
    # Skipped: the reports carry no control.
    monkeypatch.setattr(pa, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05'], loss=0.05, skip_jpeg_control=True))
    pa.main()
    points = json.loads((fake_cache / 'out' / 'operating_points.json').read_text())
    assert points['datasets']['CalDoT1']['jpeg_control'] is None


def test_main_rejects_bad_loss(fake_cache, monkeypatch):
    # A loss of 1 or more, or below 0, defines no bound.
    for loss in (1.0, -0.1):
        monkeypatch.setattr(pa, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05'], loss=loss, skip_jpeg_control=True))
        with pytest.raises(AssertionError):
            pa.main()


# The GPU job, loaded from its file (measurements-gpu is not a package).
GPU_PATH = Path(__file__).resolve().parent.parent / 'measurements-gpu' / 'r1o4_packing_detection_accuracy.py'
gpu_spec = importlib.util.spec_from_file_location('r1o4_gpu', GPU_PATH)
gpu = importlib.util.module_from_spec(gpu_spec)
gpu_spec.loader.exec_module(gpu)


def test_jpeg_round_trip_is_lossy_but_close():
    # A smooth gradient survives JPEG nearly unchanged, with the same shape and type.
    frame = np.tile(np.arange(64, dtype=np.uint8)[None, :, None] * 4, (48, 1, 3))
    out = gpu.jpeg_round_trip(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - frame.astype(int)).max() <= 8


def test_gpu_operating_frames_match_the_measurement(fake_cache, monkeypatch):
    # The GPU job picks the measurement's operating point and DC_A's detected frames.
    monkeypatch.setattr(gpu, 'load_test_tradeoff', lambda dataset: fake_tradeoff())
    row, cfg, frames = gpu.operating_frames('caldot1-y05', 0.05, ['te00.mp4', 'te01.mp4'])
    assert (row['variant_id'], cfg, frames) == (f'{DC_A}_bytetrackcython', DC_A, {'te00.mp4': [0, 2, 3], 'te01.mp4': [1]})


@pytest.fixture
def fake_detector(fake_cache, monkeypatch):
    # A detector that sees a marker for JPEG frames, no GPU, and results under the fake cache's out/.
    monkeypatch.setattr(gpu, 'load_test_tradeoff', lambda dataset: fake_tradeoff())
    monkeypatch.setattr(gpu, 'results_dir', lambda script: fake_cache / 'out')
    monkeypatch.setattr(gpu, 'warm_up', lambda dataset, video, detector: None)
    monkeypatch.setattr(gpu, 'environment', lambda g: dict(host='fake', gpu=g))
    monkeypatch.setattr(gpu, 'jpeg_round_trip', lambda frame: None)
    monkeypatch.setattr(gpu.polyis.models.detector, 'get_detector', lambda dataset, g, batch_size: 'detector')
    monkeypatch.setattr(gpu.polyis.models.detector, 'delete', lambda detector: None)
    monkeypatch.setattr(gpu.polyis.models.detector, 'detect', lambda image, detector:
                        np.array([[1, 1, 9, 9, 0.9]]) if image is not None else np.zeros((0, 5)))
    return fake_cache


def test_gpu_detect_video_writes_only_wanted_frames(fake_detector):
    path = fake_detector / 'te00.jsonl.gz'
    gpu.detect_video('caldot1-y05', 'te00.mp4', [0, 3], 'detector', path)
    with gzip.open(path, 'rt') as f:
        lines = [json.loads(l) for l in f]
    # One line per wanted frame, with both runs.
    assert lines == [dict(frame_idx=0, plain=[[1, 1, 9, 9, 0.9]], jpeg=[]), dict(frame_idx=3, plain=[[1, 1, 9, 9, 0.9]], jpeg=[])]
    # No wanted frame: an empty file.
    gpu.detect_video('caldot1-y05', 'te00.mp4', [], 'detector', path)
    with gzip.open(path, 'rt') as f:
        assert f.read() == ''
    # A frame beyond the clip.
    with pytest.raises(AssertionError):
        gpu.detect_video('caldot1-y05', 'te00.mp4', [5], 'detector', path)


def test_gpu_job_feeds_the_measurement(fake_detector, monkeypatch):
    # The GPU job on CalDoT1, then the measurement on its outputs.
    monkeypatch.setattr(gpu, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05'], loss=0.05, gpu=0))
    gpu.main()
    manifest = json.loads((fake_detector / 'out' / pa.CONTROL_DIR / 'manifest.json').read_text())
    assert manifest['datasets']['caldot1-y05']['detection_config'] == DC_A
    assert manifest['datasets']['caldot1-y05']['videos'] == {'te00.mp4': 3, 'te01.mp4': 1}
    monkeypatch.setattr(pa, 'parse_args', lambda: argparse.Namespace(datasets=['caldot1-y05'], loss=0.05, skip_jpeg_control=False))
    pa.main()
    control = json.loads((fake_detector / 'out' / 'operating_points.json').read_text())['datasets']['CalDoT1']['jpeg_control']
    # The fake detector finds the top-left box on every plain frame (te00 frames 2 and 3 have other boxes) and nothing after JPEG.
    assert control['frames'] == 4 and control['jpeg_detections'] == 0
    assert control['scores']['frame']['jpeg']['recall'] == 0.0
    assert control['scores']['frame']['rerun']['tp'] == 2
