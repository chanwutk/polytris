import json
from collections import Counter

import duckdb
import numpy as np
import pytest

from evaluation.p140_otif_transform import unwrap_leap_frames
from measurements import r1o2_leap_native_queries as lq
from polyis.io import store


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def write_tracks(path, frames):
    # Write a tracking.jsonl with one line per (frame_idx, [[id, x1, y1, x2, y2], ...]).
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for frame_idx, tracks in frames:
            f.write(json.dumps({'frame_idx': frame_idx, 'tracks': tracks}) + '\n')
    return str(path)


def set_clips(con, lengths):
    # Clips te00, te01, ... with the given lengths, pooled in order, and their frames.
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    con.execute('CREATE OR REPLACE TABLE clips (video VARCHAR, clip BIGINT, n BIGINT, pooled_offset BIGINT)')
    con.executemany('INSERT INTO clips VALUES (?, ?, ?, ?)',
                    [(f'te{c:02d}.mp4', c, n, o) for c, (n, o) in enumerate(zip(lengths, offsets))])
    lq.create_frames(con)


def set_configs(con, configs):
    # Table configs(system, config, fps, hota) from (system, config) pairs.
    con.execute('CREATE OR REPLACE TABLE configs (system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE)')
    if configs:
        con.executemany('INSERT INTO configs VALUES (?, ?, 1.0, 0.5)', configs)


def pooled_tables(con):
    # Empty pred and gt tables with the measurement's schema.
    con.execute('CREATE OR REPLACE TABLE pred (system VARCHAR, config VARCHAR, clip BIGINT, frame BIGINT, gidx BIGINT, p BIGINT)')
    con.execute('CREATE OR REPLACE TABLE gt (variant VARCHAR, clip BIGINT, frame BIGINT, gidx BIGINT, g BIGINT)')


def fill(con, table, labels, values):
    # Append one pooled series (one value per row of frames, in gidx order) under the given label columns.
    placeholders = ', '.join('?' for _ in labels)
    con.execute(f"""
        INSERT INTO {table} SELECT {placeholders}, f.clip, f.frame, f.gidx, v.v
        FROM frames f JOIN (SELECT unnest(range(?)) AS gidx, unnest(?) AS v) v USING (gidx)
    """, [*labels, len(values), [int(x) for x in values]])


def dense(con, table, column, where):
    # One configuration's or variant's pooled counts as a list.
    return [r[0] for r in con.execute(f'SELECT {column} FROM {table} WHERE {where} ORDER BY gidx').fetchall()]


def topk_scores(p, g, gid):
    # The paper's top-k scores of one prediction, as score_topk computes them.
    p, g = np.asarray(p), np.asarray(g)
    return lq.topk_paper(lq.tie_draws(p, gid), lq.separated_top(p, gid), p, g, lq.topk_tau(g, gid))


# ---------------------------------------------------------------- LEAP's code, verbatim (evaluate.py)

def leap_selection_query_1(gt_result, pred_result):
    # evaluate.py selection_query_1 with start 0 and end len(gt_result).
    TP, FP, TN, FN = 0, 0, 0, 0
    for frame_id in range(0, len(gt_result)):
        if gt_result[frame_id] >= 1 and pred_result[frame_id] >= 1:
            TP += 1
        elif gt_result[frame_id] >= 1 and pred_result[frame_id] == 0:
            FN += 1
        elif gt_result[frame_id] == 0 and pred_result[frame_id] >= 1:
            FP += 1
        else:
            TN += 1
    if TP == 0:
        TP = 1
    recall = round((TP) / (TP + FN), 4)
    precision = round((TP) / (TP + FP), 4)
    if TP == 0:
        F1 = 0
    else:
        F1 = round(2 * (precision * recall) / (precision + recall), 4)
    return F1


def leap_aggregation_query_1(gt_result, pred_result):
    # evaluate.py aggregation_query_1 with start 0 and arrays of the pooled length.
    MAE = 0
    ACC = 0
    for i in range(0, len(pred_result)):
        MAE += abs(pred_result[i] - gt_result[i])
        if gt_result[i] != 0:
            ACC += 1 - (abs(pred_result[i] - gt_result[i]) / gt_result[i])
        else:
            if pred_result[i] != 0:
                ACC += 0
            else:
                ACC += 1
    MAE = MAE / len(pred_result)
    ACC = ACC / len(pred_result)
    return MAE, ACC


def leap_sort_max_intervals(gt_result, input, gap):
    # evaluate.py _sort_max_intervals, verbatim.
    selected_id = []
    max_set = []
    for id in input:
        if len(selected_id) == 0:
            selected_id.append([id, id])
        else:
            match_flag = False
            for clus in selected_id:
                if id > clus[0] and id < clus[-1]:
                    clus.append(id)
                    clus.sort()
                    match_flag = True
                    break
                elif (id > clus[-1]) and (id < clus[-1] + int(0.25 * gap)):
                    clus.append(id)
                    match_flag = True
                    break
                elif (id < clus[0]) and (id > clus[0] - int(0.25 * gap)):
                    clus.insert(0, id)
                    match_flag = True
                    break
            if not match_flag:
                selected_id.append([id, id])
    for selected in selected_id:
        tmp = [gt_result[i] for i in range(selected[0], selected[-1] + 1)]
        max_set.append(max(tmp))
    return max_set


def leap_top_k_query_1(gt_result, pred_result, k=10, gap=300, kind='stable'):
    # evaluate.py top_k_query_1, verbatim; kind None is its argsort(), 'stable' the deterministic variant.
    gt_result = np.array(gt_result)
    pred_result = np.array(pred_result)
    topk_ids = gt_result.argsort(kind=kind)[::-1]
    topk_ids = topk_ids[:5000]
    max_gt = leap_sort_max_intervals(gt_result, topk_ids, gap)
    max_gt.sort()
    max_gt = max_gt[::-1][:k]
    topk_ids = pred_result.argsort(kind=kind)[::-1]
    topk_ids = topk_ids[:5000]
    max_pred = leap_sort_max_intervals(gt_result, topk_ids, gap)
    max_pred.sort()
    max_pred = max_pred[::-1][:k]
    counter_gt = Counter(max_gt)
    counter_pred = Counter(max_pred)
    return sum((counter_gt & counter_pred).values()) / k


# ---------------------------------------------------------------- selection

def test_selection_absent_on_both_sides_is_null():
    # No positive frame anywhere: precision, recall, and F1 are undefined; LEAP's code reports 1.0.
    m = lq.selection_metrics(0, 0, 0)
    assert (m['precision'], m['recall'], m['f1']) == (None, None, None)
    assert m['f1_leap_code'] == 1.0


def test_selection_no_true_positive_is_zero_but_code_gives_two_over_n_plus_two():
    # Ground truth positive on n frames, prediction empty: F1 0, while TP forced to 1 gives 2 / (n + 2).
    for n in (2, 10, 100):
        m = lq.selection_metrics(0, 0, n)
        assert m['f1'] == 0 and m['precision'] is None and m['recall'] == 0
        assert m['f1_leap_code'] == round(2 / (n + 2), 4)


def test_selection_plain_f1():
    # 3 TP, 1 FP, 2 FN: precision 3/4, recall 3/5, F1 6/9.
    m = lq.selection_metrics(3, 1, 2)
    assert m['precision'] == 0.75 and m['recall'] == 0.6 and m['f1'] == pytest.approx(6 / 9)


def test_selection_code_matches_leap_on_random_arrays():
    # The code variant reproduces selection_query_1 exactly, rounding included.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        g, p = rng.integers(0, 3, n) * rng.integers(0, 2, n), rng.integers(0, 3, n) * rng.integers(0, 2, n)
        tp, fp, fn = int(((g >= 1) & (p >= 1)).sum()), int(((g == 0) & (p >= 1)).sum()), int(((g >= 1) & (p == 0)).sum())
        assert lq.selection_metrics(tp, fp, fn)['f1_leap_code'] == leap_selection_query_1(g.tolist(), p.tolist())


# ---------------------------------------------------------------- aggregation

def score_pairs(con, pairs):
    # score_frames on one clip and one configuration with the given (g, p) per frame.
    set_clips(con, [len(pairs)])
    pooled_tables(con)
    fill(con, 'pred', ['s', 'c'], [p for _, p in pairs])
    fill(con, 'gt', ['all'], [g for g, _ in pairs])
    rows = lq.score_frames(con)
    assert len(rows) == 1
    return rows[0]


def test_aggregation_overcount_is_not_clipped(con):
    # Predicting 3 for a ground truth of 1 is a 200% error, so LEAP's ACC is -1.
    r = score_pairs(con, [(1, 3)])
    assert r['mape_leap'] == 2.0 and 1 - r['mape_leap'] == -1.0 and r['mae'] == 2.0


def test_aggregation_zero_frames(con):
    # g = 0, p = 1 costs 1; g = p = 0 costs 0; g = 2, p = 1 costs 0.5.
    r = score_pairs(con, [(0, 1), (0, 0), (2, 1)])
    assert r['mape_leap'] == pytest.approx(1.5 / 3)
    # mape_pos covers only g > 0 frames; zero_fp_rate is 1 of the 2 g = 0 frames.
    assert r['mape_pos'] == 0.5 and r['zero_fp_rate'] == 0.5
    # Selection counts on the same frames: one TP, one FP, no FN.
    assert (r['tp'], r['fp'], r['fn']) == (1, 1, 0)


def test_aggregation_no_zero_frames_leaves_rate_null(con):
    # Without a g = 0 frame the false-positive rate is undefined.
    r = score_pairs(con, [(1, 1), (2, 2)])
    assert r['zero_fp_rate'] is None and r['mape_leap'] == 0.0


def test_aggregation_matches_leap_on_random_arrays(con):
    # acc_leap and MAE reproduce aggregation_query_1 (start 0, end N) on random counts, zeros and overcounts included.
    rng = np.random.default_rng(4)
    for _ in range(40):
        n = int(rng.integers(1, 80))
        g = rng.integers(0, 5, n) * rng.integers(0, 2, n)
        p = rng.integers(0, 7, n) * rng.integers(0, 2, n)
        r = score_pairs(con, list(zip(g.tolist(), p.tolist())))
        metrics = lq.frame_metrics(r)['aggregation']
        mae, acc = leap_aggregation_query_1(g.tolist(), p.tolist())
        assert metrics['acc_leap'] == pytest.approx(acc, abs=1e-12) and metrics['mae'] == pytest.approx(mae, abs=1e-12)


def test_aggregation_is_identical_for_identical_answers(con):
    # Six configurations with the same counts over more than one DuckDB row group score bit-identically, so a tie
    # between LEAP and its ambiguous-dropped reading (no ambiguous object) cannot flip a bound.
    rng = np.random.default_rng(6)
    n = 300000
    set_clips(con, [n])
    pooled_tables(con)
    p, g = rng.integers(0, 25, n), rng.integers(0, 20, n)
    fill(con, 'gt', ['all'], g)
    for c in range(6):
        fill(con, 'pred', ['s', f'c{c}'], p)
    rows = lq.score_frames(con)
    assert len({(r['mape_leap'], r['mape_pos'], r['mae']) for r in rows}) == 1
    # The exact average agrees with numpy to double precision.
    errors = np.where(g > 0, np.abs(p - g) / np.maximum(g, 1), (p > 0).astype(float))
    assert rows[0]['mape_leap'] == pytest.approx(errors.mean(), rel=1e-12)


# ---------------------------------------------------------------- top-k: LEAP's code

def test_leap_clusters_never_merge_and_value_with_ground_truth():
    # Candidate order [100, 200, 300, 250, 260, 320]: 250, 260, 320 join the 200 cluster, never the 300 one.
    values = np.zeros(1000, dtype=np.int64)
    for rank, frame in enumerate([100, 200, 300, 250, 260, 320]):
        values[frame] = 6 - rank
    lo, hi = lq.leap_clusters(values, candidates=6)
    assert list(zip(lo.tolist(), hi.tolist())) == [(100, 100), (200, 320), (300, 300)]
    # A ground-truth peak of 9 at frame 300 is counted by two clusters.
    g = np.zeros(1000, dtype=np.int64)
    g[300] = 9
    assert [int(g[a:b + 1].max()) for a, b in zip(lo, hi)] == [0, 9, 9]
    assert lq.cluster_peaks((lo, hi), g) == [9, 9, 0]


def test_leap_clusters_radius_is_strict():
    # A frame exactly the radius away starts a new cluster; one frame closer joins.
    values = np.zeros(400, dtype=np.int64)
    values[[100, 175, 250 - 1]] = [3, 2, 1]
    lo, hi = lq.leap_clusters(values, candidates=3)
    assert list(zip(lo.tolist(), hi.tolist())) == [(100, 100), (175, 249)]


@pytest.mark.parametrize('kind', ['stable', None])
def test_topk_leap_code_matches_leap_on_random_arrays(kind):
    # The vectorized clustering reproduces top_k_query_1 on random counts, ties included, under both sorts.
    rng = np.random.default_rng(1)
    for n in (50, 400, 3000, 7000):
        for _ in range(4):
            g = rng.poisson(1.0, n) * (rng.random(n) < 0.6)
            p = np.maximum(g + rng.integers(-1, 2, n), 0)
            ours = lq.topk_leap_code(lq.leap_clusters(g, kind=kind), lq.leap_clusters(p, kind=kind), g)
            assert ours == leap_top_k_query_1(g.tolist(), p.tolist(), kind=kind)


# ---------------------------------------------------------------- top-k: the paper's query

def test_separated_top_respects_separation_and_tie_order():
    # Equal values: the earlier global id wins; neighbors within the separation are suppressed.
    values = np.array([5, 5, 5, 0, 5])
    gid = np.array([0, 10, 20, 30, 40])
    assert lq.separated_top(values, gid, k=3, separation=15).tolist() == [0, 2, 4]
    # Fewer separable frames than k returns what exists.
    assert lq.separated_top(values, gid, k=10, separation=100).tolist() == [0]
    # A tie-break key replaces the pooled order among equal values.
    assert lq.separated_top(values, gid, k=3, separation=15, tiebreak=np.array([4, 3, 2, 1, 0])).tolist() == [4, 2, 0]


def test_returnable_never_returns_zero_counts_and_cuts_below_the_guaranteed_level():
    # Few positive frames: every positive frame is returnable, no zero frame is.
    p = np.zeros(100, dtype=np.int64)
    p[[3, 50]] = [1, 2]
    assert lq.returnable(p).tolist() == [3, 50]
    # 1490 = 10 * (2 * 75 - 1) frames at count 3 guarantee 10 answers, so the count-1 frames are cut.
    p = np.ones(5000, dtype=np.int64)
    p[:1490] = 3
    assert len(lq.returnable(p)) == 1490
    # One frame fewer at count 3 and every positive frame stays.
    p[1489] = 1
    assert len(lq.returnable(p)) == 5000


def test_returnable_cut_does_not_change_any_answer():
    # On random counts with clip gaps, scanning only the returnable frames gives the same answer as scanning all positive ones.
    rng = np.random.default_rng(5)
    for _ in range(10):
        n = 20000
        p = rng.poisson(1.5, n)
        gid = np.arange(n) + lq.TOPK_RADIUS * (np.arange(n) // 900)
        keys = rng.random(n)
        positive = np.flatnonzero(p >= 1)
        cut = lq.returnable(p)
        assert len(cut) < len(positive)
        full = positive[lq.separated_top(p[positive], gid[positive], tiebreak=keys[positive])]
        assert (cut[lq.separated_top(p[cut], gid[cut], tiebreak=keys[cut])] == full).all()


def test_tie_draws_are_reproducible_and_padded():
    # The same counts give the same draws; an answer shorter than k is padded with -1.
    p = np.zeros(1000, dtype=np.int64)
    p[[100, 300, 500]] = 1
    a, b = lq.tie_draws(p, np.arange(1000)), lq.tie_draws(p, np.arange(1000))
    assert a.shape == (lq.TIE_DRAWS, lq.TOPK) and (a == b).all()
    assert (np.sort(a[:, :3], axis=1) == [100, 300, 500]).all() and (a[:, 3:] == -1).all()


def test_topk_paper_perfect_prediction_and_tau():
    # Twelve separable peaks of different heights: a perfect prediction scores 1.0 in every draw.
    g = np.zeros(2000, dtype=np.int64)
    g[np.arange(12) * 100 + 50] = np.arange(12) + 2
    gid = np.arange(2000)
    r = topk_scores(g.copy(), g, gid)
    assert r['topk_paper'] == 1.0 and r['topk_paper_min'] == 1.0 and r['tau'] == 4 and not r['tau_zero']
    assert r['returned_min'] == 10 and r['topk_paper_pooled_order'] == 1.0
    # Ranking the two smallest peaks first and then only non-peak frames: no returned frame reaches tau.
    p = np.ones(2000, dtype=np.int64)
    p[np.arange(12) * 100 + 50] = 0
    p[[50, 150]] = 9
    r = topk_scores(p, g, gid)
    assert r['topk_paper'] == 0.0 and r['topk_paper_max'] == 0.0 and r['topk_paper_pooled_order'] == 0.0
    # In pooled order, the last eight returned frames are tied with every other non-peak frame.
    assert r['ties_at_cutoff'] == 1988


def test_topk_paper_tau_zero_and_empty_answer():
    # Fewer than k positive separated frames: tau is 0, so any returned frame is correct, but an empty output returns none.
    g = np.zeros(2000, dtype=np.int64)
    g[[100, 900]] = 1
    r = topk_scores(np.zeros(2000, dtype=np.int64), g, np.arange(2000))
    assert r['tau'] == 0 and r['tau_zero'] and r['topk_paper'] == 0.0 and r['returned_min'] == 0
    # The earlier pooled-order score credited the empty output with its zero-count frames.
    assert r['topk_paper_pooled_order'] == 1.0


def test_topk_paper_short_answer_keeps_the_denominator():
    # Three returned frames, all at peaks reaching tau: precision 3 / 10 in every draw.
    g = np.zeros(1000, dtype=np.int64)
    g[np.arange(10) * 100] = 4
    p = np.zeros(1000, dtype=np.int64)
    p[[100, 300, 500]] = 1
    r = topk_scores(p, g, np.arange(1000))
    assert r['tau'] == 4 and r['topk_paper'] == pytest.approx(0.3) and r['topk_paper_max'] == pytest.approx(0.3)
    assert r['returned_min'] == 3


def test_topk_paper_empty_output_scores_zero_wherever_the_peaks_are():
    # An all-zero output scores 0, while the pooled-order score depends on whether the peaks sit where its ties land.
    gid = np.arange(2000)
    early = np.zeros(2000, dtype=np.int64)
    early[np.arange(10) * 75] = 5
    late = early[::-1].copy()
    zero = np.zeros(2000, dtype=np.int64)
    assert topk_scores(zero, early, gid)['topk_paper_pooled_order'] == 1.0
    assert topk_scores(zero, late, gid)['topk_paper_pooled_order'] == 0.0
    for g in (early, late):
        r = topk_scores(zero, g, gid)
        assert (r['topk_paper'], r['topk_paper_min'], r['topk_paper_max']) == (0.0, 0.0, 0.0)


def test_topk_paper_all_tied_output_does_not_depend_on_pooled_order():
    # An output tied on every frame scores near the base rate whether the peaks sit early or late in the pool.
    gid = np.arange(2000)
    early = np.zeros(2000, dtype=np.int64)
    early[np.arange(10) * 75] = 5
    late = early[::-1].copy()
    ones = np.ones(2000, dtype=np.int64)
    a, b = topk_scores(ones, early, gid), topk_scores(ones, late, gid)
    # Pooled order puts all of its answers on the early peaks and none on the late ones.
    assert a['topk_paper_pooled_order'] == 1.0 and b['topk_paper_pooled_order'] == 0.0
    # The mean over random tie orders is small and about the same either way.
    assert a['topk_paper'] < 0.1 and b['topk_paper'] < 0.1 and abs(a['topk_paper'] - b['topk_paper']) < 0.05
    assert a['topk_paper_min'] <= a['topk_paper'] <= a['topk_paper_max']


def test_clip_gap_separates_clips(con):
    # The last frame of one clip and the first of the next are far apart in gid, close in gidx.
    set_clips(con, [100, 50])
    rows = con.execute('SELECT clip, frame, gidx, gid FROM frames WHERE gidx IN (99, 100) ORDER BY gidx').fetchall()
    assert rows == [(0, 99, 99, 99), (1, 0, 100, 100 + lq.TOPK_RADIUS)]


# ---------------------------------------------------------------- clips, files, and dropped frames

def fake_clips(monkeypatch, tmp_path, frames_per_clip, video_frames):
    # Reference detection files with the given frame indices per clip, and fake video frame counts.
    monkeypatch.setattr(lq.cache, 'exec', lambda dataset, stage, video, name: tmp_path / 'naive' / video / name)
    monkeypatch.setattr(lq, 'get_video_frame_count', lambda dataset, video: video_frames[video])
    for video, frames in frames_per_clip.items():
        write_tracks(tmp_path / 'naive' / video / 'detection.jsonl', [(f, []) for f in frames])


def test_load_clips_pools_clips_in_order(con, monkeypatch, tmp_path):
    # Two clips of 3 and 2 frames: offsets 0 and 3, and a gap in gid between them.
    fake_clips(monkeypatch, tmp_path, {'te00.mp4': [0, 1, 2], 'te01.mp4': [0, 1]}, {'te00.mp4': 3, 'te01.mp4': 2})
    lq.load_clips(con, 'jnc0', ['te00.mp4', 'te01.mp4'])
    assert con.execute('SELECT video, n, pooled_offset FROM clips ORDER BY clip').fetchall() == [('te00.mp4', 3, 0), ('te01.mp4', 2, 3)]
    assert con.execute('SELECT gid FROM frames ORDER BY gidx').fetchall() == [(0,), (1,), (2,), (3 + lq.TOPK_RADIUS,), (4 + lq.TOPK_RADIUS,)]


@pytest.mark.parametrize('frames, video_frames', [
    ([0, 1, 2], 4),     # the video has a frame the reference never decoded
    ([0, 2], 2),        # a missing frame inside the clip
    ([1, 2], 2),        # the clip does not start at frame 0
    ([0, 1, 1], 3),     # a repeated frame
])
def test_load_clips_rejects_reference_frames_that_do_not_match_the_video(con, monkeypatch, tmp_path, frames, video_frames):
    # Any mismatch between the reference rows and the video frames is an error.
    fake_clips(monkeypatch, tmp_path, {'te00.mp4': frames}, {'te00.mp4': video_frames})
    with pytest.raises(AssertionError, match='reference frames do not match'):
        lq.load_clips(con, 'jnc0', ['te00.mp4'])


def test_load_clips_requires_every_reference_file(con, monkeypatch, tmp_path):
    # A clip without a reference detection file is an error, not an empty clip.
    fake_clips(monkeypatch, tmp_path, {'te00.mp4': [0]}, {'te00.mp4': 1, 'te01.mp4': 1})
    with pytest.raises(AssertionError, match='reference detection files missing'):
        lq.load_clips(con, 'jnc0', ['te00.mp4', 'te01.mp4'])


def test_file_rows_read_leap_files_for_the_ambiguous_dropped_variant(monkeypatch, tmp_path):
    # leap_ambiguous_dropped is LEAP's own file; a missing file of any configuration is an error.
    monkeypatch.setattr(lq, 'output_path', lambda dataset, system, config, video: tmp_path / system / config / video / 'tracking.jsonl')
    write_tracks(tmp_path / 'leap' / 'param 0' / 'te00.mp4' / 'tracking.jsonl', [])
    configs = [('leap', 'param 0', 1.0, 0.1), ('leap_ambiguous_dropped', 'param 0', 1.0, None)]
    rows = lq.file_rows('jnc0', ['te00.mp4'], configs)
    assert rows[0][3] == rows[1][3] == str(tmp_path / 'leap' / 'param 0' / 'te00.mp4' / 'tracking.jsonl')
    with pytest.raises(AssertionError, match='tracking files missing'):
        lq.file_rows('jnc0', ['te00.mp4', 'te01.mp4'], configs)


def test_drop_past_clip_rejects_negative_frames(con):
    # A negative frame index means a broken file.
    set_clips(con, [3])
    con.execute("CREATE TABLE t AS SELECT 'te00.mp4' AS video, -1 AS frame")
    with pytest.raises(AssertionError, match='negative frame'):
        lq.drop_past_clip(con, 't', "'x' AS system")


# ---------------------------------------------------------------- counts and ground truth

def test_counts_fill_missing_frames_drop_past_clip_and_dedupe(con, tmp_path):
    # One 6-frame clip; the file starts at frame 2, repeats an id, and runs past the clip.
    set_clips(con, [6])
    path = write_tracks(tmp_path / 't.jsonl', [
        (2, [[1, 0, 0, 1, 1], [1, 0, 0, 1, 1], [2, 0, 0, 1, 1]]),
        (3, []),
        (4, [[2, 0, 0, 1, 1]]),
        (6, [[2, 0, 0, 1, 1]]),
        (7, [[2, 0, 0, 1, 1]]),
    ])
    empty = tmp_path / 'e.jsonl'
    empty.write_text('')
    set_configs(con, [('s', 'a'), ('s', 'empty')])
    dropped = lq.load_counts(con, [('s', 'a', 'te00.mp4', path), ('s', 'empty', 'te00.mp4', str(empty))], [])
    # Frames 6 and 7 are past the 6-frame clip.
    assert dropped == [dict(system='s', config='a', frames=2, clips=1)]
    con.execute('CREATE OR REPLACE TABLE gt_counts (variant VARCHAR, video VARCHAR, frame BIGINT, g BIGINT)')
    lq.densify(con)
    # Missing leading and trailing frames are 0; a repeated id counts once.
    assert dense(con, 'pred', 'p', "config = 'a'") == [0, 0, 2, 0, 1, 0]
    # An empty file is all zeros.
    assert dense(con, 'pred', 'p', "config = 'empty'") == [0] * 6


def test_counts_exclude_ambiguous_ids_for_one_system_only(con, tmp_path):
    # LEAP's file serves both LEAP and its ambiguous-dropped variant; only the latter loses id 7.
    set_clips(con, [3])
    path = write_tracks(tmp_path / 'leap.jsonl', [(0, [[7, 0, 0, 1, 1], [8, 0, 0, 1, 1]]), (1, [[7, 0, 0, 1, 1]])])
    set_configs(con, [('leap', 'param 0'), ('leap_ambiguous_dropped', 'param 0')])
    lq.load_counts(con, [('leap', 'param 0', 'te00.mp4', path), ('leap_ambiguous_dropped', 'param 0', 'te00.mp4', path)],
                   [('leap_ambiguous_dropped', 'te00.mp4', 7)])
    con.execute('CREATE OR REPLACE TABLE gt_counts (variant VARCHAR, video VARCHAR, frame BIGINT, g BIGINT)')
    lq.densify(con)
    assert dense(con, 'pred', 'p', "system = 'leap'") == [2, 1, 0]
    assert dense(con, 'pred', 'p', "system = 'leap_ambiguous_dropped'") == [1, 0, 0]


def test_counts_keep_clips_apart(con, tmp_path):
    # The same id in two clips counts in each clip at its own frames.
    set_clips(con, [2, 2])
    a = write_tracks(tmp_path / 'a.jsonl', [(1, [[1, 0, 0, 1, 1]])])
    b = write_tracks(tmp_path / 'b.jsonl', [(0, [[1, 0, 0, 1, 1]])])
    set_configs(con, [('s', 'c')])
    lq.load_counts(con, [('s', 'c', 'te00.mp4', a), ('s', 'c', 'te01.mp4', b)], [])
    con.execute('CREATE OR REPLACE TABLE gt_counts (variant VARCHAR, video VARCHAR, frame BIGINT, g BIGINT)')
    lq.densify(con)
    assert dense(con, 'pred', 'p', 'true') == [0, 1, 1, 0]


def box(track_id, cx, cy):
    # A 10 x 10 box centered at (cx, cy).
    return [track_id, cx - 5, cy - 5, cx + 5, cy + 5]


def moving_counts(con, tmp_path, lengths, files):
    # Load ground truth for clips of the given lengths from {video: [(frame, boxes)]}; return pooled 'all' and 'moving' counts.
    set_clips(con, lengths)
    paths = [(v, write_tracks(tmp_path / v / 'gt.jsonl', sorted(rows))) for v, rows in files.items()]
    dropped = lq.load_ground_truth(con, paths)
    set_configs(con, [])
    con.execute('CREATE OR REPLACE TABLE counts (system VARCHAR, config VARCHAR, video VARCHAR, frame BIGINT, cnt BIGINT)')
    lq.densify(con)
    return dense(con, 'gt', 'g', "variant = 'all'"), dense(con, 'gt', 'g', "variant = 'moving'"), dropped


def test_moving_filter(con, tmp_path):
    # Track 1 moves 95 px over rows 0..19 (kept); track 2 is stationary for 30 frames; track 3 lasts 10 frames.
    frames = {f: [] for f in range(40)}
    for f in range(20):
        frames[f].append(box(1, 10 + 5 * f + 0.5, 50))
    for f in range(30):
        frames[f].append(box(2, 300, 300))
    for f in range(10):
        frames[f].append(box(3, 10 + 20 * f, 200))
    all_counts, moving, _ = moving_counts(con, tmp_path, [40], {'te00.mp4': list(frames.items())})
    # Variant all counts every box on its own frames.
    assert all_counts == [3] * 10 + [2] * 10 + [1] * 10 + [0] * 10
    # Variant moving keeps only track 1, counted on [first, last) = frames 0..18.
    assert moving == [1] * 19 + [0] * 21


def test_moving_filter_thresholds_are_inclusive_and_use_leap_truncated_centers(con, tmp_path):
    # All tracks start at frame 0. Track 1: span 15, travel 50 (kept, both bounds inclusive).
    # Track 2: float travel 49.5, LEAP centers 100 -> 150 (kept). Track 3: float travel 49.75, LEAP centers
    # 100 -> 149 (dropped). Track 4: travel 100 but span 14 (dropped).
    frames = {f: [] for f in range(20)}
    frames[0] += [box(1, 100.0, 10), box(2, 100.75, 50), box(3, 100.0, 90), box(4, 100.0, 130)]
    frames[14] += [box(4, 200.0, 130)]
    frames[15] += [box(1, 150.0, 10), box(2, 150.25, 50), box(3, 149.75, 90)]
    _, moving, dropped = moving_counts(con, tmp_path, [20], {'te00.mp4': list(frames.items())})
    assert dropped == []
    # Tracks 1 and 2 count on [0, 15).
    assert moving == [2] * 15 + [0] * 5


def test_moving_filter_truncates_left_and_width_before_the_center(con, tmp_path):
    # LEAP: x1 10.7, x2 20.6 gives xmin 10, xmax 10 + int(9.9) = 19, center 14; x1 59.5, x2 69.5 gives center 64.
    # Travel 50 keeps the track, while truncating float centers (15 -> 64) would give 49 and drop it.
    frames = [(0, [[1, 10.7, 0, 20.6, 10]]), (15, [[1, 59.5, 0, 69.5, 10]])]
    _, moving, _ = moving_counts(con, tmp_path, [20], {'te00.mp4': frames})
    assert moving == [1] * 15 + [0] * 5


def test_moving_filter_counts_across_gaps_and_keeps_clips_apart(con, tmp_path):
    # Track 1 in clip 0 is seen on frames 0..4 and 25..29 and travels 60 px: it counts on [0, 29), gap included.
    # Id 1 in clip 1 lasts 10 frames: it is a separate, short track, so it is dropped there.
    clip0 = [(f, [box(1, 100 + 2 * f, 50)]) for f in [*range(5), *range(25, 30)]]
    clip1 = [(f, [box(1, 100 + 10 * f, 50)]) for f in range(10)]
    all_counts, moving, _ = moving_counts(con, tmp_path, [40, 40], {'te00.mp4': clip0, 'te01.mp4': clip1})
    assert all_counts[:40] == [1] * 5 + [0] * 20 + [1] * 5 + [0] * 10 and all_counts[40:] == [1] * 10 + [0] * 30
    assert moving == [1] * 29 + [0] * 11 + [0] * 40


def test_ground_truth_past_clip_is_dropped_and_reported(con, tmp_path):
    # A ground-truth row at frame n is dropped before both variants are built.
    set_clips(con, [3])
    path = write_tracks(tmp_path / 'gt.jsonl', [(0, [box(1, 5, 5)]), (3, [box(1, 5, 5)])])
    dropped = lq.load_ground_truth(con, [('te00.mp4', path)])
    assert dropped == [dict(system='ground truth', frames=1, clips=1)]
    assert con.execute("SELECT max(frame) FROM gt_counts WHERE variant = 'all'").fetchone()[0] == 0


def automatic_ground_truth(monkeypatch, tmp_path, videos):
    # Automatic ground-truth files for the given clips under tmp_path/auto.
    monkeypatch.setattr(lq, 'output_path', lambda dataset, system, config, video: tmp_path / 'auto' / video / 'tracking.jsonl')
    for v in videos:
        write_tracks(tmp_path / 'auto' / v / 'tracking.jsonl', [])


def test_ground_truth_files_use_gt_root_only_when_it_holds_every_clip(monkeypatch, tmp_path):
    # --gt-root reads <root>/<root dataset>/teNN.mp4/tracking.jsonl when every clip is there.
    automatic_ground_truth(monkeypatch, tmp_path, ['te00.mp4', 'te01.mp4'])
    root = tmp_path / 'human'
    for v in ('te00.mp4', 'te01.mp4'):
        write_tracks(root / 'caldot1' / v / 'tracking.jsonl', [])
    rows, external = lq.ground_truth_files('caldot1-y05', ['te00.mp4', 'te01.mp4'], root)
    assert external and rows[1] == ('te01.mp4', str(root / 'caldot1' / 'te01.mp4' / 'tracking.jsonl'))
    # A dataset gt_root does not cover falls back to the automatic ground truth.
    rows, external = lq.ground_truth_files('jnc0', ['te00.mp4'], root)
    assert not external and rows == [('te00.mp4', str(tmp_path / 'auto' / 'te00.mp4' / 'tracking.jsonl'))]
    # Partial coverage is an error.
    with pytest.raises(AssertionError, match='holds 2 of 3 clips'):
        lq.ground_truth_files('caldot1-y05', ['te00.mp4', 'te01.mp4', 'te02.mp4'], root)


def test_ground_truth_files_require_automatic_files(monkeypatch, tmp_path):
    # Without --gt-root, a missing automatic file is an error.
    automatic_ground_truth(monkeypatch, tmp_path, ['te00.mp4'])
    assert not lq.ground_truth_files('jnc0', ['te00.mp4'], None)[1]
    with pytest.raises(AssertionError, match='ground-truth files missing'):
        lq.ground_truth_files('jnc0', ['te00.mp4', 'te01.mp4'], None)


def test_ground_truth_source_marks_automatic_b3d_provisional(tmp_path):
    # Automatic B3D ground truth is provisional; CalDoT's is not; external ground truth is not.
    assert lq.ground_truth_source('jnc0', None, False)['provisional']
    assert not lq.ground_truth_source('caldot1-y05', None, False)['provisional']
    assert lq.ground_truth_source('ams-y05', None, False)['dataset'] == 'ams'
    external = lq.ground_truth_source('jnc0', tmp_path, True)
    assert not external['provisional'] and external['root'] == str(tmp_path) and external['kind'] == 'external'


def test_blockers_drop_the_b3d_ground_truth_blocker_only_without_provisional_datasets():
    # The B3D blocker stays while any dataset is provisional; the others always stay.
    provisional = {'B3D1': dict(ground_truth=dict(provisional=True)), 'CalDoT1': dict(ground_truth=dict(provisional=False))}
    out = lq.blockers(provisional)
    assert set(out) == set(lq.BLOCKERS) and out['b3d_human_ground_truth'].endswith('Provisional here: B3D1.')
    final = {'B3D1': dict(ground_truth=dict(provisional=False))}
    assert set(lq.blockers(final)) == set(lq.BLOCKERS) - {'b3d_human_ground_truth'}


def test_add_baselines_are_constant_configurations(con):
    # zero, one, and each variant's lower median count become configurations of system 'baseline' on every frame.
    set_clips(con, [5])
    set_configs(con, [('s', 'c')])
    pooled_tables(con)
    fill(con, 'pred', ['s', 'c'], [0] * 5)
    fill(con, 'gt', ['all'], [0, 1, 2, 2, 3])
    fill(con, 'gt', ['moving'], [0, 0, 0, 1, 1])
    lq.add_baselines(con)
    assert con.execute("SELECT config FROM configs WHERE system = 'baseline' ORDER BY config").fetchall() == \
        [('median all',), ('median moving',), ('one',), ('zero',)]
    assert dense(con, 'pred', 'p', "config = 'median all'") == [2] * 5
    assert dense(con, 'pred', 'p', "config = 'median moving'") == [0] * 5
    assert dense(con, 'pred', 'p', "config = 'one'") == [1] * 5 and dense(con, 'pred', 'p', "config = 'zero'") == [0] * 5


# ---------------------------------------------------------------- LEAP export recovery, unwrap statistics, and the cache

def write_export(path, frames):
    # A raw LEAP export: one list of box dicts per frame.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[dict(track_id=t, left=0, top=0, right=1, bottom=1, **{'class': 'car'}) for t in f]
                                for f in frames]))
    return str(path)


def test_export_counts_contiguous_wrapped_full_span_and_past_clip(con, tmp_path):
    # L = 8, clip n = 6: id 1 on [1, 3]; id 2 wrapped (head [0, 1], tail [6, 7]); id 3 on every frame; id 4 on [4, 7].
    frames = [[2, 3], [1, 2, 3], [1, 3], [1, 3], [3, 4], [3, 4], [2, 3, 4], [2, 3, 4]]
    counts = lq.export_counts(con, 'te00.mp4', write_export(tmp_path / 'x.json', frames), 6)
    # Wrapped id 2 counts only on [0, 2); id 4 is clipped at the clip end; id 3 is kept whole.
    assert counts.tolist() == [2, 3, 2, 2, 2, 2]


def test_export_counts_rejects_other_shapes(con, tmp_path):
    # Two runs that do not touch both ends are not the exporter's wrap.
    with pytest.raises(AssertionError, match='neither one interval nor one wrapped interval'):
        lq.export_counts(con, 'te00.mp4', write_export(tmp_path / 'x.json', [[1], [], [1], []]), 4)


def test_export_counts_empty_and_null(con, tmp_path):
    # An empty export and a null file are all zeros; a null frame holds no boxes.
    assert lq.export_counts(con, 'te00.mp4', write_export(tmp_path / 'e.json', []), 3).tolist() == [0, 0, 0]
    (tmp_path / 'n.json').write_text('null')
    assert lq.export_counts(con, 'te00.mp4', str(tmp_path / 'n.json'), 2).tolist() == [0, 0]
    (tmp_path / 'f.json').write_text(json.dumps([None, [{'track_id': 5}]]))
    assert lq.export_counts(con, 'te00.mp4', str(tmp_path / 'f.json'), 3).tolist() == [0, 1, 0]


def test_export_counts_rejects_repeated_ids_and_a_missing_export(con, tmp_path):
    # The exporter never writes one object twice on a frame, and the cross-check needs the export.
    with pytest.raises(AssertionError, match='repeated id'):
        lq.export_counts(con, 'te00.mp4', write_export(tmp_path / 'x.json', [[1, 1]]), 1)
    with pytest.raises(AssertionError, match='raw LEAP exports missing'):
        lq.export_counts(con, 'te00.mp4', str(tmp_path / 'missing.json'), 1)


def test_export_recovery_equals_p140_unwrap(con, tmp_path):
    # By construction, the corrected cache (p140's unwrap) and the interval recovery give the same counts.
    rng = np.random.default_rng(2)
    length, n = 60, 50
    frames = [[] for _ in range(length)]
    for track_id in range(1, 30):
        # Random intervals, some starting before the clip (wrapped by the exporter), some ending past it; an interval
        # that wraps onto itself writes each frame once, as the exporter does (common.load_leap_exports rejects repeats).
        s, e = sorted(rng.integers(-20, length + 1, 2).tolist())
        for f in range(s, max(e, s + 1)):
            if f < length and track_id not in frames[f]:
                frames[f].append(track_id)
    unwrapped, _ = unwrap_leap_frames([[{'track_id': t} for t in f] for f in frames], n)
    cache_counts = [len({d['track_id'] for d in f}) for f in unwrapped] + [0] * (n - len(unwrapped))
    assert lq.export_counts(con, 'te00.mp4', write_export(tmp_path / 'x.json', frames), n).tolist() == cache_counts


def write_unwrap(path, clip_frames, export_frames, objects=0, wrapped=(), ambiguous=(), wrapped_boxes=0, past=0):
    # One p140 unwrap.json.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(export_frames=export_frames, clip_frames=clip_frames, objects=objects,
                                    wrapped_objects=list(wrapped), ambiguous_full_span_objects=list(ambiguous),
                                    wrapped_boxes_dropped=wrapped_boxes, boxes_past_clip_dropped=past)))
    return str(path)


def test_leap_unwrap_returns_the_totals(con, tmp_path):
    # te00's export is empty; te01 has one wrapped and two ambiguous objects (common.unwrap_totals' fields).
    set_clips(con, [5, 7])
    files = [('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 0)),
             ('te01.mp4', write_unwrap(tmp_path / 'b.json', 7, 9, objects=4, wrapped=[3], ambiguous=[4, 5], wrapped_boxes=2, past=6))]
    totals = lq.leap_unwrap(con, files)
    assert totals == dict(clips=2, empty_exports=1, objects=4, wrapped_objects=1, ambiguous_full_span_objects=2,
                          clips_with_ambiguous=1, wrapped_boxes_dropped=2, boxes_past_clip_dropped=6,
                          empty_export_videos=['te00.mp4'])
    # The table keeps the clip length under the name n, as table clips.
    assert con.execute('SELECT video, n FROM unwrap ORDER BY video').fetchall() == [('te00.mp4', 5), ('te01.mp4', 7)]


def test_leap_unwrap_rejects_missing_files_and_other_clip_lengths(con, tmp_path):
    # A missing unwrap.json means an old cache; a different clip length means p140 clipped elsewhere.
    set_clips(con, [5])
    with pytest.raises(AssertionError, match='unwrap.json files'):
        lq.leap_unwrap(con, [('te00.mp4', str(tmp_path / 'missing.json'))])
    with pytest.raises(AssertionError, match='clip lengths differ'):
        lq.leap_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'a.json', 6, 6))])


def test_leap_unwrap_rejects_missing_and_extra_clips(con, tmp_path):
    # Two clips, but statistics for only one of them, or for one that is not a clip.
    set_clips(con, [5, 6])
    with pytest.raises(AssertionError, match='clip lengths differ'):
        lq.leap_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 5))])
    with pytest.raises(AssertionError, match='clip lengths differ'):
        lq.leap_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'a.json', 5, 5)),
                             ('te01.mp4', write_unwrap(tmp_path / 'b.json', 6, 6)),
                             ('te02.mp4', write_unwrap(tmp_path / 'c.json', 4, 4))])


def cross_check_setup(con, tmp_path, cached_clip1):
    # Two 6-frame clips; te01 has a wrapped object, so the cross-check picks it; LEAP's cached counts on te01 as given.
    set_clips(con, [6, 6])
    lq.leap_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'u0.json', 6, 6, objects=1)),
                         ('te01.mp4', write_unwrap(tmp_path / 'u1.json', 6, 8, objects=2, wrapped=[2]))])
    # Raw export of te01 (video id 1): id 2 wrapped (head [0, 1], tail [6, 7]) counts on [0, 2); id 1 on [2, 4).
    write_export(tmp_path / 'sota' / 'jnc0' / 'leap' / '1.json', [[2], [2], [1], [1], [], [], [2], [2]])
    pooled_tables(con)
    fill(con, 'pred', ['leap', 'param 0'], [1] * 6 + cached_clip1)
    return lq.cross_check_leap(con, 'jnc0', tmp_path / 'sota')


def test_cross_check_picks_the_most_corrected_clip_and_matches(con, tmp_path):
    # The cached counts equal the recovery on te01.
    check = cross_check_setup(con, tmp_path, [1, 1, 1, 1, 0, 0])
    assert check['video'] == 'te01.mp4' and check['wrapped'] == 1 and check['equal'] and check['frames_differing'] == 0
    assert check['recovered_box_frames'] == check['cached_box_frames'] == 4


def test_cross_check_reports_differing_frames(con, tmp_path):
    # A cache that still holds the wrapped tail on the last frames differs on two frames.
    check = cross_check_setup(con, tmp_path, [1, 1, 1, 1, 1, 1])
    assert not check['equal'] and check['frames_differing'] == 2


def test_cross_check_requires_the_raw_export(con, tmp_path):
    # Without the raw export there is nothing to compare against.
    set_clips(con, [6])
    lq.leap_unwrap(con, [('te00.mp4', write_unwrap(tmp_path / 'u0.json', 6, 6))])
    with pytest.raises(AssertionError, match='raw LEAP exports missing'):
        lq.cross_check_leap(con, 'jnc0', tmp_path / 'sota')


# ---------------------------------------------------------------- lag diagnostic

def test_lag_correlation_finds_shift(con):
    # Ground truth is a random walk; system 'shifted' reports ground-truth frame f + 3 at frame f.
    rng = np.random.default_rng(3)
    set_clips(con, [300, 300])
    g = rng.integers(0, 6, 600)
    shifted = np.concatenate([np.roll(g[:300], -3), np.roll(g[300:], -3)])
    pooled_tables(con)
    fill(con, 'gt', ['all'], g)
    fill(con, 'pred', ['aligned', 'c'], g)
    fill(con, 'pred', ['shifted', 'c'], shifted)
    fill(con, 'pred', ['flat', 'c'], np.ones(600))
    fill(con, 'pred', ['baseline', 'one'], np.ones(600))
    out = {r['system']: r for r in lq.lag_correlations(con)}
    assert out['aligned']['best_lag'] == 0 and out['aligned']['lag0_r'] == pytest.approx(1.0) and not out['aligned']['warn']
    assert out['shifted']['best_lag'] == 3 and out['shifted']['best_r'] == pytest.approx(1.0) and out['shifted']['warn']
    # A constant count has no correlation on any clip, no warning, and best lag 0.
    assert out['flat']['lag0_r'] is None and out['flat']['clips_with_r'] == 0 and not out['flat']['warn']
    assert out['flat']['best_lag'] == 0
    # Constant answers are not diagnosed.
    assert 'baseline' not in out


def test_lag_ties_go_to_the_smallest_lag_then_the_negative_one(con):
    # Period-4 counts shifted by 2 correlate perfectly at lags -2, 2, -6, 6, ...; the tie goes to -2.
    set_clips(con, [400])
    g = np.tile([1, 2, 1, 0], 100)
    pooled_tables(con)
    fill(con, 'gt', ['all'], g)
    fill(con, 'pred', ['s', 'c'], np.roll(g, -2))
    (r,) = lq.lag_correlations(con)
    assert r['best_lag'] == -2 and r['lag0_r'] == pytest.approx(-1.0) and r['warn']


# ---------------------------------------------------------------- top-k over the pooled tables

def test_score_topk_aligns_configurations_with_their_counts(con):
    # Configuration 'b' predicts the ground truth's peaks and 'a' predicts nothing; results must not swap.
    set_clips(con, [1200])
    set_configs(con, [('s', 'b'), ('s', 'a')])
    g = np.zeros(1200, dtype=np.int64)
    g[np.arange(11) * 100 + 30] = np.arange(11) + 1
    pooled_tables(con)
    for variant in lq.GT_VARIANTS:
        fill(con, 'gt', [variant], g)
    fill(con, 'pred', ['s', 'b'], g)
    fill(con, 'pred', ['s', 'a'], np.zeros(1200))
    out, taus = lq.score_topk(con)
    assert set(out) == {(v, 's', c) for v in lq.GT_VARIANTS for c in 'ab'}
    b, a = out[('all', 's', 'b')], out[('all', 's', 'a')]
    assert b['topk_paper'] == 1.0 and b['topk_leap_code'] == 1.0 and b['topk_leap_code_default_sort'] == 1.0
    assert a['topk_paper'] == 0.0 and a['returned_min'] == 0
    # tau is the 10th largest peak (2) and does not move under random ground-truth ties.
    assert taus['all'] == dict(tau=2, tau_min_over_draws=2, tau_max_over_draws=2)


# ---------------------------------------------------------------- operating points

def record(system, config, fps, f1, acc, topk, hota=0.5):
    # A configs.json record with only the fields operating points read.
    return dict(system=system, config=config, fps=fps, hota=hota, selection=dict(f1=f1),
                aggregation=dict(acc_leap=acc), topk=dict(topk_paper=topk))


def constant_answers(f1=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 0.0), topk=(0.0, 0.0, 0.0)):
    # Baselines zero, one, and median with the given accuracies per query.
    return {name: dict(selection=dict(f1=f), aggregation=dict(acc_leap=a), topk=dict(topk_paper=t))
            for name, f, a, t in zip(('zero', 'one', 'median'), f1, acc, topk)}


def dataset(rows, baselines=None):
    # One scored dataset with the same records and baselines under both ground-truth variants.
    return dict(records={v: rows for v in lq.GT_VARIANTS}, baselines={v: baselines or constant_answers() for v in lq.GT_VARIANTS})


def test_operating_points_skip_undefined_reference_and_fit_leap(con):
    # Dataset A: well defined. Dataset B: the reference's top-k accuracy is 0, so top-k has no relative bound there.
    rows_a = [record('reference', 'naive', 10, 0.9, 0.8, 0.9), record('tetris', 't1', 100, 0.89, 0.79, 0.9),
              record('tetris', 't2', 200, 0.5, 0.5, 0.5), record('otif', 'param 0', 50, 0.88, 0.78, 0.9),
              record('leap', 'param 0', 300, 0.6, 0.4, 0.3),
              record('leap_ambiguous_dropped', 'param 0', 300, 0.61, 0.45, 0.3, hota=None)]
    rows_b = [record('reference', 'naive', 10, 0.9, 0.8, 0.0), record('tetris', 't1', 100, 0.9, 0.8, 0.2),
              record('otif', 'param 0', 50, 0.9, 0.8, 0.1), record('leap', 'param 0', 300, 0.5, 0.5, 0.1),
              record('leap_ambiguous_dropped', 'param 0', 300, 0.5, 0.5, 0.1, hota=None)]
    out = lq.compute_operating_points(con, {'A': dataset(rows_a), 'B': dataset(rows_b)}, [0.05])
    topk = out['all']['topk']
    # B is skipped for top-k only, with the reason recorded.
    assert topk['skipped'] == [dict(dataset='B', reference_accuracy=0.0, reason='reference accuracy is not positive')]
    assert set(topk['bounds']['0.05']) == {'A'} and set(out['all']['selection']['bounds']['0.05']) == {'A', 'B'}
    # At 5% on A, Tetris's fastest qualifying configuration is t1, 10x the reference and 2x OTIF; LEAP misses.
    a = out['all']['selection']['bounds']['0.05']['A']
    assert a['systems']['tetris']['config'] == 't1' and a['systems']['tetris']['speedup_vs_reference'] == 10.0
    assert a['tetris_speedup_vs_otif'] == 2.0 and not a['systems']['leap']['meets_bound']
    # At LEAP's own selection accuracy (0.6) Tetris can use t1 still (t2 at 0.5 misses), so it is 1/3 LEAP's speed.
    fit = out['all']['selection']['bounds']['leap_fit']['A']
    assert fit['bound'] == 0.6 and fit['systems']['leap']['meets_bound'] and fit['tetris_speedup_vs_leap'] == 0.3
    # The ambiguous-dropped variant is held to the same bound and has no HOTA of its own.
    assert fit['systems']['leap_ambiguous_dropped']['hota'] is None
    # Constant answers at 0 meet no bound here, and both LEAP readings agree.
    assert not a['trivial'] and not fit['trivial'] and not fit['leap_ambiguity_dependent']


def test_operating_points_with_every_reference_undefined(con):
    # When no dataset has a positive reference accuracy, the query has no bounds at all.
    rows = [record('reference', 'naive', 10, 0.0, 0.0, 0.0), record('tetris', 't1', 100, 0.0, 0.0, 0.0),
            record('otif', 'param 0', 50, 0.0, 0.0, 0.0), record('leap', 'param 0', 300, 0.0, 0.0, 0.0),
            record('leap_ambiguous_dropped', 'param 0', 300, 0.0, 0.0, 0.0)]
    out = lq.compute_operating_points(con, {'A': dataset(rows)}, [0.05])
    assert out['moving']['aggregation']['bounds'] == {} and len(out['moving']['aggregation']['skipped']) == 1


def test_operating_points_with_null_accuracies(con):
    # A null reference accuracy skips the dataset; a configuration with a null accuracy cannot qualify but does not break the rest.
    rows_a = [record('reference', 'naive', 10, 0.9, 0.8, 0.9), record('tetris', 't1', 100, None, 0.8, 0.9),
              record('tetris', 't2', 50, 0.9, 0.8, 0.9), record('otif', 'param 0', 50, None, 0.8, 0.9),
              record('leap', 'param 0', 300, 0.6, 0.4, 0.3), record('leap_ambiguous_dropped', 'param 0', 300, 0.6, 0.4, 0.3)]
    rows_b = [record('reference', 'naive', 10, None, 0.8, 0.9), *rows_a[1:]]
    out = lq.compute_operating_points(con, {'A': dataset(rows_a), 'B': dataset(rows_b)}, [0.05])
    selection = out['all']['selection']
    assert [s['dataset'] for s in selection['skipped']] == ['B'] and set(selection['bounds']['0.05']) == {'A'}
    a = selection['bounds']['0.05']['A']['systems']
    assert a['tetris']['config'] == 't2' and 'otif' not in a


def test_operating_points_flag_trivial_cells_non_informative_datasets_and_leap_ambiguity(con):
    # Answering 'one' everywhere reaches F1 0.95, above the 5% bound (0.855) but below the reference (0.9 < 0.95 is false).
    # Answering 'zero' reaches acc 0.85, at least the reference's 0.8: aggregation there is non-informative.
    rows = [record('reference', 'naive', 10, 0.9, 0.8, 0.9), record('tetris', 't1', 100, 0.89, 0.79, 0.9),
            record('otif', 'param 0', 50, 0.88, 0.78, 0.9), record('leap', 'param 0', 300, 0.6, 0.4, 0.3),
            record('leap_ambiguous_dropped', 'param 0', 300, 0.5, 0.3, 0.2)]
    baselines = constant_answers(f1=(0.0, 0.95, 0.7), acc=(0.85, 0.2, 0.5), topk=(0.0, 0.05, 0.05))
    out = lq.compute_operating_points(con, {'A': dataset(rows, baselines)}, [0.05])['all']
    sel = out['selection']['bounds']['0.05']['A']
    assert sel['trivial'] and sel['trivial_baseline'] == dict(name='one', accuracy=0.95)
    assert out['selection']['non_informative'] == [dict(dataset='A', reference_accuracy=0.9, baseline='one', baseline_accuracy=0.95)]
    assert out['aggregation']['non_informative'][0]['baseline'] == 'zero' and out['aggregation']['bounds']['0.05']['A']['trivial']
    # Top-k: the best constant reaches 0.05, below every bound.
    assert not out['topk']['bounds']['0.05']['A']['trivial'] and out['topk']['non_informative'] == []
    # At leap_fit, LEAP meets its own accuracy and the ambiguous-dropped reading does not.
    fit = out['topk']['bounds']['leap_fit']['A']
    assert fit['leap_ambiguity_dependent'] and fit['systems']['leap']['meets_bound']
    assert not fit['systems']['leap_ambiguous_dropped']['meets_bound']


def test_operating_points_use_unrounded_accuracies(con):
    # A configuration 3e-5 below the bound misses it; rounding to 4 decimals first would have let it qualify.
    rows = [record('reference', 'naive', 10, 0.9, 0.8, 0.9), record('tetris', 't1', 100, 0.855 - 3e-5, 0.8, 0.9),
            record('tetris', 't2', 50, 0.9, 0.8, 0.9), record('otif', 'param 0', 50, 0.9, 0.8, 0.9),
            record('leap', 'param 0', 300, 0.6, 0.4, 0.3), record('leap_ambiguous_dropped', 'param 0', 300, 0.6, 0.4, 0.3)]
    out = lq.compute_operating_points(con, {'A': dataset(rows)}, [0.05])
    assert out['all']['selection']['bounds']['0.05']['A']['systems']['tetris']['config'] == 't2'


# ---------------------------------------------------------------- one dataset end to end

def leap_export(n):
    # A LEAP export longer than the n-frame clip: id 1 on [10, 60), id 2 wrapped (start before the clip), id 3 on every frame.
    length = n + 10
    frames = [[] for _ in range(length)]
    for f in range(length):
        frames[f] += [1] * (10 <= f < 60) + [2] * (f < 5 or f >= length - 10) + [3]
    return [[dict(track_id=t, left=10.0, top=10.0, right=20.0, bottom=20.0, **{'class': 'car'}) for t in f] for f in frames]


def end_to_end(monkeypatch, tmp_path, leap_configs=('param 0',)):
    # A dataset 'jnc0' with clips te00 (800 frames) and te01 (600 frames), every input faked under tmp_path.
    lengths = {'te00.mp4': 800, 'te01.mp4': 600}
    (tmp_path / 'videos').mkdir()
    for v in lengths:
        (tmp_path / 'videos' / v).write_text('')
    (tmp_path / 'videos' / 'notes.txt').write_text('')
    monkeypatch.setattr(store, 'dataset', lambda dataset, *args: tmp_path / 'videos')
    fake_clips(monkeypatch, tmp_path, {v: range(n) for v, n in lengths.items()}, lengths)
    monkeypatch.setattr(lq, 'system_configs', lambda dataset: [
        dict(system='reference', config='naive', fps=10.0, hota=0.9), dict(system='tetris', config='t1', fps=100.0, hota=0.8),
        dict(system='otif', config='param 0', fps=50.0, hota=0.0),
        *[dict(system='leap', config=c, fps=300.0, hota=0.1) for c in leap_configs]])
    out = tmp_path / 'out'
    monkeypatch.setattr(lq, 'output_path', lambda dataset, system, config, video: out / system / config / video / 'tracking.jsonl')
    for v, n in lengths.items():
        # Ground truth: one track per 20 frames, each moving 3 px per frame for 40 frames.
        rows = [(f, [box(t, 100 + 3 * (f - 20 * t), 100 + 30 * t) for t in range(n // 20) if 0 <= f - 20 * t < 40])
                for f in range(n)]
        for system, config in [('ground truth', '003_groundtruth'), ('reference', 'naive'), ('tetris', 't1')]:
            write_tracks(out / system / config / v / 'tracking.jsonl', rows)
        write_tracks(out / 'otif' / 'param 0' / v / 'tracking.jsonl', [])
    # LEAP: te00 (video id 0) has a wrapped and an ambiguous object, te01's export is empty.
    for v, export in [('te00.mp4', leap_export(800)), ('te01.mp4', [])]:
        raw = tmp_path / 'sota' / 'jnc0' / 'leap' / f'{lq.extract_video_id(v)}.json'
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(export))
        unwrapped, stats = unwrap_leap_frames(export, lengths[v])
        for c in leap_configs:
            write_tracks(out / 'leap' / c / v / 'tracking.jsonl',
                         [(f, [[d['track_id'], d['left'], d['top'], d['right'], d['bottom']] for d in boxes])
                          for f, boxes in enumerate(unwrapped)])
            (out / 'leap' / c / v / 'unwrap.json').write_text(json.dumps(stats))
    return lq.score_dataset(duckdb.connect(), 'jnc0', None, tmp_path / 'sota')


def test_score_dataset_end_to_end(monkeypatch, tmp_path):
    scored, diagnostics = end_to_end(monkeypatch, tmp_path)
    assert scored['clips'] == 2 and scored['frames'] == 1400
    assert scored['ground_truth']['kind'] == 'automatic 003_groundtruth' and scored['ground_truth']['provisional']
    records = {r['system']: r for r in scored['records']['all']}
    assert set(records) == {'reference', 'tetris', 'otif', 'leap', 'leap_ambiguous_dropped'}
    # The reference copies the ground truth exactly; OTIF's empty output finds nothing.
    assert records['reference']['selection']['f1'] == 1.0 and records['reference']['aggregation']['acc_leap'] == 1.0
    assert records['reference']['topk']['topk_paper'] == 1.0
    assert records['otif']['aggregation']['mean_predicted'] == 0 and records['otif']['topk']['topk_paper'] == 0.0
    # Throughput and HOTA come from system_configs; the ambiguous-dropped reading has LEAP's throughput and no HOTA.
    assert (records['tetris']['fps'], records['tetris']['hota']) == (100.0, 0.8)
    assert (records['leap_ambiguous_dropped']['fps'], records['leap_ambiguous_dropped']['hota']) == (300.0, None)
    # LEAP on te00: id 3 on all 800 frames, id 1 on [10, 60), id 2 on [0, 5); dropping id 3 removes 800 box-frames.
    leap, dropped = records['leap']['aggregation']['mean_predicted'], records['leap_ambiguous_dropped']['aggregation']['mean_predicted']
    assert leap * 1400 == pytest.approx(855) and dropped * 1400 == pytest.approx(55)
    # Constant answers per variant, with the variant's own median.
    assert set(scored['baselines']['all']) == {'zero', 'one', 'median'}
    assert scored['baselines']['all']['one']['selection']['f1'] == 1.0 and scored['baselines']['all']['zero']['selection']['f1'] == 0.0
    assert scored['ground_truth_summary']['all']['all_frames_positive']
    # Diagnostics: the cross-check on the corrected clip, unwrap totals, LEAP without its empty export, no dropped frames.
    check = diagnostics['leap_cache_cross_check']
    assert check['video'] == 'te00.mp4' and check['wrapped'] == 1 and check['ambiguous'] == 1 and check['equal']
    assert diagnostics['leap_unwrap']['empty_export_videos'] == ['te01.mp4']
    assert diagnostics['leap_excluding_empty_exports']['all']['clips'] == 1
    assert diagnostics['leap_excluding_empty_exports']['all']['frames'] == 800
    assert diagnostics['frames_dropped_past_clip'] == []
    assert {r['system'] for r in diagnostics['lag_correlations']} == set(records)


def test_score_dataset_requires_exactly_one_leap_configuration(monkeypatch, tmp_path):
    # unwrap.json, the ambiguous ids, and the cross-check assume a single LEAP configuration.
    with pytest.raises(AssertionError, match='exactly one LEAP configuration'):
        end_to_end(monkeypatch, tmp_path, leap_configs=('param 0', 'param 1'))
