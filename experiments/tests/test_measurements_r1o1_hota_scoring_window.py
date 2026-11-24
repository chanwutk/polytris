import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from measurements import r1o1_hota_scoring_window as sw
from measurements.common import output_path


# The synthetic dataset: one name that needs no root-dataset mapping, and its label.
DATASET = 'jnc0'
LABEL = 'B3D1'
# Clip frame counts.
CLIPS = {'te00.mp4': 10, 'te01.mp4': 8, 'te02.mp4': 6}
# One object per clip, with the same id in the ground truth and in every configuration's tracks.
OBJECTS = {'te00.mp4': 1, 'te01.mp4': 2, 'te02.mp4': 3}
# The ground truth's frames per clip (None writes an empty file); te01's ends early.
GROUND_TRUTH = {'te00.mp4': range(10), 'te01.mp4': range(4), 'te02.mp4': range(6)}
# Each configuration's frames with the ground truth's box per clip (None writes an empty file), and its throughput.
#   reference: perfect.  tetris A: stops after frame 4 on te00 (5 misses not scored), fastest.  tetris B: perfect.
#   otif: runs to frame 7 on te01 past the ground truth's frame 3 (4 false positives not scored), empty te02.
#   leap: stops after frame 2 on te00 (7 misses not scored).
CONFIGS = {
    ('reference', 'naive'): dict(fps=1.0, frames={'te00.mp4': range(10), 'te01.mp4': range(4), 'te02.mp4': range(6)}),
    ('tetris', 'A'): dict(fps=100.0, frames={'te00.mp4': range(5), 'te01.mp4': range(4), 'te02.mp4': range(6)}),
    ('tetris', 'B'): dict(fps=50.0, frames={'te00.mp4': range(10), 'te01.mp4': range(4), 'te02.mp4': range(6)}),
    ('otif', 'param 0'): dict(fps=20.0, frames={'te00.mp4': range(10), 'te01.mp4': range(8), 'te02.mp4': None}),
    ('leap', 'param 0'): dict(fps=10.0, frames={'te00.mp4': range(3), 'te01.mp4': range(4), 'te02.mp4': range(6)}),
}


def box(k):
    # Object k's constant box [x1, y1, x2, y2].
    return [10.0 * k, 10.0 * k, 10.0 * k + 8, 10.0 * k + 8]


def write_lines(path, rows):
    # A tracking.jsonl with one line per (frame_idx, tracks) pair; an empty list writes an empty file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': f, 'tracks': t}) + '\n' for f, t in rows))
    return path


def clip_rows(video, frames):
    # The clip's object on the given frames, with the ground truth's id and box.
    oid = OBJECTS[video]
    return [] if frames is None else [(f, [[oid, *box(oid)]]) for f in frames]


def flat(summary):
    # The fields of a common.metric_summary the assertions read, flat.
    hota, alpha, count = summary['HOTA'], summary['HOTA']['alpha_0.05'], summary['Count']
    return dict(gt_dets=count['GT_Dets'], dets=count['Dets'], TP=alpha['HOTA_TP'], FN=alpha['HOTA_FN'],
                FP=alpha['HOTA_FP'], HOTA=hota['HOTA'], DetA=hota['DetA'], AssA=hota['AssA'])


def to_json(combined):
    # TrackEval's combined HOTA and Count as DATASET.json stores them (arrays as lists).
    return {family: {k: (np.asarray(v).tolist() if np.ndim(v) else v.item() if hasattr(v, 'item') else v)
                     for k, v in combined[family].items()} for family in ('HOTA', 'Count')}


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def build_dataset(tmp_path, monkeypatch, ground_truth):
    """The synthetic dataset in a cache under tmp_path with `ground_truth`'s frames per clip: ground truth, every
    configuration's tracks, and each configuration's DATASET.json from TrackEval on the unmodified files, as p110 and
    p141 write it; returns the rows system_configs gives, which every lookup of the module is pointed at."""
    # The cache, the split listing, the frame counts, and the tradeoff frame count point at the synthetic dataset.
    monkeypatch.setattr(sw.cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(sw, 'list_split_videos', lambda ds, split: sorted(CLIPS))
    monkeypatch.setattr(sw, 'get_video_frame_count', lambda ds, video: CLIPS[video])
    monkeypatch.setattr(sw, 'load_test_tradeoff', lambda ds: (None, {'frame_count': sum(CLIPS.values())}))
    # The ground truth of every clip.
    for video, frames in ground_truth.items():
        write_lines(output_path(DATASET, 'ground truth', '003_groundtruth', video), clip_rows(video, frames))
    rows = []
    for (system, config), spec in CONFIGS.items():
        # The configuration's tracks per clip.
        for video, frames in spec['frames'].items():
            write_lines(output_path(DATASET, system, config, video), clip_rows(video, frames))
        # Its cached combined result, from TrackEval on the files as they are.
        gt_dir, gt_file = sw.split_path(output_path(DATASET, 'ground truth', '', 'te00.mp4'), 'te00.mp4')
        track_dir, track_file = sw.split_path(output_path(DATASET, system, config, 'te00.mp4'), 'te00.mp4')
        combined = sw.trackeval_rerun(gt_dir, gt_file, track_dir, track_file, sorted(CLIPS), str(tmp_path / 'te' / system / config),
                                      pad_scoring_window=False)
        accuracy = sw.accuracy_path(DATASET, system, config)
        accuracy.parent.mkdir(parents=True, exist_ok=True)
        accuracy.write_text(json.dumps({'video': None, 'metrics': to_json(combined)}))
        # The tradeoff table's row, with HOTA as p130/p142 average it.
        rows.append(dict(system=system, config=config, fps=spec['fps'], hota=float(np.mean(combined['HOTA']['HOTA']))))
    monkeypatch.setattr(sw, 'system_configs', lambda ds: [dict(r) for r in rows])
    return rows


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # The synthetic dataset with the default ground truth.
    return build_dataset(tmp_path, monkeypatch, GROUND_TRUTH)


def analyze(con):
    # Everything before the TrackEval runs, as main does it.
    sw.register_inputs(con, [DATASET])
    sw.load_rows(con)
    sw.scoring_window(con)
    sw.load_cached(con)


def window(con, system, config):
    # One configuration's window row.
    return sw.query_dicts(con, 'SELECT * FROM windows WHERE system = ? AND config = ?', [system, config])[0]


def run_all(con, tmp_path, reproduce=frozenset()):
    # Every padded run, and unpadded runs of `reproduce`, in this process.
    tasks = sw.rescore_tasks(con, tmp_path, sw.padded_ground_truth(con, tmp_path), set(reproduce))
    runs = [sw.rescore(t) for t in tasks]
    padded = {(r['dataset'], r['system'], r['config']): r['metrics'] for r in runs if r['padded']}
    unpadded = {(r['dataset'], r['system'], r['config']): r['metrics'] for r in runs if not r['padded']}
    return padded, unpadded


# ---------------------------------------------------------------- the mechanism on TrackEval itself

def score(root, gt_rows, track_rows):
    # TrackEval's HOTA on one clip, configured as p110 and p141.
    write_lines(root / 'gt' / 'v' / 't.jsonl', gt_rows)
    write_lines(root / 'tr' / 'v' / 't.jsonl', track_rows)
    combined = sw.trackeval_rerun(str(root / 'gt'), 't.jsonl', str(root / 'tr'), 't.jsonl', ['v'], str(root / 'out'),
                                  pad_scoring_window=False)
    return sw.metric_summary(combined)


def test_trackeval_scores_only_the_frames_both_files_reach(tmp_path):
    full = [(f, [[1, *box(1)]]) for f in range(10)]
    half = full[:5]
    # The tracks end after frame 4: the five later ground-truth boxes are counted, not scored, and HOTA is perfect.
    m = score(tmp_path / 'a', full, half)
    assert (flat(m)['gt_dets'], flat(m)['TP'], flat(m)['FN'], flat(m)['HOTA']) == (10, 5, 0, 1.0)
    assert sw.unscored_boxes(m) == dict(gt=5, tracker=0)
    # Padding the tracks to the clip makes those boxes misses.
    m = flat(score(tmp_path / 'b', full, half + [(9, [])]))
    assert (m['TP'], m['FN'], m['DetA'], m['AssA'], m['HOTA']) == (5, 5, 0.5, 0.5, 0.5)
    # The ground truth ends after frame 4: the five later tracker boxes are not false positives.
    m = score(tmp_path / 'c', half, full)
    assert (flat(m)['dets'], flat(m)['TP'], flat(m)['FP'], flat(m)['HOTA']) == (10, 5, 0, 1.0)
    assert sw.unscored_boxes(m) == dict(gt=0, tracker=5)
    # Padding the ground truth makes them false positives.
    m = flat(score(tmp_path / 'd', half + [(9, [])], full))
    assert (m['TP'], m['FP'], m['DetA']) == (5, 5, 0.5)
    # Tracks without any box take the early exit: every ground-truth box is a miss, nothing is left out.
    m = score(tmp_path / 'e', full, [])
    assert (flat(m)['TP'], flat(m)['FN'], flat(m)['HOTA']) == (0, 10, 0.0) and sw.unscored_boxes(m) == dict(gt=0, tracker=0)
    # Ground truth without any box: every tracker box is a false positive, also from an empty ground-truth file.
    for name, gt in (('f', [(f, []) for f in range(10)]), ('g', [])):
        m = score(tmp_path / name, gt, half)
        assert (flat(m)['TP'], flat(m)['FP']) == (0, 5) and sw.unscored_boxes(m) == dict(gt=0, tracker=0)


# ---------------------------------------------------------------- paths and padded copies

def test_split_path():
    # The directory and the file below the clip's directory, as TrackEval's input_*_dir and input_*.
    assert sw.split_path(Path('/c/jnc0/execution/te00.mp4/060_uncompressed_tracks/A/tracking.jsonl'), 'te00.mp4') == \
        ('/c/jnc0/execution', '060_uncompressed_tracks/A/tracking.jsonl')
    # A path outside the clip's directory is rejected.
    with pytest.raises(AssertionError, match='not under'):
        sw.split_path(Path('/c/te01.mp4/tracking.jsonl'), 'te00.mp4')


def test_accuracy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sw.cache, 'CACHE_DIR', str(tmp_path))
    # p110's directory by variant id, p141's by three-digit parameter id.
    assert sw.accuracy_path('jnc0', 'tetris', 'A') == tmp_path / 'jnc0' / 'evaluation' / '070_accuracy' / 'raw' / 'test' / 'A' / 'DATASET.json'
    assert sw.accuracy_path('jnc0', 'reference', 'naive').parent.name == 'naive'
    assert sw.accuracy_path('jnc0', 'otif', 'param 7') == tmp_path / 'SOTA' / 'otif' / 'jnc0' / 'accuracy' / 'raw' / 'test' / '007' / 'DATASET.json'
    # Anything else is unknown.
    with pytest.raises(AssertionError, match='unknown system'):
        sw.accuracy_path('jnc0', 'ocsort', 'param 0')


def test_write_padded(tmp_path):
    source = write_lines(tmp_path / 's.jsonl', [(0, [[1, *box(1)]]), (1, [])])
    # A file ending before the clip's last frame gets one empty row there.
    sw.write_padded(str(source), tmp_path / 'p' / 'a.jsonl', n=5, t=2)
    lines = [json.loads(x) for x in (tmp_path / 'p' / 'a.jsonl').read_text().splitlines()]
    assert lines[:2] == [json.loads(x) for x in source.read_text().splitlines()] and lines[2] == {'frame_idx': 4, 'tracks': []}
    # A file already reaching the clip's last frame is copied unchanged.
    sw.write_padded(str(source), tmp_path / 'p' / 'b.jsonl', n=2, t=2)
    assert (tmp_path / 'p' / 'b.jsonl').read_bytes() == source.read_bytes()
    # An empty file becomes the padding row alone.
    empty = tmp_path / 'e.jsonl'
    empty.write_text('')
    sw.write_padded(str(empty), tmp_path / 'p' / 'c.jsonl', n=3, t=0)
    assert (tmp_path / 'p' / 'c.jsonl').read_text() == '{"frame_idx": 2, "tracks": []}\n'
    # A last line without its newline is terminated before the padding row.
    unterminated = tmp_path / 'u.jsonl'
    unterminated.write_text('{"frame_idx": 0, "tracks": []}')
    sw.write_padded(str(unterminated), tmp_path / 'p' / 'd.jsonl', n=2, t=1)
    assert (tmp_path / 'p' / 'd.jsonl').read_text().splitlines() == ['{"frame_idx": 0, "tracks": []}',
                                                                     '{"frame_idx": 1, "tracks": []}']
    # A file past its clip cannot be padded to it.
    with pytest.raises(AssertionError, match='past its clip'):
        sw.write_padded(str(source), tmp_path / 'p' / 'f.jsonl', n=1, t=2)


# ---------------------------------------------------------------- inputs and the scoring window

def test_register_inputs_requires_every_file_and_the_frame_count(con, dataset, monkeypatch):
    # The synthetic dataset registers every clip, configuration, and file.
    sw.register_inputs(con, [DATASET])
    assert con.execute('SELECT count(*) FROM clips').fetchone()[0] == 3
    assert con.execute('SELECT count(*) FROM track_files').fetchone()[0] == 3 * len(CONFIGS)
    assert con.execute('SELECT count(*) FROM accuracy_files').fetchone()[0] == len(CONFIGS)
    # Clips that do not add up to the tradeoff frame count are not the scored set.
    monkeypatch.setattr(sw, 'load_test_tradeoff', lambda ds: (None, {'frame_count': 25}))
    with pytest.raises(AssertionError, match='add up'):
        sw.register_inputs(con, [DATASET])
    monkeypatch.setattr(sw, 'load_test_tradeoff', lambda ds: (None, {'frame_count': 24}))
    # A missing tracking file is named.
    output_path(DATASET, 'tetris', 'B', 'te01.mp4').unlink()
    with pytest.raises(AssertionError, match='inputs missing'):
        sw.register_inputs(con, [DATASET])


def test_register_inputs_rejects_duplicate_configurations(con, dataset, monkeypatch):
    # The same configuration twice would double its clips.
    monkeypatch.setattr(sw, 'system_configs', lambda ds: [dict(r) for r in dataset] + [dict(dataset[1])])
    with pytest.raises(AssertionError, match='duplicate'):
        sw.register_inputs(con, [DATASET])


def test_scoring_window_counts(con, dataset):
    analyze(con)
    # Tetris A: te00's five boxes after its frame 4 are unscored; its te01 file ends with the ground truth's.
    a = window(con, 'tetris', 'A')
    assert (a['gt_boxes'], a['gt_boxes_unscored'], a['gt_rows_unscored'], a['gt_box_frames_unscored']) == (20, 5, 5, 5)
    assert a['gt_unscored_share'] == 0.25 and (a['tracker_boxes'], a['tracker_boxes_unscored']) == (15, 0)
    assert (a['clips_with_gt_unscored'], a['tracker_files_ending_before_clip_end'], a['gt_files_ending_before_clip_end'],
            a['tracker_frames_short_of_clip_end']) == (1, 2, 1, 9)
    # OTIF: four boxes after te01's ground truth ends; te02's empty file is scored in full (all misses), not truncated.
    o = window(con, 'otif', 'param 0')
    assert (o['tracker_boxes'], o['tracker_boxes_unscored'], o['tracker_rows_unscored'], o['tracker_box_frames_unscored'],
            o['tracker_unscored_share']) == (18, 4, 4, 4, 4 / 18)
    assert (o['gt_boxes_unscored'], o['clips_without_tracker_boxes'], o['clips_with_tracker_unscored']) == (0, 1, 1)
    # LEAP: seven boxes after te00's frame 2; the reference and Tetris B leave nothing out.
    assert window(con, 'leap', 'param 0')['gt_boxes_unscored'] == 7
    for system, config in (('reference', 'naive'), ('tetris', 'B')):
        w = window(con, system, config)
        assert (w['gt_boxes_unscored'], w['tracker_boxes_unscored']) == (0, 0)
    # No ground truth lies past any clip's frame count.
    assert (a['gt_boxes_past_frame_count'], a['gt_boxes_unscored_past_frame_count'],
            a['gt_unscored_share_within_frame_count'], a['clips_without_gt_boxes']) == (0, 0, 0.25, 0)


def test_window_of_a_clip_without_ground_truth(con, tmp_path, monkeypatch):
    # te02's ground truth has no box: TrackEval's early exit scores every tracker box there as a false positive.
    build_dataset(tmp_path, monkeypatch, {**GROUND_TRUTH, 'te02.mp4': None})
    analyze(con)
    ref = window(con, 'reference', 'naive')
    assert (ref['clips_without_gt_boxes'], ref['tracker_boxes'], ref['tracker_boxes_unscored'], ref['gt_boxes']) == (1, 20, 0, 14)
    # The clip is not truncated, and DATASET.json agrees to the box.
    assert not sw.query_dicts(con, "SELECT truncated FROM clip_windows WHERE system = 'reference' AND video = 'te02.mp4'")[0]['truncated']
    assert sw.cached_mismatches(con) == []
    # Padding changes nothing there: the six boxes stay false positives on both sides.
    padded, unpadded = run_all(con, tmp_path / 'work', {(DATASET, 'reference', 'naive')})
    metrics = sw.config_metrics(con, padded)
    sw.check_reproduction(unpadded, metrics)
    assert flat(padded[(DATASET, 'reference', 'naive')])['FP'] == flat(unpadded[(DATASET, 'reference', 'naive')])['FP'] == 6
    assert metrics[(DATASET, 'reference', 'naive')]['delta']['HOTA_change'] == pytest.approx(0.0, abs=1e-12)


def test_unscored_rows_without_boxes(con, dataset):
    # Tracks with empty rows up to the clip end: the ground truth's end leaves rows, but no box, unscored.
    write_lines(output_path(DATASET, 'tetris', 'B', 'te01.mp4'), clip_rows('te01.mp4', range(4)) + [(f, []) for f in range(4, 8)])
    sw.register_inputs(con, [DATASET])
    sw.load_rows(con)
    sw.scoring_window(con)
    b = window(con, 'tetris', 'B')
    assert (b['tracker_rows_unscored'], b['tracker_box_frames_unscored'], b['tracker_boxes_unscored']) == (4, 0, 0)
    # The ground truth starting after the tracks end: its rows past the tracks are the gap, not only its own rows.
    write_lines(output_path(DATASET, 'ground truth', '', 'te02.mp4'), [(f, [[3, *box(3)]]) for f in (4, 5)])
    write_lines(output_path(DATASET, 'tetris', 'B', 'te02.mp4'), [(0, [[3, *box(3)]])])
    sw.load_rows(con)
    sw.scoring_window(con)
    b = window(con, 'tetris', 'B')
    assert (b['gt_rows_unscored'], b['gt_box_frames_unscored'], b['gt_boxes_unscored']) == (5, 2, 2)


def test_window_matches_every_cached_result(con, dataset):
    analyze(con)
    # TrackEval's cached results leave out exactly the window's boxes.
    assert sw.cached_mismatches(con) == []


def test_cached_mismatches_catch_each_disagreement(con, dataset):
    analyze(con)
    # A tradeoff HOTA that differs from its DATASET.json.
    con.execute("UPDATE configs SET hota = hota - 0.01 WHERE system = 'tetris' AND config = 'B'")
    # A Count that does not hold every box.
    con.execute("UPDATE cached SET gt_dets = gt_dets + 1 WHERE system = 'leap'")
    # A cached result that scores a box the window says is unscored.
    con.execute("UPDATE cached SET fn = fn - 1 WHERE system = 'tetris' AND config = 'A'")
    bad = {(r['system'], r['config']) for r in sw.cached_mismatches(con)}
    assert bad == {('tetris', 'B'), ('leap', 'param 0'), ('tetris', 'A')}
    # The tracker side: a Count without every tracker box, and a false positive the window says is unscored.
    con.execute("UPDATE cached SET dets = dets - 1 WHERE system = 'reference'")
    con.execute("UPDATE cached SET fp = fp + 1 WHERE system = 'otif'")
    bad = {(r['system'], r['config']) for r in sw.cached_mismatches(con)}
    assert {('reference', 'naive'), ('otif', 'param 0')} <= bad
    # A configuration without its DATASET.json.
    con.execute("DELETE FROM cached WHERE system = 'otif'")
    assert ('otif', 'param 0') in {(r['system'], r['config']) for r in sw.cached_mismatches(con)}


def test_load_rows_rejects_what_trackeval_cannot_read(con, dataset):
    sw.register_inputs(con, [DATASET])
    path = output_path(DATASET, 'tetris', 'B', 'te00.mp4')
    # A repeated frame.
    write_lines(path, [(0, []), (0, [])])
    with pytest.raises(AssertionError, match='repeated'):
        sw.load_rows(con)
    # A line without its track list.
    path.write_text(json.dumps({'frame_idx': 0, 'tracks': None}) + '\n')
    with pytest.raises(AssertionError, match='without a track list'):
        sw.load_rows(con)
    # A negative frame, and a line without a frame.
    for line in ({'frame_idx': -1, 'tracks': []}, {'tracks': []}):
        path.write_text(json.dumps(line) + '\n')
        with pytest.raises(AssertionError, match='repeated, negative'):
            sw.load_rows(con)


def test_clip_lengths_follow_a_longer_ground_truth(con, tmp_path, monkeypatch):
    # The ground truth of te02 holds a box on frame 6 of a clip whose systems decode 6 frames.
    build_dataset(tmp_path, monkeypatch, {**GROUND_TRUTH, 'te02.mp4': range(7)})
    analyze(con)
    assert sw.long_clips(con) == [dict(dataset=LABEL, video='te02.mp4', frame_count=6, t_gt=7, gt_boxes_past_frame_count=1)]
    assert sw.query_dicts(con, "SELECT n FROM clip_lengths WHERE video = 'te02.mp4'") == [dict(n=7)]
    # The clip is 7 frames long, so no tracking file overruns it.
    assert sw.track_overruns(con) == []
    # The box on frame 6 is unscored for every configuration with boxes on te02, and counted apart.
    ref = window(con, 'reference', 'naive')
    assert (ref['gt_boxes_unscored'], ref['gt_boxes_unscored_past_frame_count'], ref['gt_boxes_past_frame_count'],
            ref['gt_unscored_share_within_frame_count']) == (1, 1, 1, 0.0)
    assert window(con, 'tetris', 'A')['gt_boxes_unscored'] == 6 and sw.cached_mismatches(con) == []
    # Padded to 7 frames, every configuration scores that box (config_metrics checks TP + FN = GT_Dets).
    padded, unpadded = run_all(con, tmp_path / 'work', {(DATASET, s, c) for s, c in CONFIGS})
    metrics = sw.config_metrics(con, padded)
    sw.check_reproduction(unpadded, metrics)
    assert (flat(padded[(DATASET, 'reference', 'naive')])['FN'], flat(unpadded[(DATASET, 'reference', 'naive')])['FN']) == (1, 0)
    assert metrics[(DATASET, 'reference', 'naive')]['delta']['DetA_padded'] == pytest.approx(20 / 21)
    # Tracks past the clip are reported.
    write_lines(output_path(DATASET, 'tetris', 'B', 'te01.mp4'), clip_rows('te01.mp4', range(9)))
    sw.load_rows(con)
    assert [(r['system'], r['config'], r['video'], r['n'], r['t']) for r in sw.track_overruns(con)] == \
        [('tetris', 'B', 'te01.mp4', 8, 9)]


def moving_rows(oid, frames, lag=0):
    # Object oid moving 10 px per frame with a 20 px box, drawn `lag` frames behind (IoU 1/3 one frame apart).
    return [(f, [[oid, 10.0 * (f - lag), 0.0, 10.0 * (f - lag) + 20, 20.0]]) for f in frames]


def test_clip_alignment(con, dataset):
    # Without a lengthened clip there is nothing to compare.
    sw.register_inputs(con, [DATASET])
    sw.load_rows(con)
    assert sw.clip_alignment(con) == {}
    # te02's ground truth runs one frame past the clip and follows its frames; te00 lags the reference by one frame.
    write_lines(output_path(DATASET, 'ground truth', '', 'te02.mp4'), moving_rows(3, range(7)))
    write_lines(output_path(DATASET, 'reference', 'naive', 'te02.mp4'), moving_rows(3, range(6)))
    write_lines(output_path(DATASET, 'ground truth', '', 'te00.mp4'), moving_rows(1, range(10), lag=1))
    write_lines(output_path(DATASET, 'reference', 'naive', 'te00.mp4'), moving_rows(1, range(10)))
    sw.load_rows(con)
    out = sw.clip_alignment(con)
    # The lengthened clip: frames 1..4 match the reference exactly at f and by 1/3 one frame either way.
    (clip,) = out[LABEL]['lengthened']
    assert (clip['video'], clip['frame_count'], clip['t_gt'], clip['reference_t']) == ('te02.mp4', 6, 7, 6)
    assert (clip['iou_prev'], clip['iou_same'], clip['iou_next']) == pytest.approx((1 / 3, 1.0, 1 / 3))
    # The other clips: te00 peaks one frame back (1, 1/3, 0); te01's constant box matches on frames 1-3 except frame
    # 3's next frame, where the reference has no box (1, 1, 2/3). The medians are the means of the two clips.
    others = out[LABEL]['other_clips']
    assert others['clips'] == 2 and others['iou_same'] == pytest.approx((1 / 3 + 1) / 2)
    assert others['iou_prev'] == pytest.approx(1.0) and others['iou_next'] == pytest.approx((0 + 2 / 3) / 2)


# ---------------------------------------------------------------- padded runs, reproduction, and checks

def test_padded_runs_score_every_box_and_reruns_reproduce(con, dataset, tmp_path):
    analyze(con)
    reproduce = {(DATASET, s, c) for s, c in CONFIGS}
    padded, unpadded = run_all(con, tmp_path / 'work', reproduce)
    # Every configuration has one padded run and the requested reruns.
    assert set(padded) == set(unpadded) == reproduce
    metrics = sw.config_metrics(con, padded)
    sw.check_reproduction(unpadded, metrics)
    # Tetris A: 15 hits and 5 misses; AssA weights te00's object (5 of its 10 frames) by its 5 hits.
    a = metrics[(DATASET, 'tetris', 'A')]
    assert flat(a['padded'])['DetA'] == pytest.approx(0.75)
    assert flat(a['padded'])['AssA'] == pytest.approx((5 * 0.5 + 10) / 15)
    assert a['delta']['HOTA_cached'] == 1.0 and a['delta']['HOTA_change'] == pytest.approx(np.sqrt(0.75 * 12.5 / 15) - 1)
    # OTIF: the four boxes past te01's ground truth become false positives (14 hits of 20 + 4).
    assert flat(metrics[(DATASET, 'otif', 'param 0')]['padded'])['DetA'] == pytest.approx(14 / 24)
    assert metrics[(DATASET, 'otif', 'param 0')]['delta']['DetA_cached'] == pytest.approx(14 / 20)
    # Nothing unscored, nothing changed.
    assert metrics[(DATASET, 'tetris', 'B')]['delta']['HOTA_change'] == 0.0
    # The window travels with the metrics.
    assert a['window']['gt_boxes_unscored'] == 5 and a['fps'] == 100.0


def edited(runs, key, family, field, value):
    # A copy of `runs` with one field of one run's metric summary replaced (family 'alpha_0.05' is HOTA's).
    m = json.loads(json.dumps(runs[key]))
    (m['HOTA']['alpha_0.05'] if family == 'alpha_0.05' else m[family])[field] = value
    return {k: (m if k == key else v) for k, v in runs.items()}


def test_config_metrics_and_reproduction_checks(con, dataset, tmp_path):
    analyze(con)
    padded, unpadded = run_all(con, tmp_path / 'work', {(DATASET, 'tetris', 'A')})
    # A configuration without a padded run.
    with pytest.raises(AssertionError, match='padded runs missing'):
        sw.config_metrics(con, {k: v for k, v in padded.items() if k[2] != 'B'})
    # A padded run that still leaves a ground-truth or a tracker box unscored, or whose Count misses a box.
    for family, field, value, message in (('alpha_0.05', 'HOTA_FN', 4.0, 'ground-truth boxes unscored'),
                                          ('alpha_0.05', 'HOTA_FP', -1.0, 'tracker boxes unscored'),
                                          ('Count', 'GT_Dets', 19, 'padded Count')):
        broken = edited(padded, (DATASET, 'tetris', 'A'), family, field, value)
        with pytest.raises(AssertionError, match=message):
            sw.config_metrics(con, broken)
    # A configuration without unscored boxes whose HOTA moved.
    with pytest.raises(AssertionError, match='HOTA moved'):
        sw.config_metrics(con, edited(padded, (DATASET, 'tetris', 'B'), 'HOTA', 'HOTA', 0.9))
    # A rerun that does not reproduce the cached result, or leaves out other boxes than the window.
    metrics = sw.config_metrics(con, padded)
    with pytest.raises(AssertionError, match='differs from DATASET.json'):
        sw.check_reproduction(edited(unpadded, (DATASET, 'tetris', 'A'), 'HOTA', 'HOTA', 0.5), metrics)
    for family, field, value in (('alpha_0.05', 'HOTA_FN', 5.0), ('Count', 'Dets', 16)):
        with pytest.raises(AssertionError, match='unscored, the window'):
            sw.check_reproduction(edited(unpadded, (DATASET, 'tetris', 'A'), family, field, value), metrics)


def test_rescore_removes_its_padded_copies(con, dataset, tmp_path):
    analyze(con)
    work = tmp_path / 'work'
    tasks = sw.rescore_tasks(con, work, sw.padded_ground_truth(con, work), set())
    # One padded task per configuration, largest first, and only the padded ground truth left behind.
    assert len(tasks) == len(CONFIGS) and all(t['padded'] for t in tasks)
    sw.rescore(tasks[0])
    assert sorted(p.name for p in work.iterdir()) == ['ground_truth']


# ---------------------------------------------------------------- operating points

def selections(con, dataset, tmp_path):
    # The cached and padded selections, the per-configuration metrics, and the changes of both rules, as main computes them.
    analyze(con)
    padded, _ = run_all(con, tmp_path / 'work')
    metrics = sw.config_metrics(con, padded)
    configs = sw.query_dicts(con, 'SELECT * FROM configs ORDER BY dataset, system, config')
    sw.load_points(con, 'cached_points', configs, {(c['dataset'], c['system'], c['config']): c['hota'] for c in configs})
    sw.load_points(con, 'padded_points', configs, {k: m['padded']['HOTA']['HOTA'] for k, m in metrics.items()})
    cached, padded = sw.selections(con, 'cached_points', [0.05]), sw.selections(con, 'padded_points', [0.05])
    hota = {(LABEL, s, c): (m['delta']['HOTA_cached'], m['delta']['HOTA_padded']) for (_, s, c), m in metrics.items()}
    return cached, padded, metrics, sw.point_changes(cached, padded, hota) + sw.matched_changes(cached, padded)


def change(changes, **match):
    # The one change whose fields include `match`.
    found = [c for c in changes if all(c.get(k) == v for k, v in match.items())]
    assert len(found) == 1, (match, changes)
    return found[0]


def test_padding_changes_the_operating_points(con, dataset, tmp_path):
    cached, padded, metrics, changes = selections(con, dataset, tmp_path)
    leap_padded = metrics[(DATASET, 'leap', 'param 0')]['padded']['HOTA']['HOTA']
    # Cached: Tetris A (inflated to HOTA 1) is the fastest within 5%, and LEAP (also inflated) qualifies.
    assert cached['points']['0.05'][LABEL]['systems']['tetris']['config'] == 'A'
    assert cached['points']['0.05'][LABEL]['systems']['leap']['meets_bound']
    assert cached['points']['0.05'][LABEL]['tetris_speedup_vs_leap'] == 10.0
    # Padded: A falls outside the bound, B takes its place, LEAP misses.
    assert padded['points']['0.05'][LABEL]['systems']['tetris']['config'] == 'B'
    assert not padded['points']['0.05'][LABEL]['systems']['leap']['meets_bound']
    t = change(changes, rule='shared_bound', bound='0.05', system='tetris', config=['A', 'B'])
    assert (t['fps'], t['speedup_vs_reference'], t['bound_hota']) == ([100.0, 50.0], [100.0, 50.0], [0.95, 0.95])
    # A was 0.05 above the bound cached and falls below it padded; B stays 0.05 above.
    a_padded = metrics[(DATASET, 'tetris', 'A')]['padded']['HOTA']['HOTA']
    assert t['hota_minus_bound']['A'] == pytest.approx([0.05, a_padded - 0.95])
    assert t['hota_minus_bound']['B'] == pytest.approx([0.05, 0.05])
    assert dict(rule='shared_bound', bound='0.05', dataset=LABEL, system='tetris', tetris_speedup_vs_leap=[10.0, None]) in changes
    # LEAP stops meeting the bound; its one configuration is measured against both bounds.
    leap = change(changes, rule='shared_bound', bound='0.05', system='leap', meets_bound=[True, False])
    assert leap['config'] == ['param 0', None] and leap['hota_minus_bound'] == {'param 0': pytest.approx([0.05, leap_padded - 0.95])}
    # OTIF misses the bound on both sides, so it is not a change at 5%.
    assert not any(c.get('system') == 'otif' and c['bound'] == '0.05' for c in changes)
    # At LEAP's own HOTA, padded A (0.79) still clears LEAP's padded 0.74, OTIF (0.71 padded) still misses, and nothing
    # changes under either rule.
    assert padded['points']['leap_fit'][LABEL]['systems']['tetris']['config'] == 'A'
    assert not padded['points']['leap_fit'][LABEL]['systems']['otif']['meets_bound']
    assert not any(c['bound'] == 'leap_fit' for c in changes)
    # The cached points' metrics, with the reference that sets the bound and the accuracy-matched prior point.
    points = sw.point_metrics(cached, metrics, {LABEL: DATASET})
    assert set(points['0.05'][LABEL]) == {'reference', 'tetris', 'leap', 'accuracy_matched_prior'}
    assert points['0.05'][LABEL]['tetris']['config'] == 'A' and points['0.05'][LABEL]['tetris']['HOTA_change'] < 0
    assert points['0.05'][LABEL]['accuracy_matched_prior']['system'] == 'leap'
    assert points['0.05'][LABEL]['accuracy_matched_prior']['HOTA_padded'] == pytest.approx(leap_padded)
    # Chosen points of every bound and rule are rerun, with the reference.
    assert sw.chosen_keys(cached, {LABEL: DATASET}) == {(DATASET, 'reference', 'naive'), (DATASET, 'tetris', 'A'),
                                                         (DATASET, 'leap', 'param 0')}


def test_padding_changes_the_accuracy_matched_rule(con, dataset, tmp_path):
    cached, padded, _, changes = selections(con, dataset, tmp_path)
    # Cached: LEAP meets 5% and is not more accurate than Tetris A, so it is the matched point (A is 10x faster).
    before = cached['matched']['0.05']['per_dataset'][LABEL]
    assert (before['prior']['system'], before['prior']['rule'], before['prior']['meets_bound'],
            before['tetris_speedup_vs_prior']) == ('leap', 'threshold_feasible_not_more_accurate', True, 10.0)
    # Padded: no prior system meets 5%, so Tetris B is matched with the most accurate prior point below it, LEAP's
    # (0.74 over OTIF's 0.71), which misses the bound.
    after = padded['matched']['0.05']['per_dataset'][LABEL]
    assert (after['tetris']['config'], after['prior']['system'], after['prior']['rule'], after['prior']['meets_bound'],
            after['tetris_speedup_vs_prior']) == ('B', 'leap', 'nearest_not_more_accurate', False, 5.0)
    # The changes: the bound's range and its count of matches outside the bound, and the dataset's pairing.
    summary = change(changes, rule='accuracy_matched', bound='0.05', dataset=None)
    assert {k: v for k, v in summary.items() if k not in ('rule', 'bound')} == dict(
        min=[10.0, 5.0], max=[10.0, 5.0], matched_outside_bound=[0, 1])
    pairing = change(changes, rule='accuracy_matched', bound='0.05', dataset=LABEL)
    assert {k: pairing[k] for k in ('tetris_config', 'prior_meets_bound', 'prior_rule', 'tetris_speedup_vs_prior')} == dict(
        tetris_config=['A', 'B'], prior_meets_bound=[True, False],
        prior_rule=['threshold_feasible_not_more_accurate', 'nearest_not_more_accurate'], tetris_speedup_vs_prior=[10.0, 5.0])
    # The unchanged fields are left out, and both sides' HOTA of the points come along.
    assert 'prior_system' not in pairing and 'prior_config' not in pairing
    assert pairing['tetris_hota'] == [1.0, 1.0] and pairing['prior_hota'][0] == 1.0 and pairing['prior_hota'][1] < 0.75


def test_matched_changes_when_tetris_stops_qualifying():
    # A dataset whose Tetris point meets the bound on one side only: every pairing field differs, None on the other side.
    entry = dict(bound=0.9, tetris=dict(config='A', hota=0.95, fps=100.0),
                 prior=dict(system='otif', config='param 1', hota=0.9, fps=10.0, meets_bound=True,
                            rule='threshold_feasible_not_more_accurate'), tetris_speedup_vs_prior=10.0)
    summary = dict(matched=1, min=10.0, max=10.0, matched_outside_bound=0, prior_systems=['otif'])
    cached = dict(matched={'0.05': dict(summary, per_dataset={LABEL: entry})})
    padded = dict(matched={'0.05': dict(summary, matched=0, min=None, max=None, prior_systems=[], per_dataset={})})
    changes = sw.matched_changes(cached, padded)
    assert changes[0] == dict(rule='accuracy_matched', bound='0.05', matched=[1, 0], min=[10.0, None], max=[10.0, None],
                              prior_systems=[['otif'], []])
    assert changes[1] == dict(rule='accuracy_matched', bound='0.05', dataset=LABEL, tetris_config=['A', None],
                              prior_system=['otif', None], prior_config=['param 1', None], prior_meets_bound=[True, None],
                              prior_rule=['threshold_feasible_not_more_accurate', None],
                              tetris_speedup_vs_prior=[10.0, None], tetris_hota=[0.95, None], prior_hota=[0.9, None])
    # A Tetris point without any prior point at or below its HOTA has an empty prior, and identical sides report nothing.
    assert sw.matched_pairing(None) == {}
    lonely = dict(entry, prior=None, tetris_speedup_vs_prior=None)
    assert sw.matched_pairing(lonely) == dict(tetris_config='A', prior_system=None, prior_config=None,
                                              prior_meets_bound=None, prior_rule=None, tetris_speedup_vs_prior=None)
    assert sw.matched_changes(cached, json.loads(json.dumps(cached))) == []


def test_point_changes_without_differences():
    # Identical selections report nothing.
    entry = {'systems': {'tetris': dict(meets_bound=True, config='A', fps=1.0, speedup_vs_reference=2.0),
                         'otif': dict(meets_bound=False, best_accuracy=0.5)}, 'tetris_speedup_vs_otif': None}
    cached = dict(points={'0.05': {LABEL: entry}}, bounds={'0.05': {LABEL: 0.6}})
    padded = dict(points=json.loads(json.dumps(cached['points'])), bounds={'0.05': {LABEL: 0.59}})
    hota = {(LABEL, 'tetris', 'A'): (0.7, 0.69)}
    assert sw.point_changes(cached, padded, hota) == []
    # A system present on only one side differs in every field it has; without a chosen configuration it has no margin.
    other = dict(points={'0.05': {LABEL: {'systems': {'tetris': entry['systems']['tetris']}}}}, bounds=padded['bounds'])
    assert sw.point_changes(cached, other, hota) == [
        dict(rule='shared_bound', bound='0.05', dataset=LABEL, system='otif', meets_bound=[False, None],
             bound_hota=[0.6, 0.59], hota_minus_bound={})]


def test_point_changes_of_a_speedup_over_a_prior_system():
    # Tetris and OTIF keep their configurations, but OTIF's throughput moves Tetris's speedup over it (5.8x to 4.6x).
    def points(ratio):
        # One dataset where both systems meet the bound, with Tetris's speedup over OTIF.
        return {'0.10': {LABEL: {'systems': {'tetris': dict(meets_bound=True, config='A', fps=58.0, speedup_vs_reference=5.8),
                                             'otif': dict(meets_bound=True, config='param 1', fps=58.0 / ratio,
                                                          speedup_vs_reference=1.0)},
                                 'tetris_speedup_vs_otif': ratio}}}
    cached = dict(points=points(5.8), bounds={'0.10': {LABEL: 0.6}})
    padded = dict(points=points(4.6), bounds={'0.10': {LABEL: 0.6}})
    hota = {(LABEL, 'tetris', 'A'): (0.7, 0.7), (LABEL, 'otif', 'param 1'): (0.65, 0.64)}
    changes = sw.point_changes(cached, padded, hota)
    # OTIF's fps changes (with its margin to the bound), and so does the ratio, reported as Tetris's.
    assert change(changes, system='otif')['fps'] == [10.0, pytest.approx(12.6087, abs=1e-4)]
    assert change(changes, system='otif')['hota_minus_bound'] == {'param 1': pytest.approx([0.05, 0.04])}
    assert dict(rule='shared_bound', bound='0.10', dataset=LABEL, system='tetris', tetris_speedup_vs_otif=[5.8, 4.6]) in changes
    assert len(changes) == 2


def test_bounds_and_speedup_ranges(con, dataset, tmp_path):
    cached, padded, metrics, _ = selections(con, dataset, tmp_path)
    # The relative bound from the reference's HOTA, and LEAP's best HOTA, unrounded.
    assert padded['bounds']['0.05'] == {LABEL: pytest.approx(0.95)}
    assert padded['bounds']['leap_fit'] == {LABEL: metrics[(DATASET, 'leap', 'param 0')]['padded']['HOTA']['HOTA']}
    assert cached['bounds']['leap_fit'] == {LABEL: 1.0}
    # Cached: Tetris A at 100x the reference, 10x LEAP and its matched point, OTIF never qualifying; padded: B at 50x,
    # LEAP out of the bound, and 5x over LEAP as the matched point below Tetris.
    assert sw.speedup_ranges(cached)['0.05'] == dict(
        vs_reference=dict(min=100.0, max=100.0, datasets=1), vs_otif=None, vs_leap=dict(min=10.0, max=10.0, datasets=1),
        vs_accuracy_matched_prior=dict(min=10.0, max=10.0, datasets=1))
    assert sw.speedup_ranges(padded)['0.05'] == dict(
        vs_reference=dict(min=50.0, max=50.0, datasets=1), vs_otif=None, vs_leap=None,
        vs_accuracy_matched_prior=dict(min=5.0, max=5.0, datasets=1))
    # A bound no Tetris configuration meets has no range.
    missed = dict(points={'x': {LABEL: {'systems': {'tetris': dict(meets_bound=False, best_accuracy=0.1)}}}},
                  matched={'x': dict(matched=0, min=None, max=None, matched_outside_bound=0, prior_systems=[], per_dataset={})})
    assert sw.speedup_ranges(missed) == {'x': dict(vs_reference=None, vs_otif=None, vs_leap=None,
                                                   vs_accuracy_matched_prior=None)}


def test_summary(con, dataset, tmp_path):
    _, _, metrics, _ = selections(con, dataset, tmp_path)
    stats = sw.summary(con, metrics)
    # Tetris on the synthetic dataset: A leaves 25% of the ground truth unscored, B nothing.
    t = stats['per_dataset'][LABEL]['tetris']
    assert (t['configs'], t['configs_with_gt_unscored'], t['gt_unscored_share_min'], t['gt_unscored_share_max']) == (2, 1, 0.0, 0.25)
    assert t['hota_change_max'] == 0.0 and t['hota_change_min'] < 0
    # Over every dataset, per system.
    assert stats['per_system']['otif']['configs_with_tracker_unscored'] == 1
    assert set(stats['per_system']) == {'reference', 'tetris', 'otif', 'leap'}


# ---------------------------------------------------------------- the script

def test_results_names(monkeypatch):
    # Every dataset (in any order) is a full run; fewer is partial.
    monkeypatch.setattr(sw, 'DATASET_LABELS', {'a': 'A', 'b': 'B'})
    assert sw.results_names(['b', 'a']) == ('results.json', 'configs.json')
    assert sw.results_names(['a']) == ('results-partial.json', 'configs-partial.json')


def test_ground_truth_labels():
    # B3D is automatic until R1.O1; the others are the root dataset's ground truth.
    assert sw.GROUND_TRUTH['B3D1'].startswith('automatic') and sw.GROUND_TRUTH['B3D4'].startswith('automatic')
    assert not sw.GROUND_TRUTH['CalDoT1'].startswith('automatic')


def all_floats(value):
    # Every float in a nested JSON value.
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for v in value.values() for f in all_floats(v)]
    if isinstance(value, list):
        return [f for v in value for f in all_floats(v)]
    return []


def test_main_writes_results_and_configs(dataset, tmp_path, monkeypatch):
    # The synthetic dataset as the only one, results into tmp_path.
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(sw, 'DATASET_LABELS', {DATASET: LABEL})
    monkeypatch.setattr(sw, 'results_dir', lambda _: out)
    monkeypatch.setattr('sys.argv', ['r1o1_hota_scoring_window.py', '--workers', '1', '--losses', '0.05'])
    sw.main()
    results = json.loads((out / 'results.json').read_text())
    configs = json.loads((out / 'configs.json').read_text())
    # Every part of the report, with the ground truth recorded as automatic.
    assert set(results) == {'method', 'ground_truth', 'summary', 'clips_lengthened', 'clip_alignment', 'reproduced',
                            'operating_points', 'accuracy_matched', 'speedup_ranges', 'operating_point_metrics', 'changes'}
    assert results['ground_truth'][LABEL].startswith('automatic')
    assert results['clips_lengthened'] == [] and results['clip_alignment'] == {}
    # The cached operating points and the reference were rerun unpadded.
    assert results['reproduced'] == sorted(f'{LABEL} {s} {c}' for s, c in
                                           [('reference', 'naive'), ('tetris', 'A'), ('leap', 'param 0')])
    assert results['operating_points']['padded']['0.05'][LABEL]['systems']['tetris']['config'] == 'B'
    assert any(c.get('system') == 'tetris' and c.get('config') == ['A', 'B'] for c in results['changes'])
    # Both rules' changes, and the padded accuracy-matched point below Tetris's HOTA.
    assert {c['rule'] for c in results['changes']} == {'shared_bound', 'accuracy_matched'}
    assert results['accuracy_matched']['padded']['0.05']['per_dataset'][LABEL]['prior']['rule'] == 'nearest_not_more_accurate'
    assert results['speedup_ranges']['padded']['0.05']['vs_accuracy_matched_prior'] == dict(min=5.0, max=5.0, datasets=1)
    # Every configuration, nested by dataset and system.
    assert {s: sorted(v) for s, v in configs[LABEL].items()} == {'leap': ['param 0'], 'otif': ['param 0'],
                                                                 'reference': ['naive'], 'tetris': ['A', 'B']}
    assert configs[LABEL]['tetris']['A']['window']['gt_boxes_unscored'] == 5
    assert results['speedup_ranges']['padded']['0.05']['vs_reference'] == dict(min=50.0, max=50.0, datasets=1)
    # Every float is rounded to 4 decimals, the changes' to 8 (bound margins can be far smaller).
    assert all(round(f, 4) == f for f in all_floats({k: v for k, v in results.items() if k != 'changes'}) + all_floats(configs))
    assert all(round(f, 8) == f for f in all_floats(results['changes']))
    assert change(results['changes'], bound='0.05', system='tetris', config=['A', 'B'])['hota_minus_bound']['A'][0] == pytest.approx(0.05)
    # A run over only some datasets writes the partial files and leaves the full ones alone.
    before = (out / 'results.json').read_text()
    monkeypatch.setattr(sw, 'DATASET_LABELS', {DATASET: LABEL, 'jnc2': 'B3D2'})
    monkeypatch.setattr('sys.argv', ['r1o1_hota_scoring_window.py', '--workers', '1', '--datasets', DATASET])
    sw.main()
    assert (out / 'results.json').read_text() == before
    assert list(json.loads((out / 'configs-partial.json').read_text())) == [LABEL]
