import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from evaluation.p140_otif_transform import transform_tracking_json, unwrap_leap_frames
from measurements import r1o2_leap_hota_conversion as hc


# The synthetic dataset: one name that needs no SOTA or root-dataset mapping, and its clips.
DATASET = 'jnc0'
LABEL = 'B3D1'
# Per clip: its length n, LEAP's predicted intervals {object: [start, end)} (None for a null export), and the ground
# truth {id: (box object, first frame, last frame)}.
CLIPS = {
    # Contiguous, wrapped (start -3), end past the clip, entirely past the clip, and ambiguous (wraps onto itself).
    'te00.mp4': dict(n=10, intervals={1: (2, 5), 2: (-3, 4), 3: (7, 12), 4: (10, 12), 5: (-5, 12)},
                     gt={101: (1, 0, 9), 102: (2, 0, 3), 103: (3, 7, 9)}),
    # An ambiguous object that ends inside the clip; the ground truth outlasts the export.
    'te01.mp4': dict(n=8, intervals={7: (-2, 6), 8: (1, 3)}, gt={201: (7, 0, 7)}),
    # A null export against ground-truth boxes.
    'te02.mp4': dict(n=6, intervals=None, gt={301: (1, 0, 2)}),
    # The tracks outlast the ground truth.
    'te03.mp4': dict(n=6, intervals={9: (1, 6), 10: (0, 2)}, gt={401: (10, 0, 2)}),
}


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def box(k):
    # Object k's constant box [x1, y1, x2, y2].
    return [10.0 * k, 10.0 * k, 10.0 * k + 8, 10.0 * k + 8]


def export(intervals):
    # What export-leap-outputs.py writes: each object's box at index start + idx (negative indices wrap), a list as
    # long as the largest end, and no repeated object within a frame.
    if intervals is None:
        return None
    frames = [[] for _ in range(max((e for _, e in intervals.values()), default=0))]
    for oid, (s, e) in intervals.items():
        for f in range(s, e):
            if any(d['track_id'] == oid for d in frames[f]):
                continue
            x1, y1, x2, y2 = box(oid)
            frames[f].append(dict(left=x1, top=y1, right=x2, bottom=y2, track_id=oid, score=1.0))
    return frames


def write_jsonl(path, frames):
    # A tracking.jsonl with one line per (frame_idx, tracks) pair.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': f, 'tracks': t}) + '\n' for f, t in frames))
    return str(path)


def gt_frames(gt):
    # Ground-truth lines from the first to the last frame with a box, as save_tracking_results writes them.
    rows = {}
    for gid, (k, first, last) in gt.items():
        for f in range(first, last + 1):
            rows.setdefault(f, []).append([gid, *box(k)])
    return [(f, rows.get(f, [])) for f in range(min(rows), max(rows) + 1)] if rows else []


def build_dataset(root: Path) -> Path:
    """Write the raw exports, the old and fixed LEAP caches (through p140's own transform), unwrap.json, and the
    ground truth under root, laid out as the cache and /sota-results; return the SOTA results directory."""
    sota_dir = root / 'sota-results'
    for video, clip in CLIPS.items():
        # The raw export, named by the numeric video id.
        raw = sota_dir / DATASET / 'leap' / f'{int(video[2:4])}.json'
        raw.parent.mkdir(parents=True, exist_ok=True)
        frames = export(clip['intervals'])
        raw.write_text(json.dumps(frames))
        # The old cache: the raw export transformed without unwrapping (p140 before 2026-10-26).
        transform_tracking_json(str(raw), str(root / 'SOTA' / hc.CACHES['before'] / DATASET / video / hc.TRACK_FILE),
                                DATASET, video)
        # The fixed cache: p140's unwrap, then the same transform, and unwrap.json next to the tracks.
        unwrapped, stats = unwrap_leap_frames(frames or [], clip['n'])
        after = root / 'SOTA' / hc.CACHES['after'] / DATASET / video / hc.TRACK_FILE
        after.parent.mkdir(parents=True, exist_ok=True)
        (after.parent / 'unwrapped.json').write_text(json.dumps(unwrapped))
        transform_tracking_json(str(after.parent / 'unwrapped.json'), str(after), DATASET, video)
        (after.parent / 'unwrap.json').write_text(json.dumps(stats))
        # The ground truth of the root dataset.
        write_jsonl(root / DATASET / 'execution' / video / hc.GT_FILE, gt_frames(clip['gt']))
    return sota_dir


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # The synthetic dataset, with the cache root (SOTA and execution paths, also common.output_path's) and the split
    # listing pointing into tmp_path.
    monkeypatch.setattr(hc.cache, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(hc, 'list_split_videos', lambda ds, split: sorted(CLIPS))
    return build_dataset(tmp_path)


def load_all(con, sota_dir):
    # Register and load every input of the synthetic dataset, as analyze_dataset does.
    hc.register_files(con, hc.input_files(DATASET, sorted(CLIPS), sota_dir))
    hc.load_unwrap(con)
    hc.load_lines(con)
    hc.load_exports(con)
    hc.classify_objects(con)


def cached_task(tmp_path, source):
    # A TrackEval task over the cached files of one LEAP cache, as p141 configures it.
    return dict(dataset=DATASET, variant=source, videos=sorted(CLIPS), gt_dir=str(tmp_path / DATASET / 'execution'),
                gt_file=str(hc.GT_FILE), track_dir=str(tmp_path / 'SOTA' / hc.CACHES[source] / DATASET),
                track_file=str(hc.TRACK_FILE), output_dir=str(tmp_path / 'p141' / source))


def write_p141_outputs(tmp_path):
    # p141's DATASET.json (HOTA and Count, arrays as lists) and p142's tradeoff.csv for both caches.
    for source, directory in hc.CACHES.items():
        combined = hc.trackeval_combined(cached_task(tmp_path, source))
        metrics = {family: {k: (np.asarray(v).tolist() if np.ndim(v) else v.item() if hasattr(v, 'item') else v)
                            for k, v in combined[family].items()} for family in ('HOTA', 'Count')}
        accuracy = tmp_path / 'SOTA' / directory / DATASET / 'accuracy' / 'raw' / 'test' / hc.LEAP_PARAM / 'DATASET.json'
        accuracy.parent.mkdir(parents=True, exist_ok=True)
        accuracy.write_text(json.dumps({'video': None, 'dataset': DATASET, 'metrics': metrics}))
        hota = {k: float(np.mean(metrics['HOTA'][k])) for k in ('HOTA', 'AssA', 'DetA')}
        (tmp_path / 'SOTA' / directory / DATASET / 'tradeoff.csv').write_text(
            f"param_id,HOTA_HOTA,HOTA_AssA,HOTA_DetA\n0,{hota['HOTA']!r},{hota['AssA']!r},{hota['DetA']!r}\n")


def native_outputs(section):
    # r1o2_leap_native_queries outputs consistent with a section: unwrap totals, frames, and rounded mean counts.
    frames = section['unwrap']['clip_frames']
    names = ('clips', 'empty_exports', 'objects', 'wrapped_objects', 'ambiguous_full_span_objects', 'clips_with_ambiguous',
             'wrapped_boxes_dropped', 'boxes_past_clip_dropped', 'empty_export_videos')
    mean = {'leap': section['box_frames']['after']['boxes_in_clip'] / frames,
            'leap_ambiguous_dropped': section['box_frames']['after_ambiguous_dropped']['boxes_in_clip'] / frames,
            'reference': 1.0, 'otif': 2.0}

    def record(system):
        return dict(system=system, config='c', hota=0.5, aggregation=dict(mean_predicted=round(mean[system], 4), mae=0.3,
                    acc_leap=0.6), selection=dict(precision=0.9, recall=0.8, f1=0.85))

    records = {v: [record(s) for s in ('reference', 'leap', 'leap_ambiguous_dropped', 'otif')] for v in ('all', 'moving')}
    configs = {'datasets': {LABEL: dict(frames=frames, ground_truth={'kind': 'automatic'}, records=records,
                                        ground_truth_summary={v: {'mean_count': 1.5} for v in records})}}
    diagnostics = {LABEL: dict(leap_unwrap={k: section['unwrap'][k] for k in names},
                               leap_excluding_empty_exports={'all': {'f1': 0.7}},
                               lag_correlations=[dict(system='leap', lag0_r=0.4), dict(system='otif', lag0_r=0.9)],
                               leap_cache_cross_check={'equal': True})}
    return configs, diagnostics


# ---------------------------------------------------------------- interval recovery against p140


def test_objects_recover_every_interval_kind(con, dataset):
    load_all(con, dataset)
    # The recovery agrees with p140's unwrap.json on every clip.
    assert hc.unwrap_mismatches(con) == []
    rows = {r['track_id']: r for r in hc.query_dicts(con, """
        SELECT track_id, wrapped, ambiguous, interval_start, interval_end, end_clipped, ambiguous_past_clip,
               entirely_past_clip, boxes, wrapped_boxes, past_clip_boxes, head_end, tail_start FROM objects""")}
    # Contiguous inside the clip.
    assert rows[1] == dict(track_id=1, wrapped=False, ambiguous=False, interval_start=2, interval_end=5, end_clipped=False,
                           ambiguous_past_clip=False, entirely_past_clip=False, boxes=3, wrapped_boxes=0,
                           past_clip_boxes=0, head_end=4, tail_start=2)
    # Wrapped: the true start -3 is recovered, its three tail boxes (two of them past the clip) count as wrapped only;
    # its head ends on frame 3 and its tail starts on frame 9.
    assert (rows[2]['wrapped'], rows[2]['interval_start'], rows[2]['interval_end'], rows[2]['wrapped_boxes'],
            rows[2]['past_clip_boxes'], rows[2]['head_end'], rows[2]['tail_start']) == (True, -3, 4, 3, 0, 3, 9)
    # End past the clip, and entirely past it.
    assert (rows[3]['end_clipped'], rows[3]['past_clip_boxes'], rows[3]['entirely_past_clip']) == (True, 2, False)
    assert (rows[4]['end_clipped'], rows[4]['past_clip_boxes'], rows[4]['entirely_past_clip']) == (True, 2, True)
    # Ambiguous: kept as [0, L); its end is never a known end. In te00 the export outlasts the clip, so it is clipped
    # at one end for certain; in te01 the export ends inside the clip.
    assert (rows[5]['ambiguous'], rows[5]['interval_start'], rows[5]['interval_end'], rows[5]['end_clipped'],
            rows[5]['ambiguous_past_clip']) == (True, 0, 12, False, True)
    assert (rows[7]['ambiguous'], rows[7]['interval_end'], rows[7]['end_clipped'], rows[7]['ambiguous_past_clip']) == \
        (True, 6, False, False)
    # Not starting at 0 is never ambiguous even when it ends on the export's last frame.
    assert (rows[9]['ambiguous'], rows[9]['wrapped'], rows[9]['interval_start']) == (False, False, 1)


def test_interval_summary_shares_bracket_ambiguous_objects(con, dataset):
    load_all(con, dataset)
    s = hc.interval_summary(con)
    # Nine objects: one known start clip (object 2), two known end clips (3 and 4), two ambiguous objects (5 in an
    # export longer than its clip, 7 in a shorter one), and one gone entirely (4).
    assert (s['objects'], s['start_clipped'], s['end_clipped'], s['both_ends_clipped'], s['ambiguous'],
            s['ambiguous_past_clip'], s['entirely_past_clip']) == (9, 1, 2, 0, 2, 1, 1)
    # Upper bounds: either ambiguous object may have wrapped; only object 5 may end past its clip.
    assert (s['start_clipped_upper'], s['end_clipped_upper']) == (3, 3)
    assert (s['start_clipped_share'], s['start_clipped_share_upper']) == (pytest.approx(1 / 9), pytest.approx(3 / 9))
    assert (s['end_clipped_share'], s['end_clipped_share_upper']) == (pytest.approx(2 / 9), pytest.approx(3 / 9))
    # Either end: object 5 is clipped for certain, object 7 only possibly.
    assert (s['clipped_either_end_lower'], s['clipped_either_end_upper']) == (4, 5)
    assert s['clipped_either_end_share_lower'] == pytest.approx(4 / 9)
    assert s['clipped_either_end_share_upper'] == pytest.approx(5 / 9)
    # 44 raw boxes; the fix removes 3 wrapped and 6 past the clip; object 2's head and tail are 5 frames apart.
    assert (s['export_boxes'], s['wrapped_boxes'], s['past_clip_boxes'], s['wrapped_gap_frames']) == (44, 3, 6, 5)
    assert s['removed_box_share'] == pytest.approx(9 / 44)
    # Clipped interval lengths leave the ambiguous objects out: objects 2 (7), 3 (5), and 4 (2) have median 5.
    assert s['median_clipped_interval_frames'] == 5


def test_unwrap_mismatch_is_reported(con, dataset):
    # A statistics file that misses the wrapped object and miscounts boxes.
    path = hc.cache.sota(hc.CACHES['after'], DATASET, 'te00.mp4', str(hc.TRACK_FILE.parent), 'unwrap.json')
    stats = json.loads(path.read_text())
    path.write_text(json.dumps(dict(stats, wrapped_objects=[], boxes_past_clip_dropped=0)))
    load_all(con, dataset)
    # Exactly that clip disagrees.
    assert [r['video'] for r in hc.unwrap_mismatches(con)] == ['te00.mp4']


def test_unwrap_mismatch_on_export_length(con, dataset):
    # An export one frame longer than p140 saw (a trailing empty frame).
    raw = dataset / DATASET / 'leap' / '3.json'
    raw.write_text(json.dumps(json.loads(raw.read_text()) + [[]]))
    load_all(con, dataset)
    assert [r['video'] for r in hc.unwrap_mismatches(con)] == ['te03.mp4']


def test_object_with_a_hole_raises(con, dataset):
    # An object with a hole that does not start at frame 0 is not the exporter's wrap.
    raw = dataset / DATASET / 'leap' / '3.json'
    frames = json.loads(raw.read_text())
    frames[3] = [d for d in frames[3] if d['track_id'] != 9]
    raw.write_text(json.dumps(frames))
    with pytest.raises(AssertionError, match='neither one interval nor one wrapped interval'):
        load_all(con, dataset)


def test_load_exports_handles_null_files_frames_and_empty_lists(con, tmp_path):
    # A null file, an empty list, and an export with a null frame.
    paths = {'te00.mp4': tmp_path / 'a.json', 'te01.mp4': tmp_path / 'b.json', 'te02.mp4': tmp_path / 'c.json'}
    paths['te00.mp4'].write_text('null')
    paths['te01.mp4'].write_text('[]')
    paths['te02.mp4'].write_text(json.dumps([None, [{'track_id': 4}], []]))
    hc.register_files(con, [('export', v, str(p)) for v, p in paths.items()])
    hc.load_exports(con)
    # Lengths 0, 0, 3, and one box on frame 1.
    assert con.execute('SELECT video, L FROM export_len ORDER BY video').fetchall() == \
        [('te00.mp4', 0), ('te01.mp4', 0), ('te02.mp4', 3)]
    assert con.execute('SELECT video, frame, track_id FROM export_obs').fetchall() == [('te02.mp4', 1, 4)]


def test_load_exports_rejects_repeated_ids_in_a_frame(con, tmp_path):
    # Two boxes of one object on one frame.
    path = tmp_path / 'a.json'
    path.write_text(json.dumps([[{'track_id': 4}, {'track_id': 4}]]))
    hc.register_files(con, [('export', 'te00.mp4', str(path))])
    with pytest.raises(AssertionError, match='repeated id'):
        hc.load_exports(con)


# ---------------------------------------------------------------- tracks and boxes


def test_load_lines_rejects_repeated_frames_and_ids(con, tmp_path):
    # Minimal unwrap table so the dropped source can be derived.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 5 AS n, []::BIGINT[] AS ambiguous_full_span_objects")
    # A frame written twice.
    path = write_jsonl(tmp_path / 'a.jsonl', [(0, [[1, *box(1)]]), (0, [])])
    hc.register_files(con, [('gt', 'te00.mp4', path)])
    with pytest.raises(AssertionError, match='repeated, negative'):
        hc.load_lines(con)
    # One id twice in a frame.
    path = write_jsonl(tmp_path / 'b.jsonl', [(0, [[1, *box(1)], [1, *box(2)]])])
    hc.register_files(con, [('gt', 'te00.mp4', path)])
    with pytest.raises(AssertionError, match='repeated track id'):
        hc.load_lines(con)


def test_box_frames_and_changes(con, dataset):
    load_all(con, dataset)
    frames = hc.box_frames(con)
    # The fixed cache holds the export minus the removals; dropping the two ambiguous objects removes 10 + 6 boxes.
    assert (frames['after']['boxes'], frames['after']['boxes_past_clip'], frames['after']['objects']) == (35, 0, 8)
    assert frames['after_ambiguous_dropped']['boxes'] == 35 - 10 - 6
    # The old cache: frames 10 and 11 of te00 hold four objects' boxes; object 4 lies only there.
    assert (frames['before']['boxes'], frames['before']['boxes_past_clip'], frames['before']['frames_past_clip'],
            frames['before']['objects'], frames['before']['objects_in_clip']) == (49, 8, 2, 9, 8)
    # Ground truth: 10 + 4 + 3 + 8 + 3 + 3 boxes of six tracks.
    assert (frames['gt']['boxes'], frames['gt']['objects_in_clip']) == (31, 6)
    changes = hc.box_changes(frames, hc.interval_summary(con))
    # The old interpolation filled frames 4-8 between the wrapped object's head and tail; inside the clips the old
    # cache held 41 boxes (49 minus te00's 8 past the clip).
    assert changes == dict(before=49, after=35, surplus=14, before_in_clip=41, after_in_clip=35, surplus_in_clip=6,
                           wrapped_tails=3, past_clip=6, gap_fill=5, inflation=pytest.approx(49 / 35 - 1),
                           inflation_in_clip=pytest.approx(41 / 35 - 1))


def test_box_changes_rejects_a_cache_that_does_not_match_the_export():
    # Raw export of 9 boxes, 1 wrapped and 1 past the clip, with 1 gap frame between the wrapped head and tail.
    intervals = dict(export_boxes=9, wrapped_boxes=1, past_clip_boxes=1, wrapped_gap_frames=1)
    # A fixed cache with one box more than the export minus the removals.
    with pytest.raises(AssertionError, match='fixed cache holds'):
        hc.box_changes({'before': dict(boxes=10), 'after': dict(boxes=8)}, intervals)
    # An old cache smaller than the raw export.
    with pytest.raises(AssertionError, match='fewer boxes than the raw export'):
        hc.box_changes({'before': dict(boxes=8), 'after': dict(boxes=7)}, intervals)
    # An old cache whose surplus over the export is not the head-to-tail gaps.
    with pytest.raises(AssertionError, match="not the wrapped objects' gap frames"):
        hc.box_changes({'before': dict(boxes=11), 'after': dict(boxes=7)}, intervals)
    # Without fixed boxes in the clip, no inflation is defined.
    changes = hc.box_changes({'before': dict(boxes=10, boxes_in_clip=3), 'after': dict(boxes=7, boxes_in_clip=0)}, intervals)
    assert changes['inflation_in_clip'] is None and changes['gap_fill'] == 1


def test_box_frames_reports_sources_without_boxes(con):
    # No box at all still yields every source with zeros.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 5 AS n")
    con.execute('CREATE TABLE boxes (source VARCHAR, video VARCHAR, frame BIGINT, track_id BIGINT)')
    frames = hc.box_frames(con)
    assert set(frames) == set(hc.BOX_SOURCES) and all(f['boxes'] == 0 for f in frames.values())


# ---------------------------------------------------------------- scoring window and TrackEval


def test_scored_window_counts(con, dataset):
    load_all(con, dataset)
    after, before = hc.scored_window(con, 'after'), hc.scored_window(con, 'before')
    # te01's ground truth outlasts the export by two boxes; te03's tracks outlast the ground truth by three.
    assert (after['gt_boxes_not_scored'], after['tracker_boxes_not_scored'],
            after['clips_with_gt_not_scored'], after['clips_with_tracker_not_scored']) == (2, 3, 1, 1)
    # The old cache also kept te00's frames 10-11 (eight boxes) past the ground truth's last row.
    assert (before['gt_boxes_not_scored'], before['tracker_boxes_not_scored']) == (2, 11)
    # te02's empty export is not truncated (TrackEval counts all its ground truth as misses).
    assert after['gt_boxes'] == 31 and after['tracker_boxes'] == 35
    # Files ending before the clip's last frame: ground truth on te02 and te03, tracks on te01 and te02 (empty).
    assert (after['gt_files_ending_before_clip_end'], after['tracker_files_ending_before_clip_end']) == (2, 2)


def test_scored_window_explains_trackeval_to_the_box(con, dataset, tmp_path):
    load_all(con, dataset)
    # Real TrackEval on both caches, as p141 runs it.
    for source in hc.CACHES:
        rerun = hc.run_trackeval(cached_task(tmp_path, source))['metrics']
        hc.check_scored_window(hc.scored_window(con, source), rerun, source)
    # A wrong count is caught.
    window = dict(hc.scored_window(con, 'after'), gt_boxes_not_scored=0)
    with pytest.raises(AssertionError, match='scored window'):
        hc.check_scored_window(window, rerun, 'after')


def test_count_ceiling_and_perfect_placement(con, tmp_path):
    # One clip of 10 frames: ground-truth objects 1 and 2 on every frame; tracks hold object 1 exactly and, on
    # frames 0-4, a second box either on object 2 (perfect placement) or far from everything.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 10 AS n, 10 AS export_frames, []::BIGINT[] AS ambiguous_full_span_objects")
    gt = write_jsonl(tmp_path / 'gt' / 'te00.mp4' / 'tracking.jsonl', [(f, [[1, *box(1)], [2, *box(2)]]) for f in range(10)])
    deta = {}
    for name, second in (('perfect', box(2)), ('far', box(50))):
        tracks = write_jsonl(tmp_path / name / 'te00.mp4' / 'tracking.jsonl',
                             [(f, [[7, *box(1)]] + ([[8, *second]] if f < 5 else [])) for f in range(10)])
        hc.register_files(con, [('gt', 'te00.mp4', gt), ('after', 'te00.mp4', tracks)])
        hc.load_lines(con)
        ceiling = hc.count_ceiling(con, 'after')
        # p = 2, 2, 2, 2, 2, 1, ..., g = 2: M = 15, G = 20, P = 15.
        assert (ceiling['matchable'], ceiling['gt_boxes'], ceiling['tracker_boxes']) == (15, 20, 15)
        assert ceiling['deta'] == pytest.approx(15 / 20) and ceiling['detre'] == pytest.approx(0.75)
        assert ceiling['detpr'] == 1.0 and ceiling['hota_perfect_association'] == pytest.approx(np.sqrt(0.75))
        task = dict(dataset='d', variant=name, videos=['te00.mp4'], gt_dir=str(tmp_path / 'gt'), gt_file='tracking.jsonl',
                    track_dir=str(tmp_path / name), track_file='tracking.jsonl', output_dir=str(tmp_path / 'te' / name))
        deta[name] = hc.run_trackeval(task)['metrics']['HOTA']['DetA']
    # Perfect placement reaches the ceiling; a misplaced box falls below it (TP 10, FN 10, FP 5).
    assert deta['perfect'] == pytest.approx(0.75)
    assert deta['far'] == pytest.approx(10 / 25)


def test_count_ceiling_without_boxes(con):
    # No box on either side has no ceiling.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 5 AS n")
    con.execute('CREATE TABLE boxes (source VARCHAR, video VARCHAR, frame BIGINT, track_id BIGINT)')
    assert hc.count_ceiling(con, 'after') == dict(tracker_boxes=0, gt_boxes=0, matchable=0, deta=None, detre=None,
                                                  detpr=None, hota_perfect_association=None)


def test_variant_lines_drop_cut_pad_and_keep_digits(con, tmp_path):
    # One clip of 4 frames with ambiguous object 2, a box with a long coordinate, and a line past the clip.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 4 AS n, [2]::BIGINT[] AS ambiguous_full_span_objects")
    con.execute("CREATE TABLE lines AS SELECT * FROM (VALUES ('after', 'te00.mp4', 1, ['[1,0.30000000000000004,1,2,2]', '[2,5,5,6,6]']::JSON[]), "
                "('after', 'te00.mp4', 5, ['[1,1,1,2,2]']::JSON[])) t(source, video, frame, tracks)")
    # Without ambiguous objects on full clips: id 2 gone, the frame past the clip cut, an empty row on frame 3.
    hc.build_variant_lines(con, 'after', drop_ambiguous=True, full_length=True)
    hc.copy_variant_lines(con, tmp_path / 'v', ['te00.mp4'])
    lines = [json.loads(x) for x in (tmp_path / 'v' / 'te00.mp4' / 'tracking.jsonl').read_text().splitlines()]
    assert lines == [{'frame_idx': 1, 'tracks': [[1, 0.1 + 0.2, 1, 2, 2]]}, {'frame_idx': 3, 'tracks': []}]
    # Unchanged reading: every line as it was, no padding.
    hc.build_variant_lines(con, 'after', drop_ambiguous=False, full_length=False)
    hc.copy_variant_lines(con, tmp_path / 'w', ['te00.mp4', 'te01.mp4'])
    assert [json.loads(x)['frame_idx'] for x in (tmp_path / 'w' / 'te00.mp4' / 'tracking.jsonl').read_text().splitlines()] == [1, 5]
    # A clip without lines still gets a file, empty.
    assert (tmp_path / 'w' / 'te01.mp4' / 'tracking.jsonl').read_text() == ''


def test_variant_lines_do_not_pad_an_existing_last_frame(con, tmp_path):
    # A file that already has a row on the clip's last frame keeps one row there.
    con.execute("CREATE TABLE unwrap AS SELECT 'te00.mp4' AS video, 2 AS n, []::BIGINT[] AS ambiguous_full_span_objects")
    con.execute("CREATE TABLE lines AS SELECT * FROM (VALUES ('gt', 'te00.mp4', 1, ['[1,1,1,2,2]']::JSON[])) t(source, video, frame, tracks)")
    hc.build_variant_lines(con, 'gt', drop_ambiguous=False, full_length=True)
    assert con.execute('SELECT frame_idx, len(tracks) FROM variant_lines').fetchall() == [(1, 1)]


# ---------------------------------------------------------------- summaries and checks


def test_metric_summary_means_loosest_alpha_and_scalars():
    # Two alphas only, for readability; counts are not averaged.
    metrics = {'HOTA': {'HOTA': [0.2, 0.4], 'HOTA_TP': np.array([3, 1]), 'HOTA(0)': np.float64(0.5)},
               'Count': {'Dets': np.int64(7)}}
    out = hc.metric_summary(metrics)
    assert out['HOTA']['HOTA'] == pytest.approx(0.3) and 'HOTA_TP' not in out['HOTA']
    assert out['HOTA']['HOTA(0)'] == 0.5 and type(out['HOTA']['HOTA(0)']) is float
    assert out['HOTA']['alpha_0.05'] == {'HOTA': 0.2, 'HOTA_TP': 3.0}
    assert out['Count'] == {'Dets': 7} and type(out['Count']['Dets']) is int


def test_same_metrics_tolerance_and_missing_fields():
    a = {'HOTA': {'HOTA': 0.3, 'alpha_0.05': {'DetA': 0.5}}, 'Count': {'Dets': 3}}
    # Within the tolerance, nothing differs.
    assert hc.same_metrics(a, {'HOTA': {'HOTA': 0.3 + 1e-12, 'alpha_0.05': {'DetA': 0.5}}, 'Count': {'Dets': 3}}) == []
    # A changed loosest-alpha value and a missing field are both reported.
    b = {'HOTA': {'alpha_0.05': {'DetA': 0.6}}, 'Count': {'Dets': 3}}
    assert hc.same_metrics(a, b) == ['HOTA.HOTA', 'HOTA.alpha_0.05.DetA']
    # The comparison is symmetric: a field only on the second side is reported too.
    assert hc.same_metrics(b, a) == ['HOTA.HOTA', 'HOTA.alpha_0.05.DetA']
    extra = {'HOTA': {'HOTA': 0.3, 'LocA': 0.9, 'alpha_0.05': {'DetA': 0.5, 'AssA': 0.1}}, 'Count': {'Dets': 3, 'IDs': 2}}
    assert hc.same_metrics(a, extra) == ['HOTA.LocA', 'HOTA.alpha_0.05.AssA', 'Count.IDs']


def test_cached_accuracy_rejects_a_stale_tradeoff_table(con, dataset, tmp_path):
    write_p141_outputs(tmp_path)
    # Consistent files load, with a file record each.
    accuracy, records = hc.cached_accuracy(con, DATASET)
    assert set(accuracy) == set(hc.CACHES) and len(records) == 4 and all(r['exists'] for r in records)
    # A tradeoff row that no longer matches DATASET.json.
    path = tmp_path / 'SOTA' / hc.CACHES['after'] / DATASET / 'tradeoff.csv'
    path.write_text('param_id,HOTA_HOTA,HOTA_AssA,HOTA_DetA\n0,0.9,0.9,0.9\n')
    with pytest.raises(AssertionError, match='differs from DATASET.json'):
        hc.cached_accuracy(con, DATASET)


def test_end_to_end_on_the_synthetic_dataset(con, dataset, tmp_path):
    # p141's cached results, then everything the script does for one dataset.
    write_p141_outputs(tmp_path)
    section, tasks, records = hc.analyze_dataset(con, DATASET, dataset, tmp_path / 'work')
    assert [t['variant'] for t in tasks] == list(hc.VARIANTS) and len(records) == 4
    section['trackeval'] = {t['variant']: hc.run_trackeval(t)['metrics'] for t in tasks}
    # The reruns reproduce the cache, and every full-clip check holds.
    hc.check_trackeval(section, section['accuracy'])
    # The unwrap totals and the box decomposition of the synthetic clips.
    u = section['unwrap']
    assert (u['clips'], u['clip_frames'], u['empty_exports'], u['empty_export_videos'], u['wrapped_objects'],
            u['ambiguous_full_span_objects'], u['clips_with_ambiguous'], u['export_frames_past_clip'],
            u['clips_exported_past_clip'], u['clips_exported_short_of_clip_end']) == \
        (4, 30, 1, ['te02.mp4'], 1, 2, 2, 2, 1, 1)
    assert section['box_changes']['gap_fill'] == 5
    # Full clips score te01's two ground-truth boxes as misses and te03's three tracker boxes as false positives.
    assert section['trackeval']['after_full_length']['Count'] == dict(
        section['trackeval']['after']['Count'], Dets=35, GT_Dets=31)
    full, cached = section['trackeval']['after_full_length']['HOTA'], section['trackeval']['after']['HOTA']
    assert full['alpha_0.05']['HOTA_FN'] == cached['alpha_0.05']['HOTA_FN'] + 2
    assert full['alpha_0.05']['HOTA_FP'] == cached['alpha_0.05']['HOTA_FP'] + 3
    # The native-queries outputs are reused after the cross-checks, and the attribution reads them.
    configs, diagnostics = native_outputs(section)
    section['native_queries'] = hc.native_summary(configs, diagnostics, LABEL, section)
    a = hc.attribution(section)
    assert a['fix_effect'] == pytest.approx(a['leap_hota']['after'] - a['leap_hota']['before'])
    assert a['deta_full_length'] <= a['deta_ceiling'] and a['deta_share_of_ceiling'] <= 1
    assert a['objects_vs_ground_truth'] == pytest.approx(8 / 6)
    assert a['reference_hota'] == 0.5 and a['per_frame_count_mae'] == 0.3
    # The DetA split adds up, and association is reported but not split.
    split = a['deta_decomposition']
    assert split['count_loss_abs'] + split['placement_loss_abs'] == pytest.approx(1 - a['deta_full_length'])
    assert a['assa_attributed'] is False and a['assa_full_length'] == section['trackeval']['after_full_length']['HOTA']['AssA']
    # Only LEAP's lag rows are kept.
    assert [r['system'] for r in section['native_queries']['lag_correlations']] == ['leap']
    # The time-base diagnostic ran on every scale, and every input source has a digest of its four clips.
    assert [r['scale'] for r in section['time_base']['scales']] == list(hc.TIME_SCALES)
    assert a['time_base_flagged'] == section['time_base']['flagged']
    assert {s: d['files'] for s, d in section['input_digests'].items()} == {s: 4 for s in hc.INPUT_SOURCES}


def test_check_trackeval_catches_each_inconsistency(con, dataset, tmp_path):
    write_p141_outputs(tmp_path)
    section, tasks, _ = hc.analyze_dataset(con, DATASET, dataset, tmp_path / 'work')
    runs = {t['variant']: hc.run_trackeval(t)['metrics'] for t in tasks}

    def check(change):
        # Apply one change to a copy of the reruns and check.
        section['trackeval'] = json.loads(json.dumps(runs))
        change(section['trackeval'])
        hc.check_trackeval(section, section['accuracy'])

    # A rerun that does not reproduce DATASET.json.
    with pytest.raises(AssertionError, match='rerun differs from DATASET.json'):
        check(lambda r: r['after']['HOTA'].update(HOTA=r['after']['HOTA']['HOTA'] + 0.1))
    # A full-clip rerun that scored other boxes than the clip holds.
    with pytest.raises(AssertionError, match='differs from the clip boxes'):
        check(lambda r: r['after_full_length']['Count'].update(Dets=0))
    # A DetA above the count ceiling.
    with pytest.raises(AssertionError, match='exceeds its ceiling'):
        check(lambda r: r['after_full_length']['HOTA'].update(DetA=1.0))
    # Without ambiguous objects, dropping them must change nothing.
    section['unwrap'] = dict(section['unwrap'], ambiguous_full_span_objects=0)
    with pytest.raises(AssertionError, match='dropping no object changed'):
        check(lambda r: None)


def test_native_summary_rejects_other_caches(con, dataset, tmp_path):
    write_p141_outputs(tmp_path)
    section, _, _ = hc.analyze_dataset(con, DATASET, dataset, tmp_path / 'work')
    configs, diagnostics = native_outputs(section)
    # A consistent run passes.
    assert hc.native_summary(configs, diagnostics, LABEL, section)['leap_cache_cross_check_equal'] is True
    # A dataset the native run did not score.
    with pytest.raises(AssertionError, match='missing from the r1o2_leap_native_queries outputs'):
        hc.native_summary(configs, diagnostics, 'CalDoT1', section)
    # Unwrap totals of another cache.
    bad = json.loads(json.dumps(diagnostics))
    bad[LABEL]['leap_unwrap']['wrapped_objects'] += 1
    with pytest.raises(AssertionError, match='unwrap totals differ'):
        hc.native_summary(configs, bad, LABEL, section)
    # A mean count off by more than the rounding.
    bad = json.loads(json.dumps(configs))
    bad['datasets'][LABEL]['records']['all'][1]['aggregation']['mean_predicted'] += 1e-3
    with pytest.raises(AssertionError, match='native mean count'):
        hc.native_summary(bad, diagnostics, LABEL, section)
    # Another pooled frame count.
    bad = json.loads(json.dumps(configs))
    bad['datasets'][LABEL]['frames'] += 1
    with pytest.raises(AssertionError, match='native-queries frames'):
        hc.native_summary(bad, diagnostics, LABEL, section)


def test_load_native_requires_both_files(tmp_path):
    # Only one of the two outputs present.
    (tmp_path / 'configs.json').write_text('{}')
    with pytest.raises(AssertionError, match='run measurements/r1o2_leap_native_queries.py first'):
        hc.load_native(tmp_path)
    (tmp_path / 'diagnostics.json').write_text('{}')
    configs, diagnostics, records = hc.load_native(tmp_path)
    assert configs == {} and diagnostics == {} and len(records) == 2


def test_input_files_require_every_clip(dataset):
    # Every clip has five inputs.
    rows = hc.input_files(DATASET, sorted(CLIPS), dataset)
    assert len(rows) == 5 * len(CLIPS)
    # A missing raw export fails before anything is read.
    (dataset / DATASET / 'leap' / '1.json').unlink()
    with pytest.raises(AssertionError, match='LEAP inputs missing'):
        hc.input_files(DATASET, sorted(CLIPS), dataset)


# ---------------------------------------------------------------- time base


def g_count(t):
    # Ground-truth count on clip frame t: a level that changes every 5 frames.
    return (t // 5) % 3 + (t // 15) % 2


def time_base_tables(con, leap_frame_of, clips=('a', 'b'), n=40, empty=False):
    # Clips of n frames whose LEAP count on LEAP frame f is the ground truth's count on clip frame leap_frame_of(f)
    # (each count k as ids 0..k-1), plus, on clip a, object 99 on clip frames 0-2 in both, with a wrapped tail on
    # LEAP frames 35-39 that must be ignored.
    length = 2 * n
    obs, gt = [], []
    for v in clips:
        if not empty:
            obs += [(v, f, k) for f in range(length) for k in range(g_count(leap_frame_of(f)) if leap_frame_of(f) < n else 0)]
        gt += [('gt', v, t, k) for t in range(n) for k in range(g_count(t))]
    head = [f for f in range(length) if leap_frame_of(f) in (0, 1, 2) and f < 35]
    obs += [('a', f, 99) for f in head + [35, 36, 37, 38, 39]] if not empty else []
    gt += [('gt', 'a', t, 99) for t in (0, 1, 2)]
    con.execute('CREATE OR REPLACE TABLE export_obs (video VARCHAR, frame BIGINT, track_id BIGINT)')
    if obs:
        con.executemany('INSERT INTO export_obs VALUES (?, ?, ?)', obs)
    con.execute('CREATE OR REPLACE TABLE boxes (source VARCHAR, video VARCHAR, frame BIGINT, track_id BIGINT)')
    con.executemany('INSERT INTO boxes VALUES (?, ?, ?, ?)', gt)
    con.execute('CREATE OR REPLACE TABLE unwrap AS SELECT unnest(?) AS video, ?::BIGINT AS n', [list(clips), n])
    con.execute('CREATE OR REPLACE TABLE export_len AS SELECT unnest(?) AS video, ?::BIGINT AS L',
                [list(clips), 0 if empty else length])
    # Objects: every id spans the export; object 99 is wrapped with its tail from frame 35.
    con.execute("""
        CREATE OR REPLACE TABLE objects AS
        SELECT DISTINCT video, track_id, track_id = 99 AS wrapped, CASE WHEN track_id = 99 THEN 35 ELSE 0 END AS tail_start,
               false AS ambiguous, 0 AS interval_start, 40 AS interval_end FROM export_obs
    """)


def test_time_base_finds_a_half_speed_time_line(con):
    # LEAP's frame f shows clip frame f // 2: clip frame t reads LEAP frame 2t at scale 0.5.
    time_base_tables(con, lambda f: f // 2)
    tb = hc.time_base(con)
    assert tb['best_scale'] == 0.5 and tb['best_r'] == pytest.approx(1.0) and tb['flagged'] is True
    assert tb['r_at_scale_1'] < 1 - hc.TIME_SCALE_WARN_GAIN
    # Both clips have a correlation at every scale; the export is twice the clip.
    assert {r['clips_with_r'] for r in tb['scales']} == {2}
    assert tb['export_over_clip_median'] == tb['export_over_clip_min'] == tb['export_over_clip_max'] == 2.0
    # Intervals of 40 frames against ground-truth tracks: ids 0 and 1 recur across the clip, 99 lasts 3 frames.
    assert tb['median_interval_frames'] == 40 and tb['median_gt_track_frames'] is not None


def test_time_base_keeps_an_aligned_time_line_and_ignores_wrapped_tails(con):
    # LEAP's frame f is clip frame f; counting object 99's wrapped tail would break the perfect correlation at 1.
    time_base_tables(con, lambda f: f)
    tb = hc.time_base(con)
    assert tb['best_scale'] == 1.0 and tb['r_at_scale_1'] == pytest.approx(1.0) and tb['flagged'] is False


def test_time_base_without_any_correlation(con):
    # Empty exports: LEAP's count is constant, so no clip has a correlation and nothing is flagged.
    time_base_tables(con, lambda f: f, empty=True)
    tb = hc.time_base(con)
    assert (tb['best_scale'], tb['best_r'], tb['r_at_scale_1'], tb['flagged']) == (None, None, None, False)
    assert tb['export_over_clip_median'] is None and tb['interval_over_gt_track'] is None


# ---------------------------------------------------------------- attribution and summary


def test_deta_decomposition_absolute_and_log():
    # CalDoT1-like numbers: DetA 0.075 under a ceiling of 0.3489.
    d = hc.deta_decomposition(0.075, 0.3489)
    assert (d['count_loss_abs'], d['placement_loss_abs']) == (pytest.approx(0.6511), pytest.approx(0.2739))
    assert d['count_share_abs'] == pytest.approx(0.6511 / 0.925) and d['placement_share_abs'] == pytest.approx(0.2739 / 0.925)
    assert d['count_loss_log'] == pytest.approx(-np.log(0.3489))
    assert d['placement_share_log'] == pytest.approx(np.log(0.3489 / 0.075) / -np.log(0.075))
    assert d['count_share_log'] + d['placement_share_log'] == pytest.approx(1)
    # A DetA of 0 has no log split; a missing ceiling has no split at all.
    zero = hc.deta_decomposition(0.0, 0.5)
    assert zero['placement_loss_abs'] == 0.5 and zero['placement_share_log'] is None and zero['placement_loss_log'] is None
    assert hc.deta_decomposition(0.1, None) == dict(deta=0.1, ceiling=None)
    # A perfect DetA has no loss to share.
    assert hc.deta_decomposition(1.0, 1.0)['count_share_abs'] is None


def attribution_section(label_dataset, hota, deta, ceiling, assa, flagged):
    # A section with hand-set reruns, ceiling, box counts, native counts, and time-base flag.
    runs = {name: {'HOTA': {'HOTA': h, 'DetA': deta, 'AssA': assa}} for name, h in hota.items()}
    return dict(dataset=label_dataset, trackeval=runs,
                count_ceiling={'after': dict(deta=ceiling, hota_perfect_association=np.sqrt(ceiling))},
                box_frames={'gt': dict(objects_in_clip=10, boxes_in_clip=100), 'after': dict(objects_in_clip=4, boxes_in_clip=50)},
                native_queries={'per_frame_counts': {'all': {'systems': {'reference': {'hota': 0.8},
                                                                          'leap': {'mae': 1.2, 'acc_leap': 0.4}}}}},
                time_base={'flagged': flagged})


HOTAS = dict(before=0.07, after=0.08, after_ambiguous_dropped=0.075, after_full_length=0.078,
             after_ambiguous_dropped_full_length=0.074)


def test_attribution_values():
    a = hc.attribution(attribution_section('caldot1-y05', HOTAS, 0.075, 0.3489, 0.08, True))
    # Each step's HOTA change, from the right pair of reruns.
    assert a['fix_effect'] == pytest.approx(0.01)
    assert a['ambiguous_dropped_effect'] == pytest.approx(-0.005)
    assert a['ambiguous_dropped_effect_full_length'] == pytest.approx(-0.004)
    assert a['full_length_effect'] == pytest.approx(-0.002)
    # The full-clip DetA against its ceiling, and the association reported unattributed.
    assert a['deta_share_of_ceiling'] == pytest.approx(0.075 / 0.3489)
    assert a['deta_decomposition'] == hc.deta_decomposition(0.075, 0.3489)
    assert (a['assa_full_length'], a['assa_attributed'], a['time_base_flagged']) == (0.08, False, True)
    # The counts LEAP's intervals imply and the reused native-queries numbers.
    assert (a['objects_vs_ground_truth'], a['box_frames_vs_ground_truth']) == (0.4, 0.5)
    assert (a['reference_hota'], a['per_frame_count_mae'], a['per_frame_count_acc_leap']) == (0.8, 1.2, 0.4)


def test_summary_ranges_and_lists():
    # A CalDoT-like dataset (counts lose more in absolute DetA, placement more in logs) and a B3D-like one (placement
    # loses more on both scales, time base flagged).
    sections = {'CalDoT1': attribution_section('caldot1-y05', HOTAS, 0.075, 0.3489, 0.08, False),
                'B3D1': attribution_section('jnc0', {k: v / 2 for k, v in HOTAS.items()}, 0.03, 0.701, 0.05, True)}
    for s in sections.values():
        s['attribution'] = hc.attribution(s)
    out = hc.summary(sections)
    assert out['measured_conversion_effect_range'] == [pytest.approx(-0.005), pytest.approx(0.01)]
    assert out['leap_hota_range'] == [pytest.approx(0.035), pytest.approx(0.08)]
    assert out['deta_share_of_ceiling_range'] == [pytest.approx(0.03 / 0.701), pytest.approx(0.075 / 0.3489)]
    assert out['assa_full_length_range'] == [0.05, 0.08]
    assert out['count_loss_larger_absolute'] == ['CalDoT1'] and out['placement_loss_larger_log'] == ['CalDoT1', 'B3D1']
    assert out['time_base_flagged'] == ['B3D1'] and out['b3d_provisional'] == ['B3D1']
    # An empty list has no range.
    assert hc.value_range([]) is None


# ---------------------------------------------------------------- provenance and the whole run


def test_input_digests_track_every_file(con, dataset):
    load_all(con, dataset)
    first = hc.input_digests(con)
    # Four clips per source, each with bytes and a combined digest.
    assert {s: d['files'] for s, d in first.items()} == {s: 4 for s in hc.INPUT_SOURCES}
    assert all(d['bytes'] > 0 and len(d['sha256']) == 64 for d in first.values())
    # Changing one ground-truth file changes only the ground truth's digest.
    gt = hc.output_path(DATASET, 'ground truth', '003_groundtruth', 'te02.mp4')
    gt.write_text(gt.read_text() + json.dumps({'frame_idx': 5, 'tracks': []}) + '\n')
    second = hc.input_digests(con)
    assert [s for s in hc.INPUT_SOURCES if first[s]['sha256'] != second[s]['sha256']] == ['gt']


def test_results_name(monkeypatch):
    # Every dataset (in any order) is a full run; fewer is partial.
    monkeypatch.setattr(hc, 'DATASET_LABELS', {'a': 'A', 'b': 'B'})
    assert hc.results_name(['b', 'a']) == 'results.json'
    assert hc.results_name(['a']) == 'results-partial.json'


def all_floats(value):
    # Every float in a nested JSON value.
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for v in value.values() for f in all_floats(v)]
    if isinstance(value, list):
        return [f for v in value for f in all_floats(v)]
    return []


def test_main_writes_full_and_partial_results(con, dataset, tmp_path, monkeypatch):
    # p141's outputs and native-queries outputs consistent with this cache.
    write_p141_outputs(tmp_path)
    section, _, _ = hc.analyze_dataset(con, DATASET, dataset, tmp_path / 'probe')
    native = tmp_path / 'native'
    native.mkdir()
    configs, diagnostics = native_outputs(section)
    (native / 'configs.json').write_text(json.dumps(configs))
    (native / 'diagnostics.json').write_text(json.dumps(diagnostics))
    # The synthetic dataset as the only one, results into tmp_path.
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(hc, 'DATASET_LABELS', {DATASET: LABEL})
    monkeypatch.setattr(hc, 'results_dir', lambda _: out)
    argv = ['r1o2_leap_hota_conversion.py', '--sota-dir', str(dataset), '--native-dir', str(native), '--workers', '1']
    monkeypatch.setattr('sys.argv', argv)
    hc.main()
    full = json.loads((out / 'results.json').read_text())
    # Every top-level part, the dataset section, and the file records (two native, two per cache).
    assert set(full) == {'method', 'blockers', 'findings', 'summary', 'inputs', 'datasets'}
    assert list(full['datasets']) == [LABEL] and len(full['inputs']) == 6
    assert set(full['datasets'][LABEL]['trackeval']) == set(hc.VARIANTS)
    assert full['summary']['datasets'] == [LABEL] and full['summary']['b3d_provisional'] == [LABEL]
    # Every float is rounded to 4 decimals.
    assert all(round(f, 4) == f for f in all_floats(full))
    # A run over only some datasets writes results-partial.json and leaves results.json alone.
    before = (out / 'results.json').read_text()
    monkeypatch.setattr(hc, 'DATASET_LABELS', {DATASET: LABEL, 'jnc2': 'B3D2'})
    monkeypatch.setattr('sys.argv', argv + ['--datasets', DATASET])
    hc.main()
    assert (out / 'results.json').read_text() == before
    assert list(json.loads((out / 'results-partial.json').read_text())['datasets']) == [LABEL]
