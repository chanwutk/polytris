import gzip
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

import evaluation.p140_otif_transform as p140
from measurements import common
from measurements import r1o2_leap_otif_rerun as m

# The host-side GPU script (standard library only), loaded from its file since measurements-gpu/ is not a package.
_spec = importlib.util.spec_from_file_location(
    'r1o2_leap_otif_rerun_gpu', Path(__file__).resolve().parent.parent / 'measurements-gpu' / 'r1o2_leap_otif_rerun.py')
gpu_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_script)


# The synthetic dataset: a name with no SOTA or root-dataset mapping, its label, and its clips (LEAP names 0 and 1).
DATASET = 'jnc0'
LABEL = 'B3D1'
VIDEOS = ('te00.mp4', 'te01.mp4')
# Clip length in frames, and LEAP's sampled frames per clip.
N = 12
SAMPLED = {'te00.mp4': [0, 4, 8], 'te01.mp4': [0, 6]}
# Per clip, LEAP's objects: id -> (interval [start, end), pattern, class id, frames it was observed on).
OBJECTS = {'te00.mp4': {1: ((0, 9), 0, 2.0, [0, 4, 8]), 2: ((4, 12), 1, 7.0, [4, 8])},
           'te01.mp4': {1: ((0, 7), 0, 2.0, [0, 6])}}
# GNU time's report of a 1:02:03.5 run.
GNU_TIME = ('\tUser time (seconds): 100.25\n\tSystem time (seconds): 9.75\n'
            '\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.5\n'
            '\tMaximum resident set size (kbytes): 2048\n\tExit status: 0\n')


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test, with the script's macros.
    connection = duckdb.connect()
    m.create_macros(connection)
    yield connection
    connection.close()


def box(oid, frame, shift=0):
    # Object oid's box on a frame: it moves 10 px per frame, so a frame offset changes the IoU.
    x = 10.0 * (frame + shift) + 100 * oid
    return [x, 20.0 * oid, x + 30, 20.0 * oid + 15]


def write_jsonl_gz(path: Path, rows: list[dict]):
    # Gzipped JSON lines, as the GPU script writes them.
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def export_frames(video: str) -> list:
    # An export-leap-outputs.py frame list: every object's box on every frame of its interval, null on empty frames.
    frames: list = [None] * N
    for oid, ((s, e), _, cls, _) in OBJECTS[video].items():
        for f in range(s, e):
            x1, y1, x2, y2 = box(oid, f)
            frames[f] = (frames[f] or []) + [dict(left=x1, top=y1, right=x2, bottom=y2, track_id=oid, score=1.0,
                                                  **{'class': 'car'})]
    return frames


def gt_lines(video: str, shift: int = 0) -> list[dict]:
    # Ground truth on every frame: each object's box (shifted in time by shift frames), [track_id, x1, y1, x2, y2].
    return [dict(frame_idx=f, tracks=[[100 + oid, *box(oid, f, shift)] for oid, ((s, e), *_ ) in OBJECTS[video].items()
                                      if s <= f < e]) for f in range(N)]


def build(tmp_path: Path, monkeypatch, differ: bool = False, gt_shift: int = 0) -> tuple[Path, Path]:
    """Every input of one synthetic dataset under tmp_path: the GPU script's raw files, /sota-results, the root
    dataset's ground truth, and p141's reported DATASET.json. Returns (raw dir, sota dir)."""
    # The cache root, the split listing, and the clip length p140's unwrap reads.
    monkeypatch.setattr(m.cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(m, 'list_split_videos', lambda ds, split: list(VIDEOS))
    monkeypatch.setattr(p140, 'get_video_frame_count', lambda ds, video: N)
    raw, sota = tmp_path / 'raw', tmp_path / 'sota'
    rerun, scored, exports, speed = [], [], [], []
    for i, video in enumerate(VIDEOS):
        name = str(i)
        objects = [dict(id=oid, interval=list(iv), pattern=p, class_id=c, matched={}, intervals_raw=[list(iv)],
                        observations=[[*box(oid, f), c, f] for f in seen])
                   for oid, (iv, p, c, seen) in OBJECTS[video].items()]
        rerun.append(dict(video_name=name, video_file=f'{name}.mp4', frames_sampled=SAMPLED[video], patterns_sha256='p',
                          timings=dict(run_session_wall_s=2.0, detector_s=1.0, differencer_s=0.0, reid_s=0.5, match_s=0.25,
                                       decode_s=0.25, frames_filtered=0), objects=objects))
        # The scored side: the same objects, one interval changed on clip 1 when asked.
        scored_objects = [{k: o[k] for k in ('id', 'interval', 'pattern', 'class_id')} for o in objects]
        if differ and i == 1:
            scored_objects[0]['interval'] = [0, 5]
        scored.append(dict(video_name=name, patterns_sha256='p', objects=scored_objects))
        exports.append(dict(video_name=name, exporter_sha256='e', frames=export_frames(video)))
        (sota / DATASET / 'leap').mkdir(parents=True, exist_ok=True)
        (sota / DATASET / 'leap' / f'{name}.json').write_text(json.dumps(export_frames(video)))
        # Two scored runs, each a block of rows in clip order with cumulative sampled frames.
        gt = Path(common.output_path(DATASET, 'ground truth', '', video))
        gt.parent.mkdir(parents=True, exist_ok=True)
        gt.write_text(''.join(json.dumps(r) + '\n' for r in gt_lines(video, gt_shift)))
    for run in range(2):
        total = 0
        for i, video in enumerate(VIDEOS):
            total += len(SAMPLED[video])
            speed.append(dict(video_name=str(i), inference_total='2.0', detector='1.0', differencer='0', reid='0.5',
                              match='0.25', decode='0.25', frames_sampled=str(total), frames_filtered='0',
                              row=run * len(VIDEOS) + i))
    write_jsonl_gz(raw / DATASET / 'leap_rerun.jsonl.gz', rerun)
    write_jsonl_gz(raw / DATASET / 'leap_scored.jsonl.gz', scored)
    write_jsonl_gz(raw / DATASET / 'leap_scored_run.jsonl.gz', [dict(source='s', patterns={}, speed=speed)])
    write_jsonl_gz(raw / DATASET / 'leap_rerun_export.jsonl.gz', exports)
    write_jsonl_gz(raw / DATASET / 'leap_scored_export.jsonl.gz', exports)
    (raw / DATASET / 'leap_rerun_launch.json').write_text(json.dumps(dict(
        run='g5-x', host='bombe', gpu=5, cpus='20-29,60-69', numa_node=1, image='i', contaminated=False, foreign_samples=0,
        exit=0, start='s', end='e', gnu_time=GNU_TIME)))
    (raw / 'manifest.json').write_text(json.dumps(dict(
        images={}, dockerfile='', scored_container={}, label_aliases={},
        video_identity=[dict(dataset=DATASET, split='test', identical=2, differing=[])])))
    return raw, sota


def write_reported(tmp_path: Path, raw: Path):
    # p141's DATASET.json for LEAP: the scored exports through p140's transform, scored by TrackEval as p141 does.
    base = tmp_path / 'reported'
    for i, video in enumerate(VIDEOS):
        source = base / f'{i}.json'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(export_frames(video)))
        p140.transform_tracking_json(str(source), str(base / 'tracks' / video / 'tracking.jsonl'), DATASET, video,
                                     unwrap_leap=True)
    gt_dir = str(Path(common.output_path(DATASET, 'ground truth', '', VIDEOS[0])).parents[2])
    metrics = common.trackeval_rerun(gt_dir, str(Path('003_groundtruth', 'tracking.jsonl')), str(base / 'tracks'),
                                     'tracking.jsonl', list(VIDEOS), str(base / 'te'), ('HOTA',))
    lists = {family: {k: (np.asarray(v).tolist() if np.ndim(v) else v.item() if hasattr(v, 'item') else v)
                      for k, v in metrics[family].items()} for family in ('HOTA', 'Count')}
    path = Path(m.cache.sota('leap', DATASET, 'accuracy', 'raw', 'test', m.LEAP_PARAM, 'DATASET.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(metrics=lists)))


def loaded(con, tmp_path, monkeypatch, **kwargs):
    # The synthetic dataset loaded into con.
    raw, sota = build(tmp_path, monkeypatch, **kwargs)
    m.load_dataset(con, DATASET, m.raw_paths(raw, DATASET), sota)
    return raw, sota


# ---------------------------------------------------------------- loading and the rerun's sections

def test_ground_truth_boxes_skip_the_track_id(con, tmp_path, monkeypatch):
    # tracking.jsonl rows are [track_id, x1, y1, x2, y2]; the box must not start at the id.
    loaded(con, tmp_path, monkeypatch)
    row = con.execute("SELECT x1, y1, x2, y2 FROM gt_boxes WHERE video = 'te00.mp4' AND frame = 4 ORDER BY x1").fetchone()
    assert list(row) == box(1, 4)


def test_observations_are_one_row_per_kept_box(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch)
    rows = con.execute('SELECT video, id, frame, class_id FROM observations ORDER BY ALL').fetchall()
    assert rows == [('te00.mp4', 1, 0, 2.0), ('te00.mp4', 1, 4, 2.0), ('te00.mp4', 1, 8, 2.0), ('te00.mp4', 2, 4, 7.0),
                    ('te00.mp4', 2, 8, 7.0), ('te01.mp4', 1, 0, 2.0), ('te01.mp4', 1, 6, 2.0)]


def test_frame_base_aligned_at_lag_zero(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch)
    fb = m.frame_base(con)
    best = fb['best']
    assert fb['aligned'] and (best['scale'], best['lag']) == (1.0, 0)
    assert best['mean_best_iou'] == pytest.approx(1.0) and best['matched_share'] == 1.0
    # Every grid point is reported, for every observation.
    assert {(r['scale'], r['lag']) for r in fb['shifts']} == {(s, 0) for s in m.SCALES} | {(1.0, lag) for lag in m.LAGS}
    assert {r['observations'] for r in fb['shifts']} == {7}


def test_frame_base_finds_a_shifted_ground_truth(con, tmp_path, monkeypatch):
    # Ground truth two frames late: the observation on frame f matches ground truth frame f + 2.
    loaded(con, tmp_path, monkeypatch, gt_shift=-2)
    fb = m.frame_base(con)
    assert not fb['aligned'] and (fb['best']['scale'], fb['best']['lag']) == (1.0, 2)


def test_agreement_identical(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch)
    a = m.agreement(con)
    assert a['objects'] == dict(rerun_objects=3, scored_objects=3, matched_ids=3, same_interval=3, same_pattern=3,
                                same_class=3, identical=3)
    assert a['clips'] == dict(clips=2, identical_clips=2, differing_clips=[])
    # Two scored runs, each differenced to per-clip sampled frames that match the rerun's slices.
    assert a['frames_sampled'] == dict(scored_runs=2, clip_runs=4, same_sampled_count=4, rerun_frames_sampled=5)
    assert all(e['identical_clips'] == 2 for e in a['exports'].values())


def test_agreement_reports_a_differing_interval(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch, differ=True)
    a = m.agreement(con)
    assert a['objects']['identical'] == 2 and a['objects']['same_interval'] == 2 and a['objects']['same_pattern'] == 3
    assert a['clips'] == dict(clips=2, identical_clips=1, differing_clips=['1'])


def test_classes_count_objects_and_observations(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch)
    c = m.classes(con)
    per_object = c.pop('observations_per_object')
    assert c == dict(rerun_objects={'2': 2, '7': 1}, scored_objects={'2': 2, '7': 1}, observations={'2': 5, '7': 2})
    # Objects seen on 3, 2, and 2 sampled frames: none once, all three at least twice.
    assert per_object == dict(objects=3, observations=7, objects_unobserved=0, objects_observed_once=0,
                              objects_observed_twice_or_more=3, mean_observations=pytest.approx(7 / 3), max_observations=3)


def test_observations_per_object_counts_single_and_missing_observations(con, tmp_path, monkeypatch):
    loaded(con, tmp_path, monkeypatch)
    # One object observed once, one never (an empty list), one with a NULL list.
    con.execute("""CREATE OR REPLACE TABLE rerun_objects AS SELECT * FROM (VALUES
        (2.0, [[1.0, 2.0, 3.0, 4.0, 2.0, 0.0]]), (2.0, []::DOUBLE[][]), (7.0, NULL::DOUBLE[][])) t(class_id, observations)""")
    per_object = m.classes(con)['observations_per_object']
    assert (per_object['objects'], per_object['observations'], per_object['objects_unobserved'],
            per_object['objects_observed_once'], per_object['objects_observed_twice_or_more']) == (3, 1, 2, 1, 0)


def test_timing_parses_gnu_time(con, tmp_path, monkeypatch):
    raw, _ = loaded(con, tmp_path, monkeypatch)
    gpus = raw / DATASET / 'leap_rerun_gpus.jsonl.gz'
    t = m.timing(con, json.loads((raw / DATASET / 'leap_rerun_launch.json').read_text()), gpus)
    assert t['gnu_time_wall_s'] == pytest.approx(3723.5) and t['max_rss_kb'] == 2048
    assert t['leap_timers']['scored_runs_min'] == 2 and t['leap_timers']['rerun_detector_s'] == 2.0
    # No GPU log yet (the GPU script's gpus action not run): no neighbors, not an error.
    assert t['neighbors'] is None


def test_timing_reports_neighbor_gpus(con, tmp_path, monkeypatch):
    raw, _ = loaded(con, tmp_path, monkeypatch)
    gpus = raw / DATASET / 'leap_rerun_gpus.jsonl.gz'
    # The run's own GPU 5 (left out), GPU 6 on its node busy in 1 of 2 samples, GPU 0 on the other node always busy.
    write_jsonl_gz(gpus, [dict(t='a', gpu=g, util=u, mem_mib=10.0 * u) for g, u in ((5, 90), (6, 0), (6, 40), (0, 100), (0, 100))])
    t = m.timing(con, json.loads((raw / DATASET / 'leap_rerun_launch.json').read_text()), gpus)
    assert t['neighbors']['node_neighbor_busy_share'] == 0.5
    assert t['neighbors']['other_gpus'] == [dict(gpu=0, util_mean=100.0, busy_share=1.0, mem_max_mib=1000.0),
                                           dict(gpu=6, util_mean=20.0, busy_share=0.5, mem_max_mib=400.0)]


def test_gnu_macros(con):
    # h:mm:ss, m:ss, and a missing field.
    assert con.execute("SELECT gnu_elapsed_s('Elapsed (wall clock) time (h:mm:ss or m:ss): 6:01.31')").fetchone()[0] == \
        pytest.approx(361.31)
    assert con.execute('SELECT gnu_elapsed_s(?)', [GNU_TIME]).fetchone()[0] == pytest.approx(3723.5)
    assert con.execute("SELECT gnu_elapsed_s('nothing'), gnu_number('nothing', $p)",
                       {'p': m.GNU_FIELDS['user_s']}).fetchone() == (None, None)
    assert con.execute('SELECT gnu_number(?, ?)', [GNU_TIME, m.GNU_FIELDS['user_s']]).fetchone()[0] == 100.25
    # cpulists: two ranges, one CPU, and a range plus a single CPU.
    assert con.execute("SELECT cpu_count('20-29,60-69'), cpu_count('5'), cpu_count('0-3,8')").fetchone() == (20, 1, 5)


def test_raw_and_linear_readings(con, tmp_path, monkeypatch):
    raw, _ = loaded(con, tmp_path, monkeypatch)
    tasks = m.write_variants(con, DATASET, list(VIDEOS), raw / DATASET / 'leap_rerun_export.jsonl.gz', tmp_path / 'v')
    assert [t['variant'] for t in tasks] == list(m.HOTA_VARIANTS)
    read = lambda variant: [json.loads(l) for l in (tmp_path / 'v' / variant / 'te00.mp4' / 'tracking.jsonl').read_text().splitlines()]
    # raw: only the observed frames carry boxes; linear: object 1 on every frame from 0 to 8, interpolated.
    raw_frames = {r['frame_idx'] for r in read('raw') if r['tracks']}
    assert raw_frames == {0, 4, 8}
    linear = {r['frame_idx']: r['tracks'] for r in read('linear')}
    assert sorted(f for f, t in linear.items() if any(b[0] == 1 for b in t)) == list(range(9))
    assert [b for b in linear[2] if b[0] == 1][0][1:] == pytest.approx(box(1, 2))


def test_hota_section_difference_and_flag():
    def metrics(hota):
        return dict(HOTA=dict(HOTA=hota, DetA=0.1, AssA=0.2, LocA=0.9), Count=dict(Dets=1, GT_Dets=2, IDs=3))
    reruns = [dict(variant=v, metrics=metrics(h)) for v, h in (('raw', 0.1), ('linear', 0.3), ('pattern', 0.2 + 3e-6))]
    h = m.hota_section(reruns, metrics(0.2))
    assert h['pattern_minus_reported_e6'] == pytest.approx(3.0) and not h['pattern_reproduces_reported']
    assert m.hota_section([dict(r, metrics=metrics(0.2)) for r in reruns], metrics(0.2))['pattern_reproduces_reported']


def test_results_name():
    assert m.results_name(list(reversed(list(m.DATASET_LABELS)))) == 'results.json'
    assert m.results_name([DATASET]) == 'results-partial.json'


def test_missing_raw_files_name_the_gpu_script(tmp_path):
    with pytest.raises(AssertionError, match='measurements-gpu/r1o2_leap_otif_rerun.py'):
        m.raw_paths(tmp_path, DATASET)


def test_mismatched_clips_raise(con, tmp_path, monkeypatch):
    raw, sota = build(tmp_path, monkeypatch)
    # The scored side lacks clip 1.
    rows = [json.loads(l) for l in gzip.open(raw / DATASET / 'leap_scored.jsonl.gz', 'rt')]
    write_jsonl_gz(raw / DATASET / 'leap_scored.jsonl.gz', rows[:1])
    with pytest.raises(AssertionError, match='clips missing'):
        m.load_dataset(con, DATASET, m.raw_paths(raw, DATASET), sota)


def test_end_to_end_reproduces_the_reported_hota(tmp_path, monkeypatch):
    raw, sota = build(tmp_path, monkeypatch)
    write_reported(tmp_path, raw)
    # The results directory, as results_dir creates it, outside the checkout.
    (tmp_path / 'out').mkdir()
    monkeypatch.setattr(m, 'results_dir', lambda script: tmp_path / 'out')
    monkeypatch.setattr(sys, 'argv', ['x', '--datasets', DATASET, '--raw-dir', str(raw), '--sota-dir', str(sota),
                                      '--workers', '1'])
    m.main()
    out = json.loads((tmp_path / 'out' / 'results-partial.json').read_text())
    section = out['datasets'][LABEL]
    # The pattern reading is the reported LEAP tracks; the observations are a strict subset of them.
    assert section['hota']['pattern_reproduces_reported'] and section['hota']['pattern']['HOTA'] == section['hota']['reported']['HOTA']
    assert section['hota']['raw']['dets'] == 7 and section['hota']['linear']['dets'] > 7
    assert section['frame_base']['aligned'] and out['preparation'] is None


# ---------------------------------------------------------------- preparation

PREP_STEPS = {'leap': ['labels_train', 'labels_test', 'reid_data', 'reid_train', 'motion_patterns'],
              'otif': ['detect_train', 'tune_valid']}


def record(system, dataset, step, wall='0:10.0', exit=0, contaminated=False):
    # One step record as the GPU script writes it.
    gnu = f'\tUser time (seconds): 4.0\n\tSystem time (seconds): 1.0\n\tElapsed (wall clock) time (h:mm:ss or m:ss): {wall}\n'
    return dict(run=f'{system}-{step}', host='bombe', gpu=5, gpu_uuid='u', numa_node=1, cpus='20-29,60-69', image='i',
                start='s', end='e', exit=exit, samples=3, foreign_samples=int(contaminated), contaminated=contaminated,
                command='c', system=system, dataset=dataset, step=step, wrapped='w', output='o',
                output_size=dict(exists=True, files=2, bytes=10), gnu_time=gnu, compare_error=None, stdout_tail='')


def build_prep(tmp_path: Path) -> Path:
    """raw/prep with: LEAP jnc0 labels (one clip differs by a line), crops, a failed re-id training, no motion step;
    OTIF caldot1-y05 complete with a contaminated detection step and both validation curves."""
    prep = tmp_path / 'prep'
    prep.mkdir()
    (prep / 'steps.json').write_text(json.dumps(PREP_STEPS))
    leap, otif = prep / 'leap' / DATASET, prep / 'otif' / 'caldot1-y05'
    leap.mkdir(parents=True)
    otif.mkdir(parents=True)
    for step, wall, exit in (('labels_train', '1:40.0', 0), ('labels_test', '0:20.5', 0), ('reid_data', '0:05', 0),
                             ('reid_train', '1:00:00', 1)):
        (leap / f'{step}.json').write_text(json.dumps(record('leap', DATASET, step, wall, exit)))
    for step, contaminated in (('detect_train', True), ('tune_valid', False)):
        (otif / f'{step}.json').write_text(json.dumps(record('otif', 'caldot1-y05', step, '2:00.0', 0, contaminated)))
    # GPU samples of one step.
    write_jsonl_gz(leap / 'labels_train_samples.jsonl.gz', [dict(t='t', util=u, mem_mib=100.0 * u, pids=[], foreign=0,
                                                                 load_1m=2.0 * u) for u in (0, 50, 100)])
    # Labels ('frame,track id,x,y,w,h,confidence,class'): clip 0 identical; clip 1 shares C and one D, repeats D (key
    # 2,1 twice on the rerun side, so not paired), has E only on the scored side, and G and H moved by 2.5 and 0.6 px;
    # clip 2 exists only on the scored side. Scored files written at 100, 130, 160 s.
    line = lambda frame, x, w=30.0: f'{frame},1,{x},20.0,{w},40.0,0.9,2.00,-1,-1,-1'
    a, b, c, d, e = line(0, 10.0), '0,2,50.0,20.0,30.0,40.0,0.8,2.00,-1,-1,-1', line(1, 11.0), line(2, 12.0), line(3, 13.0)
    write_jsonl_gz(leap / 'labels_train_compare.jsonl.gz', [
        dict(source='rerun', clip=0, mtime=1.0, lines=[a, b]), dict(source='scored', clip=0, mtime=100.0, lines=[a, b]),
        dict(source='rerun', clip=1, mtime=2.0, lines=[c, d, d, line(4, 14.0), line(5, 15.0)]),
        dict(source='scored', clip=1, mtime=130.0, lines=[c, d, e, line(4, 14.4, 32.5), line(5, 15.6)]),
        dict(source='scored', clip=2, mtime=160.0, lines=[line(0, 1.0)])])
    write_jsonl_gz(leap / 'reid_data_compare.jsonl.gz', [
        dict(source=s, dir=d, names=n) for d, (r, sc) in {'image_train': (['x', 'y'], ['x', 'y']),
                                                          'image_query': (['q1'], ['q2'])}.items()
        for s, n in (('rerun', r), ('scored', sc))])
    (leap / 'reid_train_compare.json').write_text(json.dumps(dict(
        rerun=dict(metrics=[{'total_loss': 1.0}], time_lines=[]),
        scored=dict(metrics=[{'Rank-1': 0.5, 'mAP': 0.4}, {'total_loss': 2.0}, {'Rank-1': 0.9, 'mAP': 0.8}],
                    time_lines=['Total training time: 0:10:03 (0:00:00 on hooks)', 'Total training time: 5:46:28 (x)']))))
    (leap / 'motion_patterns_compare.json').write_text(json.dumps(dict(
        rerun={'jnc0_0_0_tracks_clustered.npy': [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]},
        scored={'jnc0_0_0_tracks_clustered.npy': [[[1, 2, 3, 4]]]}, scored_dir='d')))
    (otif / 'tune_valid_compare.json').write_text(json.dumps(dict(
        rerun=['d1\ts1\tt1\t100\t0.70', 'd2\ts2\tt2\t50\t0.75'], scored=['d1\ts1\tt1\t90\t0.72'], scored_path='p')))
    return prep


def test_preparation_totals(tmp_path):
    p = m.preparation(build_prep(tmp_path))
    totals = {(t['system'], t['dataset']): t for t in p['totals']}
    leap, otif = totals[('leap', DATASET)], totals[('otif', 'caldot1-y05')]
    # LEAP: the failed training and the missing motion step make it incomplete; its successful steps add up.
    assert leap['missing_steps'] == ['motion_patterns'] and leap['failed_steps'] == ['reid_train']
    assert not leap['complete'] and leap['clean'] and leap['wall_s'] == pytest.approx(100 + 20.5 + 5)
    assert leap['cpu_s'] == pytest.approx(15.0) and leap['cpu_sets'] == ['20-29,60-69']
    # CPU time without the evaluation-only step, and the busy share of the 20-thread set over the succeeded steps.
    assert leap['preparation_cpu_s'] == pytest.approx(10.0)
    assert leap['cpu_busy_share'] == pytest.approx(15.0 / ((100 + 20.5 + 5) * 20))
    # The test-clip labels are evaluation-only: in the total, not in the preparation total.
    assert leap['evaluation_only_steps'] == ['labels_test'] and leap['preparation_wall_s'] == pytest.approx(105.0)
    assert otif['evaluation_only_steps'] == [] and otif['preparation_wall_s'] == pytest.approx(240.0)
    # No GPU logs in this fixture: no neighbor activity known.
    assert leap['node_neighbor_busy_share_max'] is None
    # OTIF: complete, but one step saw a foreign GPU process.
    assert otif['complete'] and not otif['clean'] and otif['contaminated_steps'] == ['detect_train']
    assert otif['missing_steps'] == [] and otif['failed_steps'] == [] and otif['wall_s'] == pytest.approx(240.0)


def test_preparation_lists_a_missing_evaluation_only_step(tmp_path):
    prep = build_prep(tmp_path)
    (prep / 'leap' / DATASET / 'labels_test.json').unlink()
    leap = [t for t in m.preparation(prep)['totals'] if t['system'] == 'leap'][0]
    # Missing, still named as evaluation-only, and the preparation total is unchanged.
    assert leap['missing_steps'] == ['labels_test', 'motion_patterns'] and leap['evaluation_only_steps'] == ['labels_test']
    assert leap['preparation_wall_s'] == pytest.approx(105.0) and leap['wall_s'] == pytest.approx(105.0)


def test_preparation_neighbors_and_tracked_frames(tmp_path):
    prep = build_prep(tmp_path)
    # labels_train ran on GPU 5 (node 1): GPU 4 on its node busy in half the samples, GPU 1 (node 0) idle.
    write_jsonl_gz(prep / 'leap' / DATASET / 'labels_train_gpus.jsonl.gz',
                   [dict(t='a', gpu=g, util=u, mem_mib=1.0) for g, u in ((5, 80), (4, 30), (4, 0), (1, 0), (1, 0))])
    # OTIF's frame extraction printed how many frames came from tracks.
    otif = prep / 'otif' / 'caldot1-y05'
    (otif / 'extract_frames.json').write_text(json.dumps(dict(
        record('otif', 'caldot1-y05', 'extract_frames'), stdout_tail='x\ngot 1234 frames with track\nextracting 2000 frames\n')))
    steps_json = json.loads((prep / 'steps.json').read_text())
    (prep / 'steps.json').write_text(json.dumps(dict(steps_json, otif=['detect_train', 'extract_frames', 'tune_valid'])))
    p = m.preparation(prep)
    step = {(s['system'], s['step']): s for s in p['steps']}
    assert step[('leap', 'labels_train')]['node_neighbor_busy_share'] == 0.5
    assert [g['gpu'] for g in step[('leap', 'labels_train')]['other_gpus']] == [1, 4]
    # Steps without a GPU log have none; only the extraction has a tracked-frame count.
    assert step[('leap', 'labels_test')]['other_gpus'] is None and step[('leap', 'labels_test')]['tracked_frames'] is None
    assert step[('otif', 'extract_frames')]['tracked_frames'] == 1234
    totals = {t['system']: t for t in p['totals']}
    assert totals['leap']['node_neighbor_busy_share_max'] == 0.5


def test_preparation_steps_carry_samples(tmp_path):
    p = m.preparation(build_prep(tmp_path))
    step = [s for s in p['steps'] if s['system'] == 'leap' and s['step'] == 'labels_train'][0]
    assert (step['samples'], step['gpu_util_mean'], step['gpu_mem_max_mib'], step['load_1m_max']) == (3, 50.0, 10000.0, 200.0)
    assert step['gpu_util_busy_share'] == pytest.approx(2 / 3) and step['wall_s'] == 100.0
    # 5 CPU seconds over 100 s on 20 threads.
    assert step['cpu_threads'] == 20 and step['cpu_busy_share'] == pytest.approx(5 / 100 / 20)
    # Steps without samples have none, and the order follows steps.json.
    assert [s['step'] for s in p['steps'] if s['system'] == 'leap'] == PREP_STEPS['leap'][:4]


def test_preparation_reproduction_checks(tmp_path):
    p = m.preparation(build_prep(tmp_path))
    labels = p['leap_labels'][0]
    # 4 lines are shared (the rerun's second D has no scored partner); clip 2 has no rerun file.
    assert (labels['rerun_clips'], labels['scored_clips'], labels['identical_clips']) == (2, 3, 1)
    assert (labels['rerun_lines'], labels['scored_lines'], labels['identical_lines']) == (7, 8, 4)
    # Paired by clip, frame, and track id: a, b, c (identical), G (2.5 px off), H (0.6 px); D's key repeats and E is unpaired.
    assert (labels['paired_lines'], labels['paired_within_px'], labels['paired_max_px']) == (5, 4, pytest.approx(2.5))
    assert labels['scored_span_s'] == 60.0 and labels['scored_duration_estimate_s'] == pytest.approx(90.0)
    crops = {c['dir']: c for c in p['leap_crops']}
    assert crops['image_train']['common_crops'] == 2 and crops['image_query']['common_crops'] == 0
    reid = {r['source']: r for r in p['leap_reid_training']}
    # The last total and the last evaluation; the rerun's log has neither.
    assert reid['scored']['total_training_s'] == 5 * 3600 + 46 * 60 + 28 and reid['scored']['rank1'] == 0.9
    assert reid['rerun']['total_training_s'] is None and reid['rerun']['rank1'] is None
    # The last training row of each (the scored one follows an evaluation; neither logs cls_accuracy here).
    assert (reid['scored']['last_total_loss'], reid['rerun']['last_total_loss']) == (2.0, 1.0)
    assert reid['scored']['last_cls_accuracy'] is None
    assert p['leap_motion_patterns'] == [dict(dataset=DATASET, scored_dir='d', rerun_clusters=2, scored_clusters=1)]
    curves = {c['source']: c for c in p['otif_curves']}
    assert (curves['rerun']['points'], curves['rerun']['best_valid_accuracy'], curves['rerun']['last_config']) == \
        (2, 0.75, 'd2 s2 t2')
    assert curves['scored']['last_runtime'] == 90.0


def test_preparation_rejects_misplaced_and_unknown_records(tmp_path):
    prep = build_prep(tmp_path)
    # A record whose content names another step than its file.
    bad = json.loads((prep / 'leap' / DATASET / 'reid_data.json').read_text())
    (prep / 'leap' / DATASET / 'reid_data.json').write_text(json.dumps(dict(bad, step='labels_test')))
    with pytest.raises(AssertionError, match='misplaced'):
        m.preparation(prep)
    # A step the GPU script does not define.
    (prep / 'leap' / DATASET / 'reid_data.json').write_text(json.dumps(bad))
    (prep / 'leap' / DATASET / 'distill.json').write_text(json.dumps(record('leap', DATASET, 'distill')))
    with pytest.raises(AssertionError, match='unknown steps'):
        m.preparation(prep)


def test_preparation_requires_steps_json(tmp_path):
    with pytest.raises(AssertionError, match='steps.json'):
        m.preparation(tmp_path)


# ---------------------------------------------------------------- the GPU script's commands and GPU log


def test_preparation_metadata_writes_are_safe_across_processes(tmp_path):
    # Force writers to reach the atomic replace together, as four dataset queues do at startup.
    ctx = multiprocessing.get_context('fork')
    barrier = ctx.Barrier(4)
    replace = gpu_script.os.replace
    def writer(value):
        def simultaneous_replace(source, target):
            barrier.wait(timeout=10)
            replace(source, target)
        gpu_script.os.replace = simultaneous_replace
        gpu_script.write_json(tmp_path / 'steps.json', {'writer': value})
    processes = [ctx.Process(target=writer, args=(i,)) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert json.loads((tmp_path / 'steps.json').read_text())['writer'] in range(4)
    assert list(tmp_path.glob('*.tmp')) == []


def test_otif_commands_follow_the_scored_runs():
    # Every dataset tracks with [0,1,1,1,1,1,1] (the scored shell history, Amsterdam included) and tunes from 640_352_-1.
    for dataset in gpu_script.OTIF_DATASETS:
        tune, _ = gpu_script.otif_prep_command(dataset, 'tune_valid')
        detector = 'retinanet-1080x720' if dataset.startswith('jnc') else 'yolov5-704x480'
        assert f" ours-simple {detector} 640_352_-1 '{{\"Thresholds\":[0,1,1,1,1,1,1]}}' 10 " in tune
        tracks, out = gpu_script.otif_prep_command(dataset, 'tracks_train')
        # The output directory is created before iou-tracker2.go writes into it.
        assert tracks.startswith(f'mkdir -p {out} && ') and tracks.endswith(gpu_script.OTIF_THRESHOLDS + "'")
    # Frame extraction runs in GOPATH mode, so its `go run iou-tracker.go` builds.
    extract, out = gpu_script.otif_prep_command('ams-y05', 'extract_frames')
    assert extract.startswith('export GO111MODULE=off GOFLAGS= CGO_ENABLED=0 GOPATH=/scratch/env/otif/gopath && ')
    assert out == '/otif-data/dataset/amsterdam-y5/train/seg-train'
    # Every step's output is under the dataset's clone (/otif-data) or the scratch mount, where it may be removed.
    for dataset in gpu_script.OTIF_DATASETS:
        for step in gpu_script.OTIF_PREP_STEPS:
            assert gpu_script.otif_prep_command(dataset, step)[1].startswith(('/otif-data/dataset/', '/scratch/prep/otif/'))


def test_leap_commands_name_the_scored_inputs():
    # CalDoT labels with COCO YOLOv5x at 15 fps, B3D with RetinaNet; both splits write under the same label directory.
    caldot, out = gpu_script.leap_prep_command('caldot1-y05', 'labels_train')
    assert '--yolo_weight weights/yolov5x.pt' in caldot and '--fps 15' in caldot and out.endswith('/labels/caldot1-y5')
    b3d, _ = gpu_script.leap_prep_command('jnc0', 'labels_test')
    assert '--use_retinanet' in b3d and '/scratch/videos/jnc0/test' in b3d
    # The motion patterns cluster every train track (k=0) with a minimum of 3.
    motion, _ = gpu_script.leap_prep_command('ams-y05', 'motion_patterns')
    assert '--video_name amsterdam-y5' in motion and '--k 0 --cluster_min_num 3' in motion


def test_otif_root_removes_an_interrupted_clone(tmp_path, monkeypatch):
    # Commands are recorded, not run (ZFS reflink copies are not available here).
    calls = []
    monkeypatch.setattr(gpu_script, 'SCRATCH', tmp_path)
    monkeypatch.setattr(gpu_script, 'sh', lambda argv, run, **kwargs: calls.append(argv))
    # A clone directory without its marker, as a killed clone leaves it.
    root = tmp_path / 'prep' / 'otif' / 'caldot2-y05' / 'otif-dataset'
    (root / 'dataset' / 'caldot2-y5').mkdir(parents=True)
    assert gpu_script.otif_root('caldot2-y05', run=True) == root
    # The partial clone is removed before the first copy, and the marker is written.
    assert calls[0] == ['rm', '-rf', str(root)] and calls[1][:3] == ['cp', '-a', '--reflink=always']
    assert (root / 'clone.json').exists()
    # With the marker present, nothing is removed or copied again.
    calls.clear()
    gpu_script.otif_root('caldot2-y05', run=True)
    assert calls == []


@pytest.mark.parametrize('dataset', ['jnc0', 'jnc2', 'jnc6', 'jnc7'])
def test_otif_b3d_uses_retinanet_and_clears_generated_outputs(dataset, tmp_path, monkeypatch):
    # B3D must regenerate RetinaNet detections and use the original weights, with no YOLOv5 dependency.
    calls = []
    monkeypatch.setattr(gpu_script, 'SCRATCH', tmp_path)
    monkeypatch.setattr(gpu_script, 'sh', lambda argv, run, **kwargs: calls.append(argv))
    root = tmp_path / 'prep' / 'otif' / dataset / 'otif-dataset'
    for split in ('train', 'valid'):
        (root / 'dataset' / dataset / split / 'retinanet-1080x720').mkdir(parents=True)
        command, out = gpu_script.otif_prep_command(dataset, f'detect_{split}')
        assert command.endswith(f'/{split}/retinanet-1080x720/ 1080x720')
        assert out == f'/otif-data/dataset/{dataset}/{split}/retinanet-1080x720'
    gpu_script.otif_root(dataset, run=True)
    copies = [argv for argv in calls if argv[0] == 'cp']
    assert any(argv[-2:] == [str(gpu_script.OTIF_SOURCE / 'retinanet/b3d'), str(root / 'retinanet/b3d')] for argv in copies)
    assert not any('yolov5' in arg for argv in copies for arg in argv)
    marker = json.loads((root / 'clone.json').read_text())
    assert marker['removed'] == ['train/retinanet-1080x720', 'valid/retinanet-1080x720']


def test_gpu_rows_skip_incomplete_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_script, 'SCRATCH', tmp_path)
    (tmp_path / 'samples').mkdir()
    (tmp_path / 'samples' / 'g6-x.gpus.csv').write_text(
        '2026/10/27 06:20:56.791, 0, 90, 1883\n2026/10/27 06:20:56.804, 6, [N/A], 3\n2026/10/27 06:20:57.1, 5, 7, 12\n'
        '2026/10/27 06:20:58.0, 4, 1')
    assert gpu_script.gpu_rows('g6-x') == [dict(t='2026/10/27 06:20:56.791', gpu=0, util=90.0, mem_mib=1883.0),
                                          dict(t='2026/10/27 06:20:57.1', gpu=5, util=7.0, mem_mib=12.0)]
