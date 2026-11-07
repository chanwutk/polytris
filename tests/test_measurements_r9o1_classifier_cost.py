import csv
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from measurements import r9o1_classifier_cost as rc


# The two synthetic test videos of every dataset and their frame counts.
VIDEOS = {'te00.mp4': 20, 'te01.mp4': 33}
# Frames per dataset.
FRAMES = sum(VIDEOS.values())
# Reference HOTA; the 5% bound is 0.76.
REFERENCE_HOTA = 0.8
# Per-video p121 seconds at the operating point (cfg_s4), 0.04 s per video in every dataset, and the logs behind them.
#   s1: ms of each sample-rate-1 classifier batch per video (te00: 16 + 4 frames, te01: 16 + 16 + 1 frames).
#   det_ms: reference detection per frame; dataset 'a' has one 90 ms frame at the end of te01.
SPECS = {
    'a': dict(classify=0.010, prune=None, pack=0.005, detect=0.020, read=0.0025, unpack=0.0005, track=0.002,
              det_ms=30.0, s1=([16.0, 8.0], [16.0, 32.0, 5.0])),
    'b': dict(classify=0.006, prune=0.002, pack=0.004, detect=0.022, read=0.002, unpack=0.001, track=0.003,
              det_ms=60.0, s1=([32.0, 32.0], [32.0, 32.0, 2.0])),
    'c': dict(classify=0.018, prune=None, pack=0.007, detect=0.012, read=0.001, unpack=0.0004, track=0.0016,
              det_ms=10.0, s1=([8.0, 2.0], [8.0, 8.0, 0.5])),
}
# p020's last line per video: video decoding and copy-back, both excluded.
TAIL_LINE = [{'op': 'read', 'time': 900.0}, {'op': 'retrieve', 'time': 2.0}]
# Columns of p121's per-op table.
PER_OP_HEADER = ['op', 'time', 'count', 'stage', 'dataset', 'videoset', 'video', 'variant', 'variant_id']


@pytest.fixture
def con():
    # A fresh in-memory DuckDB connection per test.
    connection = duckdb.connect()
    yield connection
    connection.close()


def write_jsonl(path: Path, lines: list, blank_tail: bool = True):
    # Write one JSON value per line, optionally with a trailing blank line as some logs have.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines) + ('\n' if blank_tail else ''))


def batch_line(ms: float) -> list[dict]:
    # A p020 batch line whose ops add up to ms (inference and send carry the time).
    return [{'op': op, 'time': ms / 2 if op in ('inference', 'send') else 0.0} for op in rc.CLASSIFIER_BATCH_OPS]


def detector_line(ms: float) -> list[dict]:
    # A p002 line: video decoding (excluded) and detection.
    return [{'op': 'read', 'time': 7.0}, {'op': 'detect', 'time': ms}]


def write_csv(path: Path, header: list[str], rows: list[tuple]):
    # Write a small CSV with a header row.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def p020_sampled(frames: int, sample_rate: int) -> int:
    # p020's sampling rule, transcribed: every sample_rate-th frame, plus the last frame.
    indices = [i for i in range(frames) if i % sample_rate == 0]
    if frames > 0 and indices[-1] != frames - 1:
        indices.append(frames - 1)
    return len(indices)


def tradeoff(dataset: str, frame_count: int = FRAMES) -> tuple[pd.DataFrame, pd.Series]:
    # Three Tetris configurations: too inaccurate, the operating point (s=4), and slow but accurate.
    spec = SPECS[dataset]
    threshold = 0.4 if spec['prune'] is not None else np.nan
    tetris = pd.DataFrame({
        'variant_id': ['cfg_fast', 'cfg_s4', 'cfg_slow'],
        'throughput_fps': [2000.0, 662.5, 2.0],
        'HOTA_HOTA': [0.5, 0.78, 0.8],
        'classifier': ['Net'] * 3, 'tilesize': [60.0] * 3, 'sample_rate': [1.0, 4.0, 1.0],
        'tracking_accuracy_threshold': [np.nan, threshold, np.nan],
        'frame_count': [frame_count] * 3, 'time': [0.0265, 0.04 * len(VIDEOS), 26.5],
    })
    # The reference: 1 s per video.
    naive = pd.Series({'throughput_fps': 26.5, 'HOTA_HOTA': REFERENCE_HOTA, 'frame_count': frame_count, 'time': 1.0 * len(VIDEOS)})
    return tetris, naive


def per_op_rows(dataset: str) -> list[tuple]:
    # p121's per-op rows (op, seconds, records, stage, video, variant_id) of the operating point and the reference.
    spec, rows = SPECS[dataset], []
    for video, frames in VIDEOS.items():
        # One classifier batch per video at sample rate 4 (6 and 9 classified frames), split over inference and send.
        rows += [('inference', spec['classify'] / 2, 1, '020_exec_classify', video, 'cfg_s4'),
                 ('send', spec['classify'] / 2, 1, '020_exec_classify', video, 'cfg_s4')]
        # Pruning only when the configuration sets a threshold.
        if spec['prune'] is not None:
            rows += [('solve_ilp', spec['prune'], 1, '022_exec_prune_polyominoes', video, 'cfg_s4')]
        rows += [('render_copy', spec['pack'], 3, '030_exec_compress', video, 'cfg_s4'),
                 ('detect', spec['detect'], 2, '040_exec_detect', video, 'cfg_s4'),
                 ('read', spec['read'], 2, '040_exec_detect', video, 'cfg_s4'),
                 ('unpack_detections', spec['unpack'], 2, '050_exec_uncompress', video, 'cfg_s4'),
                 ('tracker_update', spec['track'], frames, '060_exec_track', video, 'cfg_s4')]
        # The reference: detection with decoding, then tracking (1 s per video).
        rows += [('detect', 0.8, frames, '001_preprocess_groundtruth_detection', video, 'naive'),
                 ('read', 0.1, frames, '001_preprocess_groundtruth_detection', video, 'naive'),
                 ('tracker_update', 0.1, frames, '002_preprocess_groundtruth_tracking', video, 'naive')]
        # Another configuration's rows must be ignored.
        rows += [('detect', 99.0, 1, '040_exec_detect', video, 'cfg_slow')]
    return rows


def write_per_op(path: Path, dataset: str, rows: list[tuple]):
    # p121's table: the given test rows plus a validation row that must be ignored.
    write_csv(path, PER_OP_HEADER,
              [(op, t, n, stage, dataset, 'test', video, 'naive' if cfg == 'naive' else 'polytris', cfg)
               for op, t, n, stage, video, cfg in rows]
              + [('detect', 50.0, 1, '040_exec_detect', dataset, 'valid', 'va00.mp4', 'polytris', 'cfg_s4')])


def classifier_log(dataset: str, video: str, rate: int) -> list:
    # Sample rate 1 uses the spec's batches; any other rate classifies at most 16 frames, so one batch per video.
    if rate == 1:
        return [batch_line(ms) for ms in SPECS[dataset]['s1'][list(VIDEOS).index(video)]] + [TAIL_LINE]
    assert p020_sampled(VIDEOS[video], rate) <= rc.CLASSIFIER_BATCH
    # The operating point's batch must equal p121's classify time for the video.
    return [batch_line(1000 * SPECS[dataset]['classify'])] + [TAIL_LINE]


def build_world(tmp_path: Path, monkeypatch, datasets=('a', 'b', 'c'), rates=(1, 4), skip_log=None, extra_run=None,
                frame_count: int = FRAMES) -> dict:
    """Synthetic datasets: tradeoff rows, throughput manifests, logs, per-op tables, and tile masks under tmp_path."""
    # Tradeoff rows come from the patched loader.
    monkeypatch.setattr(rc, 'load_test_tradeoff', lambda dataset: tradeoff(dataset, frame_count))
    # Cache paths resolve under tmp_path.
    monkeypatch.setattr(rc.cache, 'exec', lambda dataset, stage, video, *args: tmp_path / dataset / video / stage / Path(*args))
    monkeypatch.setattr(rc.cache, 'index', lambda dataset, stage, *args: tmp_path / dataset / 'index' / stage / Path(*args))
    # The detector config maps every dataset to one model whose weights do not exist.
    monkeypatch.setattr(rc, 'get_config', lambda name: {
        'dataset_name_mapping': {d: 'm' for d in SPECS},
        'dataset_detector_mapping': {'m': {'detector': 'yolo', 'model_path': str(tmp_path / 'none.pt'), 'width': 1}}})
    manifests, per_ops = {}, {}
    for d in datasets:
        # A 2x3 tile grid with 4 tiles ever relevant.
        mask = tmp_path / d / 'index' / 'never-relevant' / '60_all.npy'
        mask.parent.mkdir(parents=True)
        np.save(mask, np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8))
        detector_logs = {}
        for video, frames in VIDEOS.items():
            # Detector log: a flat cost per frame, with one 90 ms frame at the end of te01 in dataset 'a'.
            lines = [detector_line(SPECS[d]['det_ms'])] * frames
            if d == 'a' and video == 'te01.mp4':
                lines[-1] = detector_line(90.0)
            detector_logs[video] = tmp_path / d / video / 'naive' / 'detection_runtime.jsonl'
            write_jsonl(detector_logs[video], lines)
            # One classifier log per run, unless the test leaves it out; the run directory always exists.
            for rate in rates:
                run_dir = tmp_path / d / video / 'relevancy' / f'Net_60_{rate}' / 'score'
                run_dir.mkdir(parents=True)
                if skip_log != (d, video, rate):
                    write_jsonl(run_dir / 'runtime.jsonl', classifier_log(d, video, rate), blank_tail=video == 'te00.mp4')
            # Directories of other classifiers, tile sizes, and names must be ignored.
            for other in ('Net_120_1', 'Other_60_2', 'Net_60_4_old'):
                (tmp_path / d / video / 'relevancy' / other).mkdir()
        # A run that exists for one video only.
        if extra_run is not None and extra_run[0] == d:
            run_dir = tmp_path / d / 'te01.mp4' / 'relevancy' / f'Net_60_{extra_run[1]}' / 'score'
            write_jsonl(run_dir / 'runtime.jsonl', [batch_line(1.0), TAIL_LINE])
        # The throughput manifest lists both reference logs and an unrelated stage and split.
        manifests[d] = str(tmp_path / d / 'query_execution.csv')
        write_csv(Path(manifests[d]), ['dataset', 'videoset', 'video', 'variant', 'variant_id', 'stage', 'runtime_file'],
                  [(d, 'test', v, 'naive', 'naive', rc.DETECTOR_STAGE, str(p)) for v, p in detector_logs.items()]
                  + [(d, 'test', 'te00.mp4', 'naive', 'naive', '002_preprocess_groundtruth_tracking', '/elsewhere'),
                     (d, 'valid', 'va00.mp4', 'naive', 'naive', rc.DETECTOR_STAGE, '/missing')])
        # p121's per-op table.
        per_ops[d] = str(tmp_path / d / 'query_execution_per_op.csv')
        write_per_op(Path(per_ops[d]), d, per_op_rows(d))
    return dict(manifests=manifests, per_ops=per_ops, tmp=tmp_path, datasets=list(datasets))


@pytest.fixture
def world(tmp_path, monkeypatch):
    # Three datasets: 'b' prunes, 'a' and 'c' do not.
    return build_world(tmp_path, monkeypatch)


def run_per_frame(con, world) -> dict:
    # The per-frame half of main(): operating points, logs, frame tables, run and frame summaries, report.
    rc.load_configs(con, world['datasets'])
    rc.choose_operating_points(con, 0.05)
    rc.load_runtime_lines(con, rc.runtime_files(con, world['manifests']))
    rc.build_frame_tables(con)
    rc.summarize_runs(con)
    rc.summarize_frames(con)
    return rc.per_frame_report(con)


def prepared(con, world):
    # Tables ready for check_operator_ops.
    run_per_frame(con, world)
    rc.load_operator_ops(con, world['per_ops'])


def run_shares(con, world) -> dict:
    # The share half of main(), after the per-frame half.
    prepared(con, world)
    rc.check_operator_ops(con)
    rc.check_classify_against_runs(con)
    return rc.share_report(con, 0.05)


def load_logs(con, tmp_path, detector_lines: list, classifier_lines: list, raw: str | None = None):
    # One video with one detector log and one sample-rate-1 classifier log; raw text replaces the classifier log.
    det, cls = tmp_path / 'det.jsonl', tmp_path / 'cls.jsonl'
    write_jsonl(det, detector_lines)
    if raw is None:
        write_jsonl(cls, classifier_lines, blank_tail=False)
    else:
        cls.write_text(raw)
    rc.load_runtime_lines(con, [('d', 'v', 'detector', None, str(det)), ('d', 'v', 'classifier', 1, str(cls))])


# Operating point.

def test_operating_point_is_fastest_within_bound(con, world):
    rc.load_configs(con, ['a', 'b'])
    rc.choose_operating_points(con, 0.05)
    chosen = con.execute('SELECT dataset, config, sample_rate, tracking_accuracy_threshold, reference_accuracy '
                         'FROM chosen ORDER BY dataset').fetchall()
    # cfg_fast misses the 0.76 bound, so the next fastest qualifying configuration is chosen; NaN threshold is NULL.
    assert chosen == [('a', 'cfg_s4', 4, None, REFERENCE_HOTA), ('b', 'cfg_s4', 4, 0.4, REFERENCE_HOTA)]


def test_operating_point_missing_fails(con, monkeypatch):
    # Every configuration is more than 5% below the reference: there is no operating point to break down.
    tetris, naive = tradeoff('a')
    tetris['HOTA_HOTA'] = [0.5, 0.6, 0.7]
    monkeypatch.setattr(rc, 'load_test_tradeoff', lambda dataset: (tetris, naive))
    rc.load_configs(con, ['a'])
    with pytest.raises(AssertionError, match='no Tetris configuration'):
        rc.choose_operating_points(con, 0.05)


def test_operating_point_bound_is_inclusive(con, monkeypatch):
    # A configuration exactly at reference x 0.95 qualifies, and it is the fastest.
    tetris, naive = tradeoff('a')
    tetris['HOTA_HOTA'] = [REFERENCE_HOTA * (1 - 0.05), 0.78, 0.8]
    monkeypatch.setattr(rc, 'load_test_tradeoff', lambda dataset: (tetris, naive))
    rc.load_configs(con, ['a'])
    rc.choose_operating_points(con, 0.05)
    assert con.execute('SELECT config FROM chosen').fetchall() == [('cfg_fast',)]


# Per-frame comparison.

def test_per_frame_splits_batches_over_their_frames(con, world):
    out = run_per_frame(con, world)['datasets']['a']
    classifier, detector = out['classifier'], out['detector']
    # te00: 16 frames at 1 ms and 4 at 2 ms; te01: 16 at 1 ms, 16 at 2 ms, 1 at 5 ms (77 ms over 53 frames).
    assert (out['frames'], out['videos'], classifier['batches']) == (53, 2, 5)
    assert classifier['median_ms_per_frame'] == 1.0
    assert classifier['mean_ms_per_frame'] == round(77 / 53, 3)
    # fps follows the mean, consistent with throughput; fps_at_median follows the median.
    assert classifier['fps'] == round(53 / 77 * 1000, 1) and classifier['fps_at_median'] == 1000.0
    # Host intervals per op: inference and send carry half of every batch each.
    assert classifier['host_interval_ms_per_frame_by_op'] == {
        op: round(38.5 / 53, 4) if op in ('inference', 'send') else 0.0 for op in rc.CLASSIFIER_BATCH_OPS}
    # Retrieve is excluded from the classifier and reported per frame as a bound.
    assert classifier['retrieve_ms_per_frame'] == round(4.0 / 53, 4)
    # The detector's read is excluded; the 90 ms frame moves the mean, not the median.
    assert detector['median_ms_per_frame'] == 30.0
    assert detector['mean_ms_per_frame'] == round((52 * 30 + 90) / 53, 3)
    # Headline ratio of means, and the ratio of medians as a robustness check.
    assert out['ratio_of_means'] == round(((52 * 30 + 90) / 53) / (77 / 53), 1)
    assert out['ratio_of_medians'] == 30.0
    # Tile mask and log path are recorded for the disclosure.
    assert (classifier['grid'], classifier['tiles'], classifier['tiles_inferred_at_most']) == ([2, 3], 6, 4)
    assert classifier['log'] == '020_relevancy/Net_60_1/score/runtime.jsonl'


def test_classifier_runs_and_cost_across_runs(con, world):
    out = run_per_frame(con, world)['datasets']['a']
    runs = out['classifier_runs']
    # Only this classifier and tile size's runs, with p020's sampled frame counts (6 + 9 at s=4).
    assert sorted(runs) == ['1', '4']
    assert (runs['1']['classified_frames'], runs['1']['batches']) == (53, 5)
    assert (runs['4']['classified_frames'], runs['4']['batches'], runs['4']['videos']) == (15, 2, 2)
    # The s=4 run costs 10 ms per batch of 6 or 9 frames: 20 ms over 15 frames.
    assert runs['4']['mean_ms_per_classified_frame'] == round(20 / 15, 4)
    detector_mean = (52 * 30 + 90) / 53
    assert runs['4']['ratio_of_means'] == round(detector_mean / (20 / 15), 1)
    # The range across runs: s=1 is the dearer run here, s=4 the operating point and the cheaper one.
    across = out['cost_across_runs']
    assert across['sample_rates'] == [1, 4] and across['operating_point_sample_rate'] == 4
    assert across['mean_ms_per_classified_frame'] == dict(min=round(20 / 15, 4), max=round(77 / 53, 4),
                                                          operating_point=round(20 / 15, 4))
    assert across['ratio_of_means'] == dict(min=round(detector_mean / (77 / 53), 1), max=round(detector_mean / (20 / 15), 1),
                                            operating_point=round(detector_mean / (20 / 15), 1))
    assert across['spread_pct'] == round(100 * ((77 / 53) / (20 / 15) - 1), 1)


def test_per_frame_summary_across_datasets(con, world):
    summary = run_per_frame(con, world)['summary']['ratio_of_means']
    # Every-frame ratios of means: a 1650/77, b 60/(130/53), c 10/0.5.
    every = {'a': 1650 / 77, 'b': 60 / (130 / 53), 'c': 20.0}
    # Operating-point ratios: detector mean over 1000 x classify x 2 / 15 ms per classified frame.
    operating = {'a': (1650 / 53) / (20 / 15), 'b': 60 / (12 / 15), 'c': 10 / (36 / 15)}
    assert summary['every_frame_run'] == dict(median=round(every['a'], 1), min=20.0, max=round(every['b'], 1), datasets=3)
    assert summary['operating_point_run'] == dict(median=round(operating['a'], 1), min=round(operating['c'], 1),
                                                  max=round(operating['b'], 1), datasets=3)
    assert summary['all_runs'] == dict(min=round(operating['c'], 1), max=round(operating['b'], 1), runs=6)


def test_detector_model_is_labeled_as_current_config(con, world):
    model = run_per_frame(con, world)['datasets']['a']['detector']['model']
    # The model comes from today's config, not the logs; its weights are absent here.
    assert model['detector'] == 'yolo' and model['model_path'].endswith('none.pt')
    assert 'not recorded in the logs' in model['source'] and model['weights_found'] is False
    assert 'weights_predate_first_log' not in model


@pytest.mark.parametrize('offset, predates', [(-10, True), (10, False)])
def test_detector_model_weights_date(monkeypatch, tmp_path, offset, predates):
    # Weights modified 10 s before or after the first detector log.
    weights = tmp_path / 'w.pt'
    weights.write_bytes(b'')
    os.utime(weights, (1_000_000 + offset, 1_000_000 + offset))
    monkeypatch.setattr(rc, 'get_config', lambda name: {'dataset_name_mapping': {'d': 'm'},
                                                        'dataset_detector_mapping': {'m': {'detector': 'yolo', 'model_path': str(weights)}}})
    model = rc.detector_model('d', 1_000_000.0)
    assert model['weights_found'] is True and model['weights_predate_first_log'] is predates


def test_retina_weights_come_from_retinanet_config(monkeypatch, tmp_path):
    # B3D's RetinaNet entry has no model_path; its weights are named in configs/retinanet.json.
    weights = tmp_path / 'retina.pth'
    weights.write_bytes(b'')
    (tmp_path / 'retinanet.json').write_text(json.dumps({'config': 'x.yaml', 'weights': str(weights)}))
    monkeypatch.setattr(rc, 'CONFIG_DIR', str(tmp_path))
    monkeypatch.setattr(rc, 'get_config', lambda name: {'dataset_name_mapping': {'d': 'b3d'},
                                                        'dataset_detector_mapping': {'b3d': {'detector': 'retina'}}})
    model = rc.detector_model('d', os.path.getmtime(weights) + 1)
    assert model['model_path'] == str(weights) and 'retinanet.json' in model['source']
    assert model['weights_found'] is True and model['weights_predate_first_log'] is True


def test_full_last_batch_has_sixteen_frames(con, tmp_path):
    # 32 frames make exactly two full batches.
    load_logs(con, tmp_path, [detector_line(10.0)] * 32, [batch_line(16.0), batch_line(32.0), TAIL_LINE])
    rc.build_frame_tables(con)
    assert con.execute('SELECT ms, count(*) FROM classifier_frames GROUP BY ms ORDER BY ms').fetchall() == [(1.0, 16), (2.0, 16)]
    # Only the second batch is the video's last.
    assert con.execute('SELECT frames, last FROM classifier_batches ORDER BY line').fetchall() == [(16, False), (16, True)]


def test_single_frame_video(con, tmp_path):
    # One frame: one batch of one frame.
    load_logs(con, tmp_path, [detector_line(10.0)], [batch_line(3.0), TAIL_LINE])
    rc.build_frame_tables(con)
    assert con.execute('SELECT frames, last, ms FROM classifier_batches').fetchall() == [(1, True, 3.0)]


def test_sampling_macro_matches_p020(con):
    # p020's sampled frame count for every (frames, sample rate) pair, including empty and single-frame videos.
    cases = [(n, s) for n in list(range(0, 40)) + [900, 901] for s in (1, 2, 3, 4, 8, 16, 17)]
    rc.create_sampling_macro(con)
    got = con.execute('SELECT n, s, p020_sampled(n, s) FROM (SELECT unnest($n) AS n, unnest($s) AS s)',
                      {'n': [n for n, _ in cases], 's': [s for _, s in cases]}).fetchall()
    assert {(n, s): k for n, s, k in got} == {(n, s): p020_sampled(n, s) for n, s in cases}


@pytest.mark.parametrize('classifier_lines, detector_frames, message', [
    # One batch too many for 20 frames.
    ([batch_line(1.0)] * 3 + [TAIL_LINE], 20, 'one line per'),
    # One batch too few.
    ([batch_line(1.0)] + [TAIL_LINE], 20, 'one line per'),
    # The read/retrieve line is not last.
    ([batch_line(1.0), TAIL_LINE, batch_line(1.0)], 20, 'one line per'),
    # No read/retrieve line.
    ([batch_line(1.0)] * 2, 20, 'one line per'),
    # Two read/retrieve lines.
    ([batch_line(1.0)] * 2 + [TAIL_LINE, TAIL_LINE], 20, 'one line per'),
    # An unknown line, while the batch count and tail position are right: only the unknown-op check can fire.
    ([batch_line(1.0), batch_line(1.0), [{'op': 'weird', 'time': 1.0}], TAIL_LINE], 20, 'one line per'),
    # A batch line with an extra op.
    ([batch_line(1.0) + [{'op': 'extra', 'time': 1.0}], batch_line(1.0), TAIL_LINE], 20, 'one line per'),
    # A negative time.
    ([[{'op': op, 'time': -1.0} for op in rc.CLASSIFIER_BATCH_OPS], batch_line(1.0), TAIL_LINE], 20, 'negative'),
    # A missing time.
    ([[{'op': 'send'}] + batch_line(1.0)[1:], batch_line(1.0), TAIL_LINE], 20, 'negative'),
    # A missing op name.
    ([[{'time': 1.0}] + batch_line(1.0)[1:], batch_line(1.0), TAIL_LINE], 20, 'missing op names'),
    # An empty classifier log.
    ([], 20, 'one line per'),
])
def test_bad_classifier_logs_fail(con, tmp_path, classifier_lines, detector_frames, message):
    load_logs(con, tmp_path, [detector_line(10.0)] * detector_frames, classifier_lines)
    with pytest.raises(AssertionError, match=message):
        rc.build_frame_tables(con)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '1e400', '"slow"'])
def test_non_finite_times_fail(con, tmp_path, value):
    # A batch whose inference time is NaN, infinite, overflowing, or a string.
    bad = json.dumps(batch_line(1.0)).replace('0.5', value, 1)
    raw = bad + '\n' + json.dumps(batch_line(1.0)) + '\n' + json.dumps(TAIL_LINE) + '\n'
    load_logs(con, tmp_path, [detector_line(10.0)] * 20, [], raw=raw)
    with pytest.raises(AssertionError, match='non-finite'):
        rc.build_frame_tables(con)


@pytest.mark.parametrize('line', ['{"op": "send", "time": 1}', '[]', 'null'])
def test_lines_that_are_not_op_lists_fail(con, tmp_path, line):
    # A JSON object, an empty list, or null would vanish when unnested.
    raw = line + '\n' + json.dumps(batch_line(1.0)) + '\n' + json.dumps(TAIL_LINE) + '\n'
    with pytest.raises(AssertionError, match='not a non-empty'):
        load_logs(con, tmp_path, [detector_line(10.0)] * 16, [], raw=raw)


def test_malformed_json_fails(con, tmp_path):
    # A line that is not JSON stops the read.
    with pytest.raises(duckdb.InvalidInputException, match='Malformed JSON'):
        load_logs(con, tmp_path, [detector_line(10.0)] * 16, [], raw='not json\n')


def test_bad_detector_lines_fail(con, tmp_path):
    # A detector line without its detect op cannot be timed.
    load_logs(con, tmp_path, [detector_line(10.0)] * 3 + [[{'op': 'read', 'time': 1.0}]], [batch_line(1.0), TAIL_LINE])
    with pytest.raises(AssertionError, match='unexpected detector'):
        rc.build_frame_tables(con)


def test_null_detector_time_fails(con, tmp_path):
    # A detector time that is null must not be summed as zero.
    load_logs(con, tmp_path, [[{'op': 'read', 'time': 1.0}, {'op': 'detect', 'time': None}]], [batch_line(1.0), TAIL_LINE])
    with pytest.raises(AssertionError, match='negative'):
        rc.build_frame_tables(con)


def test_classifier_without_detector_log_fails(con, tmp_path):
    # A classifier log for a video the detector side lacks.
    cls = tmp_path / 'cls.jsonl'
    write_jsonl(cls, [batch_line(1.0), TAIL_LINE])
    det = tmp_path / 'det.jsonl'
    write_jsonl(det, [detector_line(10.0)])
    rc.load_runtime_lines(con, [('d', 'v', 'detector', None, str(det)), ('d', 'w', 'classifier', 1, str(cls))])
    with pytest.raises(AssertionError, match='one line per'):
        rc.build_frame_tables(con)


def test_inference_spikes_counted(con, tmp_path):
    # 80 frames, 5 batches; inference 1, 15, 1, 1, 20 ms: the second and the last batch exceed 10x the 1 ms median.
    load_logs(con, tmp_path, [detector_line(10.0)] * 80, [batch_line(ms) for ms in (2, 30, 2, 2, 40)] + [TAIL_LINE])
    rc.build_frame_tables(con)
    rc.summarize_runs(con)
    run = rc.query_dicts(con, 'SELECT * FROM classifier_runs')[0]
    assert (run['spike_batches'], run['spike_last_batches'], run['spike_excess_ms']) == (2, 1, 14 + 19)
    entry = rc.run_entry(dict(run, ratio_of_means=1.0))['inference_spikes']
    assert entry['excess_share_pct'] == round(100 * 33 / 76, 2)
    assert entry['mean_ms_per_classified_frame_without_excess'] == round((76 - 33) / 80, 4)


def test_no_spikes_when_median_inference_is_zero(con, tmp_path):
    # With a zero median, no batch is a spike rather than every positive one.
    lines = [[{'op': op, 'time': 5.0 if (op == 'inference' and i == 2) else 0.0} for op in rc.CLASSIFIER_BATCH_OPS] for i in range(3)]
    load_logs(con, tmp_path, [detector_line(10.0)] * 48, lines + [TAIL_LINE])
    rc.build_frame_tables(con)
    rc.summarize_runs(con)
    assert con.execute('SELECT spike_batches, spike_excess_ms FROM classifier_runs').fetchall() == [(0, 0.0)]


def test_frame_counts_must_match_tradeoff(con, tmp_path, monkeypatch):
    # The tradeoff table claims one more frame than the logs hold.
    world = build_world(tmp_path, monkeypatch, datasets=('a',), frame_count=FRAMES + 1)
    with pytest.raises(AssertionError, match='frame counts differ'):
        run_per_frame(con, world)


def test_missing_classifier_log_fails(con, tmp_path, monkeypatch):
    # A run directory without its log must stop the comparison rather than drop the video.
    world = build_world(tmp_path, monkeypatch, datasets=('a', 'b'), skip_log=('b', 'te01.mp4', 4))
    rc.load_configs(con, world['datasets'])
    rc.choose_operating_points(con, 0.05)
    with pytest.raises(AssertionError, match='runtime logs missing'):
        rc.runtime_files(con, world['manifests'])


@pytest.mark.parametrize('rates', [(4,), (1,)])
def test_missing_every_frame_or_operating_run_fails(con, tmp_path, monkeypatch, rates):
    # Without the s=1 run there is no every-frame cost; without the s=4 run no operating-point run.
    world = build_world(tmp_path, monkeypatch, datasets=('a',), rates=rates)
    rc.load_configs(con, world['datasets'])
    rc.choose_operating_points(con, 0.05)
    with pytest.raises(AssertionError, match='lack the every-frame or operating-point rate'):
        rc.runtime_files(con, world['manifests'])


def test_runs_differ_across_videos_fails(con, tmp_path, monkeypatch):
    # A sample-rate-8 run on te01 only.
    world = build_world(tmp_path, monkeypatch, datasets=('a',), extra_run=('a', 8))
    rc.load_configs(con, world['datasets'])
    rc.choose_operating_points(con, 0.05)
    with pytest.raises(AssertionError, match='differ across test videos'):
        rc.runtime_files(con, world['manifests'])


def test_manifest_without_test_videos_fails(con, world, tmp_path):
    # A manifest listing no test-split reference logs for the dataset.
    empty = tmp_path / 'empty.csv'
    write_csv(empty, ['dataset', 'videoset', 'video', 'variant', 'variant_id', 'stage', 'runtime_file'],
              [('a', 'valid', 'va00.mp4', 'naive', 'naive', rc.DETECTOR_STAGE, '/x')])
    rc.load_configs(con, ['a'])
    rc.choose_operating_points(con, 0.05)
    with pytest.raises(AssertionError, match='no test videos'):
        rc.runtime_files(con, {'a': str(empty)})


def test_missing_manifest_or_per_op_table_fails(con, world, tmp_path):
    # Absent inputs fail with their paths, not a bare DuckDB IOException.
    rc.load_configs(con, ['a'])
    rc.choose_operating_points(con, 0.05)
    with pytest.raises(AssertionError, match='throughput manifests .* missing'):
        rc.runtime_files(con, {'a': str(tmp_path / 'nope.csv')})
    with pytest.raises(AssertionError, match='per-op throughput tables .* missing'):
        rc.load_operator_ops(con, {'a': str(tmp_path / 'nope.csv')})


# Operator shares.

def test_shares_sum_to_extraction_and_ignore_other_rows(con, world):
    out = run_shares(con, world)['datasets']['a']
    ops = out['operators']
    # 0.08 s in total: classify 0.02, pack 0.01, detect 0.045 (0.005 of it canvas reads), unpack 0.001, track 0.004.
    assert out['extraction_seconds'] == 0.08 and out['frames'] == FRAMES
    # Pruning did not run, so it is null rather than zero.
    assert ops['prune'] is None and out['pruning_threshold'] is None
    expected = dict(classify=25.0, pack=12.5, detect=56.25, unpack=1.25, track=5.0)
    assert {k: ops[k]['share_pct'] for k in expected} == {k: pytest.approx(v, abs=0.051) for k, v in expected.items()}
    assert ops['detect']['seconds'] == 0.045 and ops['classify']['ms_per_frame'] == round(20 / 53, 4)
    assert out['detect_canvas_read_share_pct'] == 6.25
    # Per-op detail keyed by operator, then op.
    assert out['ops']['detect']['read'] == dict(seconds=0.005, share_pct=6.25)
    # The reference's own breakdown: detection 1.8 s of which 0.2 s decoding, tracking 0.2 s.
    assert out['reference']['operators']['detect']['seconds'] == 1.8
    assert out['reference']['detect_frame_read_share_pct'] == 10.0
    # The operating point and its bound.
    assert (out['config'], out['sample_rate'], out['hota_loss_pct']) == ('cfg_s4', 4, 2.5)


def test_share_summary_across_datasets(con, world):
    report = run_shares(con, world)
    summary = report['summary']
    # Classify shares 25 (a), 15 (b), 45 (c); detect 56.25, 60, 32.5; pack 12.5, 10, 17.5.
    assert summary['classify'] == dict(median_pct=25.0, min_pct=15.0, max_pct=45.0, datasets=3)
    assert summary['detect']['median_pct'] == pytest.approx(56.25, abs=0.051)
    assert (summary['detect']['min_pct'], summary['detect']['max_pct'], summary['detect']['datasets']) == (32.5, 60.0, 3)
    assert summary['pack'] == dict(median_pct=12.5, min_pct=10.0, max_pct=17.5, datasets=3)
    # Pruning only over the dataset that prunes.
    assert summary['prune'] == dict(median_pct=5.0, min_pct=5.0, max_pct=5.0, datasets=1)
    # The reference rows (detect 90%) stay out of the Tetris summary.
    assert all(v['datasets'] == 3 for k, v in summary.items() if k != 'prune')
    # Per dataset: prune null where it did not run, present where it did.
    assert report['datasets']['b']['operators']['prune']['share_pct'] == 5.0
    assert report['datasets']['b']['pruning_threshold'] == 0.4
    assert report['datasets']['a']['operators']['prune'] is None and report['datasets']['c']['operators']['prune'] is None


def test_ratio_vs_share_separates_sampling_from_run_cost(con, world):
    p = run_shares(con, world)['datasets']['b']['per_frame_vs_share']
    # 6 + 9 frames classified at sample rate 4 out of 53.
    assert (p['classified_frames'], p['frames'], p['sample_rate']) == (15, 53, 4)
    assert p['sampling_factor'] == round(53 / 15, 3)
    # The s=4 run costs 0.8 ms per classified frame, against 130/53 ms in the every-frame run.
    every_classify, run_classify = 130 / 53, 12 / 15
    assert p['classifier_ms_per_frame_every_frame'] == round(every_classify, 4)
    assert p['classifier_ms_per_classified_frame'] == round(run_classify, 4)
    assert p['per_classified_frame_cost_vs_every_frame'] == round(run_classify / every_classify, 3)
    # Classification per original frame shrinks by sampling over relative run cost.
    reduction = (53 / 15) / (run_classify / every_classify)
    assert p['classification_reduction'] == round(reduction, 2)
    assert p['classify_ms_per_original_frame'] == round(12 / 53, 4)
    # Detection per original frame: 60 ms per reference frame against 48 ms of canvas detection over 53 frames.
    assert p['detection_reduction'] == pytest.approx(60 / (48 / 53), abs=0.051)
    assert p['per_frame_ratio_of_means'] == round(60 / every_classify, 1)
    assert p['share_if_every_frame_and_nothing_else_pct'] == round(100 / (1 + 60 / every_classify), 1)
    # The explanation names both factors and the measured share.
    assert '3.53x fewer' in p['explanation'] and '-67% against the every-frame run' in p['explanation']
    assert '15.0% of extraction' in p['explanation'] and '(means;' in p['explanation']


def test_check_passes_on_consistent_tables(con, world):
    prepared(con, world)
    rc.check_operator_ops(con)
    rc.check_classify_against_runs(con)


def test_check_rejects_unknown_stage(con, world):
    prepared(con, world)
    con.execute("UPDATE operator_ops SET stage = '099_mystery' WHERE stage = '050_exec_uncompress' AND dataset = 'c'")
    with pytest.raises(AssertionError, match='unknown stages'):
        rc.check_operator_ops(con)


def test_check_rejects_missing_required_stage(con, world):
    prepared(con, world)
    con.execute("DELETE FROM operator_ops WHERE stage = '060_exec_track' AND dataset = 'b'")
    with pytest.raises(AssertionError, match='missing stages'):
        rc.check_operator_ops(con)


def test_check_rejects_pruning_without_threshold(con, world):
    prepared(con, world)
    # A pruning stage on a configuration without a threshold means the wrong rows were joined.
    con.execute("INSERT INTO operator_ops VALUES ('a', 'tetris', '022_exec_prune_polyominoes', 'solve_ilp', 0.0, 1)")
    with pytest.raises(AssertionError, match='missing stages'):
        rc.check_operator_ops(con)


def test_check_rejects_missing_threshold_stage(con, world):
    prepared(con, world)
    # A threshold without a pruning stage.
    con.execute("UPDATE chosen SET tracking_accuracy_threshold = 0.4 WHERE dataset = 'c'")
    with pytest.raises(AssertionError, match='missing stages'):
        rc.check_operator_ops(con)


def test_check_rejects_missing_reference(con, world):
    prepared(con, world)
    con.execute("DELETE FROM operator_ops WHERE system = 'reference' AND dataset = 'a'")
    with pytest.raises(AssertionError, match='missing stages'):
        rc.check_operator_ops(con)


def test_check_allows_an_op_absent_from_some_videos(con, world):
    # te01 logs no canvas read (its time folded into detect) and packs with a different op than te00.
    rows = []
    for op, t, n, stage, video, cfg in per_op_rows('a'):
        if (stage, video, cfg) == ('040_exec_detect', 'te01.mp4', 'cfg_s4') and op == 'read':
            continue
        if (stage, op, video, cfg) == ('040_exec_detect', 'detect', 'te01.mp4', 'cfg_s4'):
            t += SPECS['a']['read']
        if (stage, video) == ('030_exec_compress', 'te01.mp4'):
            op = 'render_fetch'
        rows.append((op, t, n, stage, video, cfg))
    write_per_op(Path(world['per_ops']['a']), 'a', rows)
    prepared(con, world)
    rc.check_operator_ops(con)


def test_check_rejects_stage_missing_from_a_video(con, world):
    # te01 has no unpacking rows at all.
    rows = [r for r in per_op_rows('a') if not (r[3] == '050_exec_uncompress' and r[4] == 'te01.mp4')]
    write_per_op(Path(world['per_ops']['a']), 'a', rows)
    prepared(con, world)
    with pytest.raises(AssertionError, match='every test video'):
        rc.check_operator_ops(con)


@pytest.mark.parametrize('replace', [False, True])
def test_check_rejects_stage_on_an_unknown_video(con, world, replace):
    # Tracking rows for a video the comparison does not have: added (3 videos) or in place of te01 (2 videos, 1 known).
    rows = per_op_rows('c')
    if replace:
        rows = [(op, t, n, stage, 'te99.mp4' if (stage, video) == ('060_exec_track', 'te01.mp4') else video, cfg)
                for op, t, n, stage, video, cfg in rows]
    else:
        rows += [('tracker_update', 0.0, 1, '060_exec_track', 'te99.mp4', 'cfg_s4')]
    write_per_op(Path(world['per_ops']['c']), 'c', rows)
    prepared(con, world)
    with pytest.raises(AssertionError, match='every test video'):
        rc.check_operator_ops(con)


def test_check_rejects_runtime_mismatch(con, world):
    prepared(con, world)
    # The tradeoff runtime disagrees with the summed per-op table.
    con.execute("UPDATE chosen SET seconds = 0.085 WHERE dataset = 'b'")
    with pytest.raises(AssertionError, match='tradeoff runtime'):
        rc.check_operator_ops(con)


@pytest.mark.parametrize('update', [
    # p121's classify time no longer equals the raw log.
    "UPDATE operator_ops SET seconds = seconds * 2 WHERE stage = '020_exec_classify' AND dataset = 'a'",
    # p121's batch count no longer equals the raw log.
    "UPDATE operator_ops SET records = records + 1 WHERE op = 'inference' AND dataset = 'c'",
    # p121 has no classify rows.
    "DELETE FROM operator_ops WHERE stage = '020_exec_classify' AND dataset = 'b'",
])
def test_classify_must_match_operating_point_log(con, world, update):
    prepared(con, world)
    con.execute(update)
    with pytest.raises(AssertionError, match='differs from its p020 log'):
        rc.check_classify_against_runs(con)


# Hardware notes and main.

@pytest.mark.parametrize('detector_time, overlap', [('2026-03-05 19:30:00+00', False), ('2026-04-19 23:06:00+00', True)])
def test_hardware_notes_from_log_dates(con, monkeypatch, detector_time, overlap):
    # Two classifier sets that overlap each other, one detector set before or inside their window.
    monkeypatch.setattr(rc, 'host_info', lambda: {'hostname': 'h'})
    con.execute('CREATE TABLE runtime_logs (dataset VARCHAR, video VARCHAR, kind VARCHAR, sample_rate INTEGER, '
                'last_modified TIMESTAMPTZ)')
    con.executemany('INSERT INTO runtime_logs VALUES (?, ?, ?, ?, ?)', [
        ('a', 'v1', 'classifier', 1, '2026-04-19 23:05:00+00'), ('a', 'v2', 'classifier', 1, '2026-04-19 23:08:00+00'),
        ('a', 'v1', 'classifier', 4, '2026-04-19 23:06:00+00'), ('a', 'v1', 'detector', None, detector_time)])
    notes = rc.hardware_notes(con)
    assert notes['logs_written']['classifier'] == dict(first='2026-04-19 23:05:00+00', last='2026-04-19 23:08:00+00',
                                                       log_sets=2, sets_overlapping_another_set=2)
    assert notes['logs_written']['detector']['sets_overlapping_another_set'] == 0
    assert ('the two windows overlap' in notes['note']) is overlap
    assert ('different runs' in notes['note']) is not overlap
    assert notes['host_that_read_the_cache'] == {'hostname': 'h'} and 'Not recorded' in notes['recorded_in_logs']


def test_results_written_with_method_notes(world, tmp_path, monkeypatch):
    # main() end to end on the three synthetic datasets, writing into a temporary results root.
    monkeypatch.setattr(rc, 'DATASET_LABELS', {'a': 'A', 'b': 'B', 'c': 'C'})
    monkeypatch.setattr(rc, 'results_dir', lambda script: tmp_path / 'out')
    monkeypatch.setattr(rc, 'host_info', lambda: {'hostname': 'h', 'cpus': 1, 'gpus': {}})
    (tmp_path / 'out').mkdir()
    monkeypatch.setattr(rc.cache, 'eval', lambda dataset, stage, *args: Path(world['manifests'][dataset])
                        if args[-1] == 'query_execution.csv' else Path(world['per_ops'][dataset]))
    monkeypatch.setattr('sys.argv', ['r9o1', '--datasets', 'a', 'b', 'c'])
    rc.main()
    per_frame = json.loads((tmp_path / 'out' / 'classifier_vs_detector.json').read_text())
    shares = json.loads((tmp_path / 'out' / 'operator_share.json').read_text())
    # Both files are keyed by the dataset label and carry the hardware and boundary notes.
    assert list(per_frame['datasets']) == ['A', 'B', 'C'] and list(shares['datasets']) == ['A', 'B', 'C']
    assert per_frame['method']['hardware']['host_that_read_the_cache']['hostname'] == 'h'
    assert 'Reference video read/decode is also excluded' in shares['method']['share_boundary']
    assert 'synchronize' in shares['method']['gpu_synchronization']
    # The summaries span all three datasets.
    assert per_frame['summary']['ratio_of_means']['every_frame_run']['datasets'] == 3
    assert shares['summary']['classify']['datasets'] == 3 and shares['summary']['prune']['datasets'] == 1


def test_unknown_dataset_rejected(monkeypatch):
    monkeypatch.setattr('sys.argv', ['r9o1', '--datasets', 'nope'])
    with pytest.raises(AssertionError, match='unknown datasets'):
        rc.main()
