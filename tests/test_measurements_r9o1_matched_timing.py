import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pytest

from measurements import r9o1_matched_timing as mt


# The GPU job, loaded from its file (measurements-gpu is not a package).
GPU_PATH = Path(__file__).resolve().parent.parent / 'measurements-gpu' / 'r9o1_matched_timing.py'
gpu_spec = importlib.util.spec_from_file_location('r9o1_matched_timing_gpu', GPU_PATH)
gpu = importlib.util.module_from_spec(gpu_spec)
gpu_spec.loader.exec_module(gpu)

# Timed videos per dataset and their frame counts: CalDoT1 has a full and a partial batch plus a short video.
VIDEOS = {'caldot1-y05': {'te00.mp4': 20, 'te29.mp4': 3}, 'jnc0': {'te01.mp4': 17}}
# Operating points, as the GPU job and r9o1_classifier_cost record them.
POINTS = {'caldot1-y05': dict(config='Net_60_4_r025_bl_s100_sort', classifier='Net', tilesize=60),
          'jnc0': dict(config='Net_60_8_r025_none_s100_sort', classifier='Net', tilesize=60)}
# Per-frame costs: the classifier's batch costs 1 ms per frame except a partial batch of a video longer than one batch
# at 2 ms; one frame per call costs 4 ms (CalDoT1) or 5 ms (B3D1); the detector costs 40 ms, with one 400 ms frame in
# CalDoT1's te00; the other runs cost the same on every frame.
CLASSIFIER_1_MS = {'caldot1-y05': 4.0, 'jnc0': 5.0}
FLAT_MS = {'decode': 3.0, 'rgb_copy': 0.5, 'classifier_stream': 0.9, 'detector': 40.0, 'detector_16': 20.0}
SLOW_FRAME = ('caldot1-y05', 'te00.mp4', 7, 400.0)
# Each dataset's window in the manifest (epoch ns; the host-side samples below are in epoch ms).
WINDOWS = {'caldot1-y05': (1_000_000_000 * 10**6, 1_000_060_000 * 10**6), 'jnc0': (1_000_100_000 * 10**6, 1_000_160_000 * 10**6)}
# The physical GPU the synthetic job ran on, and another GPU of the host.
TIMED_GPU, OTHER_GPU = 6, 5


def per_frame_ms(dataset: str, video: str, run: str, start: int, frames: int, n: int) -> float:
    # A call's cost per frame under the rules above.
    if run == 'classifier':
        return 1.0 if frames == 16 or n < 16 else 2.0
    if run == 'classifier_1':
        return CLASSIFIER_1_MS[dataset]
    if run == 'detector' and (dataset, video, start) == SLOW_FRAME[:3]:
        return SLOW_FRAME[3]
    return FLAT_MS[run]


def calls(dataset: str) -> list[dict]:
    # Every timed call of a dataset, in the GPU job's runs and batching (None: one call for the whole video).
    lines = []
    for video, n in VIDEOS[dataset].items():
        for run, per_call in gpu.RUNS.items():
            step = per_call or n
            for start in range(0, n, step):
                frames = min(step, n - start)
                lines.append(dict(dataset=dataset, video=video, run=run, frame_idx=start, frames=frames, start_ns=0,
                                  ms=frames * per_frame_ms(dataset, video, run, start, frames, n)))
    return lines


def sample(phase, util, clock, memory=1000.0, window_phase=None, gpu_index=TIMED_GPU):
    # One nvidia-smi sample as the GPU job writes it; its window starts in the same phase unless told otherwise.
    return dict(t_ns=0, phase=phase, window_phase=window_phase or phase, gpu_index=gpu_index, utilization_gpu_pct=util,
                utilization_memory_pct=1.0, memory_used_mib=memory, sm_clock_mhz=clock, memory_clock_mhz=7000.0,
                power_w=100.0, temperature_c=40.0, pstate='P0', clock_event_reasons='0x0000000000000000')


def samples() -> list[dict]:
    # Phases outside the runs (load, decode, warm-up) carry values that must be ignored; one classifier sample straddles
    # the boundary from decode; classifier_1 has none, as a run shorter than the window would; another GPU is busy
    # during the detector and during loading.
    return [sample('load', 0.0, 300.0, 50000.0), sample('decode', 0.0, 300.0), sample('warmup_detector', 99.0, 2000.0),
            sample('classifier', 0.0, 300.0, window_phase='decode'),
            sample('classifier', 30.0, 1500.0), sample('classifier', 50.0, 1700.0),
            sample('detector', 85.0, 1800.0), sample('detector', 87.0, 1900.0), sample('detector', 90.0, 1850.0),
            sample('detector', 100.0, 1900.0, 20000.0, gpu_index=OTHER_GPU),
            sample('load', 70.0, 1900.0, 30000.0, gpu_index=OTHER_GPU)]


def manifest() -> dict:
    # The GPU job's manifest for both datasets; B3D1's te01 has 20% of its pages off the GPU's node.
    def video(dataset, v, n):
        # A timed video's record.
        pages = {'N0': 20, 'N1': 80} if dataset == 'jnc0' else {'N1': 100}
        return dict(frames=n, anon_pages_by_node=pages, file=dict(path=f'/polyis-data/{v}', sha256='0'))
    return dict(method=dict(timing='synthetic'), datasets={d: dict(
        label=mt.DATASET_LABELS[d], operating_point=dict(POINTS[d], sample_rate=4, hota=0.78, reference_hota=0.8, loss=0.05),
        classifier=dict(name='Net', tilesize=60, optimization='torchscript_optimize', grid=[8, 12], tiles=96, tiles_inferred_at_most=29),
        detector=dict(detector='ultralytics', weights=dict(path='/w.pt')), runs=dict(gpu.RUNS),
        videos={v: video(d, v, n) for v, n in VIDEOS[d].items()}, frames=sum(VIDEOS[d].values()),
        started='s', finished='f', started_ns=WINDOWS[d][0], finished_ns=WINDOWS[d][1],
        idle_before_launch=dict(utilization_gpu_pct=0.0, memory_used_mib=0.0),
        environment=dict(hostname='bombe', cpu='Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz', gpu_index=TIMED_GPU,
                         gpu=dict(name='Quadro RTX 6000', uuid='GPU-6'),
                         placement=dict(gpu_numa_node=1, gpu_local_cpus=[20, 21, 22], cpus=[20, 21], numactl=dict(membind='1'))))
        for d in VIDEOS})


def write_log_derived(directory: Path, points: dict | None = None, labels=None):
    # r9o1_classifier_cost's two reports, with what the comparison reads.
    points = points or POINTS
    labels = labels or [mt.DATASET_LABELS[d] for d in POINTS]
    per_frame = {mt.DATASET_LABELS[d]: dict(
        videos=60, frames=54000, classifier=dict(mean_ms_per_frame=1.25, median_ms_per_frame=1.2),
        detector=dict(mean_ms_per_frame=50.0, median_ms_per_frame=49.0), ratio_of_means=40.0, ratio_of_medians=40.8,
        cost_across_runs=dict(ratio_of_means=dict(min=30.0, max=44.0, operating_point=35.0))) for d in POINTS}
    shares = {mt.DATASET_LABELS[d]: dict(points[d], operators=dict(classify=dict(seconds=20.0, share_pct=20.0)),
                                         extraction_seconds=100.0) for d in POINTS}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'classifier_vs_detector.json').write_text(json.dumps(dict(datasets={k: v for k, v in per_frame.items() if k in labels})))
    (directory / 'operator_share.json').write_text(json.dumps(dict(datasets={k: v for k, v in shares.items() if k in labels})))


@pytest.fixture
def world(tmp_path):
    # The GPU job's raw directory, host-side samples, and the log-derived results.
    raw = tmp_path / 'raw'
    raw.mkdir()
    for d in VIDEOS:
        gpu.write_jsonl_gz(raw / f'{d}.jsonl.gz', calls(d))
        gpu.write_jsonl_gz(raw / f'{d}.gpu.jsonl.gz', samples())
    (raw / mt.MANIFEST).write_text(json.dumps(manifest()))
    # Two launches' host samples (epoch ms): CalDoT1 always one process; B3D1 once two; samples outside both windows
    # (before and between datasets) must not count.
    def host(t, apps, apps_all, load):
        # One host-side sample.
        return dict(t_ms=t, apps=apps, apps_all=apps_all, load_1m=load)
    gpu.write_jsonl_gz(raw / 'host_compute_apps.1.jsonl.gz', [host(999_000_000, 5, 9, 50.0), host(1_000_000_000, 1, 1, 7.0),
                                                              host(1_000_030_000, 1, 2, 9.0), host(1_000_080_000, 3, 9, 60.0)])
    gpu.write_jsonl_gz(raw / 'host_compute_apps.2.jsonl.gz', [host(1_000_100_500, 1, 1, 7.0), host(1_000_120_000, 2, 3, 8.0),
                                                              host(1_000_160_000, 0, 0, 6.0)])
    write_log_derived(tmp_path / 'logs')
    return dict(raw=raw, logs=tmp_path / 'logs', tmp=tmp_path)


def build(world, datasets=None) -> dict:
    # The measurement's steps on the synthetic world.
    manifest_, datasets = mt.load_manifest(world['raw'], datasets)
    con = duckdb.connect()
    mt.load_calls(con, world['raw'], manifest_, datasets)
    return mt.report(con, world['raw'], manifest_, datasets, world['logs'])


# CalDoT1's classifier: te00 has a full batch at 1 ms and 4 frames at 2 ms (24 ms over 20 frames), te29 3 frames at 1 ms.
CAL_CLASSIFIER = 27 / 23
# CalDoT1's detector: 22 frames at 40 ms and one at 400 ms; te00 averages 58 ms, te29 40 ms.
CAL_DETECTOR = (22 * 40 + 400) / 23


def test_run_statistics(world):
    r = build(world)['datasets']
    cal = r['CalDoT1']['runs']
    # Classifier: 19 frames at 1 ms and 4 (te00's partial batch) at 2 ms.
    values = np.array([1.0] * 19 + [2.0] * 4)
    c = cal['classifier']
    assert (c['videos'], c['frames'], c['calls'], c['frames_per_call']) == (2, 23, 3, 16)
    assert c['mean_ms_per_frame'] == pytest.approx(CAL_CLASSIFIER)
    assert (c['median_ms_per_frame'], c['min_ms_per_frame'], c['max_ms_per_frame']) == (1.0, 1.0, 2.0)
    # p95 by linear interpolation, as NumPy's default.
    assert c['p95_ms_per_frame'] == pytest.approx(np.percentile(values, 95))
    # fps is frames over summed time.
    assert c['fps'] == pytest.approx(1000 / CAL_CLASSIFIER) and c['fps_at_median'] == pytest.approx(1000.0)
    # Each video's mean.
    assert c['per_video_mean_ms'] == {'te00.mp4': pytest.approx(1.2), 'te29.mp4': pytest.approx(1.0)}
    # One frame per call: 23 calls of 4 ms.
    assert (cal['classifier_1']['calls'], cal['classifier_1']['mean_ms_per_frame']) == (23, 4.0)
    # Whole-video runs: one call per video, split evenly over its frames.
    for run in ('decode', 'rgb_copy', 'classifier_stream'):
        assert (cal[run]['calls'], cal[run]['frames'], cal[run]['frames_per_call']) == (2, 23, None)
        assert cal[run]['mean_ms_per_frame'] == pytest.approx(FLAT_MS[run])
    # Detector: the 400 ms frame moves the mean and the max, not the median.
    d = cal['detector']
    assert d['mean_ms_per_frame'] == pytest.approx(CAL_DETECTOR)
    assert (d['median_ms_per_frame'], d['max_ms_per_frame']) == (40.0, 400.0)
    assert d['p95_ms_per_frame'] == pytest.approx(np.percentile([40.0] * 22 + [400.0], 95))
    assert d['per_video_mean_ms'] == {'te00.mp4': pytest.approx(58.0), 'te29.mp4': pytest.approx(40.0)}
    # The batched detector: te00 in a call of 16 and one of 4, te29 in one call.
    assert (cal['detector_16']['calls'], cal['detector_16']['mean_ms_per_frame']) == (3, pytest.approx(20.0))
    # B3D1: one video of 17 frames, a full batch (1 ms) and a partial one (2 ms).
    b = r['B3D1']['runs']['classifier']
    assert (b['frames'], b['calls'], b['mean_ms_per_frame']) == (17, 2, pytest.approx(18 / 17))


def test_ratios(world):
    cal = build(world)['datasets']['CalDoT1']['ratios']
    # As deployed: the ratio of means equals the ratio of throughputs; the ratio of medians is the robustness check.
    a = cal['as_deployed']
    assert (a['tetris_mean_ms'], a['reference_mean_ms']) == (pytest.approx(CAL_CLASSIFIER), pytest.approx(CAL_DETECTOR))
    assert a['ratio_of_means'] == pytest.approx(CAL_DETECTOR / CAL_CLASSIFIER)
    assert a['ratio_of_medians'] == pytest.approx(40.0)
    # Per video: te00 58 / 1.2, te29 40 / 1.0; the range spans them.
    assert a['per_video'] == {'te00.mp4': pytest.approx(58 / 1.2), 'te29.mp4': pytest.approx(40.0)}
    assert (a['ratio_of_means_video_min'], a['ratio_of_means_video_max']) == (pytest.approx(40.0), pytest.approx(58 / 1.2))
    # The RGB copy adds to Tetris's side (per frame and per video); a side of two runs has no single median.
    w = cal['as_deployed_with_rgb_copy']
    assert w['tetris_mean_ms'] == pytest.approx(CAL_CLASSIFIER + 0.5) and w['ratio_of_medians'] is None
    assert w['ratio_of_means'] == pytest.approx(CAL_DETECTOR / (CAL_CLASSIFIER + 0.5))
    assert w['per_video']['te00.mp4'] == pytest.approx(58 / 1.7)
    # The unsynchronized stream and both same-batch comparisons.
    assert cal['as_deployed_unsynchronized']['ratio_of_means'] == pytest.approx(CAL_DETECTOR / 0.9)
    assert cal['same_batch_1']['ratio_of_means'] == pytest.approx(CAL_DETECTOR / 4)
    s16 = cal['same_batch_16']
    assert (s16['ratio_of_means'], s16['ratio_of_medians']) == (pytest.approx(20 / CAL_CLASSIFIER), pytest.approx(20.0))
    # Every comparison is reported.
    assert set(cal) == set(mt.PAIRS)


def test_summary(world):
    result = build(world)
    s = result['summary']['ratio_of_means']
    assert set(s) == set(mt.PAIRS)
    # As deployed: CalDoT1 and B3D1 (40 / (18 / 17)), and the extremes over all three videos.
    pair = sorted([CAL_DETECTOR / CAL_CLASSIFIER, 40 / (18 / 17)])
    a = s['as_deployed']
    assert (a['min'], a['max'], a['datasets']) == (pytest.approx(pair[0]), pytest.approx(pair[1]), 2)
    assert a['median'] == pytest.approx(sum(pair) / 2)
    assert (a['video_min'], a['video_max']) == (pytest.approx(40 / (18 / 17)), pytest.approx(58 / 1.2))


def test_gpu_conditions_use_only_each_runs_samples(world):
    runs = build(world)['datasets']['CalDoT1']['runs']
    # Load, decode, and warm-up samples (50000 MiB, 99%, 2000 MHz) are not part of any run; the other GPU is not the
    # timed one.
    g = runs['detector']['gpu']
    assert (g['samples'], g['utilization_median_pct'], g['utilization_min_pct'], g['utilization_max_pct']) == (3, 87.0, 85.0, 90.0)
    assert (g['sm_clock_min_mhz'], g['sm_clock_median_mhz'], g['sm_clock_max_mhz'], g['memory_used_max_mib']) == (1800.0, 1850.0, 1900.0, 1000.0)
    assert (g['pstates'], g['clock_event_reasons']) == (['P0'], ['0x0000000000000000'])
    # The other GPU's load during the detector is reported beside it; its load phase is not.
    assert (g['other_gpu_samples'], g['other_gpus_utilization_max_pct'], g['other_gpus_memory_used_max_mib']) == (1, 100.0, 20000.0)
    # The sample whose window started in decode is dropped from the classifier.
    c = runs['classifier']['gpu']
    assert (c['samples'], c['utilization_median_pct'], c['utilization_min_pct'], c['other_gpu_samples']) == (2, 40.0, 30.0, 0)
    assert c['other_gpus_utilization_max_pct'] is None
    # A run without samples is reported with none, not dropped.
    assert runs['classifier_1']['gpu']['samples'] == 0 and runs['classifier_1']['gpu']['utilization_median_pct'] is None


def test_host_processes_count_samples_inside_each_window(world):
    r = build(world)['datasets']
    # CalDoT1's window holds two one-process samples; the 5 before it and the 3 between datasets are outside.
    assert r['CalDoT1']['host_processes'] == dict(samples=2, max_processes=1, samples_with_other_processes=0,
                                                  max_processes_all_gpus=2, load_1m_median=8.0, load_1m_max=9.0)
    # B3D1's window (end inclusive) holds 1, 2, and 0 processes.
    assert r['B3D1']['host_processes'] == dict(samples=3, max_processes=2, samples_with_other_processes=1,
                                               max_processes_all_gpus=3, load_1m_median=7.0, load_1m_max=8.0)


def test_host_processes_absent_without_sampler_files(world):
    # Without the host-side sampler there is nothing to say about other processes.
    for path in world['raw'].glob(mt.HOST_APPS_GLOB):
        path.unlink()
    assert build(world)['datasets']['CalDoT1']['host_processes'] is None


def test_placement(world):
    r = build(world)['datasets']
    # CalDoT1's pages are all on the GPU's node; B3D1's te01 has 80% there.
    assert r['CalDoT1']['placement'] == dict(gpu_numa_node=1, cpus=2, cpus_local_to_gpu=True, membind='1',
                                             anon_pages_on_gpu_node_share={'te00.mp4': 1.0, 'te29.mp4': 1.0})
    assert r['B3D1']['placement']['anon_pages_on_gpu_node_share'] == {'te01.mp4': pytest.approx(0.8)}
    # The environment is carried into the report.
    assert (r['CalDoT1']['host'], r['CalDoT1']['gpu']['name'], r['CalDoT1']['cpu']) == ('bombe', 'Quadro RTX 6000', manifest()['datasets']['jnc0']['environment']['cpu'])


def test_log_derived_comparison(world):
    cal = build(world)['datasets']['CalDoT1']
    # The log-derived numbers are copied and divided into, for the as-deployed comparison.
    assert cal['log_derived']['ratio_of_means'] == 40.0 and cal['log_derived']['ratio_of_means_across_runs']['min'] == 30.0
    m = cal['matched_over_log']
    assert m['classifier_mean'] == pytest.approx(CAL_CLASSIFIER / 1.25)
    assert m['detector_mean'] == pytest.approx(CAL_DETECTOR / 50.0)
    assert m['ratio_of_means'] == pytest.approx(cal['ratios']['as_deployed']['ratio_of_means'] / 40.0)
    # The share sensitivity scales only classification: 20 s of 100 s becomes 20 x factor of (80 + 20 x factor).
    factor = CAL_CLASSIFIER / 1.25
    assert cal['classify_share_pct'] == dict(log_derived=20.0, with_matched_classifier_cost=pytest.approx(100 * 20 * factor / (80 + 20 * factor)))
    # The manifest's operating point is carried into the report.
    assert cal['operating_point']['config'] == POINTS['caldot1-y05']['config']


@pytest.mark.parametrize('field, value', [('config', 'other'), ('classifier', 'ShuffleNet05'), ('tilesize', 30)])
def test_log_derived_must_describe_the_same_operating_point(world, field, value):
    # Another configuration, classifier, or tile size in r9o1_classifier_cost makes the comparison meaningless.
    write_log_derived(world['logs'], dict(POINTS, **{'caldot1-y05': dict(POINTS['caldot1-y05'], **{field: value})}))
    with pytest.raises(AssertionError, match='operating point differs'):
        build(world)


def test_log_derived_must_cover_every_dataset(world):
    # A dataset label missing from r9o1_classifier_cost.
    write_log_derived(world['logs'], labels=['CalDoT1'])
    with pytest.raises(AssertionError, match='has no B3D1'):
        build(world)
    # Missing reports fail with the producing script's name.
    (world['logs'] / 'operator_share.json').unlink()
    with pytest.raises(AssertionError, match='r9o1_classifier_cost'):
        build(world)


def test_missing_gpu_outputs_name_the_gpu_job(world):
    # No manifest.
    (world['raw'] / mt.MANIFEST).unlink()
    with pytest.raises(AssertionError, match='measurements-gpu/r9o1_matched_timing.py'):
        mt.load_manifest(world['raw'], None)


def test_manifest_from_an_older_job_fails(world):
    # A dataset timed without the batched detector (the first version of the job) cannot be compared at batch 16.
    m = manifest()
    del m['datasets']['jnc0']['runs']['detector_16']
    (world['raw'] / mt.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(AssertionError, match=r"jnc0: the manifest has no runs \['detector_16'\]"):
        build(world)


def test_missing_raw_files_fail(world):
    # A dataset's timings missing.
    (world['raw'] / 'jnc0.jsonl.gz').unlink()
    with pytest.raises(AssertionError, match='raw timing files'):
        build(world)


def test_missing_sample_files_fail(world):
    # A dataset's GPU samples missing.
    (world['raw'] / 'jnc0.gpu.jsonl.gz').unlink()
    with pytest.raises(AssertionError, match='GPU sample files'):
        build(world)


def test_dataset_selection(world):
    # By default every timed dataset; a subset on request; one the GPU job did not time is an error.
    assert mt.load_manifest(world['raw'], None)[1] == ['caldot1-y05', 'jnc0']
    assert set(build(world, ['jnc0'])['datasets']) == {'B3D1'}
    with pytest.raises(AssertionError, match='not timed'):
        mt.load_manifest(world['raw'], ['ams-y05'])


def edit_calls(world, dataset: str, edit):
    # Rewrite a dataset's raw timings through `edit(lines) -> lines`.
    gpu.write_jsonl_gz(world['raw'] / f'{dataset}.jsonl.gz', edit(calls(dataset)))


def is_call(c: dict, run: str, video: str, frame: int) -> bool:
    # Whether a raw line is the given run's call starting at the frame.
    return (c['run'], c['video'], c['frame_idx']) == (run, video, frame)


def split_stream(lines: list[dict]) -> list[dict]:
    # te00's whole-video stream recorded as two calls of 10 frames.
    out = [c for c in lines if not is_call(c, 'classifier_stream', 'te00.mp4', 0)]
    return out + [dict(dataset='caldot1-y05', video='te00.mp4', run='classifier_stream', frame_idx=s, frames=10, ms=9.0, start_ns=0)
                  for s in (0, 10)]


@pytest.mark.parametrize('edit, message', [
    # A detector frame missing.
    (lambda ls: [c for c in ls if not is_call(c, 'detector', 'te00.mp4', 3)], 'exactly once'),
    # A frame timed twice.
    (lambda ls: ls + [c for c in ls if is_call(c, 'classifier_1', 'te29.mp4', 0)], 'exactly once'),
    # A whole run missing for one video.
    (lambda ls: [c for c in ls if not (c['run'] == 'classifier_1' and c['video'] == 'te29.mp4')], 'exactly once'),
    # A whole-video run missing for one video.
    (lambda ls: [c for c in ls if not is_call(c, 'rgb_copy', 'te29.mp4', 0)], 'exactly once'),
    # A frame beyond the video's end.
    (lambda ls: ls + [dict(c, frame_idx=3) for c in ls if is_call(c, 'detector', 'te29.mp4', 2)], 'batching'),
    # Batches of 15 frames do not follow p020.
    (lambda ls: [dict(c, frames=15) if is_call(c, 'classifier', 'te00.mp4', 0) else c for c in ls], 'batching'),
    # A detector call holding two frames.
    (lambda ls: [dict(c, frames=2) if is_call(c, 'detector', 'te00.mp4', 0) else c for c in ls], 'batching'),
    # A batched detector call of 8 frames.
    (lambda ls: [dict(c, frames=8) if is_call(c, 'detector_16', 'te00.mp4', 0) else c for c in ls], 'batching'),
    # A whole-video run split into two calls.
    (split_stream, 'batching'),
    # Zero, negative, and missing times.
    (lambda ls: [dict(c, ms=0.0) if is_call(c, 'detector', 'te00.mp4', 1) else c for c in ls], 'non-positive'),
    (lambda ls: [dict(c, ms=-1.0) if is_call(c, 'classifier', 'te00.mp4', 0) else c for c in ls], 'non-positive'),
    (lambda ls: [dict(c, ms=None) if is_call(c, 'classifier_1', 'te00.mp4', 0) else c for c in ls], 'non-positive'),
    # An unknown run, a video the manifest does not list, a line of another dataset.
    (lambda ls: ls + [dict(ls[0], run='classifier_4')], 'unknown video or run'),
    (lambda ls: [dict(c, video='te99.mp4') if is_call(c, 'detector', 'te29.mp4', 0) else c for c in ls], 'unknown video or run'),
    (lambda ls: [dict(c, dataset='jnc0') if is_call(c, 'detector', 'te29.mp4', 0) else c for c in ls], 'wrong dataset'),
])
def test_bad_raw_timings_fail(world, edit, message):
    edit_calls(world, 'caldot1-y05', edit)
    with pytest.raises(AssertionError, match=message):
        build(world)


def test_main_writes_the_report(world, monkeypatch, capsys):
    # main() on the synthetic world, with outputs under tmp_path.
    monkeypatch.setattr(mt, 'parse_args', lambda: argparse.Namespace(datasets=None))
    monkeypatch.setattr(mt, 'results_dir', lambda script: world['tmp'])
    monkeypatch.setattr(mt, 'LOG_DERIVED_DIR', world['logs'])
    mt.main()
    result = json.loads((world['tmp'] / 'matched_timing.json').read_text())
    # Both datasets, rounded, with the method of the GPU job, the comparisons, and the aggregation.
    assert set(result['datasets']) == {'CalDoT1', 'B3D1'}
    assert result['datasets']['CalDoT1']['runs']['classifier']['mean_ms_per_frame'] == round(CAL_CLASSIFIER, 4)
    assert result['method']['timing'] == 'synthetic' and 'ratio of means equals' in result['method']['aggregation']
    assert result['method']['pairs']['as_deployed_with_rgb_copy'] == dict(
        tetris=['classifier', 'rgb_copy'], reference='detector', description=mt.PAIR_DESCRIPTIONS['as_deployed_with_rgb_copy'])
    # One printed line per dataset, then the summary.
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith(('CalDoT1 ', 'B3D1 ')) for line in lines) == 2 and lines[-1].startswith('summary')


# ---- the GPU job ----


def test_gpu_and_measurement_agree_on_raw_layout():
    # The GPU job writes where the measurement reads, and times every run the comparisons need.
    assert (gpu.RAW_DIR, gpu.MANIFEST) == (mt.RAW_DIR, mt.MANIFEST)
    assert set(mt.REQUIRED_RUNS) <= set(gpu.RUNS) and set(gpu.RUNS) == set(gpu.RUN_DESCRIPTIONS)
    assert gpu.RUNS['classifier'] == gpu.RUNS['detector_16'] == gpu.BATCH_SIZE == 16 and gpu.RUNS['detector'] == 1


def test_gpu_copies_of_p020_and_p002_still_match():
    # load_classifier and run_video repeat p020's preparation (classify_all cannot be called without its worker queue)
    # and p002's detector calls; fail when the pipeline's source changes underneath them.
    root = GPU_PATH.parent.parent
    p020 = (root / 'scripts' / 'p020_exec_classify.py').read_text()
    for snippet in ('[0.485, 0.456, 0.406] * 2', '[0.229, 0.224, 0.225] * 2', "'never-relevant', f'{tile_size}_all.npy'",
                    'repeat_interleave(grid_width)', '.repeat(grid_height)', 'np.ascontiguousarray(frame[:, :, ::-1])',
                    '[frames[idx - 1] if idx > 0 else frames[idx + 1] for idx in batch_indices]', 'BATCH_SIZE: int = 16',
                    'model = load_model(dataset, tile_size, classifier, device)', 'select_model_optimization(model, benchmark_results'):
        assert snippet in p020, f'p020 no longer contains {snippet!r}'
    p002 = (root / 'preprocess' / 'p002_preprocess_naive_detection.py').read_text()
    for snippet in ('get_detector(dataset, gpu_id, batch_size=1)', 'polyis.models.detector.detect(frame, detector)',
                    f'for _ in range({gpu.DETECTOR_WARMUP_FRAMES}):'):
        assert snippet in p002, f'p002 no longer contains {snippet!r}'


def test_gpu_select_videos():
    # Evenly spaced from first to last, in name order.
    sixty = [f'te{i:02d}.mp4' for i in range(60)]
    assert gpu.select_videos(list(reversed(sixty)), 3) == ['te00.mp4', 'te29.mp4', 'te59.mp4']
    b3d = ['te13.mp4', 'te01.mp4', 'te08.mp4', 'te04.mp4', 'te12.mp4']
    assert gpu.select_videos(b3d, 3) == ['te01.mp4', 'te08.mp4', 'te13.mp4']
    # One video, all videos when there are fewer than asked, and no duplicates.
    assert gpu.select_videos(b3d, 1) == ['te01.mp4']
    assert gpu.select_videos(['b', 'a'], 3) == ['a', 'b']
    assert gpu.select_videos(b3d, 5) == sorted(b3d)
    # Nothing to select.
    with pytest.raises(AssertionError):
        gpu.select_videos([], 3)
    with pytest.raises(AssertionError):
        gpu.select_videos(b3d, 0)


# An nvidia-smi line of GPU_FIELDS for GPU 6.
LINE = '6, 87, 40, 1191, 1875, 7000, [N/A], 61, P0, 0x0000000000000000'


def test_gpu_parse_sample():
    # Numbers become floats (the index an int), pstate and hex flags stay text, bracketed values become None.
    assert gpu.parse_sample(LINE) == dict(gpu_index=6, utilization_gpu_pct=87.0, utilization_memory_pct=40.0, memory_used_mib=1191.0,
                                          sm_clock_mhz=1875.0, memory_clock_mhz=7000.0, power_w=None, temperature_c=61.0,
                                          pstate='P0', clock_event_reasons='0x0000000000000000')
    # A line with another field count.
    with pytest.raises(AssertionError):
        gpu.parse_sample('87, 40')


def test_gpu_idle_check(monkeypatch):
    # An idle GPU passes and is recorded with the time.
    monkeypatch.setattr(gpu, 'query_gpu', lambda g, fields: '6, 0, 0, 0, 300, 405, 22.9, 34, P8, 0x0000000000000001')
    assert gpu.idle_check(6)['pstate'] == 'P8'
    # Any utilization or memory in use refuses to start.
    for line in ('6, 3, 0, 0, 300, 405, 22.9, 34, P8, 0x1', '6, 0, 0, 1156, 300, 405, 22.9, 34, P8, 0x1'):
        monkeypatch.setattr(gpu, 'query_gpu', lambda g, fields, line=line: line)
        with pytest.raises(AssertionError, match='not idle'):
            gpu.idle_check(6)


@pytest.mark.parametrize('visible, order, message', [('0', 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES=6'),
                                                     ('6', None, 'CUDA_DEVICE_ORDER=PCI_BUS_ID'),
                                                     ('6', 'FASTEST_FIRST', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')])
def test_gpu_main_requires_the_checked_device(monkeypatch, visible, order, message):
    # The models use cuda:0, so CUDA must number GPUs as nvidia-smi does and expose only the checked one.
    monkeypatch.setattr(gpu, 'parse_args', lambda: argparse.Namespace(datasets=['jnc0'], loss=0.05, gpu=6, videos=3))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    if order is None:
        monkeypatch.delenv('CUDA_DEVICE_ORDER', raising=False)
    else:
        monkeypatch.setenv('CUDA_DEVICE_ORDER', order)
    with pytest.raises(AssertionError, match=message):
        gpu.main()


def test_gpu_main_checks_idle_before_touching_cuda(monkeypatch):
    # A busy GPU stops the job before any CUDA call or environment query (which initializes CUDA).
    monkeypatch.setattr(gpu, 'parse_args', lambda: argparse.Namespace(datasets=['jnc0'], loss=0.05, gpu=6, videos=3))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '6')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    touched = []
    for name in ('synchronize', 'get_device_properties', 'get_device_name', 'init', 'empty_cache'):
        monkeypatch.setattr(gpu.torch.cuda, name, lambda *a, name=name, **k: touched.append(name))
    monkeypatch.setattr(gpu, 'environment', lambda g: touched.append('environment'))

    def busy(g):
        # The GPU shows another job.
        raise AssertionError(f'GPU {g} is not idle')
    monkeypatch.setattr(gpu, 'idle_check', busy)
    with pytest.raises(AssertionError, match='not idle'):
        gpu.main()
    assert touched == []


def environment(**placement) -> dict:
    # The parts of environment() that check_environment reads, for a job pinned to node 1.
    return dict(gpu_index=6, gpu=dict(uuid='GPU-a8e8'), torch_device_uuid='GPU-a8e8',
                placement=dict(dict(gpu_numa_node=1, gpu_local_cpus=[20, 21, 60, 61], cpus=[20, 61], numactl=dict(membind='1')), **placement))


def test_gpu_check_environment():
    # cuda:0 is the checked GPU, CPUs are local to it, and memory is bound to its node.
    gpu.check_environment(environment())
    # Another device behind cuda:0.
    with pytest.raises(AssertionError, match='cuda:0 is GPU-other'):
        gpu.check_environment(dict(environment(), torch_device_uuid='GPU-other'))
    # A CPU on the other node, as an unpinned launch has.
    with pytest.raises(AssertionError, match='not all local'):
        gpu.check_environment(environment(cpus=[0, 20]))
    # Memory not bound, or bound to the other node.
    for numactl in (dict(), dict(membind='0'), dict(membind='0 1')):
        with pytest.raises(AssertionError, match='memory bound'):
            gpu.check_environment(environment(numactl=numactl))


def test_gpu_placement_helpers(monkeypatch, tmp_path):
    # nvidia-smi's bus id as sysfs names it, and kernel CPU lists.
    assert gpu.sysfs_bus_id('00000000:B2:00.0') == '0000:b2:00.0'
    assert gpu.sysfs_bus_id('00000001:3D:00.0') == '0001:3d:00.0'
    assert gpu.parse_cpulist('20-22,60-61\n') == [20, 21, 22, 60, 61] and gpu.parse_cpulist('5') == [5]
    # placement() reads the GPU's node and local CPUs from sysfs and numactl's view of this process.
    device = tmp_path / '0000:b2:00.0'
    device.mkdir()
    (device / 'numa_node').write_text('1\n')
    (device / 'local_cpulist').write_text('20-21\n')
    monkeypatch.setattr(gpu, 'SYSFS_PCI', tmp_path)
    monkeypatch.setattr(gpu.shutil, 'which', lambda name: '/usr/bin/numactl')
    shown = 'policy: bind\npreferred node: 1\nphyscpubind: 20 21\ncpubind: 1\nnodebind: 1\nmembind: 1\n'
    monkeypatch.setattr(gpu.subprocess, 'run', lambda *a, **k: argparse.Namespace(stdout=shown))
    monkeypatch.setattr(gpu.os, 'sched_getaffinity', lambda pid: {21, 20})
    assert gpu.placement('00000000:B2:00.0') == dict(gpu_numa_node=1, gpu_local_cpus=[20, 21], cpus=[20, 21], numactl=dict(
        policy='bind', **{'preferred node': '1'}, physcpubind='20 21', cpubind='1', nodebind='1', membind='1'))
    # Without numactl the job cannot say where its memory is.
    monkeypatch.setattr(gpu.shutil, 'which', lambda name: None)
    with pytest.raises(AssertionError, match='numactl not found'):
        gpu.numactl_show()


def test_gpu_anon_pages_by_node(monkeypatch, tmp_path):
    # Only anonymous mappings count; file-backed pages (libraries) do not.
    maps = tmp_path / 'numa_maps'
    maps.write_text('7f00 bind:1 anon=10 dirty=10 N1=10 kernelpagesize_kB=4\n'
                    '7f10 bind:1 file=/usr/lib/libtorch.so mapped=50 N0=50 kernelpagesize_kB=4\n'
                    '7f20 bind:1 anon=7 dirty=7 N0=2 N1=5 kernelpagesize_kB=4\n'
                    '7f30 bind:1 stack anon=3 dirty=3 N1=3 kernelpagesize_kB=4\n')
    real_open = open
    monkeypatch.setattr(gpu, 'open', lambda path, *a, **k: real_open(maps if path == '/proc/self/numa_maps' else path, *a, **k), raising=False)
    assert gpu.anon_pages_by_node() == {'N0': 2, 'N1': 18}


def test_gpu_timed_synchronizes_around_every_call(monkeypatch):
    # The call runs strictly between two synchronizations, and its time and start are returned.
    events = []
    monkeypatch.setattr(gpu.torch.cuda, 'synchronize', lambda: events.append('sync'))

    def fn(x, y=0):
        # The timed work.
        events.append('fn')
        return x + y
    out, ms, start_ns = gpu.timed(fn, 2, y=3)
    assert (out, events) == (5, ['sync', 'fn', 'sync']) and ms >= 0 and start_ns > 0


def test_gpu_sampler_labels_samples_by_their_window():
    # Phases started at 1 s (classifier) and 3 s (detector); utilization averages over the preceding second.
    sampler = gpu.GpuSampler.__new__(gpu.GpuSampler)
    sampler.phases = [(0, 'load'), (1_000_000_000, 'classifier'), (3_000_000_000, 'detector')]
    s = int(1e9)
    # Inside the classifier, on its boundary with load, inside the detector, and before the first phase.
    assert (sampler.label(int(2.5 * s), LINE)['phase'], sampler.label(int(2.5 * s), LINE)['window_phase']) == ('classifier', 'classifier')
    assert (sampler.label(int(1.5 * s), LINE)['phase'], sampler.label(int(1.5 * s), LINE)['window_phase']) == ('classifier', 'load')
    assert sampler.label(5 * s, LINE)['window_phase'] == 'detector' and sampler.label(-s, LINE)['phase'] == 'load'
    # A phase starting exactly at the sample time is the running one; the sample keeps nvidia-smi's fields.
    record = sampler.label(3 * s, LINE)
    assert (record['phase'], record['window_phase'], record['t_ns'], record['gpu_index']) == ('detector', 'classifier', 3 * s, 6)


def test_gpu_sampler_reads_every_gpu_until_stopped(monkeypatch):
    # A fake nvidia-smi that prints two GPUs' lines (and a blank one) and records how it was called and stopped.
    calls = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            # The command, and its output stream.
            calls.append(args)
            self.stdout = iter([LINE + '\n', '\n', LINE.replace('6,', '5,', 1) + '\n'])

        def terminate(self):
            # The sampler stops nvidia-smi before joining its reader.
            calls.append('terminate')

        def wait(self):
            # Nothing to wait for.
            return 0
    monkeypatch.setattr(gpu.subprocess, 'Popen', FakePopen)
    sampler = gpu.GpuSampler()
    sampler.set_phase('detector')
    out = sampler.stop()
    # Every GPU is sampled (no -i), every non-blank line becomes a sample, and the process is terminated.
    assert '-i' not in calls[0] and '-lms' in calls[0] and calls[-1] == 'terminate'
    assert [r['gpu_index'] for r in out] == [6, 5]
    assert [p for _, p in sampler.phases] == ['start', 'detector']


def test_gpu_decode(tmp_path, monkeypatch):
    # A real video: every frame decoded, in BGR, at the header's count.
    path = tmp_path / 'v.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
    writer.release()
    frames = gpu.decode(path)
    assert len(frames) == 5 and frames[0].shape == (48, 64, 3)
    assert gpu.video_size(path) == (64, 48)
    # A missing file cannot be opened.
    with pytest.raises(AssertionError, match='cannot open'):
        gpu.decode(tmp_path / 'missing.mp4')

    class FakeCapture:
        # A stream of `n` frames whose header declares `declared`.
        def __init__(self, n, declared):
            self.n, self.declared = n, declared

        def isOpened(self):
            # The file opens.
            return True

        def get(self, prop):
            # The header's frame count (the only property decode reads).
            return self.declared

        def read(self):
            # One frame per read until the stream ends.
            self.n -= 1
            return (True, np.zeros((2, 2, 3), np.uint8)) if self.n >= 0 else (False, None)

        def release(self):
            # Nothing to close.
            pass
    # A header that disagrees with the stream, and a one-frame video, which p020 cannot pair.
    for n, declared, message in ((4, 5, 'decoded 4 of 5'), (1, 1, 'fewer than two')):
        monkeypatch.setattr(gpu.cv2, 'VideoCapture', lambda p, n=n, declared=declared: FakeCapture(n, declared))
        with pytest.raises(AssertionError, match=message):
            gpu.decode(path)


def test_gpu_run_video_feeds_the_measurement(world, monkeypatch):
    # No GPU: synchronization is recorded, decoding returns synthetic frames, and both detector calls are stubs.
    events = []
    monkeypatch.setattr(gpu.torch.cuda, 'synchronize', lambda: events.append('sync'))
    # Frames whose blue (BGR) channel holds the frame index and red channel 255.
    frames = [np.dstack([np.full((2, 3), i, np.uint8), np.zeros((2, 3), np.uint8), np.full((2, 3), 255, np.uint8)]) for i in range(40)]
    monkeypatch.setattr(gpu, 'decode', lambda path: (events.append('decode'), frames)[1])
    detected = []
    monkeypatch.setattr(gpu.polyis.models.detector, 'detect', lambda frame, detector: (events.append('detect'), detected.append([int(frame[0, 0, 0])]))[0])
    monkeypatch.setattr(gpu.polyis.models.detector, 'detect_batch',
                        lambda batch, detector: (events.append('detect_batch'), detected.append([int(f[0, 0, 0]) for f in batch]))[0])
    batches = []

    def classify(batch_frames, batch_prev_frames):
        # The classifier gets RGB copies (red first) and each frame's previous frame (the second for the first).
        events.append('classify')
        batches.append(([int(f[0, 0, 2]) for f in batch_frames], [int(f[0, 0, 2]) for f in batch_prev_frames], int(batch_frames[0][0, 0, 0])))
        return None, [{'op': 'inference', 'time': 1.0}]

    class Sampler:
        # Records every phase the job starts.
        def __init__(self):
            # Phases in the order the job starts them.
            self.phases = []

        def set_phase(self, phase):
            # Keep the name only; times do not matter here.
            self.phases.append(phase)
    sampler = Sampler()
    # Frames of another size than the first video's are refused.
    with pytest.raises(AssertionError, match='frame size differs'):
        gpu.run_video('jnc0', 'te01.mp4', Path('v'), (2, 3), classify, 'det', sampler, warmup=False)
    # Warm-up: every run, but both detector runs only on the first frames.
    sampler.phases.clear()
    warm, _ = gpu.run_video('jnc0', 'te01.mp4', Path('v'), (3, 2), classify, 'det', sampler, warmup=True)
    assert sum(r['run'] == 'detector' for r in warm) == gpu.DETECTOR_WARMUP_FRAMES
    assert [r['frames'] for r in warm if r['run'] == 'detector_16'] == [16, 16]
    assert sampler.phases == [f'warmup_{run}' for run in gpu.RUNS]
    # Timed: every run over all 40 frames, in RUNS' order.
    for log in (batches, detected, events, sampler.phases):
        log.clear()
    records, video = gpu.run_video('jnc0', 'te01.mp4', Path('v'), (3, 2), classify, 'det', sampler, warmup=False)
    assert sampler.phases == list(gpu.RUNS) and list(dict.fromkeys(r['run'] for r in records)) == list(gpu.RUNS)
    assert [(r['frame_idx'], r['frames']) for r in records if r['run'] in ('classifier', 'detector_16')] == [(0, 16), (16, 16), (32, 8)] * 2
    assert [(r['frame_idx'], r['frames']) for r in records if r['run'] in ('decode', 'rgb_copy', 'classifier_stream')] == [(0, 40)] * 3
    assert sum(r['run'] == 'classifier_1' for r in records) == 40 and sum(r['run'] == 'detector' for r in records) == 40
    # Every call is synchronized before and after, except inside the stream, whose 3 calls run between one pair.
    timed_calls = [e for e in events if e != 'sync']
    assert timed_calls == ['decode'] + ['classify'] * (3 + 40 + 3) + ['detect'] * 40 + ['detect_batch'] * 3
    assert events[:3] == ['sync', 'decode', 'sync'] and events.count('sync') == 2 * len(records)
    # Events before the stream: decode (3), the RGB copy (2), and 43 per-call classifier calls (3 each).
    before = 3 + 2 + 3 * (3 + 40)
    assert events[before:before + 5] == ['sync', 'classify', 'classify', 'classify', 'sync']
    # p020's pairing at sample rate 1, on RGB copies; the stream sees the same batches as the per-call run.
    assert batches[0][0] == list(range(16)) and batches[0][1] == [1] + list(range(15)) and batches[0][2] == 255
    assert batches[3][0] == [0] and batches[3][1] == [1] and batches[3 + 40] == batches[0]
    # The detector gets the BGR frames, one per call, then in the classifier's batches.
    assert detected[:40] == [[i] for i in range(40)] and detected[40:] == [list(range(16)), list(range(16, 32)), list(range(32, 40))]
    # Every record has a non-negative time; per-call classifier records carry p020's op intervals, the others none.
    assert all(r['ms'] >= 0 for r in records)
    assert all(('ops' in r) == (r['run'] in gpu.OPS_RUNS) for r in records)
    # The video's record: its frames and where its pages are (a dict of node -> pages on Linux).
    assert video['frames'] == 40 and isinstance(video['anon_pages_by_node'], dict)
    # The measurement accepts the records as the GPU job writes them (times made positive for the synthetic run).
    raw = world['tmp'] / 'gpu_raw'
    raw.mkdir()
    gpu.write_jsonl_gz(raw / 'jnc0.jsonl.gz', [dict(r, ms=r['ms'] + 1.0) for r in records])
    m = manifest()
    m['datasets'] = {'jnc0': dict(m['datasets']['jnc0'], videos={'te01.mp4': video})}
    con = duckdb.connect()
    mt.load_calls(con, raw, m, ['jnc0'])
    assert {r['run']: r['frames'] for r in mt.run_stats(con)} == {run: 40 for run in gpu.RUNS}
    assert {r['pair'] for r in mt.ratios(con)} == set(mt.PAIRS)
