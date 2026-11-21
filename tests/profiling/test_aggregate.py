"""Aggregator: dataset HOTA as TrackEval combines it, times, Pareto union, YAML, page (profiling/PLAN.md 15A)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from execution2.config import parse_config
from profiling import aggregate as ag, stages as st
from profiling.config import Artifacts, Evaluation, Params, StageSystem, SweepConfig, System
from execution2.config import Detector
from polyis.utilities import save_tracking_results


def real_cfg(tmp_path: Path, tolerances=(None, 0.4)) -> SweepConfig:
    """A sweep config whose artifact paths exist, so the generated execution2 YAML validates."""
    paths = {}
    for n in ['w.pth', 'b.jsonl', 'm.pt']:
        (tmp_path / n).write_text('x')
        paths[n] = str(tmp_path / n)
    # Real arrays: the aggregator reads them to expand tolerance groups.  Gaps: 0.4 -> 12, 0.6/0.8 -> 8 frames.
    np.save(tmp_path / 'n.npy', np.ones((2, 2), np.uint8))
    np.save(tmp_path / 'sort.npy', np.broadcast_to(np.array([16, 12, 10, 8, 8, 8, 4, 2, 1]), (2, 2, 9)).astype(np.int64))
    paths['n.npy'], paths['sort.npy'] = str(tmp_path / 'n.npy'), str(tmp_path / 'sort.npy')
    (tmp_path / 'ref').mkdir(exist_ok=True)
    sys_ = {n: StageSystem(1, (0,) if n in ('classify', 'detect') else (), 1)
            for n in ('decode', 'classify', 'prune', 'pack', 'detect', 'track', 'evaluate')}
    return SweepConfig('ds', 'valid', str(tmp_path / 'out'),
                       Params('Fake', 60, 'jpeg', (1, 4), (0.5,), tolerances, ('none',), (1.0,), ('sortcython',)),
                       Artifacts(paths['w.pth'], paths['b.jsonl'], paths['n.npy'], {'sortcython': paths['sort.npy']},
                                 Detector('ultralytics', {'model_path': paths['m.pt']})),
                       Evaluation(str(tmp_path / 'ref'), '002_naive/tracking.jsonl'), System(2, 8, 2, sys_))


def tracks(shift: float, n: int = 20) -> dict:
    return {f: [[1, 10 + f + shift, 10, 40 + f + shift, 30], [2, 100, 50 + shift, 130, 80 + shift]] for f in range(n)}


def per_video_hota(cfg: SweepConfig, tmp_path: Path, shifts: dict[str, float]) -> dict[str, dict]:
    """Each video's raw HOTA result from the evaluate stage (reference = unshifted tracks)."""
    from profiling.messages import Branch, Tracked
    state = st.evaluate_setup(cfg, type('W', (), {'index': 0})())
    out = {}
    for video, shift in shifts.items():
        ref = Path(cfg.evaluation.reference_dir) / video / '002_naive'
        ref.mkdir(parents=True, exist_ok=True)
        save_tracking_results(tracks(0.0), ref / 'tracking.jsonl')
        result = st.evaluate(state, Tracked(f'/v/{video}', Branch(), (), tracks(shift)))
        out[video] = ag_to_lists(result.metrics['HOTA'])
    return out


def ag_to_lists(d: dict) -> dict:
    return {k: np.asarray(v).tolist() for k, v in d.items()}


def test_combined_hota_equals_trackevals_own_combination(tmp_path):
    from evaluation.p110_accuracy_compute import build_metric_objects, run_trackeval
    cfg = real_cfg(tmp_path)
    shifts = {'a.mp4': 3.0, 'b.mp4': 9.0}
    per_video = per_video_hota(cfg, tmp_path, shifts)
    combined = ag.combine_hota(list(per_video.values()))
    # TrackEval scoring both videos at once (p110's way) gives its COMBINED_SEQ row.
    track_dir = tmp_path / 'both'
    for video, shift in shifts.items():
        (track_dir / video).mkdir(parents=True)
        save_tracking_results(tracks(shift), track_dir / video / 'tracking.jsonl')
    results = run_trackeval({'output_fol': str(tmp_path / 'te'), 'output_sub_fol': 'x', 'input_gt': '002_naive/tracking.jsonl',
                             'input_track': 'tracking.jsonl', 'skip': 1, 'tracker': 'x', 'seq_list': sorted(shifts),
                             'input_dir': str(track_dir), 'input_gt_dir': cfg.evaluation.reference_dir,
                             'input_track_dir': str(track_dir)}, build_metric_objects(['HOTA']), str(tmp_path / 'te'), 2)
    expected = results['COMBINED_SEQ']['vehicle']['HOTA']
    assert np.allclose(combined['HOTA'], expected['HOTA'], rtol=0, atol=1e-12)
    assert 0 < ag.hota_score(combined) < 1


def write_sweep(cfg: SweepConfig, tmp_path: Path, configs: list[tuple[dict, dict[str, float], float]]) -> None:
    """records.jsonl and run.json for configs given as (branch, per-video shifts, charged seconds per video)."""
    out = Path(cfg.output_root) / cfg.dataset / 'profiling'
    out.mkdir(parents=True)
    shifts_all = {v for _, s, _ in configs for v in s}
    with open(out / 'records.jsonl', 'w') as f:
        for branch, shifts, charged in configs:
            hota = per_video_hota(cfg, tmp_path, shifts)
            for video in shifts:
                timings = [{'stage': 'decode', 'start': 0.0, 'end': 2.0, 'charged': 0.0},
                           {'stage': 'detect', 'start': 2.0, 'end': 2.0 + charged, 'charged': charged}]
                f.write(json.dumps({'video': video, 'branch': branch, 'timings': timings,
                                    'metrics': {'HOTA': hota[video]}}) + '\n')
    (out / 'run.json').write_text(json.dumps({'videos': len(shifts_all), 'video_frames': {v: 20 for v in shifts_all}}))


def branch(s, tol, tracker='sortcython'):
    return {'classifier': 'Fake', 'tile_size': 60, 'sample_rate': s, 'relevance_threshold': 0.5,
            'mistrack_tolerance': tol, 'tracker': tracker, 'tile_padding': 'none', 'canvas_scale': 1.0}


def test_aggregate_times_throughput_pareto_and_yaml(tmp_path):
    cfg = real_cfg(tmp_path)
    videos = ('a.mp4', 'b.mp4')
    # accurate-but-slow, fast-but-less-accurate, and one dominated by the accurate one.
    write_sweep(cfg, tmp_path, [
        (branch(1, None), {v: 1.0 for v in videos}, 4.0),
        (branch(4, 0.4), {v: 6.0 for v in videos}, 1.0),
        (branch(4, None), {v: 6.0 for v in videos}, 5.0),
    ])
    configs, pareto = ag.aggregate(cfg)
    by = {(c['sample_rate'], c['tracking_accuracy_threshold']): c for c in configs}
    assert by[(1, None)]['time'] == pytest.approx(8.0) and by[(1, None)]['decode_s'] == pytest.approx(4.0)
    assert by[(1, None)]['throughput_fps'] == pytest.approx(40 / 8.0)
    assert by[(1, None)]['throughput_fps_with_decode'] == pytest.approx(40 / 12.0)
    assert by[(1, None)]['HOTA_HOTA'] > by[(4, 0.4)]['HOTA_HOTA']
    on_full = {(p['sample_rate'], p['tracking_accuracy_threshold']) for p in pareto if 'full' in p['pareto_conditions']}
    assert on_full == {(1, None), (4, 0.4)}
    for c in configs:
        e2 = yaml.safe_load(c['execution2_yaml'])
        parsed = parse_config(e2)
        assert parsed.params.sample_rate == c['sample_rate'] and parsed.use_prune == (c['tracking_accuracy_threshold'] is not None)
    out = Path(cfg.output_root) / cfg.dataset / 'profiling'
    assert json.loads((out / 'pareto.json').read_text()) == pareto
    html = (out / 'pareto.html').read_text()
    spec = json.loads(re.search(r'const SPEC = (.*);\n', html).group(1))
    embedded_yaml = json.loads(re.search(r'const YAML = (.*);\n', html).group(1))
    assert len(spec['data']['values']) == 3 and set(embedded_yaml) == {str(c['id']) for c in configs}
    assert embedded_yaml[str(by[(4, 0.4)]['id'])] == by[(4, 0.4)]['execution2_yaml']


def test_aggregate_refuses_incomplete_configurations(tmp_path):
    cfg = real_cfg(tmp_path)
    write_sweep(cfg, tmp_path, [(branch(1, None), {'a.mp4': 1.0, 'b.mp4': 1.0}, 1.0),
                                (branch(4, None), {'a.mp4': 1.0}, 1.0)])
    with pytest.raises(RuntimeError, match='1 videos recorded, expected 2'):
        ag.aggregate(cfg)


def test_aggregate_expands_a_tolerance_group_to_all_its_tolerances(tmp_path):
    # 0.6 and 0.8 have identical gaps, so only 0.6 ran; the aggregator gives 0.8 the same row.
    cfg = real_cfg(tmp_path, tolerances=(0.4, 0.6, 0.8))
    write_sweep(cfg, tmp_path, [(branch(1, 0.4), {'a.mp4': 2.0}, 1.0), (branch(1, 0.6), {'a.mp4': 5.0}, 1.0)])
    configs, _ = ag.aggregate(cfg)
    rows = {c['tracking_accuracy_threshold']: c for c in configs}
    assert set(rows) == {0.4, 0.6, 0.8} and rows[0.8]['ran_as'] == 0.6 and rows[0.4]['ran_as'] == 0.4
    assert rows[0.8]['HOTA_HOTA'] == rows[0.6]['HOTA_HOTA'] and rows[0.8]['time'] == rows[0.6]['time']
    assert yaml.safe_load(rows[0.8]['execution2_yaml'])['params']['mistrack_tolerance'] == 0.8
    assert sorted(c['id'] for c in configs) == [0, 1, 2]


def test_combine_hota_works_in_a_fresh_process():
    # Regression: combine_hota must not rely on another module having put TrackEval on sys.path first.
    import subprocess
    import sys
    code = ("from profiling.aggregate import combine_hota\n"
            "r = {'HOTA_TP': [5] * 19, 'HOTA_FN': [1] * 19, 'HOTA_FP': [1] * 19, 'AssRe': [0.9] * 19,"
            " 'AssPr': [0.9] * 19, 'AssA': [0.8] * 19, 'LocA': [0.9] * 19}\n"
            "print(round(float(combine_hota([r, r])['DetA'][0]), 4))\n")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd='/polyis')
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == str(round(10 / 14, 4))
