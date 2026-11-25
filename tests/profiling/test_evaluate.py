"""profiling/evaluate.py: which runs get scored, failure handling, and TrackEval on real track files."""

import json
from pathlib import Path

import pytest

from profiling import evaluate as ev, main as pm
from profiling.config import parse_sweep_config
from tests.profiling.fixtures import FakeRunner, sweep_doc, videos_with_references


def swept(tmp_path, **params):
    cfg = parse_sweep_config(sweep_doc(tmp_path, **params))
    pm.sweep(cfg, runner=FakeRunner(), videos=videos_with_references(tmp_path))
    return cfg


def fake_scores(run_dir, reference_dir, reference_file):
    """evaluate_run stand-in: a HOTA result per video, tagged with the run."""
    result = json.loads(Path(run_dir, 'result.json').read_text())
    return {v: {'HOTA': {'HOTA': [0.5]}, 'Count': {'run': Path(run_dir).name}} for v in result['video_files']}


def test_pending_runs_are_the_configurations_that_ran_and_are_not_scored(tmp_path):
    cfg = swept(tmp_path)
    pending = [Path(p).name for p in ev.pending_runs(cfg)]
    # 16 configurations, 2 of them duplicates (scored through their representatives).
    assert len(pending) == 14 and 'C_60_4_080_r050_none_s100_sortcython' not in pending
    Path(ev.pending_runs(cfg)[0], 'evaluation.json').write_text('{}')
    assert len(ev.pending_runs(cfg)) == 13 and len(ev.pending_runs(cfg, force=True)) == 14


def test_failed_runs_have_no_result_and_are_not_scored(tmp_path):
    cfg = parse_sweep_config(sweep_doc(tmp_path, sample_rates=[1], mistrack_tolerances=[None]))
    pm.sweep(cfg, runner=FakeRunner(fail={'C_60_1_r050_none_s100_sortcython'}), videos=videos_with_references(tmp_path))
    assert [Path(p).name for p in ev.pending_runs(cfg)] == ['C_60_1_r050_none_s100_bytetrackcython']


def test_evaluate_sweep_writes_each_evaluation_and_reports_failures(tmp_path, monkeypatch):
    cfg = swept(tmp_path, sample_rates=[1], mistrack_tolerances=[None])
    bad = 'C_60_1_r050_none_s100_bytetrackcython'

    def scores_or_fail(run_dir, reference_dir, reference_file):
        if Path(run_dir).name == bad:
            raise RuntimeError('trackeval broke')
        return fake_scores(run_dir, reference_dir, reference_file)

    monkeypatch.setattr(ev, 'evaluate_run', scores_or_fail)
    summary = ev.evaluate_sweep(cfg, procs=1)
    assert summary == {'evaluated': 1, 'failed': [bad]}
    runs = Path(cfg.profiling_dir, 'runs')
    good = json.loads((runs / 'C_60_1_r050_none_s100_sortcython' / 'evaluation.json').read_text())
    assert sorted(good) == ['va00.mp4', 'va01.mp4']
    assert 'trackeval broke' in (runs / bad / 'evaluation-error.txt').read_text()
    assert not (runs / bad / 'evaluation.json').exists()
    # A retry that succeeds removes the old error.
    monkeypatch.setattr(ev, 'evaluate_run', fake_scores)
    assert ev.evaluate_sweep(cfg, procs=1) == {'evaluated': 1, 'failed': []}
    assert not (runs / bad / 'evaluation-error.txt').exists() and (runs / bad / 'evaluation.json').exists()


def test_missing_references_lists_each_missing_file(tmp_path):
    videos_with_references(tmp_path, names=('va00.mp4',))
    missing = ev.missing_references(str(tmp_path / 'reference'), '002_naive/tracking.jsonl', ['va00.mp4', 'va01.mp4'])
    assert missing == [str(tmp_path / 'reference' / 'va01.mp4' / '002_naive' / 'tracking.jsonl')]


def test_to_json_converts_numpy_recursively():
    import numpy as np
    assert ev.to_json({'a': np.array([1.5, 2.0]), 'b': (np.int64(3), [np.float32(0.5)])}) == {'a': [1.5, 2.0], 'b': [3, [0.5]]}


def write_tracks(path: Path, boxes_per_frame: list[list[list[float]]]) -> None:
    """p060-format tracks: one line per frame with [id, x1, y1, x2, y2] rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': i, 'tracks': boxes}) + '\n' for i, boxes in enumerate(boxes_per_frame)))


@pytest.mark.skipif(not Path('/polyis/modules/TrackEval').exists(), reason='needs the vendored TrackEval (container)')
def test_evaluate_run_scores_real_tracks_against_the_references(tmp_path):
    cfg = swept(tmp_path, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'])
    frames = [[[1, 10.0 + 2 * i, 10.0, 60.0 + 2 * i, 50.0], [2, 200.0, 100.0, 260.0, 150.0]] for i in range(20)]
    run = Path(cfg.profiling_dir, 'runs', 'C_60_1_r050_none_s100_sortcython')
    for video in ('va00.mp4', 'va01.mp4'):
        write_tracks(Path(cfg.evaluation.reference_dir, video, '002_naive', 'tracking.jsonl'), frames)
    # va00 matches its reference exactly; va01 lost its second object.
    write_tracks(run / 'tracks' / 'va00.jsonl', frames)
    write_tracks(run / 'tracks' / 'va01.jsonl', [f[:1] for f in frames])
    scores = ev.evaluate_run(str(run), cfg.evaluation.reference_dir, cfg.evaluation.reference_file)
    assert sorted(scores) == ['va00.mp4', 'va01.mp4'] and set(scores['va00.mp4']) == {'HOTA', 'Count'}
    hota = {v: sum(s['HOTA']['HOTA']) / len(s['HOTA']['HOTA']) for v, s in scores.items()}
    assert hota['va00.mp4'] == pytest.approx(1.0) and hota['va01.mp4'] < hota['va00.mp4']
    # JSON-ready (numpy converted), as evaluation.json stores it.
    json.dumps(scores)
