"""experiments.fanout_profiling.main: startup checks, admission, records, never overwriting (PLAN step 5).  Fake pipeline."""

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from execution.main import StartupError
from experiments.fanout_profiling import main as m, stages as st
from tests.profiling import fake_stages as pfs
from tests.profiling.test_stages import make_cfg


def write_video(path: Path, n: int, size=(80, 60)) -> str:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for t in range(n):
        w.write(np.full((size[1], size[0], 3), t * 5 % 256, np.uint8))
    w.release()
    return str(path)


@pytest.fixture
def setup(tmp_path):
    cfg = make_cfg(tmp_path)
    vdir = tmp_path / 'videos'
    vdir.mkdir()
    videos = [write_video(vdir / f'va{i:02d}.mp4', 6 + i) for i in range(3)]
    for v in videos:
        ref = Path(cfg.evaluation.reference_dir) / Path(v).name / '002_naive'
        ref.mkdir(parents=True)
        (ref / 'tracking.jsonl').write_text('')
    return cfg, videos


def test_execute_writes_one_record_per_configuration_and_video(setup):
    cfg, videos = setup
    summary = m.execute(cfg, 'c.yaml', build=pfs.fake_sweep_build, videos=videos)
    out = Path(cfg.output_root) / cfg.dataset / 'profiling'
    records = [json.loads(l) for l in (out / 'records.jsonl').read_text().splitlines()]
    expected = st.expected_results(cfg)
    assert summary['records'] == len(records) == expected * 3 and summary['configurations'] == expected
    assert sorted({r['video'] for r in records}) == ['va00.mp4', 'va01.mp4', 'va02.mp4']
    assert records[0]['metrics'] == {'HOTA': {'HOTA': [0.5, 0.7]}} and records[0]['timings'][0]['stage'] == 'detect'
    run = json.loads((out / 'run.json').read_text())
    assert run['videos'] == 3 and run['reduce_search_space'] is cfg.params.reduce_search_space


def test_wall_clock_includes_the_startup_checks(setup, monkeypatch):
    # The wall clock starts before probing and the reference/shm checks, not when the pipeline starts.
    import time
    cfg, videos = setup
    real_probe = m.probe
    monkeypatch.setattr(m, 'probe', lambda v: (time.sleep(0.2), real_probe(v))[1])
    summary = m.execute(cfg, 'c.yaml', build=pfs.fake_sweep_build, videos=videos)
    assert summary['wall_s'] >= 0.6


def test_an_existing_output_is_never_overwritten(setup):
    cfg, videos = setup
    (Path(cfg.output_root) / cfg.dataset).mkdir(parents=True)
    with pytest.raises(StartupError, match='already exists'):
        m.execute(cfg, build=pfs.fake_sweep_build, videos=videos)


def test_missing_reference_tracks_fail_before_running(setup):
    cfg, videos = setup
    (Path(cfg.evaluation.reference_dir) / 'va01.mp4' / '002_naive' / 'tracking.jsonl').unlink()
    with pytest.raises(StartupError, match='reference tracks missing for 1 videos'):
        m.execute(cfg, build=pfs.fake_sweep_build, videos=videos)
    assert not (Path(cfg.output_root) / cfg.dataset).exists()


def test_shm_estimate_refuses_an_oversized_sweep(setup, monkeypatch):
    cfg, videos = setup
    infos = [m.probe(v) for v in videos]
    m.check_shm(cfg, infos)
    monkeypatch.setattr(m, 'worst_case_shm_bytes', lambda cfg, infos: 10**18)
    with pytest.raises(StartupError, match='exceeds'):
        m.check_shm(cfg, infos)


def test_shm_estimate_counts_admitted_frames_and_in_flight_canvases(setup):
    cfg, videos = setup
    infos = [m.probe(v) for v in videos]
    # 3 videos up to 8 frames of 80x60; canvases: densest rate 1, 8 collages of an 8x6-tile canvas.
    frames_max = 8 * 60 * 80 * 3
    holders = 1 * 2 + cfg.system.queue_maxsize + 1
    assert m.worst_case_shm_bytes(cfg, infos) == cfg.system.max_videos_in_flight * frames_max + holders * 8 * 60 * 80 * 3


def test_admission_never_exceeds_the_slots():
    slots, stopped = threading.Semaphore(2), threading.Event()
    gen = m.admitted(['a', 'b', 'c'], slots, stopped)
    assert next(gen) == 'a' and next(gen) == 'b'
    got = []
    t = threading.Thread(target=lambda: got.append(next(gen)))
    t.start()
    t.join(timeout=1.0)
    assert got == []            # blocked: both slots taken
    slots.release()
    t.join(timeout=2.0)
    assert got == ['c']
    stopped.set()


def test_json_conversion_handles_numpy():
    assert m.to_json({'a': np.array([1.5]), 'b': (np.int64(3), np.float32(0.5))}) == {'a': [1.5], 'b': [3, 0.5]}
