"""main.py tests: startup checks, dummy-video timing, outputs (PLAN.md step 5).  The pipeline is a fake."""

import glob
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from execution import main as m
from execution.config import Config, StageSystem, System
from tests.execution import fake_stages as fs
from tests.execution.test_stages import make_cfg


def write_video(path: Path, n_frames: int, size=(32, 24)) -> str:
    """A small MJPG video whose pixels encode the frame index."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for t in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), t * 5 % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def cfg_with(tmp_path: Path, queue_maxsize=2, mistrack_tolerance=None) -> Config:
    base = make_cfg(tmp_path, mistrack_tolerance=mistrack_tolerance)
    stages = {'decode': StageSystem(2, None), 'classify': StageSystem(1, 0), 'pack': StageSystem(2, None),
              'detect': StageSystem(1, 0), 'track': StageSystem(3, None)}
    if mistrack_tolerance is not None:
        stages['prune'] = StageSystem(1, None)
    return Config(base.params, base.artifacts, System(queue_maxsize, stages))


@pytest.fixture
def videos(tmp_path) -> Path:
    d = tmp_path / 'videos'
    d.mkdir()
    for name, n in [('a.avi', 5), ('b.avi', 8), ('c.avi', 3)]:
        write_video(d / name, n)
    (d / 'README.txt').write_text('not a video')
    return d


def test_execute_writes_one_file_per_video_and_drops_the_dummy(tmp_path, videos):
    out = tmp_path / 'out'
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), 'execution-*')))
    summary = m.execute(cfg_with(tmp_path), str(videos), str(out), build=fs.fake_build)
    assert sorted(os.listdir(out)) == ['a.jsonl', 'b.jsonl', 'c.jsonl']
    # Each file holds that video's own result (the fake tags it with the video's frame count).
    for stem, n in [('a', 5), ('b', 8), ('c', 3)]:
        rows = [json.loads(line) for line in (out / f'{stem}.jsonl').read_text().splitlines()]
        assert rows == [{'frame_idx': 0, 'tracks': [[1.0, 0.0, 0.0, 10.0, float(n)]]}]
    assert summary['videos'] == 3 and summary['frames'] == 16 and summary['runtime_s'] >= 0
    # The dummy copy's temp dir is gone.
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), 'execution-*'))) == before


def test_single_video_still_times_one_interval(tmp_path):
    d = tmp_path / 'one'
    d.mkdir()
    write_video(d / 'only.avi', 4)
    summary = m.execute(cfg_with(tmp_path), str(d), str(tmp_path / 'out'), build=fs.fake_build)
    assert summary['videos'] == 1 and os.listdir(tmp_path / 'out') == ['only.jsonl']


def test_list_videos_ignores_other_files_and_sorts(videos):
    assert [Path(v).name for v in m.list_videos(str(videos))] == ['a.avi', 'b.avi', 'c.avi']


def test_empty_or_missing_video_dir_is_rejected(tmp_path):
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'empty' / 'notes.txt').write_text('x')
    with pytest.raises(m.StartupError, match='no .mp4'):
        m.list_videos(str(tmp_path / 'empty'))
    with pytest.raises(m.StartupError, match='not a directory'):
        m.list_videos(str(tmp_path / 'missing'))


def test_videos_sharing_a_stem_are_rejected(videos):
    write_video(videos / 'a.mp4', 3)
    with pytest.raises(m.StartupError, match=r"share a stem.*\['a'\]"):
        m.list_videos(str(videos))


def test_probe_rejects_unreadable_and_one_frame_videos(tmp_path):
    bad = tmp_path / 'broken.mp4'
    bad.write_text('not a video')
    with pytest.raises(m.StartupError, match='cannot open'):
        m.probe(str(bad))
    with pytest.raises(m.StartupError, match='at least 2'):
        m.probe(write_video(tmp_path / 'one.avi', 1))
    info = m.probe(write_video(tmp_path / 'ok.avi', 6))
    assert (info.width, info.height, info.frame_count) == (32, 24, 6)


def test_execute_fails_before_running_on_a_bad_video(tmp_path, videos):
    write_video(videos / 'z.avi', 1)
    with pytest.raises(m.StartupError, match='z.avi'):
        m.execute(cfg_with(tmp_path), str(videos), str(tmp_path / 'out'), build=fs.fake_build)
    assert not (tmp_path / 'out').exists()


def test_videos_holding_blocks(tmp_path):
    # Frames: decode 2 + classify 1 + pack 2 workers, 2 queues between them.  Canvases: pack 2 + detect 1, 1 queue.
    cfg = cfg_with(tmp_path, queue_maxsize=2)
    assert (m.videos_holding_frames(cfg), m.videos_holding_canvases(cfg)) == (5 + 2 * 2, 3 + 2)
    # Prune adds a worker and a queue to the frames region only.
    cfg = cfg_with(tmp_path, queue_maxsize=2, mistrack_tolerance=0.4)
    assert (m.videos_holding_frames(cfg), m.videos_holding_canvases(cfg)) == (6 + 2 * 3, 3 + 2)


def test_block_size_estimates():
    info = m.VideoInfo('v', 130, 70, 10)
    # s=4 over 10 frames: sampled [0,4,8,9], needed [0,1,3,4,7,8,9].
    assert m.frames_block_bytes(info, 4) == 7 * 70 * 130 * 3
    # Canvases: one per sampled frame, sized as pack sizes them (7x13 grid of 10 px tiles at scale 1).
    assert m.canvases_block_bytes(info, 4, 10, 1.0) == 4 * 70 * 130 * 3
    # Scale 1.5: grid round(7*1.5)=10 by round(13*1.5)=20 tiles.
    assert m.canvases_block_bytes(info, 4, 10, 1.5) == 4 * 100 * 200 * 3


def test_worst_case_uses_the_largest_video_in_each_region(tmp_path):
    cfg = cfg_with(tmp_path, queue_maxsize=2)
    small, large = m.VideoInfo('s', 80, 60, 10), m.VideoInfo('l', 160, 120, 40)
    expected = (m.videos_holding_frames(cfg) * m.frames_block_bytes(large, 2)
                + m.videos_holding_canvases(cfg) * m.canvases_block_bytes(large, 2, 10, 1.0))
    assert m.worst_case_shm_bytes(cfg, [small, large]) == expected


def test_shm_budget_refuses_an_oversized_run(tmp_path, videos):
    infos = [m.probe(v) for v in m.list_videos(str(videos))]
    m.check_shm_budget(cfg_with(tmp_path), infos)
    with pytest.raises(m.StartupError, match='exceeds'):
        m.check_shm_budget(cfg_with(tmp_path, queue_maxsize=10**9), infos)


def test_cli_reports_config_errors_with_exit_code_2(tmp_path, monkeypatch, capsys, videos):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('params: {}\n')
    monkeypatch.setattr('sys.argv', ['main', '--config', str(bad), '--videos', str(videos), '--out', str(tmp_path / 'o')])
    assert m.main() == 2
    assert 'error: config: missing keys' in capsys.readouterr().err
