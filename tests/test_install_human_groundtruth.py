import hashlib
import json
from pathlib import Path

import pytest

from polyis import groundtruth
from polyis.io import cache, store
from preprocess import p004_install_human_groundtruth as p004


FRAMES = 3


@pytest.fixture
def env(tmp_path, monkeypatch):
    # Isolated cache, datasets and source roots.
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(store, 'DATASETS_DIR', str(tmp_path / 'datasets'))
    monkeypatch.setattr(p004, 'get_video_frame_count', lambda dataset, video: FRAMES)
    # One dataset with two test videos, each with automatic ground truth.
    for video in ('te01.mp4', 'te04.mp4'):
        (tmp_path / 'datasets' / 'jnc0' / 'test').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'datasets' / 'jnc0' / 'test' / video).write_bytes(b'')
        auto = groundtruth.groundtruth_dir('jnc0', video)
        auto.mkdir(parents=True)
        (auto / 'tracking.jsonl').write_text('auto\n')
        write_source(tmp_path / 'source', 'jnc0', video, [[[1, 0, 0, 10, 10]], [], [[1, 2, 2, 12, 12], [2, 5, 5, 9, 9]]])
    return tmp_path


def write_source(root: Path, dataset: str, video: str, frames, start: int = 0) -> Path:
    path = root / dataset / video / 'tracking.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_idx': start + i, 'tracks': t}) + '\n' for i, t in enumerate(frames)))
    return path


def install(env, **kwargs):
    plan = p004.plan_dataset(env / 'source', 'jnc0')
    return [p004.install_video('jnc0', video, path, frames, kwargs.get('commit', 'abc123')) for video, path, frames in plan]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_first_install_backs_up_and_writes_labels(env):
    assert install(env) == ['installed', 'installed']
    target = groundtruth.groundtruth_dir('jnc0', 'te01.mp4')
    # The automatic labels move to the backup untouched.
    assert (groundtruth.backup_dir('jnc0', 'te01.mp4') / 'tracking.jsonl').read_text() == 'auto\n'
    # Tracking is copied frame for frame; detection is derived with score 1.0 on the same frame index.
    assert [r['tracks'] for r in read_jsonl(target / 'tracking.jsonl')] == [[[1, 0, 0, 10, 10]], [], [[1, 2, 2, 12, 12], [2, 5, 5, 9, 9]]]
    assert read_jsonl(target / 'detection.jsonl') == [
        {'frame_idx': 0, 'detections': [[0, 0, 10, 10, 1.0]]},
        {'frame_idx': 1, 'detections': []},
        {'frame_idx': 2, 'detections': [[2, 2, 12, 12, 1.0], [5, 5, 9, 9, 1.0]]},
    ]
    # Provenance names the commit and hashes every file.
    source = json.loads((target / groundtruth.MARKER).read_text())
    assert source['source_commit'] == 'abc123'
    for name in p004.LABEL_FILES:
        assert source['files'][name] == hashlib.sha256((target / name).read_bytes()).hexdigest()
    assert source['source_sha256'] == hashlib.sha256((env / 'source/jnc0/te01.mp4/tracking.jsonl').read_bytes()).hexdigest()
    # No staging directory is left behind.
    assert not target.with_name(p004.STAGING_NAME).exists()


def test_reinstall_keeps_the_original_backup(env):
    install(env)
    # New labels replace the installed ones; the backup still holds the automatic labels.
    write_source(env / 'source', 'jnc0', 'te01.mp4', [[], [], [[7, 1, 1, 2, 2]]])
    assert install(env, commit='def456') == ['reinstalled', 'reinstalled']
    target = groundtruth.groundtruth_dir('jnc0', 'te01.mp4')
    assert read_jsonl(target / 'tracking.jsonl')[2]['tracks'] == [[7, 1, 1, 2, 2]]
    assert json.loads((target / groundtruth.MARKER).read_text())['source_commit'] == 'def456'
    assert (groundtruth.backup_dir('jnc0', 'te01.mp4') / 'tracking.jsonl').read_text() == 'auto\n'


def test_interrupted_install_resumes(env):
    # Simulate a crash after the automatic labels were moved but before the new directory landed.
    groundtruth.groundtruth_dir('jnc0', 'te01.mp4').rename(groundtruth.backup_dir('jnc0', 'te01.mp4'))
    assert install(env) == ['resumed', 'installed']
    assert groundtruth.is_installed('jnc0', 'te01.mp4')


def test_refuses_ambiguous_state(env):
    # Uninstalled labels next to an existing backup: unclear which is automatic.
    groundtruth.backup_dir('jnc0', 'te01.mp4').mkdir()
    with pytest.raises(RuntimeError, match='resolve by hand'):
        install(env)
    assert (groundtruth.groundtruth_dir('jnc0', 'te01.mp4') / 'tracking.jsonl').read_text() == 'auto\n'


def test_refuses_installed_without_backup(env):
    install(env)
    import shutil
    shutil.rmtree(groundtruth.backup_dir('jnc0', 'te01.mp4'))
    with pytest.raises(RuntimeError, match='missing'):
        install(env)


def test_refuses_missing_groundtruth(env):
    import shutil
    shutil.rmtree(groundtruth.groundtruth_dir('jnc0', 'te01.mp4'))
    with pytest.raises(RuntimeError, match='does not exist'):
        install(env)


@pytest.mark.parametrize('videos', [['te01.mp4'], ['te01.mp4', 'te04.mp4', 'te08.mp4']])
def test_refuses_mismatched_clip_set(env, videos):
    import shutil
    shutil.rmtree(env / 'source' / 'jnc0')
    for video in videos:
        write_source(env / 'source', 'jnc0', video, [[], [], []])
    with pytest.raises(ValueError, match='test split has'):
        p004.plan_dataset(env / 'source', 'jnc0')


@pytest.mark.parametrize('frames, start, message', [
    ([[], []], 0, 'clip has 3'),                                   # too few frames
    ([[], [], [], []], 0, 'clip has 3'),                           # too many frames
    ([[], [], []], 1, 'expected 0'),                               # does not start at zero
    ([[[1, 0, 0, 1, 1], [1, 2, 2, 3, 3]], [], []], 0, 'duplicate'),  # duplicate id in a frame
    ([[[-1, 0, 0, 1, 1]], [], []], 0, 'bad or duplicate'),        # negative id
    ([[[1.5, 0, 0, 1, 1]], [], []], 0, 'bad or duplicate'),       # non-integer id
    ([[[1, 0, 0, 1]], [], []], 0, 'bad or duplicate'),            # wrong arity
    ([[[1, 5, 0, 1, 1]], [], []], 0, 'invalid box'),              # inverted x
    ([[[1, 0, 5, 1, 1]], [], []], 0, 'invalid box'),              # inverted y
    ([[[1, 0, 0, 0, 1]], [], []], 0, 'invalid box'),              # zero width
])
def test_rejects_invalid_source(env, frames, start, message):
    write_source(env / 'source', 'jnc0', 'te04.mp4', frames, start)
    with pytest.raises(ValueError, match=message):
        p004.plan_dataset(env / 'source', 'jnc0')


def test_rejects_non_finite_box(env):
    path = env / 'source/jnc0/te04.mp4/tracking.jsonl'
    path.write_text('{"frame_idx": 0, "tracks": [[1, 0, 0, NaN, 1]]}\n{"frame_idx": 1, "tracks": []}\n{"frame_idx": 2, "tracks": []}\n')
    with pytest.raises(ValueError, match='invalid box'):
        p004.plan_dataset(env / 'source', 'jnc0')


def test_validation_failure_writes_nothing(env, monkeypatch):
    # main validates every dataset before installing any video.
    write_source(env / 'source', 'jnc0', 'te04.mp4', [[], []])
    monkeypatch.setattr('sys.argv', ['p004', '--source', str(env / 'source'), '--source-commit', 'abc', '--datasets', 'jnc0'])
    with pytest.raises(ValueError):
        p004.main()
    assert not groundtruth.is_installed('jnc0', 'te01.mp4')
    assert not groundtruth.backup_dir('jnc0', 'te01.mp4').exists()


def test_p003_guard_refuses_installed_videos(env):
    install(env)
    with pytest.raises(RuntimeError, match='te01.mp4, te04.mp4'):
        groundtruth.assert_not_installed('jnc0', ['te04.mp4', 'te01.mp4'])


def test_p003_guard_allows_automatic_ground_truth(env):
    groundtruth.assert_not_installed('jnc0', ['te01.mp4', 'te04.mp4'])


@pytest.mark.parametrize('script', ['p003_preprocess_groundtruth_detection.py', 'p003_preprocess_groundtruth_tracking.py'])
def test_p003_scripts_call_the_guard_before_scheduling(script):
    # Both p003 entry points must check every video before queueing work.
    source = (Path(__file__).parent.parent / 'preprocess' / script).read_text()
    main = source[source.index('def main'):]
    assert 'assert_not_installed(dataset,' in main
    assert main.index('assert_not_installed(dataset,') < main.index('ProgressBar(')


def test_p003_skips_human_ground_truth_datasets(capsys):
    assert groundtruth.without_human_groundtruth(['caldot1', 'jnc0', 'ams', 'jnc7', 'jnc2', 'jnc6']) == ['caldot1', 'ams']
    assert 'Skipping jnc0' in capsys.readouterr().out


@pytest.mark.parametrize('script', ['p003_preprocess_groundtruth_detection.py', 'p003_preprocess_groundtruth_tracking.py'])
def test_p003_scripts_drop_human_ground_truth_datasets(script):
    # Both p003 entry points filter the configured datasets before any work.
    source = (Path(__file__).parent.parent / 'preprocess' / script).read_text()
    assert 'datasets_to_process = without_human_groundtruth(dedupe_datasets_by_root(EXEC_DATASETS))' in source
