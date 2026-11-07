import hashlib
import importlib.util
import os
import sys
import types
import urllib.request

import pytest

# Load scripts/weights.py as a module (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    'weights', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'weights.py'))
weights = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(weights)

# Payload served by the fake host and its digest.
PAYLOAD = b'checkpoint bytes' * 1000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = 'https://example.invalid/m.pt'


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Serve URL from tmp_path/host/m.pt and point WEIGHTS at a not-yet-existing nested target."""
    # Write the hosted copy.
    host_file = tmp_path / 'host' / 'm.pt'
    host_file.parent.mkdir()
    host_file.write_bytes(PAYLOAD)
    # Target lives in a nested directory that does not exist yet.
    target = tmp_path / 'out' / 'a' / 'b' / 'best.pt'
    monkeypatch.setattr(weights, 'WEIGHTS', {
        str(target): dict(repo='org/m', private=False, url=URL, sha256=DIGEST)})
    # Redirect the URL to the local file, counting fetches.
    calls = []
    real_urlopen = urllib.request.urlopen

    def fake_urlopen(url):
        calls.append(url)
        assert url == URL
        return real_urlopen(host_file.as_uri())

    monkeypatch.setattr(weights.urllib.request, 'urlopen', fake_urlopen)
    return host_file, target, calls


def test_download_creates_dirs_and_verifies(host):
    _, target, calls = host
    weights.download()
    assert target.read_bytes() == PAYLOAD
    assert not os.path.exists(str(target) + '.part')
    assert calls == [URL]


def test_download_is_idempotent(host):
    _, target, calls = host
    weights.download()
    weights.download()
    assert target.read_bytes() == PAYLOAD
    assert calls == [URL]


def test_download_skips_correct_existing_file(host):
    _, target, calls = host
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    weights.download()
    assert calls == []


def test_download_replaces_wrong_existing_file(host):
    _, target, calls = host
    target.parent.mkdir(parents=True)
    target.write_bytes(b'stale')
    weights.download()
    assert target.read_bytes() == PAYLOAD
    assert calls == [URL]


def test_html_error_page_raises_and_keeps_existing(host):
    # A host serving an HTML page (e.g. a Drive quota notice) instead of the file must not replace the target.
    host_file, target, _ = host
    host_file.write_bytes(b'<html>quota exceeded</html>')
    target.parent.mkdir(parents=True)
    target.write_bytes(b'stale')
    with pytest.raises(ValueError, match='sha256'):
        weights.download()
    assert target.read_bytes() == b'stale'
    assert not os.path.exists(str(target) + '.part')


def test_missing_remote_file_raises_and_cleans_up(host):
    host_file, target, _ = host
    host_file.unlink()
    with pytest.raises(OSError):
        weights.download()
    assert not target.exists()
    assert not os.path.exists(str(target) + '.part')


def test_check_rejects_mismatch(tmp_path):
    p = tmp_path / 'f'
    p.write_bytes(b'x')
    weights.check(str(p), hashlib.sha256(b'x').hexdigest())
    with pytest.raises(ValueError):
        weights.check(str(p), DIGEST)


@pytest.fixture
def fake_hf(monkeypatch):
    """Install a fake huggingface_hub whose HfApi records calls."""
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, private, exist_ok):
            calls.append(('create', repo_id, private, exist_ok))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
            calls.append(('upload', repo_id, path_in_repo, path_or_fileobj))

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(HfApi=FakeApi))
    return calls


def test_upload_creates_repo_with_visibility_and_uploads(tmp_path, monkeypatch, fake_hf):
    pub = tmp_path / 'pub' / 'best.pt'
    priv = tmp_path / 'priv' / 'retinanet_b3d.pth'
    for p in (pub, priv):
        p.parent.mkdir()
        p.write_bytes(PAYLOAD)
    monkeypatch.setattr(weights, 'WEIGHTS', {
        str(pub): dict(repo='org/pub', private=False, url=URL, sha256=DIGEST),
        str(priv): dict(repo='org/priv', private=True, url=URL, sha256=DIGEST)})
    weights.upload()
    assert fake_hf == [
        ('create', 'org/pub', False, True), ('upload', 'org/pub', 'best.pt', str(pub)),
        ('create', 'org/priv', True, True), ('upload', 'org/priv', 'retinanet_b3d.pth', str(priv))]


def test_upload_refuses_wrong_checkpoint_before_touching_hub(tmp_path, monkeypatch, fake_hf):
    p = tmp_path / 'best.pt'
    p.write_bytes(b'not the experiment checkpoint')
    monkeypatch.setattr(weights, 'WEIGHTS', {str(p): dict(repo='org/m', private=False, url=URL, sha256=DIGEST)})
    with pytest.raises(ValueError, match='sha256'):
        weights.upload()
    assert fake_hf == []


def test_weights_table_is_consistent():
    repos = [w['repo'] for w in weights.WEIGHTS.values()]
    # One repo per checkpoint, all under tetris-db.
    assert len(set(repos)) == len(repos)
    assert all(r.startswith('tetris-db/') for r in repos)
    for path, w in weights.WEIGHTS.items():
        assert os.path.isabs(path)
        assert len(w['sha256']) == 64 and int(w['sha256'], 16) >= 0
        if w['private']:
            # Private repos need auth, so they must download from somewhere else.
            assert not w['url'].startswith('https://huggingface.co/')
        else:
            # Public downloads come from the same repo and file name that upload writes.
            assert w['url'] == f"https://huggingface.co/{w['repo']}/resolve/main/{os.path.basename(path)}"


def test_b3d_is_private_and_yolo_repos_are_named():
    by_repo = {w['repo']: w for w in weights.WEIGHTS.values()}
    assert set(by_repo) == {'tetris-db/caldot1-yolov5', 'tetris-db/caldot2-yolov5',
                            'tetris-db/amsterdam-yolov5', 'tetris-db/b3d'}
    assert by_repo['tetris-db/b3d']['private'] is True
    assert not any(w['private'] for r, w in by_repo.items() if r != 'tetris-db/b3d')
