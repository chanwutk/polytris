"""Support for re-scoring B3D on human ground truth: dataset subsetting, scoped p110 cleanup, snapshots."""
import pytest

from evaluation import p110_accuracy_compute as p110
from measurements.snapshot_outputs import snapshot
from polyis.io import cache
from polyis.utilities import get_config


def test_polyis_datasets_restricts_the_configured_datasets(monkeypatch):
    monkeypatch.setenv('POLYIS_DATASETS', 'jnc6, jnc0')
    assert get_config()['EXEC']['DATASETS'] == ['jnc6', 'jnc0']


def test_without_polyis_datasets_the_config_is_unchanged(monkeypatch):
    monkeypatch.delenv('POLYIS_DATASETS', raising=False)
    assert 'caldot1-y05' in get_config()['EXEC']['DATASETS']


@pytest.mark.parametrize('value', ['', ' , ', 'jnc0,jnc9', 'JNC0'])
def test_polyis_datasets_rejects_empty_or_unknown_names(monkeypatch, value):
    monkeypatch.setenv('POLYIS_DATASETS', value)
    with pytest.raises(ValueError, match='POLYIS_DATASETS'):
        get_config()


def test_p110_clears_only_the_requested_split(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(p110, 'DATASETS', ['jnc0'])
    monkeypatch.setattr(p110, 'pareto_params_exist', lambda dataset: True)
    monkeypatch.setattr(p110, 'build_accuracy_task_rows', lambda dataset, videoset: [])
    acc = cache.eval('jnc0', 'acc')
    for split in ('test', 'valid'):
        (acc / 'raw' / split / 'variant').mkdir(parents=True)
        (acc / 'raw' / split / 'variant' / 'DATASET.json').write_text('{}')
    (acc / 'accuracy.csv').write_text('kept')
    # With no tasks p110 stops after cleanup; only raw/test is cleared.
    with pytest.raises(AssertionError, match='No tracking accuracy tasks'):
        p110.main(type('Args', (), {'metrics': 'HOTA', 'no_parallel': True, 'test': True, 'valid': False})())
    assert (acc / 'raw' / 'test').is_dir() and not any((acc / 'raw' / 'test').iterdir())
    assert (acc / 'raw' / 'valid' / 'variant' / 'DATASET.json').exists()
    assert (acc / 'accuracy.csv').read_text() == 'kept'


def make_outputs(tmp_path):
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'dir' / 'a.csv').write_text('a')
    (tmp_path / 'b.json').write_text('b')
    return [tmp_path / 'dir', tmp_path / 'b.json']


def test_snapshot_copies_files_and_directories(tmp_path):
    paths = make_outputs(tmp_path)
    assert snapshot(paths, '.auto-gt') == [tmp_path / 'dir.auto-gt', tmp_path / 'b.json.auto-gt']
    assert (tmp_path / 'dir.auto-gt' / 'a.csv').read_text() == 'a'
    assert (tmp_path / 'b.json.auto-gt').read_text() == 'b'
    # The originals are untouched.
    assert (tmp_path / 'dir' / 'a.csv').read_text() == 'a'


def test_snapshot_never_replaces_an_existing_snapshot(tmp_path):
    paths = make_outputs(tmp_path)
    snapshot(paths, '.auto-gt')
    (tmp_path / 'b.json').write_text('changed')
    with pytest.raises(FileExistsError, match='already exists'):
        snapshot(paths, '.auto-gt')
    assert (tmp_path / 'b.json.auto-gt').read_text() == 'b'


def test_snapshot_checks_every_path_before_copying(tmp_path):
    paths = make_outputs(tmp_path)
    # One missing path: nothing is copied, not even the valid ones.
    with pytest.raises(FileExistsError, match='missing'):
        snapshot([*paths, tmp_path / 'absent.csv'], '.auto-gt')
    assert not (tmp_path / 'dir.auto-gt').exists() and not (tmp_path / 'b.json.auto-gt').exists()


@pytest.mark.parametrize('suffix, paths, message', [('', None, 'suffix'), ('.x', 'repeat', 'repeat')])
def test_snapshot_rejects_bad_arguments(tmp_path, suffix, paths, message):
    outputs = make_outputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        snapshot(outputs + outputs[:1] if paths == 'repeat' else outputs, suffix)
