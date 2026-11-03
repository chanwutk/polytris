import json

import pandas as pd
import pytest

from polyis import leap_results
from evaluation import p200_compare_compute as compare


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path/'leap_canonical.json'
    monkeypatch.setattr(leap_results, 'ROOT', tmp_path)
    monkeypatch.setattr(leap_results, 'MANIFEST', path)
    entry = dict(param_id=0, frames=900, compute_s=2.0,
                 hota=dict(HOTA=.5, DetA=.6, AssA=.7, LocA=.8),
                 tracking_root='runs/tracks', native_intervals='runs/native.gz',
                 source_comparison='runs/comparison.json')
    path.write_text(json.dumps(dict(datasets={'caldot1-y05': entry})))
    return path, entry


def test_unselected_dataset_and_missing_registry_keep_existing_behavior(registry):
    path, _ = registry
    assert leap_results.canonical_leap_result('jnc0') is None
    path.unlink()
    assert leap_results.canonical_leap_result('caldot1-y05') is None


def test_invalid_registry_fails_instead_of_silently_using_old_results(registry):
    path, entry = registry
    entry['tracking_root'] = '../outside'
    path.write_text(json.dumps(dict(datasets={'caldot1-y05': entry})))
    with pytest.raises(AssertionError, match='Invalid canonical'):
        leap_results.canonical_leap_result('caldot1-y05')


def test_promoted_point_changes_only_test_results_and_preserves_cache(registry, tmp_path, monkeypatch):
    path = tmp_path/'tradeoff.csv'
    original = pd.DataFrame(dict(param_id=[0, 0], videoset=['valid', 'test'], sample_rate=[1, 1],
                                 time=[9., 10.], HOTA_HOTA=[.1, .2], MOTA_MOTA=[.3, .4]))
    original.to_csv(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(compare.cache, 'sota', lambda *args: str(path))
    result = compare.load_sota_tradeoff_data(['caldot1-y05'], 'leap')
    assert result.loc[result.videoset.eq('valid'), 'HOTA_HOTA'].iloc[0] == .1
    selected = result.loc[result.videoset.eq('test')].iloc[0]
    assert selected['time'] == 2. and selected.HOTA_HOTA == .5
    assert pd.isna(selected.MOTA_MOTA)
    assert path.read_bytes() == before


def test_canonical_track_path_uses_reviewed_output_without_replacing_old_cache(registry, tmp_path, monkeypatch):
    from measurements import common
    target = tmp_path/'runs/tracks/te00.mp4/tracking.jsonl'
    target.parent.mkdir(parents=True)
    target.write_text('{"frame_idx":0,"tracks":[]}\n')
    monkeypatch.setattr(common.cache, 'sota', lambda *args: str(tmp_path/'original-cache'))
    assert common.output_path('caldot1-y05','leap','param 0','te00.mp4') == target
    assert common.output_path('caldot1-y05','leap','param 1','te00.mp4') == tmp_path/'original-cache'
