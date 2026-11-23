"""execution2.tools.scripts_equivalence: linking p022's time-limited outputs, and the configuration grid."""

from pathlib import Path

import pytest

from execution2.tools import scripts_equivalence as se


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    # cache.exec(dataset, stage, video, *rest) -> tmp/<video>/<stage>/<rest...>
    monkeypatch.setattr(se.cache, 'exec', lambda ds, stage, video, *rest: tmp_path.joinpath(video, stage, *rest))
    return tmp_path


def write_tl(root: Path, video: str, param: str, limit: str) -> Path:
    d = root / video / 'pruned-polyominoes-tl' / param / f'tl{limit}' / 'score'
    d.mkdir(parents=True)
    (d / 'score.jsonl').write_text('timed\n')
    return d


def test_links_point_p030s_path_at_the_time_limited_result(fake_cache):
    write_tl(fake_cache, 'va00.mp4', 'P_1_040_r025_sortcython', '5.0')
    write_tl(fake_cache, 'va00.mp4', 'P_4_080_r050_bytetrackcython', '5.0')
    assert se.link_time_limited_prunes('ds', ['va00.mp4', 'va01.mp4'], 5.0) == 2
    read = fake_cache / 'va00.mp4' / 'pruned-polyominoes' / 'P_1_040_r025_sortcython' / 'score' / 'score.jsonl'
    assert read.read_text() == 'timed\n' and read.parent.parent.is_symlink()
    # Relinking (a rerun) replaces the links instead of failing.
    assert se.link_time_limited_prunes('ds', ['va00.mp4'], 5.0) == 2


def test_an_untimed_result_in_the_way_is_an_error(fake_cache):
    write_tl(fake_cache, 'va00.mp4', 'P_1_040_r025_sortcython', '5.0')
    (fake_cache / 'va00.mp4' / 'pruned-polyominoes' / 'P_1_040_r025_sortcython').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='not a link'):
        se.link_time_limited_prunes('ds', ['va00.mp4'], 5.0)


def test_a_missing_time_limit_is_an_error(fake_cache):
    write_tl(fake_cache, 'va00.mp4', 'P_1_040_r025_sortcython', '0.1')
    with pytest.raises(FileNotFoundError, match='tl5.0'):
        se.link_time_limited_prunes('ds', ['va00.mp4'], 5.0)


def test_grid_has_every_configuration_once():
    branches = se.grid_branches()
    # 2 sample rates x 2 T_r x 3 tolerance choices x 2 paddings x 1 scale x 2 trackers.
    assert len(branches) == 48
    names = {se.config_name(b) for b in branches}
    assert len(names) == 48 and 's1_r025_tnone_none_c100_sortcython' in names and 's4_r050_t080_tr_c100_bytetrackcython' in names


def test_p022_runs_one_grid_combination_at_a_time(tmp_path, monkeypatch):
    # Each generated config holds one (sample rate, tolerance, T_r, tracker), pruned branches only; DATA is kept.
    import yaml
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs' / 'equivalence.yaml').write_text((repo / 'configs' / 'equivalence.yaml').read_text())
    monkeypatch.chdir(tmp_path)
    names = se.single_combo_configs()
    assert len(names) == 2 * 2 * 2 * 2
    combos = set()
    for name in names:
        c = yaml.safe_load((tmp_path / 'configs' / name).read_text())
        ex = c['EXEC']
        assert all(len(ex[k]) == 1 for k in ('SAMPLE_RATES', 'TRACKING_ACCURACY_THRESHOLDS', 'RELEVANCE_THRESHOLDS',
                                             'TRACKERS'))
        assert ex['TRACKING_ACCURACY_THRESHOLDS'][0] is not None and c['DATA']['CACHE_DIR'] == '/polyis-cache/'
        combos.add((ex['SAMPLE_RATES'][0], ex['TRACKING_ACCURACY_THRESHOLDS'][0], ex['RELEVANCE_THRESHOLDS'][0],
                    ex['TRACKERS'][0]))
    assert len(combos) == 16
