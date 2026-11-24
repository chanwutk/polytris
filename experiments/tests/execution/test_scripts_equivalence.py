"""experiments.tools.execution.scripts_equivalence: keeping every p022 run's results, and the configuration grid."""

from pathlib import Path

import pytest

from experiments.tools.execution import scripts_equivalence as se


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    # cache.exec(dataset, stage, video, *rest) -> tmp/<video>/<stage>/<rest...>
    monkeypatch.setattr(se.cache, 'exec', lambda ds, stage, video, *rest: tmp_path.joinpath(video, stage, *rest))
    return tmp_path


def p022_run(root: Path, video: str, param: str) -> None:
    """What one p022 run does to a video: delete its 022 folder, then write one combination."""
    import shutil
    pruned = root / video / 'pruned-polyominoes'
    shutil.rmtree(pruned, ignore_errors=True)
    (pruned / param / 'score').mkdir(parents=True)
    (pruned / param / 'score' / 'score.jsonl').write_text(param + '\n')


def test_every_p022_runs_results_survive_the_next_run(fake_cache):
    params = ['P_1_040_r025_sortcython', 'P_1_080_r025_sortcython', 'P_4_040_r050_bytetrackcython']
    for param in params:
        p022_run(fake_cache, 'va00.mp4', param)
        se.set_aside_prunes('ds', ['va00.mp4', 'va01.mp4'])
    assert se.restore_prunes('ds', ['va00.mp4', 'va01.mp4']) == 3
    pruned = fake_cache / 'va00.mp4' / 'pruned-polyominoes'
    assert sorted(d.name for d in pruned.iterdir()) == sorted(params)
    assert all((pruned / p / 'score' / 'score.jsonl').read_text() == p + '\n' for p in params)
    assert not (pruned.parent / 'pruned-polyominoes.hold').exists()


def test_a_combination_written_twice_is_an_error(fake_cache):
    p022_run(fake_cache, 'va00.mp4', 'P_1_040_r025_sortcython')
    se.set_aside_prunes('ds', ['va00.mp4'])
    p022_run(fake_cache, 'va00.mp4', 'P_1_040_r025_sortcython')
    with pytest.raises(FileExistsError, match='already held'):
        se.set_aside_prunes('ds', ['va00.mp4'])


def test_grid_has_every_configuration_once():
    branches = se.grid_branches()
    # 2 sample rates x 2 T_r x 3 tolerance choices x 2 paddings x 1 scale x 2 trackers.
    assert len(branches) == 48
    names = {se.config_name(b) for b in branches}
    assert len(names) == 48 and 's1_r025_tnone_none_c100_sortcython' in names and 's4_r050_t080_tr_c100_bytetrackcython' in names


def test_p022_runs_one_grid_combination_at_a_time(tmp_path, monkeypatch):
    # Each generated config holds one (sample rate, tolerance, T_r, tracker), pruned branches only; DATA is kept.
    import yaml
    repo = Path(__file__).resolve().parents[3]
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs' / 'equivalence.yaml').write_text((repo / 'experiments' / 'configs' / 'equivalence.yaml').read_text())
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
