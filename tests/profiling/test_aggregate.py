"""profiling/aggregate.py: rows from results and evaluations, duplicates, Pareto fronts, and the outputs."""

import json
from pathlib import Path

import pytest
import yaml

from execution.config import parse_config
from profiling import aggregate as ag, main as pm
from profiling.config import parse_sweep_config
from tests.profiling.fixtures import FakeRunner, sweep_doc, videos_with_references


def swept(tmp_path, runtimes: dict[str, float] | None = None, hota: dict[str, float] | None = None, save=False, **params):
    """A finished sweep with fake runtimes, and an evaluation per run whose combined HOTA is controlled."""
    cfg = parse_sweep_config(sweep_doc(tmp_path, **params))
    pm.sweep(cfg, save=save, runner=FakeRunner(runtime_s=10.0), videos=videos_with_references(tmp_path))
    runs = Path(cfg.profiling_dir, 'runs')
    for run in runs.iterdir():
        result = json.loads((run / 'result.json').read_text())
        if runtimes and run.name in runtimes:
            result['runtime_s'] = runtimes[run.name]
            (run / 'result.json').write_text(json.dumps(result))
        if result['duplicate_of'] is None:
            (run / 'evaluation.json').write_text(json.dumps({v: {'HOTA': {'value': (hota or {}).get(run.name, 0.5)}}
                                                             for v in result['video_files']}))
    return cfg


@pytest.fixture
def fake_hota(monkeypatch):
    """Combining is TrackEval's (tested in the container); here each video's 'value' passes straight through."""
    monkeypatch.setattr(ag, 'combine_hota', lambda per_video: {'HOTA': [per_video[0]['value']]})


def test_rows_carry_parameters_hota_time_throughput_and_provenance(tmp_path, fake_hota):
    cfg = swept(tmp_path, runtimes={'C_60_1_040_r050_none_s100_sortcython': 4.0},
                hota={'C_60_1_040_r050_none_s100_sortcython': 0.9})
    rows = ag.configuration_rows(cfg)
    assert len(rows) == 16 and [r['id'] for r in rows] == list(range(16))
    row = next(r for r in rows if r['config_id'] == 'C_60_1_040_r050_none_s100_sortcython')
    assert row['tracking_accuracy_threshold'] == 0.4 and row['tilesize'] == 60 and row['tilepadding'] == 'none'
    assert row['HOTA_HOTA'] == 0.9 and row['time'] == 4.0 and row['frames'] == 20 and row['throughput_fps'] == 5.0
    assert row['saved'] is False and row['duplicate_of'] is None
    # The embedded YAML is the configuration's execution config.
    assert parse_config(yaml.safe_load(row['execution_yaml'])).params.mistrack_tolerance == 0.4


def test_duplicates_take_their_representatives_score_and_time(tmp_path, fake_hota):
    rep = 'C_60_4_060_r050_none_s100_sortcython'
    cfg = swept(tmp_path, hota={rep: 0.8})
    rows = {r['config_id']: r for r in ag.configuration_rows(cfg)}
    dup = rows['C_60_4_080_r050_none_s100_sortcython']
    assert dup['duplicate_of'] == rep and dup['HOTA_HOTA'] == 0.8 and dup['time'] == rows[rep]['time']
    assert dup['tracking_accuracy_threshold'] == 0.8


def test_missing_results_or_evaluations_stop_aggregation_with_a_list(tmp_path, fake_hota):
    cfg = swept(tmp_path)
    runs = Path(cfg.profiling_dir, 'runs')
    (runs / 'C_60_1_r050_none_s100_sortcython' / 'result.json').unlink()
    (runs / 'C_60_4_060_r050_none_s100_sortcython' / 'evaluation.json').unlink()
    with pytest.raises(RuntimeError, match='3 configurations are not ready') as e:
        ag.configuration_rows(cfg)
    # The representative and its duplicate both lack a score; the deleted result is named too.
    assert 'C_60_1_r050_none_s100_sortcython: no result.json' in str(e.value)
    assert 'C_60_4_080_r050_none_s100_sortcython: no evaluation.json in C_60_4_060' in str(e.value)


def test_an_evaluation_covering_fewer_videos_is_not_aggregated(tmp_path, fake_hota):
    cfg = swept(tmp_path, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'])
    path = Path(cfg.profiling_dir, 'runs', 'C_60_1_r050_none_s100_sortcython', 'evaluation.json')
    scores = json.loads(path.read_text())
    del scores['va01.mp4']
    path.write_text(json.dumps(scores))
    with pytest.raises(RuntimeError, match='covers 1 of 2 videos'):
        ag.configuration_rows(cfg)


def test_pareto_fronts_cover_full_no_sampling_and_no_sampling_or_pruning(tmp_path, fake_hota):
    # s=4 pruned: fastest and best; s=1 unpruned is the only no-sampling-no-pruning point family.
    runtimes = {'C_60_4_040_r050_none_s100_sortcython': 1.0, 'C_60_1_040_r050_none_s100_sortcython': 3.0,
                'C_60_1_r050_none_s100_sortcython': 6.0}
    hota = {'C_60_4_040_r050_none_s100_sortcython': 0.95, 'C_60_1_040_r050_none_s100_sortcython': 0.9,
            'C_60_1_r050_none_s100_sortcython': 0.85}
    cfg = swept(tmp_path, runtimes=runtimes, hota=hota)
    configs, pareto = ag.aggregate(cfg)
    on = {p['config_id']: p['pareto_conditions'] for p in pareto}
    assert on['C_60_4_040_r050_none_s100_sortcython'] == ['full']
    assert 'no_sampling' in on['C_60_1_040_r050_none_s100_sortcython']
    assert 'no_both' in on['C_60_1_r050_none_s100_sortcython']
    # Every front member of no_sampling has s=1; of no_both, s=1 and no pruning.
    by_id = {c['config_id']: c for c in configs}
    for cid, conditions in on.items():
        if 'no_sampling' in conditions:
            assert by_id[cid]['sample_rate'] == 1
        if 'no_both' in conditions:
            assert by_id[cid]['sample_rate'] == 1 and by_id[cid]['tracking_accuracy_threshold'] is None
    # The files: every configuration, the front, and a page embedding each YAML.
    out = Path(cfg.profiling_dir)
    assert len(json.loads((out / 'configurations.json').read_text())) == 16
    assert json.loads((out / 'pareto.json').read_text()) == pareto
    html = (out / 'pareto.html').read_text()
    assert 'end-to-end runtime' in html and 'C_60_4_040_r050_none_s100_sortcython' in html


def test_saved_runs_are_flagged_with_a_warning(tmp_path, fake_hota, capsys):
    cfg = swept(tmp_path, save=True, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'])
    configs, _ = ag.aggregate(cfg)
    assert all(c['saved'] for c in configs)
    assert 'ran with --save' in capsys.readouterr().err


# TrackEval's combination itself (combine_hota vs TrackEval on real tracks) is tested by
# experiments/tests/fanout_profiling/test_aggregate.py, which exercises this same shared function.
