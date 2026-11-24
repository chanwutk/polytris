"""The fixed-test ILP benchmark's clip subset, no-solution rows, gap, resume, and the 1-hour rerun's clip choice."""
import json

import numpy as np
import pytest

from experiments.measurements import fixed_ilp_1h
from experiments.measurements import fixed_ilp_unlimited as benchmark
from tetris.sample.ilp.gurobi import ILPResult


EXPECTED = {'variables': 3, 'constraints': 4, 'runtime': {'runtime': [{'op': 'solve_ilp', 'time': 100.0}]}}


@pytest.mark.parametrize('objective, bound, expected', [
    (100, 90.0, 0.1), (100, 100.0, 0.0), (0, 0.0, 0.0), (None, 5.0, None), (10, None, None)])
def test_gap(objective, bound, expected):
    assert benchmark.gap(objective, bound) == (pytest.approx(expected) if expected is not None else None)


def test_row_without_a_solution_keeps_the_bound_and_no_objective(monkeypatch):
    # selected_stats would reject an empty selection; a run without a solution must not call it.
    monkeypatch.setattr(benchmark, 'selected_stats', lambda *_: pytest.fail('selected_stats called'))
    result = ILPResult(selected=set(), build_ms=1.0, solve_ms=3_600_000.0, status=9, sol_count=0, obj_bound=42.5)
    row = benchmark.clip_row('ams-y05', 'te00.mp4', 900, {}, EXPECTED, result, 3_600_001.0, 'sha')
    assert (row['objective_tiles'], row['obj_bound'], row['gap'], row['status']) == (None, 42.5, None, 9)
    assert row['old_solve_ms'] == 100.0


def test_row_with_a_solution_records_objective_and_gap(monkeypatch):
    monkeypatch.setattr(benchmark, 'selected_stats', lambda structure, selected: 200)
    result = ILPResult(selected={(0, 0)}, build_ms=1.0, solve_ms=5.0, status=2, sol_count=1, obj_bound=200.0)
    row = benchmark.clip_row('ams-y05', 'te00.mp4', 900, {}, EXPECTED, result, 6.0, 'sha')
    assert (row['objective_tiles'], row['gap'], row['selected']) == (200, 0.0, 1)


def run_main(tmp_path, monkeypatch, clips, done=()):
    """Run the benchmark's main on three fake clips with every expensive step replaced; return solved clips."""
    experiment, output = tmp_path / 'experiment', tmp_path / 'output'
    experiment.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    videos = {'caldot1-y05': ['te00.mp4', 'te01.mp4'], 'ams-y05': ['te00.mp4']}
    (experiment / 'inputs.json').write_text(json.dumps({d: {'videos': v} for d, v in videos.items()}))
    (experiment / 'operator_details.json').write_text(json.dumps(
        [dict(EXPECTED, dataset=d, video=v) for d, vs in videos.items() for v in vs]))
    # Clips already written by an earlier, interrupted run.
    (output / 'results.jsonl').write_text(''.join(json.dumps({'dataset': d, 'video': v}) + '\n' for d, v in done))
    score = tmp_path / 'score.jsonl'
    score.write_text('{}\n')
    solved = []
    monkeypatch.setattr(benchmark, 'EXPERIMENT', experiment)
    monkeypatch.setattr(benchmark, 'OUTPUT', output)
    monkeypatch.setattr(benchmark, 'CACHE_DIR', None)
    monkeypatch.setattr(benchmark, 'CLIPS', clips)
    monkeypatch.setattr(benchmark.cache, 'exec', lambda *args: score)
    monkeypatch.setattr(benchmark, 'decode_grids', lambda rows: (np.zeros((1, 1, 1)), [0]))
    monkeypatch.setattr(benchmark.sizes, 'max_distance', lambda *args: np.ones((1, 1)))
    monkeypatch.setattr(benchmark.sizes, 'group', lambda grids: (grids, [[1]]))
    monkeypatch.setattr(benchmark.sizes, 'ilp_structure', lambda *args: {})
    monkeypatch.setattr(benchmark.sizes, 'instance_size', lambda structure: {'variables': 3})
    monkeypatch.setattr(benchmark, 'selected_stats', lambda structure, selected: 1)
    monkeypatch.setattr(benchmark, 'write_summary', lambda path: None)

    def solve(*args, time_limit_seconds):
        solved.append(len(solved))
        return ILPResult(selected={(0, 0)}, build_ms=1.0, solve_ms=2.0, status=2, sol_count=1, obj_bound=1.0)
    monkeypatch.setattr(benchmark, 'solve_ilp', solve)
    benchmark.main()
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    return [(r['dataset'], r['video']) for r in rows if 'status' in r], json.loads((output / 'metadata.json').read_text())


def test_clips_subset_solves_only_listed_clips_and_records_them(tmp_path, monkeypatch):
    solved, metadata = run_main(tmp_path, monkeypatch, {('ams-y05', 'te00.mp4')})
    assert solved == [('ams-y05', 'te00.mp4')]
    assert metadata['clips'] == [['ams-y05', 'te00.mp4']] and metadata['cache_dir'] is None


def test_resume_skips_clips_already_written(tmp_path, monkeypatch):
    clips = {('caldot1-y05', 'te01.mp4'), ('ams-y05', 'te00.mp4')}
    solved, _ = run_main(tmp_path, monkeypatch, clips, done=[('caldot1-y05', 'te01.mp4')])
    assert solved == [('ams-y05', 'te00.mp4')]


def test_clips_outside_the_fixed_test_set_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        run_main(tmp_path, monkeypatch, {('ams-y05', 'te99.mp4')})


def write_sixty_second_results(path, solve_ms):
    path.write_text(''.join(json.dumps({'dataset': 'ams-y05', 'video': f'te{i:02d}.mp4', 'solve_ms': ms}) + '\n'
                            for i, ms in enumerate(solve_ms)))


def test_capped_clips_are_runs_at_or_above_sixty_seconds(tmp_path):
    path = tmp_path / 'results.jsonl'
    write_sixty_second_results(path, [59_999.9, 60_000.0, 60_412.0, 10.0])
    assert fixed_ilp_1h.capped_clips(path) == {('ams-y05', 'te01.mp4'), ('ams-y05', 'te02.mp4')}


def test_configure_sets_the_hour_limit_default_cache_and_capped_clips(tmp_path, monkeypatch):
    path = tmp_path / 'results.jsonl'
    write_sixty_second_results(path, [60_000.0] * 48 + [5.0] * 152)
    monkeypatch.setattr(fixed_ilp_1h, 'SIXTY_SECOND_RESULTS', path)
    for name in ('OUTPUT', 'TIME_LIMIT_SECONDS', 'CACHE_DIR', 'CLIPS'):
        monkeypatch.setattr(benchmark, name, getattr(benchmark, name))
    fixed_ilp_1h.configure()
    assert (benchmark.TIME_LIMIT_SECONDS, benchmark.CACHE_DIR, len(benchmark.CLIPS)) == (3600, None, 48)
    assert benchmark.OUTPUT.name == 'fixed_ilp_1h'


def test_configure_refuses_an_unexpected_number_of_capped_clips(tmp_path, monkeypatch):
    path = tmp_path / 'results.jsonl'
    write_sixty_second_results(path, [60_000.0] * 47 + [5.0])
    monkeypatch.setattr(fixed_ilp_1h, 'SIXTY_SECOND_RESULTS', path)
    with pytest.raises(AssertionError, match='expected 48'):
        fixed_ilp_1h.configure()
