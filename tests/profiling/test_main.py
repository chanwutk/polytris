"""profiling/main.py: one execution per configuration, dedup, resume, failures, and the output contract."""

import json
import os
from functools import partial
from pathlib import Path

import pytest
import yaml

from execution.config import parse_config
from execution.main import StartupError
from profiling import main as pm
from profiling.config import parse_sweep_config
from tests.profiling.fixtures import FakeRunner, sweep_doc, videos_with_references

DUPLICATES = {'C_60_4_080_r050_none_s100_sortcython': 'C_60_4_060_r050_none_s100_sortcython',
              'C_60_4_080_r050_none_s100_bytetrackcython': 'C_60_4_060_r050_none_s100_bytetrackcython'}


def setup(tmp_path, **params):
    cfg = parse_sweep_config(sweep_doc(tmp_path, **params))
    return cfg, videos_with_references(tmp_path)


def result(cfg, run_id) -> dict:
    return json.loads(Path(cfg.profiling_dir, 'runs', run_id, 'result.json').read_text())


def test_every_non_duplicate_configuration_runs_once_with_its_own_config(tmp_path):
    cfg, videos = setup(tmp_path)
    runner = FakeRunner()
    summary = pm.sweep(cfg, runner=runner, videos=videos)
    assert summary == {'configurations': 16, 'duplicates': 2, 'ran': 14, 'skipped': 0, 'duplicates_copied': 2,
                       'failed': []}
    ran = {c['id'] for c in runner.calls}
    assert len(ran) == 14 and not ran & set(DUPLICATES)
    for call in runner.calls:
        # The run's execution config is written beside it, and is valid; saving is off by default.
        config = yaml.safe_load(Path(call['config']).read_text())
        assert parse_config(config) and call['save'] is None and call['videos'] == videos
        assert call['out'] == os.path.join(cfg.profiling_dir, 'runs', call['id'], 'tracks')


def test_result_records_execution_runtime_and_the_video_files(tmp_path):
    cfg, videos = setup(tmp_path)
    pm.sweep(cfg, runner=FakeRunner(runtime_s=4.25), videos=videos)
    r = result(cfg, 'C_60_1_040_r050_none_s100_sortcython')
    assert r['runtime_s'] == 4.25 and r['frames'] == 20 and r['videos'] == 2
    assert r['video_files'] == ['va00.mp4', 'va01.mp4'] and r['saved'] is False and r['duplicate_of'] is None
    assert r['params']['mistrack_tolerance'] == 0.4 and r['wall_s'] >= 0


def test_duplicates_copy_their_representatives_result(tmp_path):
    cfg, videos = setup(tmp_path)
    pm.sweep(cfg, runner=FakeRunner(), videos=videos)
    for dup, rep in DUPLICATES.items():
        d, r = result(cfg, dup), result(cfg, rep)
        assert d['duplicate_of'] == rep and d['id'] == dup and d['params']['mistrack_tolerance'] == 0.8
        assert d['runtime_s'] == r['runtime_s'] and d['video_files'] == r['video_files']
        # Duplicates hold only their result: no tracks, no config of their own.
        assert sorted(os.listdir(Path(cfg.profiling_dir, 'runs', dup))) == ['result.json']


def test_without_dedup_every_configuration_runs(tmp_path):
    cfg, videos = setup(tmp_path, dedup_pruning=False)
    summary = pm.sweep(cfg, runner=FakeRunner(), videos=videos)
    assert summary['ran'] == 16 and summary['duplicates'] == 0


def test_a_selection_runs_only_its_configurations_with_dedup_among_them(tmp_path):
    selected = ['C_60_1_r050_none_s100_sortcython', *DUPLICATES, *DUPLICATES.values()]
    cfg = parse_sweep_config({**sweep_doc(tmp_path), 'configurations': selected})
    runner = FakeRunner()
    summary = pm.sweep(cfg, runner=runner, videos=videos_with_references(tmp_path))
    assert summary == {'configurations': 5, 'duplicates': 2, 'ran': 3, 'skipped': 0, 'duplicates_copied': 2,
                       'failed': []}
    assert sorted(c['id'] for c in runner.calls) == sorted(['C_60_1_r050_none_s100_sortcython', *DUPLICATES.values()])
    # Nothing outside the selection gets a run directory.
    assert sorted(os.listdir(Path(cfg.profiling_dir, 'runs'))) == sorted(selected)


def test_save_passes_cache_shaped_root_and_marks_results(tmp_path):
    cfg, videos = setup(tmp_path, sample_rates=[1], mistrack_tolerances=[None])
    runner = FakeRunner()
    pm.sweep(cfg, save=True, runner=runner, videos=videos)
    assert {c['save'] for c in runner.calls} == {os.path.join(cfg.output_root, 'ds')}
    assert result(cfg, 'C_60_1_r050_none_s100_sortcython')['saved'] is True


def test_a_rerun_skips_finished_configurations(tmp_path):
    cfg, videos = setup(tmp_path)
    pm.sweep(cfg, runner=FakeRunner(), videos=videos)
    runner = FakeRunner()
    summary = pm.sweep(cfg, runner=runner, videos=videos)
    assert runner.calls == [] and summary['skipped'] == 14 and summary['ran'] == 0 and summary['duplicates_copied'] == 0


def test_failures_are_recorded_the_sweep_continues_and_a_rerun_retries_them(tmp_path):
    cfg, videos = setup(tmp_path)
    bad = 'C_60_4_060_r050_none_s100_sortcython'
    summary = pm.sweep(cfg, runner=FakeRunner(fail={bad}), videos=videos)
    assert summary['failed'] == [bad] and summary['ran'] == 14
    run = Path(cfg.profiling_dir, 'runs', bad)
    error = (run / 'error.txt').read_text()
    assert 'exited with 3' in error and f'stderr of {bad}' in error and not (run / 'result.json').exists()
    # Its duplicate waits for it.
    assert not Path(cfg.profiling_dir, 'runs', 'C_60_4_080_r050_none_s100_sortcython', 'result.json').exists()
    # The rerun cleans the failed directory, runs only it, then fills its duplicate.
    runner = FakeRunner()
    summary = pm.sweep(cfg, runner=runner, videos=videos)
    assert [c['id'] for c in runner.calls] == [bad] and summary['failed'] == [] and summary['duplicates_copied'] == 1
    assert not (run / 'error.txt').exists()
    assert result(cfg, 'C_60_4_080_r050_none_s100_sortcython')['duplicate_of'] == bad


def test_a_crashed_attempt_without_result_is_started_over(tmp_path):
    cfg, videos = setup(tmp_path, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'])
    run = Path(cfg.profiling_dir, 'runs', 'C_60_1_r050_none_s100_sortcython')
    run.mkdir(parents=True)
    (run / 'leftover.txt').write_text('from a killed run')
    # The sweep must first be claimed by the same document; claim it as a previous sweep would have.
    pm.claim_output(cfg)
    pm.sweep(cfg, runner=FakeRunner(), videos=videos)
    assert not (run / 'leftover.txt').exists() and (run / 'result.json').exists()


def test_a_zero_exit_without_summary_is_a_failure(tmp_path):
    cfg, videos = setup(tmp_path, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'])
    summary = pm.sweep(cfg, runner=lambda config, v, out, save, log: 0, videos=videos)
    assert summary['failed'] == ['C_60_1_r050_none_s100_sortcython']
    assert 'no summary.json' in Path(cfg.profiling_dir, 'runs', summary['failed'][0], 'error.txt').read_text()


def test_a_different_sweep_is_refused_in_the_same_output(tmp_path):
    cfg, videos = setup(tmp_path)
    pm.sweep(cfg, runner=FakeRunner(), videos=videos)
    other = parse_sweep_config(sweep_doc(tmp_path, sample_rates=[1]))
    with pytest.raises(StartupError, match='different sweep'):
        pm.sweep(other, runner=FakeRunner(), videos=videos)
    recorded = json.loads(Path(cfg.profiling_dir, 'sweep.json').read_text())
    assert recorded['sweep'] == cfg.raw


def test_missing_references_stop_the_sweep_before_anything_runs(tmp_path):
    cfg, videos = setup(tmp_path)
    os.remove(Path(cfg.evaluation.reference_dir, 'va01.mp4', '002_naive', 'tracking.jsonl'))
    runner = FakeRunner()
    with pytest.raises(StartupError, match='reference tracks missing for 1 videos'):
        pm.sweep(cfg, runner=runner, videos=videos)
    assert runner.calls == [] and not os.path.exists(cfg.profiling_dir)


def test_the_real_subprocess_path_runs_the_command_line_and_reads_its_files(tmp_path):
    cfg, videos = setup(tmp_path, sample_rates=[1], mistrack_tolerances=[None], trackers=['sortcython'],
                        relevance_thresholds=[0.5, 0.75])
    runner = partial(pm.run_execution, module='tests.profiling.stand_in_execution')
    summary = pm.sweep(cfg, save=True, runner=runner, videos=videos)
    # 0.75 is the stand-in's failing configuration: exit code and stderr reach error.txt.
    assert summary['failed'] == ['C_60_1_r075_none_s100_sortcython']
    error = Path(cfg.profiling_dir, 'runs', 'C_60_1_r075_none_s100_sortcython', 'error.txt').read_text()
    assert 'exited with 5' in error and 'stand-in stderr line' in error
    ok = Path(cfg.profiling_dir, 'runs', 'C_60_1_r050_none_s100_sortcython')
    argv = json.loads((ok / 'tracks' / 'argv.json').read_text())
    assert argv == {'config': str(ok / 'config.yaml'), 'videos': videos, 'out': str(ok / 'tracks'),
                    'save': os.path.join(cfg.output_root, 'ds')}
    assert result(cfg, ok.name)['runtime_s'] == 3.5 and (ok / 'stderr.log').read_text() == 'stand-in stderr line\n'
