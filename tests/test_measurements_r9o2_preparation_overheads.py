import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from measurements import r9o2_preparation_overheads as overheads


@pytest.fixture
def gpu():
    # Import the standalone runner without executing a benchmark or touching CUDA.
    path = Path(__file__).resolve().parents[1] / 'measurements-gpu/r9o2_preparation_overheads.py'
    spec = importlib.util.spec_from_file_location('preparation_overheads_gpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    # Small synthetic saved report whose shared p040 outputs represent two configurations on five videos.
    report = {'datasets': {'demo': {
        'video': {'train': {'videos': 3}, 'valid': {'videos': 5}},
        'components': {
            'classifier_optimization': {'instrumented_s': 10},
            'validation_sweep': {'coverage': {'complete_videos': 5},
                                 'stages': {'040_exec_detect': {'work_units': 10}}},
        },
        'retiming': {'totals_with_retiming': {'instrumented_s': 100, 'process_s': 200}},
        'break_even': {b: {'saving_s_per_video_minute': 2} for b in ('instrumented', 'process')},
    }}}
    samples = [dict(dataset='demo', step=step, cuda_init_s=1, load_s=2, warmup_s=warmup, seconds=3 + warmup)
               for step, warmup in [('train', 0), ('reference', 3), ('sweep', 1), ('persistent', 4)]]
    samples += [dict(dataset='demo', step='classifier', seconds=20),
                dict(dataset='demo', step='tables', seconds=5, tradeoff_s=3, pareto_s=2)]
    base = tmp_path / 'base.json'
    timings = tmp_path / 'timings.jsonl'
    base.write_text(json.dumps(report))
    timings.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    return base, timings, report, samples


def test_account_replaces_benchmark_and_counts_shared_detector_tasks_once(inputs):
    base, timings, _, _ = inputs
    rows = overheads.account(duckdb.connect(), base, timings)['rows']
    actual = {(r['boundary'], r['scenario']): r for r in rows}
    # The old benchmark is replaced: +20 -10, with +3 +2 for table processing.
    once = actual['instrumented', 'once_per_dataset']
    assert once['total_s'] == 122
    assert once['startup_s'] == 7
    # Three stage CUDA contexts, three training loads, five reference loads/warm-ups, two sweep loads/warm-ups.
    repeated = actual['instrumented', 'per_video_and_configuration']
    assert repeated['sweep_detector_loads'] == 2
    assert repeated['startup_s'] == 3 + 3 * 2 + 5 * 5 + 2 * 3
    assert repeated['total_s'] == 155
    # Reference warm-ups are already present in the process spans, unlike instrumented ops.
    assert actual['process', 'once_per_dataset']['already_counted_reference_warmup_s'] == 15
    assert actual['process', 'once_per_dataset']['total_s'] == 207
    assert actual['process', 'per_video_and_configuration']['total_s'] == 240
    assert repeated['break_even_video_hours'] == pytest.approx(155 / 120)


def test_account_scales_tradeoff_probe_to_missing_validation_videos(inputs):
    base, timings, report, _ = inputs
    report['datasets']['demo']['video']['valid']['videos'] = 6
    base.write_text(json.dumps(report))
    rows = overheads.account(duckdb.connect(), base, timings)['rows']
    row = next(r for r in rows if r['boundary'] == 'instrumented' and r['scenario'] == 'once_per_dataset')
    assert row['measured_tradeoff_s'] == 3
    assert row['tradeoff_s'] == pytest.approx(3.6)
    assert row['pareto_s'] == 2
    assert row['total_s'] == pytest.approx(122.6)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'fractional_jobs', 'negative'])
def test_account_refuses_incomplete_or_invalid_measurements(inputs, change):
    base, timings, report, samples = inputs
    if change == 'missing':
        samples.pop()
    elif change == 'duplicate':
        samples.append(samples[0])
    elif change == 'negative':
        samples[0]['seconds'] = -1
    else:
        report['datasets']['demo']['components']['validation_sweep']['stages']['040_exec_detect']['work_units'] = 11
    base.write_text(json.dumps(report))
    timings.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    with pytest.raises(AssertionError):
        overheads.account(duckdb.connect(), base, timings)


@pytest.mark.parametrize('profile,single,batches,batch_size', [
    ('train', 0, 0, 16), ('reference', 32, 0, 1), ('sweep', 0, 1, 4), ('persistent', 32, 1, 4),
])
def test_startup_matches_stage_warmup_without_charging_training_twice(tmp_path, monkeypatch, gpu, profile,
                                                                    single, batches, batch_size):
    import cv2
    import torch
    from polyis.io import store
    from polyis.models import detector

    # Replace only hardware operations, checking the actual stage warm-up loops.
    (tmp_path / 'va00.mp4').touch()
    for i in range(4):
        (tmp_path / f'packed_{i}.jpg').touch()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(store, 'dataset', lambda *args: tmp_path)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: type('Capture', (), {
        'read': lambda self: (True, frame), 'release': lambda self: None,
    })())
    monkeypatch.setattr(cv2, 'imread', lambda *args: frame)
    monkeypatch.setattr(torch, 'zeros', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    loads, calls = [], []
    monkeypatch.setattr(detector, 'get_detector', lambda *args, **kwargs: loads.append(kwargs))
    monkeypatch.setattr(detector, 'detect', lambda *args: calls.append('single'))
    monkeypatch.setattr(detector, 'detect_batch', lambda images, model: calls.append(('batch', len(images))))
    result = gpu.startup('demo', profile, tmp_path)
    assert calls.count('single') == single
    assert calls.count(('batch', 4)) == batches
    assert loads == [dict(batch_size=batch_size, num_images=0)]
    assert result['warmup_s'] == 0 if profile == 'train' else result['warmup_s'] >= 0


def test_classifier_reuses_checkpoint_and_cleans_up_private_compiler_workers(tmp_path, monkeypatch, gpu):
    import os
    from polyis.io import cache
    from scripts import p015_tune_optimize_classifier as benchmark
    from torch._inductor import async_compile

    # The real benchmark is replaced with a tiny writer; no model is retrained or inference run.
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'model_best.pth').write_bytes(b'saved checkpoint')
    target = tmp_path / 'cache'
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(cache, 'index', lambda *args: target)
    monkeypatch.setattr(benchmark, 'get_video_resolution', lambda dataset: (1920, 1080))
    calls = []

    def fake_benchmark(*args):
        calls.append(args)
        assert (target / 'model_best.pth').read_bytes() == b'saved checkpoint'
        (target / 'model_compilation.jsonl').write_text('{"method":"baseline","runtime_ms":1}\n')

    monkeypatch.setattr(benchmark, 'benchmark_classifier', fake_benchmark)
    monkeypatch.setattr(async_compile, 'shutdown_compile_workers', lambda: calls.append('cleanup'))
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', 'previous')
    monkeypatch.setenv('TRITON_CACHE_DIR', 'previous')
    result = gpu.classifier('demo', inputs, work, {'operating_point': {'classifier': 'C', 'tilesize': 60}})
    assert calls == [(['demo'], 1920, 1080, 'C', 60, 128, 0, None), 'cleanup']
    assert os.environ['TORCHINDUCTOR_CACHE_DIR'] == str(work / 'inductor')
    assert os.environ['TRITON_CACHE_DIR'] == str(work / 'triton')
    assert result['methods'] == [dict(method='baseline', runtime_ms=1)]


@pytest.mark.parametrize('recomputed', ['original', 'different'])
def test_tables_keep_saved_frontier_even_when_cost_probe_differs(tmp_path, monkeypatch, gpu, recomputed):
    from evaluation import p130_tradeoff_compute as tradeoff, p135_pareto_extract as pareto
    from polyis.io import cache

    # Isolate all output paths and keep the original selection CSV immutable.
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for name in ('accuracy.csv', 'query_execution_overall.csv'):
        (inputs / name).write_text('saved validation rows\n')
    saved = inputs / 'pareto_params.csv'
    saved.write_text('variant_id\noriginal\n')
    monkeypatch.setattr(cache, 'eval', lambda dataset, stage, *parts: tmp_path / 'scratch' / stage / Path(*parts))
    monkeypatch.setattr(tradeoff, 'process_dataset', lambda dataset: None)
    monkeypatch.setattr(pareto, 'PARETO_PARAM_COLS', ['variant_id'])

    def select_scratch(args):
        assert args.test is False
        target = cache.eval('demo', 'pareto-params', 'pareto_params.csv')
        target.parent.mkdir(parents=True)
        target.write_text(f'variant_id\n{recomputed}\n')

    monkeypatch.setattr(pareto, 'main', select_scratch)
    result = gpu.tables('demo', inputs)
    assert saved.read_text() == 'variant_id\noriginal\n'
    assert result['source_selections_preserved'] is True
    assert result['recomputed_frontier_configuration_differences'] == (0 if recomputed == 'original' else 2)


def test_runner_sets_isolated_config_before_importing_pipeline(tmp_path, monkeypatch, gpu):
    import os
    from types import SimpleNamespace
    import yaml

    # CONFIG must resolve to scratch before pipeline imports initialize their cache paths.
    config_path = tmp_path / 'original.yaml'
    source = tmp_path / 'source-cache'
    config_path.write_text(yaml.safe_dump({'DATA': {'CACHE_DIR': str(source)}, 'EXEC': {'DATASETS': ['demo']}}))
    inputs = tmp_path / 'inputs' / 'demo'
    inputs.mkdir(parents=True)
    (inputs / 'metadata.json').write_text('{"input_files": {}}')
    scratch = tmp_path / 'scratch'
    monkeypatch.setenv('CONFIG', str(config_path))
    monkeypatch.setattr(gpu, 'parse_args', lambda: SimpleNamespace(dataset='demo', step='tables',
                                                               inputs=str(inputs.parent), scratch=str(scratch)))

    def isolated_tables(dataset, paths):
        configured = yaml.safe_load(Path(os.environ['CONFIG']).read_text())['DATA']['CACHE_DIR']
        assert configured == str(scratch / 'demo' / 'tables' / 'cache')
        assert paths == inputs
        return dict(seconds=0)

    monkeypatch.setattr(gpu, 'tables', isolated_tables)
    gpu.main()
    assert not source.exists()
    assert (scratch / 'demo' / 'tables' / 'result.json').is_file()


def test_tradeoff_loader_reads_the_writers_configured_cache(tmp_path, monkeypatch):
    from polyis.io import cache
    from polyis import utilities

    # Exercise the real reader: its former hardcoded /polyis-cache path broke scratch p130/p135 runs.
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(utilities, 'CACHE_DIR', '/unused-hardcoded-root')
    path = cache.eval('demo', 'tradeoff', 'tradeoff.csv')
    path.parent.mkdir(parents=True)
    path.write_text('dataset,videoset,variant,variant_id\ndemo,valid,polytris,C\n')
    assert utilities.load_tradeoff_data('demo').to_dict('records') == [
        dict(dataset='demo', videoset='valid', variant='polytris', variant_id='C'),
    ]
