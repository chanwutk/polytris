"""Engine tests with fake stages: shapes, failures, backpressure, shared-memory cleanup (PLAN.md 12A)."""

import multiprocessing as mp
import threading
import time
from collections import Counter
from functools import partial

import pytest

from execution2 import frames
from execution2.pipeline import PipelineError, Stage, run
from tests.execution2 import fake_stages as fs


def inputs(n):
    return [{'id': i, 'path': []} for i in range(n)]


def stage(name, process=fs.tag, procs=1, setup=fs.worker_setup, gpu=None):
    return Stage(name=name, setup=setup, process=process, procs=procs, gpu=gpu)


def assert_each_once(results, n, stage_names):
    """Every input came out exactly once, having passed every stage in order."""
    assert sorted(r['id'] for r in results) == list(range(n))
    for r in results:
        assert [s for s, _ in r['path']] == stage_names


@pytest.mark.parametrize('procs', [(1, 1), (1, 3), (3, 1), (3, 2), (2, 2, 2)])
def test_every_input_passes_every_stage_once(procs):
    names = [f's{i}' for i in range(len(procs))]
    results = list(run([stage(n, procs=p) for n, p in zip(names, procs)], inputs(40), queue_maxsize=2))
    assert_each_once(results, 40, names)


def test_several_workers_share_a_stage():
    # Slow work spread over 3 workers: more than one of them must take messages.
    results = list(run([stage('a', partial(fs.sleep_then_tag, 0.05), procs=3)], inputs(30), queue_maxsize=2))
    assert len({idx for r in results for _, idx in r['path']}) > 1


def test_empty_input_finishes_with_no_results():
    assert list(run([stage('a', procs=2), stage('b', procs=3)], [], queue_maxsize=1)) == []


def test_results_stream_before_inputs_are_exhausted():
    # The input generator yields one item, then waits until that item's result has come out. A feeder
    # that read every input before putting any would never let the first result through, so the
    # generator times out and the run fails.
    first_result = threading.Event()

    def lazy_inputs():
        yield {'id': 0, 'path': []}
        if not first_result.wait(timeout=20):
            raise RuntimeError('the feeder read ahead: no result before the input stream was exhausted')
        yield from ({'id': i, 'path': []} for i in range(1, 50))

    results = []
    for r in run([stage('a'), stage('b')], lazy_inputs(), queue_maxsize=1):
        results.append(r)
        first_result.set()
    assert_each_once(results, 50, ['a', 'b'])


def test_ordinary_payloads_pass_through():
    (result,) = list(run([stage('a', fs.np_payload)], inputs(1), queue_maxsize=1))
    assert result['array'].tolist() == [0, 0, 0]


@pytest.mark.parametrize('failing_stage', [0, 1, 2])
def test_exception_in_any_stage_raises_with_its_report(failing_stage):
    stages = [stage(f's{i}', procs=2) for i in range(3)]
    stages[failing_stage] = stage(f's{failing_stage}', partial(fs.fail_on, 7), procs=2)
    start = time.monotonic()
    with pytest.raises(PipelineError) as e:
        list(run(stages, inputs(20), queue_maxsize=2))
    assert f's{failing_stage}[' in str(e.value) and 'bad input 7' in str(e.value)
    assert "'id': 7" in str(e.value)
    assert time.monotonic() - start < 30


@pytest.mark.parametrize('killed_stage', [0, 1, 2])
def test_silent_death_in_any_stage_raises(killed_stage):
    stages = [stage(f's{i}', procs=2) for i in range(3)]
    stages[killed_stage] = stage(f's{killed_stage}', partial(fs.kill_on, 5), procs=2)
    start = time.monotonic()
    with pytest.raises(PipelineError, match=r'died with exit code -9 without reporting'):
        list(run(stages, inputs(20), queue_maxsize=2))
    assert time.monotonic() - start < 30


def test_setup_failure_raises():
    with pytest.raises(PipelineError, match='setup exploded in b') as e:
        list(run([stage('a'), stage('b', setup=fs.failing_setup)], inputs(3), queue_maxsize=1))
    assert 'no message (setup or shutdown)' in str(e.value)


def test_process_returning_none_is_an_error():
    with pytest.raises(PipelineError, match='returned None'):
        list(run([stage('a', fs.return_none)], inputs(2), queue_maxsize=1))


def test_feeder_failure_raises():
    def broken_inputs():
        yield {'id': 0, 'path': []}
        raise OSError('cannot list videos')
    with pytest.raises(PipelineError, match='cannot list videos'):
        list(run([stage('a')], broken_inputs(), queue_maxsize=1))


def test_gpu_stage_sees_its_gpu_as_the_only_device():
    (result,) = list(run([stage('a', fs.report_gpu, gpu=3)], inputs(1), queue_maxsize=1))
    assert result['cuda_visible_devices'] == '3' and result['gpu'] == 3


def test_cpu_stage_does_not_pin_a_gpu():
    (result,) = list(run([stage('a', fs.report_gpu)], inputs(1), queue_maxsize=1))
    assert result['gpu'] is None


def test_bounded_queues_stop_a_fast_stage_running_ahead():
    # Stage a is instant, stage b slow. a can be ahead of b by at most: the queue between them,
    # the message b is working on, and the one a holds while blocked on put.
    ctx = mp.get_context('spawn')
    made, done = ctx.Value('i', 0), ctx.Value('i', 0)
    maxsize = 2
    stages = [stage('a', partial(fs.count_then_tag, made)),
              stage('b', partial(fs.slow_count_then_tag, done, 0.05))]
    worst = 0
    for _ in run(stages, inputs(30), queue_maxsize=maxsize):
        worst = max(worst, made.value - done.value)
    assert worst <= maxsize + 2


def test_blocks_flow_between_stages_and_are_freed():
    prefix = frames.new_prefix()
    stages = [stage('make', partial(fs.make_block, prefix), procs=2), stage('use', fs.read_and_free_block, procs=2)]
    results = list(run(stages, inputs(10), queue_maxsize=2, shm_prefix=prefix))
    assert sorted(r['value'] for r in results) == list(range(10))
    assert frames.leftover(prefix) == []


def test_unfreed_blocks_fail_a_clean_run_and_are_swept():
    prefix = frames.new_prefix()
    stages = [stage('make', partial(fs.make_block, prefix)), stage('use', fs.read_and_keep_block)]
    with pytest.raises(PipelineError, match='never freed'):
        list(run(stages, inputs(3), queue_maxsize=1, shm_prefix=prefix))
    assert frames.leftover(prefix) == []


@pytest.mark.parametrize('failure', ['exception', 'sigkill'])
def test_failed_run_frees_every_block(failure):
    prefix = frames.new_prefix()
    downstream = partial(fs.fail_on, 6) if failure == 'exception' else partial(fs.kill_on, 6)
    stages = [stage('make', partial(fs.make_block, prefix), procs=2), stage('crash', downstream)]
    with pytest.raises(PipelineError):
        list(run(stages, inputs(20), queue_maxsize=2, shm_prefix=prefix))
    assert frames.leftover(prefix) == []


def test_stopping_early_tears_down_and_frees_blocks():
    prefix = frames.new_prefix()
    stages = [stage('make', partial(fs.make_block, prefix)), stage('use', fs.read_and_keep_block)]
    gen = run(stages, inputs(20), queue_maxsize=1, shm_prefix=prefix)
    next(gen)
    gen.close()
    assert frames.leftover(prefix) == []


def test_invalid_stage_lists_are_rejected():
    with pytest.raises(ValueError, match='at least one stage'):
        list(run([], inputs(1), queue_maxsize=1))
    with pytest.raises(ValueError, match='procs must be >= 1'):
        list(run([stage('a', procs=0)], inputs(1), queue_maxsize=1))


def test_worker_usage_counts_add_up():
    # With 3 workers on each of 2 stages, every message is tagged by exactly one worker per stage.
    results = list(run([stage('a', procs=3), stage('b', procs=3)], inputs(60), queue_maxsize=3))
    per_stage = Counter(s for r in results for s, _ in r['path'])
    assert per_stage == {'a': 60, 'b': 60}


# ---------------------------------------------------------------- fan-out, GPU tuples, savers (profiling/PLAN.md 13A)

def fan_inputs(ns):
    return [{'id': i, 'path': [], 'n': n} for i, n in enumerate(ns)]


@pytest.mark.parametrize('procs', [(1, 1), (3, 2), (2, 3)])
def test_fan_out_delivers_every_child_once(procs):
    ns = [0, 1, 3, 5, 2, 0, 4]
    stages = [Stage('fan', fs.worker_setup, fs.fan_out, procs[0], fan_out=True), stage('after', procs=procs[1])]
    results = list(run(stages, fan_inputs(ns), queue_maxsize=2))
    assert sorted((r['id'], r['child']) for r in results) == sorted((i, c) for i, n in enumerate(ns) for c in range(n))
    assert all([s for s, _ in r['path']] == ['fan', 'after'] for r in results)


def test_fan_out_of_nothing_still_shuts_down():
    stages = [Stage('fan', fs.worker_setup, fs.fan_out, 2, fan_out=True), stage('after')]
    assert list(run(stages, fan_inputs([0, 0, 0]), queue_maxsize=1)) == []


def test_fan_out_none_output_is_an_error():
    with pytest.raises(PipelineError, match='None output'):
        list(run([Stage('fan', fs.worker_setup, fs.none_in_fan_out, fan_out=True)], inputs(2), queue_maxsize=1))


def test_gpu_tuple_is_taken_by_workers_in_turn():
    # One message per worker would not guarantee every worker got one, so check the assignment rule itself
    # and that results only ever report GPUs from the tuple.
    from execution2.pipeline import worker_gpu
    s = Stage('a', fs.worker_setup, fs.report_gpu, procs=5, gpu=(4, 6))
    assert [worker_gpu(s, k) for k in range(5)] == [4, 6, 4, 6, 4]
    results = list(run([s], inputs(20), queue_maxsize=2))
    assert {r['gpu'] for r in results} <= {4, 6}
    assert all(r['cuda_visible_devices'] == str(r['gpu']) for r in results)


def test_empty_gpu_tuple_is_rejected():
    with pytest.raises(ValueError, match='gpu tuple must not be empty'):
        list(run([Stage('a', fs.worker_setup, fs.tag, gpu=())], inputs(1), queue_maxsize=1))


@pytest.mark.parametrize('savers', [1, 3])
def test_savers_save_every_output_including_fan_out(tmp_path, savers):
    ns = [2, 0, 3, 1]
    stages = [Stage('fan', fs.worker_setup, fs.fan_out, 2, fan_out=True,
                    save=partial(fs.save_json, str(tmp_path)), savers=savers)]
    results = list(run(stages, fan_inputs(ns), queue_maxsize=2))
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f'{r["id"]}-{r["child"]}.json' for r in results)
    assert len(results) == sum(ns)


def test_slow_saving_is_not_counted_in_process_time_and_finishes_before_run_returns(tmp_path):
    # process records its own duration; a saver taking 0.3 s per message must not show up in it,
    # and run must not return until every save has been written.
    stages = [Stage('a', fs.worker_setup, fs.timed_tag, save=partial(fs.slow_save_json, str(tmp_path), 0.3))]
    results = list(run(stages, inputs(6), queue_maxsize=2))
    assert max(r['process_s'] for r in results) < 0.1
    assert len(list(tmp_path.iterdir())) == 6


def test_saver_failure_fails_the_run():
    stages = [Stage('a', fs.worker_setup, fs.tag, save=partial(fs.failing_save, 3))]
    with pytest.raises(PipelineError, match=r'(?s)a\.saver\[0\] failed.*disk full on 3'):
        list(run(stages, inputs(8), queue_maxsize=1))


def test_savers_must_be_positive_when_saving():
    with pytest.raises(ValueError, match='savers must be >= 1'):
        list(run([Stage('a', fs.worker_setup, fs.tag, save=fs.tag, savers=0)], inputs(1), queue_maxsize=1))
