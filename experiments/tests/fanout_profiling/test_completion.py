"""VideoCompletion: frames freed exactly once, at a video's last child, across workers (experiments/fanout_profiling/PLAN.md 13A)."""

import multiprocessing as mp
from collections import Counter
from functools import partial

import pytest

from execution import frames
from execution.pipeline import PipelineError, Stage, run
from experiments.fanout_profiling.completion import VideoCompletion
from tests.execution import fake_stages as efs
from experiments.tests.fanout_profiling import fake_stages as pfs


def inputs(n):
    return [{'id': i, 'path': []} for i in range(n)]


@pytest.fixture
def manager():
    m = mp.get_context('spawn').Manager()
    yield m
    m.shutdown()


def test_finish_counts_to_expected_once(manager):
    c = VideoCompletion(manager)
    assert [c.finish('a', 3) for _ in range(3)] == [False, False, True]
    assert c.pending() == {}
    # A video that reappears starts over (its entry was removed).
    assert c.finish('a', 1) is True


def test_interleaved_videos_are_counted_separately(manager):
    c = VideoCompletion(manager)
    assert [c.finish(v, 2) for v in ['a', 'b', 'a', 'c', 'b']] == [False, False, True, False, True]
    assert c.pending() == {'c': 1}


def test_finishing_too_many_is_an_error(manager):
    c = VideoCompletion(manager)
    c.finish('a', 1)
    c2 = VideoCompletion(manager)
    c2.finish('b', 1)
    with pytest.raises(ValueError, match='expected must be >= 1'):
        c.finish('a', 0)


@pytest.mark.parametrize('pack_procs', [1, 4])
def test_blocks_are_freed_exactly_once_after_every_child(manager, pack_procs):
    # Every child attaches the video's block before finishing; if the block were freed early, a later
    # child's attach would fail and the run would raise.
    prefix = frames.new_prefix()
    completion = VideoCompletion(manager)
    n_children = 7
    stages = [Stage('decode', efs.worker_setup, partial(pfs.make_video_block, prefix, n_children), 2),
              Stage('fan', efs.worker_setup, pfs.fan_children, 2, fan_out=True),
              Stage('pack', efs.worker_setup, partial(pfs.read_then_finish, completion), pack_procs)]
    results = list(run(stages, inputs(12), queue_maxsize=2, shm_prefix=prefix))
    assert len(results) == 12 * n_children
    assert all(r['value'] == r['id'] for r in results)
    assert Counter(r['id'] for r in results if r['freed']) == {i: 1 for i in range(12)}
    assert frames.leftover(prefix) == [] and completion.pending() == {}


def test_a_crash_mid_video_leaves_no_blocks(manager):
    prefix = frames.new_prefix()
    completion = VideoCompletion(manager)
    stages = [Stage('decode', efs.worker_setup, partial(pfs.make_video_block, prefix, 5), 1),
              Stage('fan', efs.worker_setup, pfs.fan_children, 1, fan_out=True),
              Stage('pack', efs.worker_setup, partial(efs.kill_on, 3), 2)]
    with pytest.raises(PipelineError):
        list(run(stages, inputs(6), queue_maxsize=1, shm_prefix=prefix))
    assert frames.leftover(prefix) == []
