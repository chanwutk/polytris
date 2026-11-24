"""Fake stages for profiling's frame-freeing tests; top-level so spawned workers can import them."""

import numpy as np

from execution import frames


def make_video_block(prefix, n_children, state, msg):
    """Like decode: one shared block per video holding the video id, plus how many children it will have."""
    with frames.create(prefix, (8,), 'uint8') as m:
        m.array[:] = msg['id']
    return {**msg, 'block': m.block, 'n': n_children}


def fan_children(state, msg):
    """Like classify/prune: one child per configuration of the video."""
    return [{**msg, 'child': i} for i in range(msg['n'])]


def read_then_finish(completion, state, msg):
    """Like pack: read the video's block, then free it if this was the video's last child."""
    with frames.attach(msg['block']) as m:
        value = int(m.array.max())
    last = completion.finish(str(msg['id']), msg['n'])
    if last:
        frames.unlink(msg['block'])
    return {'id': msg['id'], 'child': msg['child'], 'value': value, 'freed': last}


def emit_results(n, state, video):
    """Stand-in for the whole sweep: n Evaluated results for the video."""
    from profiling.messages import Branch, Evaluated, StageTiming
    return [Evaluated(video, Branch().with_(classifier='F', tile_size=10, sample_rate=i + 1),
                      (StageTiming('detect', 0.0, 1.0, 1.0),), {'HOTA': {'HOTA': np.array([0.5, 0.7])}})
            for i in range(n)]


def fake_sweep_build(cfg, prefix, completion):
    """build_stages replacement: one fan-out stage emitting expected_results(cfg) per video."""
    from functools import partial
    from execution.pipeline import Stage
    from profiling.stages import expected_results
    from tests.execution.fake_stages import worker_setup
    return [Stage('emit', worker_setup, partial(emit_results, expected_results(cfg)), 2, fan_out=True)]
