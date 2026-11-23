"""Pure-Python stages for engine tests; top-level so spawned workers can import them."""

import os
import signal
import time

import numpy as np

from execution2 import frames


def worker_setup(worker):
    """State is just the worker's identity."""
    return worker


def failing_setup(worker):
    raise RuntimeError(f'setup exploded in {worker.stage}')


def tag(state, msg):
    """Append (stage, worker index) to the message's path."""
    return {**msg, 'path': [*msg['path'], (state.stage, state.index)]}


def fail_on(target, state, msg):
    """Raise on one specific input; tag everything else."""
    if msg['id'] == target:
        raise ValueError(f'bad input {target}')
    return tag(state, msg)


def kill_on(target, state, msg):
    """SIGKILL this worker on one specific input (no report, like an OOM kill)."""
    if msg['id'] == target:
        os.kill(os.getpid(), signal.SIGKILL)
    return tag(state, msg)


def return_none(state, msg):
    return None


def report_gpu(state, msg):
    """Record what the worker sees as its GPU."""
    return {**msg, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': state.gpu}


def count_then_tag(counter, state, msg):
    """Increment a shared counter as each message is processed."""
    with counter.get_lock():
        counter.value += 1
    return tag(state, msg)


def slow_count_then_tag(counter, delay_s, state, msg):
    """Like count_then_tag, after sleeping delay_s (a slow downstream stage)."""
    time.sleep(delay_s)
    return count_then_tag(counter, state, msg)


def make_block(prefix, state, msg):
    """Write the message id into a new shared block and pass the handle on."""
    with frames.create(prefix, (4, 4), 'uint8') as m:
        m.array[:] = msg['id']
    return {**msg, 'block': m.block}


def read_and_free_block(state, msg):
    """Read the block back, free it, and report the value seen."""
    with frames.attach(msg['block']) as m:
        value = int(m.array.max())
    frames.unlink(msg['block'])
    return {**msg, 'value': value}


def read_and_keep_block(state, msg):
    """Read the block back but forget to free it (a lifetime bug)."""
    with frames.attach(msg['block']) as m:
        value = int(m.array.max())
    return {**msg, 'value': value}


def make_block_then_fail(prefix, target, state, msg):
    """Create a block, then fail on one input after the block exists."""
    out = make_block(prefix, state, msg)
    if msg['id'] == target:
        raise ValueError('failed after creating a block')
    return out


def sleep_then_tag(delay_s, state, msg):
    time.sleep(delay_s)
    return tag(state, msg)


def np_payload(state, msg):
    """Return a numpy array, to check ordinary pickled payloads pass through."""
    return {**msg, 'array': np.full(3, msg['id'])}


def fake_tracked(state, video):
    """Stand-in for the whole pipeline: one track per video, tagged with the video's frame count."""
    from execution2.stages import Tracked
    import cv2
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return Tracked(video, {0: [[1.0, 0.0, 0.0, 10.0, float(n)]]})


def fake_build(cfg, prefix):
    """build_stages replacement for main tests: one two-process stage, no models."""
    from execution2.pipeline import Stage
    return [Stage('fake', worker_setup, fake_tracked, 2)]


def fan_out(state, msg):
    """One output per child, msg['n'] of them (possibly none), each tagged with its child index."""
    return [{**tag(state, msg), 'child': i} for i in range(msg['n'])]


def timed_tag(state, msg):
    """Tag, recording how long process itself took (start and end inside process)."""
    start = time.perf_counter()
    out = tag(state, msg)
    return {**out, 'process_s': time.perf_counter() - start}


def save_json(directory, msg):
    """Saver: one JSON file per message, named by id (and child when present)."""
    import json
    name = f"{msg['id']}" + (f"-{msg['child']}" if 'child' in msg else '')
    with open(os.path.join(directory, f'{name}.json'), 'w') as f:
        json.dump({k: v for k, v in msg.items() if k != 'path'}, f)


def slow_save_json(directory, delay_s, msg):
    """Saver that takes delay_s per message before writing."""
    time.sleep(delay_s)
    save_json(directory, msg)


def failing_save(target, msg):
    """Saver that raises on one message."""
    if msg['id'] == target:
        raise OSError(f'disk full on {target}')


def none_in_fan_out(state, msg):
    """A fan-out stage that wrongly emits None among its outputs."""
    return [tag(state, msg), None]
