"""
Pipeline-parallel engine (PLAN.md "Topology", "Stage contract", "Failure handling").

    caller ─q0→ stage 0 ─q1→ stage 1 ─q2→ ... ─qN→ caller

Each stage runs `procs` worker processes that all read the stage's one input queue.  Every
queue is bounded by `queue_maxsize`, so a slow stage stalls the ones above it.

A stage is two plain functions:
    setup(worker) -> state         once per worker process (load models here)
    process(state, msg) -> msg     once per input message, returns exactly one output

Shutdown: a finished producer sends one STOP per consumer process; a consumer stops after it
has received one STOP per producer process, then sends its own STOPs downstream.  Queues are
FIFO, so every message a producer sent is read before the last of its STOPs.

Failure: a worker that raises reports (stage, message, traceback) and exits 1; the caller
also notices workers that die without reporting (OOM kill, segfault).  On any failure the
caller stops every worker, frees the run's shared-memory blocks, and raises PipelineError.
The engine records no timing; the caller times the results it receives.
"""

import os
import queue
import sys
import threading
import traceback
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import multiprocessing as mp

from execution2 import frames

# Shutdown marker on every queue.
STOP = None


class Worker(NamedTuple):
    """What setup() learns about the process it runs in."""
    stage: str
    index: int
    gpu: int | None


class Stage(NamedTuple):
    name: str
    setup: Callable[[Worker], Any]
    process: Callable[[Any, Any], Any]
    procs: int = 1
    # Physical GPU index; the worker sees it as cuda:0 (CUDA_VISIBLE_DEVICES is set before setup).
    gpu: int | None = None


class PipelineError(RuntimeError):
    """A worker failed or died; the message carries every report received."""


def _describe(msg: Any) -> str:
    """Short, safe description of the message a worker was handling."""
    return 'no message (setup or shutdown)' if msg is None else repr(msg)[:300]


def _worker(stage: Stage, index: int, in_q: mp.Queue, out_q: mp.Queue,
            n_in_stops: int, n_out_stops: int, err_q: mp.Queue) -> None:
    """The one loop every stage process runs."""
    # Pin the GPU before setup imports CUDA, so the worker addresses it as cuda:0.
    if stage.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(stage.gpu)
    msg = None
    try:
        state = stage.setup(Worker(stage.name, index, stage.gpu))
        stops = 0
        while stops < n_in_stops:
            msg = in_q.get()
            # Count STOPs from upstream processes; data keeps flowing until all have finished.
            if msg is STOP:
                stops += 1
                continue
            out = stage.process(state, msg)
            if out is STOP:
                raise RuntimeError(f'{stage.name}.process returned None; every message must produce an output')
            out_q.put(out)
            msg = None
        # Tell every downstream process this one is done.
        for _ in range(n_out_stops):
            out_q.put(STOP)
    except BaseException:
        err_q.put(f'{stage.name}[{index}] failed on {_describe(msg)}:\n{traceback.format_exc()}')
        # Exiting flushes err_q's feeder thread, so the report reaches the caller.
        sys.exit(1)


def _drain(err_q: mp.Queue, wait_s: float) -> list[str]:
    """Every report on err_q, waiting up to wait_s for the first one."""
    reports = []
    try:
        reports.append(err_q.get(timeout=wait_s))
        while True:
            reports.append(err_q.get_nowait())
    except queue.Empty:
        return reports


def _feed(inputs: Iterable, q: mp.Queue, n_stops: int, stopped: threading.Event, errors: list) -> None:
    """Put every input then the STOPs; give up quietly once the run is stopped."""
    try:
        for item in [*inputs, *([STOP] * n_stops)]:
            # Bounded queue: retry until there is room or the run is being torn down.
            while not stopped.is_set():
                try:
                    q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if stopped.is_set():
                return
    except BaseException:
        errors.append(f'input feeder failed:\n{traceback.format_exc()}')


def run(stages: list[Stage], inputs: Iterable, queue_maxsize: int,
        shm_prefix: str | None = None, poll_s: float = 0.2) -> Iterator[Any]:
    """Run inputs through stages; yield each final output as soon as it arrives (in completion order)."""
    if not stages:
        raise ValueError('a pipeline needs at least one stage')
    for s in stages:
        if s.procs < 1:
            raise ValueError(f'stage {s.name}: procs must be >= 1, got {s.procs}')

    ctx = mp.get_context('spawn')
    # queues[i] feeds stage i; queues[-1] returns results to the caller.
    queues = [ctx.Queue(maxsize=queue_maxsize) for _ in range(len(stages) + 1)]
    err_q = ctx.Queue()
    # (process, label) pairs; label matches the prefix of that worker's error report.
    workers: list[tuple[Any, str]] = []
    stopped = threading.Event()
    feeder_errors: list[str] = []
    failed = True
    try:
        for i, stage in enumerate(stages):
            # STOPs expected from upstream (the caller counts as one producer) and owed downstream.
            n_in = 1 if i == 0 else stages[i - 1].procs
            n_out = 1 if i == len(stages) - 1 else stages[i + 1].procs
            for k in range(stage.procs):
                p = ctx.Process(target=_worker, name=f'{stage.name}-{k}', daemon=True,
                                args=(stage, k, queues[i], queues[i + 1], n_in, n_out, err_q))
                p.start()
                workers.append((p, f'{stage.name}[{k}]'))

        # Feed from a thread so the caller can collect results while inputs are still going in.
        feeder = threading.Thread(target=_feed, daemon=True,
                                  args=(inputs, queues[0], stages[0].procs, stopped, feeder_errors))
        feeder.start()

        stops = 0
        while stops < stages[-1].procs:
            _raise_on_failure(err_q, workers, feeder_errors)
            try:
                msg = queues[-1].get(timeout=poll_s)
            except queue.Empty:
                continue
            if msg is STOP:
                stops += 1
                continue
            yield msg

        # Every worker has sent its STOPs; wait for clean exits and re-check.
        for p, _ in workers:
            p.join(timeout=30)
        _raise_on_failure(err_q, workers, feeder_errors)
        if alive := [label for p, label in workers if p.is_alive()]:
            raise PipelineError(f'workers still alive 30 s after the last result: {alive}')
        # A clean run frees every block it creates; anything left is a lifetime bug.
        if shm_prefix is not None and (leaked := frames.leftover(shm_prefix)):
            raise PipelineError(f'{len(leaked)} shared-memory blocks were never freed, e.g. {leaked[0]}')
        failed = False
    finally:
        stopped.set()
        for p, _ in workers:
            if p.is_alive():
                p.terminate()
        for p, _ in workers:
            p.join(timeout=5)
        # Never block interpreter exit on unread queue contents.
        for q in [*queues, err_q]:
            q.cancel_join_thread()
        if failed and shm_prefix is not None:
            frames.unlink_all(shm_prefix)


def _raise_on_failure(err_q: mp.Queue, workers: list[tuple[Any, str]], feeder_errors: list[str]) -> None:
    """Raise PipelineError if any worker reported, died abnormally, or the feeder failed."""
    dead = [(p, label) for p, label in workers if p.exitcode not in (None, 0)]
    # A reporting worker exits right after putting its report; give the report a moment to arrive.
    reports = _drain(err_q, wait_s=2.0 if dead else 0.0)
    if not (reports or dead or feeder_errors):
        return
    lines = [*feeder_errors, *reports]
    # Workers that died without a report (e.g. SIGKILL, segfault).
    for p, label in dead:
        if not any(r.startswith(f'{label} ') for r in reports):
            lines.append(f'{label} died with exit code {p.exitcode} without reporting')
    raise PipelineError('\n\n'.join(lines))
