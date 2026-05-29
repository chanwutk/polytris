"""Unit tests for ``execution.pool``: fan-out / fan-in relays and spawn_pool.

Uses tiny worker functions that echo input + 1 so we can assert message flow
end-to-end without involving any heavy stage code.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import threading
import time

import pytest

from execution import pool as pool_mod


def test_fan_out_relay_broadcasts_none():
    """fan_out_relay forwards messages and broadcasts num_workers Nones."""
    upstream: _queue.Queue = _queue.Queue()
    pool_in: _queue.Queue = _queue.Queue()
    num_workers = 4

    # Run the relay in a thread.
    t = threading.Thread(
        target=pool_mod._fan_out_relay,
        args=(upstream, pool_in, num_workers),
        daemon=True,
    )
    t.start()

    # Send a few payloads followed by the shutdown sentinel.
    upstream.put('a')
    upstream.put('b')
    upstream.put(None)

    # Expect a, b, and then 4 Nones in the pool_in queue.
    assert pool_in.get(timeout=2) == 'a'
    assert pool_in.get(timeout=2) == 'b'
    for _ in range(num_workers):
        assert pool_in.get(timeout=2) is None
    # Relay thread should have exited.
    t.join(timeout=2)
    assert not t.is_alive()


def test_fan_in_relay_counts_and_emits_single_none():
    """fan_in_relay forwards messages and emits a single None after N Nones."""
    pool_out: _queue.Queue = _queue.Queue()
    downstream: _queue.Queue = _queue.Queue()
    num_workers = 3

    t = threading.Thread(
        target=pool_mod._fan_in_relay,
        args=(pool_out, downstream, num_workers),
        daemon=True,
    )
    t.start()

    pool_out.put('msg1')
    pool_out.put(None)        # 1 of 3 workers done
    pool_out.put('msg2')
    pool_out.put(None)        # 2 of 3
    pool_out.put(None)        # 3 of 3 -> should emit a single None downstream

    assert downstream.get(timeout=2) == 'msg1'
    assert downstream.get(timeout=2) == 'msg2'
    assert downstream.get(timeout=2) is None
    t.join(timeout=2)
    assert not t.is_alive()


def _echo_plus_one_worker(in_q: mp.Queue, out_q: mp.Queue):
    """Tiny worker: pull ints, emit (n + 1), exit on None."""
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            return
        out_q.put(item + 1)


def test_spawn_pool_round_trip():
    """spawn_pool wires upstream -> N workers -> downstream cleanly."""
    upstream: _queue.Queue = _queue.Queue()
    downstream: _queue.Queue = _queue.Queue()

    workers, _, _, relays = pool_mod.spawn_pool(
        name='test',
        worker_target=_echo_plus_one_worker,
        worker_args=(),
        num_workers=2,
        upstream_q=upstream,
        downstream_q=downstream,
    )

    # Send three messages plus shutdown.
    for v in [10, 20, 30]:
        upstream.put(v)
    upstream.put(None)

    # Collect downstream until we see the single shutdown sentinel.
    seen: list[int] = []
    while True:
        msg = downstream.get(timeout=10)
        if msg is None:
            break
        seen.append(msg)
    assert sorted(seen) == [11, 21, 31]

    # Workers + relays should all exit cleanly.
    for w in workers:
        w.join(timeout=5)
        assert w.exitcode == 0
    for r in relays:
        r.join(timeout=5)
        assert not r.is_alive()
