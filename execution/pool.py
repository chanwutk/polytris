"""Process pool spawner with fan-out / fan-in relay threads.

Multi-consumer queues break the simple ``None`` shutdown sentinel pattern:
only one consumer ever sees the single ``None``, leaving the rest blocked.
We solve this with two small relay threads per pool:

- **fan-out relay**: reads from the upstream queue, forwards messages to
  the pool's internal input queue, and on ``None`` broadcasts ``N`` ``None``
  sentinels (one per worker).
- **fan-in relay**: reads from the pool's internal output queue, forwards
  messages to the downstream queue, counts ``N`` ``None`` sentinels (one
  per worker), then sends a single ``None`` downstream.

The result is that callers of :func:`spawn_pool` see a clean
single-producer-single-consumer interface from the outside.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from typing import Any, Callable


def _fan_out_relay(upstream_q, pool_in_q: mp.Queue, num_workers: int) -> None:
    """Forward messages 1:1; broadcast ``None`` to all workers on shutdown."""
    while True:
        msg = upstream_q.get()
        if msg is None:
            for _ in range(num_workers):
                pool_in_q.put(None)
            return
        pool_in_q.put(msg)


def _fan_in_relay(pool_out_q: mp.Queue, downstream_q, num_workers: int) -> None:
    """Forward messages 1:1; collect N ``None``s, emit a single one downstream."""
    nones_seen = 0
    while True:
        msg = pool_out_q.get()
        if msg is None:
            nones_seen += 1
            if nones_seen >= num_workers:
                downstream_q.put(None)
                return
            continue
        downstream_q.put(msg)


def spawn_pool(
    *,
    name: str,
    worker_target: Callable,
    worker_args: tuple,
    num_workers: int,
    upstream_q,
    downstream_q,
) -> tuple[list[mp.Process], mp.Queue, mp.Queue, list[threading.Thread]]:
    """Spawn ``num_workers`` worker processes wired through relay threads.

    ``worker_target`` is called with ``(pool_in_q, pool_out_q, *worker_args)``.
    Returns the list of worker processes, the internal in/out queues, and the
    list of relay threads (so the caller can ``join()`` them on shutdown).
    """
    pool_in_q: mp.Queue = mp.Queue()
    pool_out_q: mp.Queue = mp.Queue()

    workers: list[mp.Process] = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_target,
            args=(pool_in_q, pool_out_q, *worker_args),
            daemon=True,
            name=f'{name}-{i}',
        )
        p.start()
        workers.append(p)

    fan_out_t = threading.Thread(
        target=_fan_out_relay,
        args=(upstream_q, pool_in_q, num_workers),
        daemon=True,
        name=f'{name}-fanout',
    )
    fan_in_t = threading.Thread(
        target=_fan_in_relay,
        args=(pool_out_q, downstream_q, num_workers),
        daemon=True,
        name=f'{name}-fanin',
    )
    fan_out_t.start()
    fan_in_t.start()

    return workers, pool_in_q, pool_out_q, [fan_out_t, fan_in_t]
