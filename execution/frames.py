"""
Named shared-memory blocks for frames and canvases (PLAN.md "Large data").

One process creates a block and writes pixels through a numpy view; others attach by the
small Block handle that travels through queues; the last user unlinks it.  Nothing large
is pickled.

Every block name starts with the run's prefix, so main can sweep a failed run's leftovers
with unlink_all(prefix).  Blocks are created with track=False: their lifetime is managed
explicitly here, not by multiprocessing's resource tracker (which would unlink them when
the creating process exits, while downstream stages still need them).

Usage:
    with frames.create(prefix, shape, 'uint8') as m:
        m.array[...] = pixels          # use m.array only inside the block
    queue.put(m.block)                  # the handle outlives the context
    with frames.attach(block) as m:
        result = process(m.array)       # copy anything that must outlive the context
    frames.unlink(block)                # last user frees it

Pools (execution.main): a run can instead create, before anything is timed, a fixed set of
equal-size slots per kind of block, with every page already allocated.  A stage then takes a
slot (acquire) instead of creating a block and gives it back (release) instead of unlinking it,
and each process keeps its mapping of a slot for its whole life.  So the timed run allocates no
shared memory and takes no page faults on it, which on a host short of free memory (ace, whose
ZFS cache holds most of RAM) cost the stages minutes of kernel time per run.  A pooled Block says
so (pooled=True); attach then uses the process's lasting mapping.

Operators take a store, either a run prefix (a new block each time) or a Pool (a reused slot),
and use new_block / free, so the same code serves both.
"""

import glob
import os
import queue
import sys
import uuid
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any, Iterator, NamedTuple

import multiprocessing as mp
import numpy as np

SHM_DIR = '/dev/shm'
# madvise advice that allocates every page of a mapping for writing in one call (Linux 5.14+);
# Python's mmap module does not define the constant.
MADV_POPULATE_WRITE = 23
# A pool holds one slot per video that can be in flight, so acquire never waits unless a slot was
# never released; this bounds that wait with a clear error instead of a hang.
ACQUIRE_TIMEOUT_S = 60.0


class Block(NamedTuple):
    """Picklable handle to a shared-memory array."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    # True for a pool slot: attach maps it once per process and keeps the mapping.
    pooled: bool = False


class Pool(NamedTuple):
    """Equal-size pre-allocated slots and the queue of free slot names.

    Picklable only when handed to a process at start (it holds a multiprocessing queue), as stage
    setups are; it never travels through a pipeline queue (blocks carry only the slot name).
    """
    kind: str                   # 'frames' or 'canvases', for error messages
    slots: tuple[str, ...]
    slot_bytes: int
    free: Any                   # multiprocessing Queue of free slot names (spawn context)


class Stores(NamedTuple):
    """Where each kind of block comes from: a run prefix (a new block per video) or a Pool."""
    frames: 'str | Pool'
    canvases: 'str | Pool'


class Mapped:
    """An open block: .block is the handle, .array the numpy view (valid only inside the context)."""

    def __init__(self, block: Block, shm: shared_memory.SharedMemory, lasting: bool = False):
        self.block = block
        self._shm = shm
        # A lasting mapping (a pool slot) stays open after the context; only the view is dropped.
        self._lasting = lasting
        # numpy arrays over shm.buf reference the underlying mmap object but hold no buffer export,
        # so shm.close() would succeed and unmap memory under a surviving view (a segfault on its
        # next use).  Survivors are detected instead by the mmap's reference count, against this
        # baseline taken before any view exists.  For a slot, a survivor would silently read the
        # next video's pixels, so the same check applies.
        self._baseline_refs = sys.getrefcount(shm._mmap)
        # A slot may be larger than the block: the view covers its first nbytes.
        self.array: np.ndarray = np.ndarray(block.shape, dtype=np.dtype(block.dtype), buffer=shm.buf)

    def _close(self) -> None:
        # Drop our view first; any remaining reference to the mmap is a view that escaped.
        del self.array
        if sys.getrefcount(self._shm._mmap) > self._baseline_refs:
            raise BufferError(f'a view of shared block {self.block.name} outlived its context; '
                              f'copy the data instead of keeping a reference')
        if not self._lasting:
            self._shm.close()


def new_prefix() -> str:
    """A run-unique name prefix; pass it to every create() of the run and to unlink_all()."""
    return f'tetris-{uuid.uuid4().hex[:12]}-'


@contextmanager
def create(prefix: str, shape: tuple[int, ...], dtype: str | np.dtype) -> Iterator[Mapped]:
    """Create a block and yield it open for writing.  The block outlives the context; unlink() frees it."""
    block = Block(name=f'{prefix}{uuid.uuid4().hex}', shape=tuple(int(s) for s in shape), dtype=np.dtype(dtype).str)
    # SharedMemory rejects size 0 (e.g. a video with zero collages), so allocate at least one byte.
    nbytes = max(_nbytes(block.shape, dtype), 1)
    mapped = Mapped(block, shared_memory.SharedMemory(name=block.name, create=True, size=nbytes, track=False))
    try:
        yield mapped
    except BaseException:
        # A failed writer leaves nothing behind.
        del mapped.array
        mapped._shm.close()
        unlink(block)
        raise
    mapped._close()


@contextmanager
def attach(block: Block) -> Iterator[Mapped]:
    """Yield an existing block open for reading and writing (a slot through this process's lasting mapping)."""
    if block.pooled:
        mapped = Mapped(block, _slot(block.name), lasting=True)
    else:
        mapped = Mapped(block, shared_memory.SharedMemory(name=block.name, track=False))
    try:
        yield mapped
    finally:
        mapped._close()


# ---------------------------------------------------------------- pools

# This process's mappings of pool slots, opened on first use and kept until the process exits.
_slots: dict[str, shared_memory.SharedMemory] = {}


def _slot(name: str) -> shared_memory.SharedMemory:
    """The lasting mapping of slot `name` in this process."""
    shm = _slots.get(name)
    if shm is None:
        shm = _slots[name] = shared_memory.SharedMemory(name=name, track=False)
    return shm


def _nbytes(shape: tuple[int, ...], dtype: str | np.dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def create_pool(prefix: str, kind: str, count: int, slot_bytes: int) -> Pool:
    """count slots of slot_bytes each, every page allocated now, all free; names start with prefix.

    unlink_all(prefix) frees them.  The free queue uses the spawn context, as the pipeline does.
    """
    if count < 1:
        raise ValueError(f'a {kind} pool needs at least one slot, got {count}')
    names = tuple(f'{prefix}{kind}-{i}' for i in range(count))
    free = mp.get_context('spawn').Queue()
    for name in names:
        # SharedMemory rejects size 0, so a slot has at least one byte.
        shm = shared_memory.SharedMemory(name=name, create=True, size=max(slot_bytes, 1), track=False)
        try:
            # Allocate every page now, outside the timed run; the pages stay with the file after close.
            shm._mmap.madvise(MADV_POPULATE_WRITE)
        finally:
            shm.close()
        free.put(name)
    return Pool(kind, names, max(slot_bytes, 1), free)


@contextmanager
def acquire(pool: Pool, shape: tuple[int, ...], dtype: str | np.dtype) -> Iterator[Mapped]:
    """Take a free slot and yield it open for writing as a block of shape.  release() gives it back.

    Raises ValueError when the block does not fit a slot, and RuntimeError when no slot frees up
    within ACQUIRE_TIMEOUT_S (a slot that was never released).
    """
    shape = tuple(int(s) for s in shape)
    nbytes = _nbytes(shape, dtype)
    if nbytes > pool.slot_bytes:
        raise ValueError(f'a {pool.kind} block of shape {shape} needs {nbytes} bytes, but the pool\'s slots '
                         f'hold {pool.slot_bytes}')
    try:
        name = pool.free.get(timeout=ACQUIRE_TIMEOUT_S)
    except queue.Empty:
        raise RuntimeError(f'no free {pool.kind} slot after {ACQUIRE_TIMEOUT_S:.0f} s: '
                           f'a slot was never released') from None
    mapped = Mapped(Block(name, shape, np.dtype(dtype).str, pooled=True), _slot(name), lasting=True)
    try:
        yield mapped
    except BaseException:
        # A failed writer gives its slot back.
        del mapped.array
        release(pool, mapped.block)
        raise
    mapped._close()


def release(pool: Pool, block: Block) -> None:
    """Give a block's slot back to its pool.  Releasing a slot twice is caught by check_returned."""
    if not block.pooled or block.name not in pool.slots:
        raise ValueError(f'block {block.name} is not a slot of the {pool.kind} pool')
    pool.free.put(block.name)


def check_returned(pool: Pool, wait_s: float = 5.0) -> None:
    """After a run: every slot is free exactly once.  Drains the free queue (the pool is done after this)."""
    names = []
    try:
        for _ in pool.slots:
            names.append(pool.free.get(timeout=wait_s))
    except queue.Empty:
        missing = sorted(set(pool.slots) - set(names))
        raise RuntimeError(f'{len(missing)} {pool.kind} slots were never released, e.g. {missing[0]}') from None
    # A slot released twice is in the queue twice: one more name is waiting.
    try:
        extra = pool.free.get(timeout=0.5)
    except queue.Empty:
        extra = None
    if extra is not None or sorted(names) != sorted(pool.slots):
        raise RuntimeError(f'a {pool.kind} slot was released more than once '
                           f'(e.g. {extra or next(n for n in names if names.count(n) > 1)})')


@contextmanager
def new_block(store: 'str | Pool', shape: tuple[int, ...], dtype: str | np.dtype) -> Iterator[Mapped]:
    """create() from a run prefix, or acquire() from a Pool."""
    with (acquire(store, shape, dtype) if isinstance(store, Pool) else create(store, shape, dtype)) as m:
        yield m


def free(store: 'str | Pool', block: Block) -> None:
    """The last user's free: unlink() a prefix's block, release() a Pool's slot."""
    if isinstance(store, Pool):
        release(store, block)
    else:
        unlink(block)


def unlink(block: Block) -> None:
    """Free a block.  Safe to call on an already-freed block (cleanup may race a normal free)."""
    try:
        os.unlink(os.path.join(SHM_DIR, block.name))
    except FileNotFoundError:
        pass


def unlink_all(prefix: str) -> int:
    """Free every block whose name starts with prefix; returns how many were freed."""
    freed = 0
    for name in leftover(prefix):
        try:
            os.unlink(os.path.join(SHM_DIR, name))
            freed += 1
        except FileNotFoundError:
            pass
    return freed


def leftover(prefix: str) -> list[str]:
    """Names of blocks still present for prefix (used by tests and by main's exit check)."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(SHM_DIR, f'{glob.escape(prefix)}*')))
