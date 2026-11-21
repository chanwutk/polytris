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
"""

import glob
import os
import sys
import uuid
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Iterator, NamedTuple

import numpy as np

SHM_DIR = '/dev/shm'


class Block(NamedTuple):
    """Picklable handle to a shared-memory array."""
    name: str
    shape: tuple[int, ...]
    dtype: str


class Mapped:
    """An open block: .block is the handle, .array the numpy view (valid only inside the context)."""

    def __init__(self, block: Block, shm: shared_memory.SharedMemory):
        self.block = block
        self._shm = shm
        # numpy arrays over shm.buf reference the underlying mmap object but hold no buffer export,
        # so shm.close() would succeed and unmap memory under a surviving view (a segfault on its
        # next use).  Survivors are detected instead by the mmap's reference count, against this
        # baseline taken before any view exists.
        self._baseline_refs = sys.getrefcount(shm._mmap)
        self.array: np.ndarray | None = np.ndarray(block.shape, dtype=np.dtype(block.dtype), buffer=shm.buf)

    def _close(self) -> None:
        # Drop our view first; any remaining reference to the mmap is a view that escaped.
        self.array = None
        if sys.getrefcount(self._shm._mmap) > self._baseline_refs:
            raise BufferError(f'a view of shared block {self.block.name} outlived its context; '
                              f'copy the data instead of keeping a reference')
        self._shm.close()


def new_prefix() -> str:
    """A run-unique name prefix; pass it to every create() of the run and to unlink_all()."""
    return f'tetris-{uuid.uuid4().hex[:12]}-'


@contextmanager
def create(prefix: str, shape: tuple[int, ...], dtype: str | np.dtype) -> Iterator[Mapped]:
    """Create a block and yield it open for writing.  The block outlives the context; unlink() frees it."""
    block = Block(name=f'{prefix}{uuid.uuid4().hex}', shape=tuple(int(s) for s in shape), dtype=np.dtype(dtype).str)
    # SharedMemory rejects size 0 (e.g. a video with zero collages), so allocate at least one byte.
    nbytes = max(int(np.prod(block.shape)) * np.dtype(dtype).itemsize, 1)
    mapped = Mapped(block, shared_memory.SharedMemory(name=block.name, create=True, size=nbytes, track=False))
    try:
        yield mapped
    except BaseException:
        # A failed writer leaves nothing behind.
        mapped.array = None
        mapped._shm.close()
        unlink(block)
        raise
    mapped._close()


@contextmanager
def attach(block: Block) -> Iterator[Mapped]:
    """Yield an existing block open for reading and writing."""
    mapped = Mapped(block, shared_memory.SharedMemory(name=block.name, track=False))
    try:
        yield mapped
    finally:
        mapped._close()


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
