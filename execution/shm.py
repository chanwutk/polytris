"""Shared memory helpers for the pipeline.

Allocator + registry that unlinks any blocks still tracked when the main
process exits (atexit + SIGTERM).  Worker processes only call :func:`attach`
or :func:`attached_view`, never :func:`alloc`, so leaked blocks are always
the main process's responsibility.

All blocks are uint8 (frames and canvases).
"""

from __future__ import annotations

import atexit
import contextlib
import signal
import threading
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Map of name -> SharedMemory for blocks created in this process.
# Used by atexit and SIGTERM to clean up anything still alive on exit.
_registry: dict[str, SharedMemory] = {}
_lock = threading.Lock()


def _cleanup_all() -> None:
    """Unlink every registered block.  Safe to call multiple times."""
    with _lock:
        names = list(_registry.keys())
    for name in names:
        unlink(name)


def _sigterm_handler(signum, frame):
    """Run cleanup on SIGTERM, then re-raise so the default action takes over."""
    _cleanup_all()
    # Restore the default handler and re-raise.
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)


# Install handlers exactly once when the module is imported in the main
# process.  Worker processes inherit the registry empty (each process has its
# own _registry dict), so installing the handler there is harmless.
atexit.register(_cleanup_all)
try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
except (ValueError, OSError):
    # SIGTERM handler can only be installed in the main thread of the main
    # process.  Worker threads that import this module skip the handler.
    pass


def alloc(shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8)) -> tuple[SharedMemory, np.ndarray]:
    """Allocate a new shared memory block and return ``(shm, view)``.

    The block is registered for atexit cleanup.  Caller should call
    :func:`unlink` when the block is no longer needed (which also removes it
    from the registry).
    """
    # Total byte count for the buffer.
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    # SharedMemory creates a uniquely-named POSIX shared block in /dev/shm.
    shm = SharedMemory(create=True, size=nbytes)
    with _lock:
        _registry[shm.name] = shm
    # NumPy view backed by the shared buffer; valid as long as shm is open.
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


def alloc_handoff(shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8)) -> tuple[SharedMemory, np.ndarray]:
    """Allocate a block whose lifecycle is owned by a *different* process.

    Unlike :func:`alloc`, the block is *not* registered for atexit cleanup
    by this process.  After populating the view and emitting a handoff
    message that references the block by name, the creator should call
    ``shm.close()`` to release the local mapping.  The consumer process is
    responsible for unlinking the block.

    Used for canvas blocks: the compress worker creates and fills the canvas,
    hands the reference to detect via a queue, then closes its mapping.
    Detect calls :func:`unlink` after copying the canvas to GPU.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


def attach(name: str, shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8)) -> tuple[SharedMemory, np.ndarray]:
    """Attach to an existing block by name.  Caller must call ``shm.close()``."""
    # SharedMemory(name=...) attaches to the existing block.
    shm = SharedMemory(name=name)
    # NumPy view backed by the same shared buffer.
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


@contextlib.contextmanager
def attached_view(name: str, shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8)):
    """Context manager that attaches, yields the numpy view, then closes."""
    shm, arr = attach(name, shape, dtype)
    try:
        yield arr
    finally:
        # close() releases this process's mapping but does not unlink the
        # block from the OS (only the creator should unlink).
        shm.close()


def unlink(name: str) -> None:
    """Close and unlink the named block.  Idempotent."""
    with _lock:
        shm = _registry.pop(name, None)
    if shm is not None:
        try:
            shm.close()
            shm.unlink()
        except (FileNotFoundError, BufferError):
            # Already unlinked or buffer still referenced; nothing to do.
            pass
    else:
        # Block was not allocated by this process; attempt a best-effort
        # unlink via a transient handle.
        try:
            tmp = SharedMemory(name=name)
            tmp.close()
            tmp.unlink()
        except (FileNotFoundError, BufferError):
            pass
