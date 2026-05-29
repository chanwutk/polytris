"""Unit tests for ``execution.shm``.

Covers alloc + attach round-trip, the attached_view context manager, unlink
idempotency, and the alloc_handoff variant that bypasses the cleanup registry.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from execution import shm as shm_mod


def test_alloc_then_attach_round_trip():
    """A block allocated in one place can be attached by name elsewhere."""
    shape = (4, 8, 3)
    creator_shm, creator_view = shm_mod.alloc(shape=shape, dtype=np.dtype(np.uint8))
    try:
        # Write a recognizable pattern through the creator view.
        pattern = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        creator_view[:] = pattern

        # Attach by name and confirm the contents survive the round-trip.
        att_shm, att_view = shm_mod.attach(creator_shm.name, shape=shape)
        try:
            np.testing.assert_array_equal(att_view, pattern)
        finally:
            att_shm.close()
    finally:
        shm_mod.unlink(creator_shm.name)


def test_attached_view_context_manager_releases_handle():
    """attached_view must close the mmap on exit even when the body raises."""
    shape = (3, 3, 3)
    shm, view = shm_mod.alloc(shape=shape)
    try:
        view[:] = 42
        with shm_mod.attached_view(shm.name, shape) as v:
            np.testing.assert_array_equal(v, 42)
        # If we got here without an OSError, the context manager closed cleanly.
    finally:
        shm_mod.unlink(shm.name)


def test_unlink_is_idempotent():
    """unlink should not raise when called twice or on an unknown name."""
    shm, _ = shm_mod.alloc(shape=(2, 2, 3))
    name = shm.name
    shm_mod.unlink(name)
    # Second call must be safe.
    shm_mod.unlink(name)
    # Third call on a fresh, never-allocated name.
    shm_mod.unlink('nonexistent-block-12345')


def test_alloc_handoff_is_not_in_registry():
    """alloc_handoff allocates a block but skips the atexit registry."""
    shape = (2, 2, 3)
    shm, _ = shm_mod.alloc_handoff(shape=shape)
    try:
        # Internal registry must not track the handoff block.
        assert shm.name not in shm_mod._registry, \
            "alloc_handoff should not register the block"
    finally:
        shm_mod.unlink(shm.name)


def test_alloc_registers_in_registry():
    """alloc should register the block for atexit cleanup."""
    shape = (2, 2, 3)
    shm, _ = shm_mod.alloc(shape=shape)
    try:
        assert shm.name in shm_mod._registry
    finally:
        shm_mod.unlink(shm.name)
        assert shm.name not in shm_mod._registry


def _worker_writes_and_exits(name: str, shape: tuple, value: int):
    """Subprocess target: attach, write a constant, exit."""
    shm, view = shm_mod.attach(name, shape)
    try:
        view[:] = value
    finally:
        shm.close()


def test_cross_process_view():
    """A worker process can attach to a block created in the parent."""
    shape = (4, 4, 3)
    shm, view = shm_mod.alloc(shape=shape)
    try:
        view[:] = 0
        # Spawn a child process to write a recognizable value.
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=_worker_writes_and_exits, args=(shm.name, shape, 77))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0, f"Worker exited with {p.exitcode}"
        # Verify the parent observes the worker's write.
        assert (view == 77).all()
    finally:
        shm_mod.unlink(shm.name)
