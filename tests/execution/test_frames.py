"""Shared-memory block and pool lifetime tests (PLAN.md "Large data")."""

import os
import threading
import time

import numpy as np
import pytest

from execution import frames


@pytest.fixture
def prefix():
    p = frames.new_prefix()
    yield p
    frames.unlink_all(p)


def test_create_attach_unlink_round_trip(prefix):
    data = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    with frames.create(prefix, data.shape, 'uint8') as m:
        m.array[:] = data
    block = m.block
    assert frames.leftover(prefix) == [block.name]
    with frames.attach(block) as m2:
        assert m2.array.dtype == np.uint8 and np.array_equal(m2.array, data)
    frames.unlink(block)
    assert frames.leftover(prefix) == []


def test_block_handle_is_small_and_plain(prefix):
    with frames.create(prefix, (1080, 1920, 3), np.uint8) as m:
        pass
    assert m.block.shape == (1080, 1920, 3) and m.block.dtype == '|u1' and m.block.name.startswith(prefix)


def test_zero_sized_block_is_allowed(prefix):
    with frames.create(prefix, (0, 60, 60, 3), 'uint8') as m:
        assert m.array.shape == (0, 60, 60, 3)
    with frames.attach(m.block) as m2:
        assert m2.array.size == 0


@pytest.mark.parametrize('opener', ['create', 'attach'])
def test_view_escaping_its_context_is_a_clear_error(prefix, opener):
    with frames.create(prefix, (4,), 'uint8') as m:
        block = m.block
    ctx = frames.create(prefix, (4,), 'uint8') if opener == 'create' else frames.attach(block)
    with pytest.raises(BufferError, match='outlived its context'):
        with ctx as m2:
            kept = m2.array[1:]  # a derived view, kept alive past the context on purpose
    # The memory is still mapped, so the escaped view stays safe to read.
    assert kept.tolist() == [0, 0, 0]


def test_dropping_views_inside_the_context_closes_cleanly(prefix):
    with frames.create(prefix, (4,), 'uint8') as m:
        total = int(m.array.sum())  # plain values, no views, survive the context
    assert total == 0


def test_failed_writer_leaves_no_block(prefix):
    with pytest.raises(RuntimeError):
        with frames.create(prefix, (4,), 'uint8') as m:
            m.array[:] = 1
            raise RuntimeError('decode failed')
    assert frames.leftover(prefix) == []


def test_unlink_is_idempotent(prefix):
    with frames.create(prefix, (4,), 'uint8') as m:
        pass
    frames.unlink(m.block)
    frames.unlink(m.block)
    assert frames.leftover(prefix) == []


def test_attach_to_freed_block_fails(prefix):
    with frames.create(prefix, (4,), 'uint8') as m:
        pass
    frames.unlink(m.block)
    with pytest.raises(FileNotFoundError):
        with frames.attach(m.block):
            pass


def test_unlink_all_frees_only_its_prefix(prefix):
    other = frames.new_prefix()
    try:
        for _ in range(3):
            with frames.create(prefix, (4,), 'uint8'):
                pass
        with frames.create(other, (4,), 'uint8'):
            pass
        assert frames.unlink_all(prefix) == 3
        assert frames.leftover(prefix) == [] and len(frames.leftover(other)) == 1
    finally:
        frames.unlink_all(other)


# ---------------------------------------------------------------- pools

def test_pool_slots_are_allocated_up_front_and_named_under_the_prefix(prefix):
    pool = frames.create_pool(prefix, 'frames', 3, 64 * 1024)
    assert pool.slots == tuple(f'{prefix}frames-{i}' for i in range(3)) and pool.slot_bytes == 64 * 1024
    assert frames.leftover(prefix) == sorted(pool.slots)
    # Every page is allocated now (tmpfs counts allocated 512-byte blocks), not on first use in the timed run.
    for name in pool.slots:
        assert os.stat(os.path.join(frames.SHM_DIR, name)).st_blocks * 512 >= pool.slot_bytes
    frames.check_returned(pool)
    assert frames.unlink_all(prefix) == 3


def test_a_pool_needs_a_slot(prefix):
    with pytest.raises(ValueError, match='at least one slot'):
        frames.create_pool(prefix, 'frames', 0, 16)


def test_acquire_attach_release_round_trip_through_a_smaller_block(prefix):
    pool = frames.create_pool(prefix, 'frames', 1, 100)
    data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    with frames.acquire(pool, data.shape, 'uint8') as m:
        m.array[:] = data
    block = m.block
    assert block.pooled and block.name == pool.slots[0] and block.shape == (2, 3, 4)
    with frames.attach(block) as m2:
        assert np.array_equal(m2.array, data)
    frames.release(pool, block)
    frames.check_returned(pool)


def test_a_released_slot_is_reused_and_keeps_the_previous_pixels_until_overwritten(prefix):
    # Writers of a reused slot must set every byte they hand on (pack clears its canvases for this reason).
    pool = frames.create_pool(prefix, 'canvases', 1, 16)
    with frames.acquire(pool, (16,), 'uint8') as m:
        m.array[:] = 7
    frames.release(pool, m.block)
    with frames.acquire(pool, (8,), 'uint8') as m2:
        assert m2.block.name == m.block.name and m2.array.tolist() == [7] * 8
    frames.release(pool, m2.block)


def test_a_block_larger_than_a_slot_is_a_clear_error_and_takes_no_slot(prefix):
    pool = frames.create_pool(prefix, 'canvases', 1, 10)
    with pytest.raises(ValueError, match=r'needs 12 bytes, but the pool\'s slots hold 10'):
        with frames.acquire(pool, (3, 4), 'uint8'):
            pass
    frames.check_returned(pool)


def test_acquire_waits_for_a_release(prefix):
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    with frames.acquire(pool, (8,), 'uint8') as m:
        pass
    releaser = threading.Timer(0.3, frames.release, args=(pool, m.block))
    releaser.start()
    start = time.monotonic()
    with frames.acquire(pool, (8,), 'uint8') as m2:
        pass
    assert time.monotonic() - start >= 0.25 and m2.block.name == m.block.name
    frames.release(pool, m2.block)
    releaser.join()


def test_acquire_gives_up_with_a_clear_error_when_no_slot_is_released(prefix, monkeypatch):
    monkeypatch.setattr(frames, 'ACQUIRE_TIMEOUT_S', 0.2)
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    with frames.acquire(pool, (8,), 'uint8'):
        pass
    with pytest.raises(RuntimeError, match='no free frames slot after 0 s: a slot was never released'):
        with frames.acquire(pool, (8,), 'uint8'):
            pass


def test_a_failed_writer_gives_its_slot_back(prefix):
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    with pytest.raises(RuntimeError, match='decode failed'):
        with frames.acquire(pool, (8,), 'uint8') as m:
            m.array[:] = 1
            raise RuntimeError('decode failed')
    frames.check_returned(pool)


@pytest.mark.parametrize('opener', ['acquire', 'attach'])
def test_a_view_escaping_a_slot_context_is_a_clear_error(prefix, opener):
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    with frames.acquire(pool, (4,), 'uint8') as m:
        m.array[:] = 3
    ctx = frames.acquire(pool, (4,), 'uint8') if opener == 'acquire' else frames.attach(m.block)
    if opener == 'acquire':
        frames.release(pool, m.block)
    with pytest.raises(BufferError, match='outlived its context'):
        with ctx as m2:
            kept = m2.array[1:]
    # The slot stays mapped in this process, so the escaped view is still safe to read.
    assert kept.tolist() == [3, 3, 3]


def test_attach_reuses_the_processs_mapping_of_a_slot(prefix):
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    with frames.acquire(pool, (8,), 'uint8') as m:
        pass
    with frames.attach(m.block):
        pass
    with frames.attach(m.block):
        pass
    # One lasting mapping per slot, kept open after the contexts.
    assert m.block.name in frames._slots and not frames._slots[m.block.name]._mmap.closed
    frames.release(pool, m.block)


def test_release_refuses_a_block_that_is_not_one_of_the_pools_slots(prefix):
    pool = frames.create_pool(prefix, 'frames', 1, 8)
    other = frames.create_pool(prefix, 'canvases', 1, 8)
    with frames.create(prefix, (4,), 'uint8') as plain:
        pass
    with frames.acquire(other, (4,), 'uint8') as foreign:
        pass
    for block in (plain.block, foreign.block):
        with pytest.raises(ValueError, match='is not a slot of the frames pool'):
            frames.release(pool, block)


def test_check_returned_catches_a_slot_never_released(prefix):
    pool = frames.create_pool(prefix, 'frames', 2, 8)
    with frames.acquire(pool, (8,), 'uint8') as m:
        pass
    with pytest.raises(RuntimeError, match=f'1 frames slots were never released, e.g. {m.block.name}'):
        frames.check_returned(pool, wait_s=0.2)


def test_check_returned_catches_a_slot_released_twice(prefix):
    pool = frames.create_pool(prefix, 'frames', 2, 8)
    with frames.acquire(pool, (8,), 'uint8') as m:
        pass
    frames.release(pool, m.block)
    frames.release(pool, m.block)
    with pytest.raises(RuntimeError, match='released more than once'):
        frames.check_returned(pool, wait_s=0.2)


@pytest.mark.parametrize('pooled', [False, True])
def test_new_block_and_free_serve_a_prefix_or_a_pool(prefix, pooled):
    store = frames.create_pool(prefix, 'frames', 1, 8) if pooled else prefix
    with frames.new_block(store, (8,), 'uint8') as m:
        m.array[:] = 5
    assert m.block.pooled is pooled
    with frames.attach(m.block) as m2:
        assert m2.array.tolist() == [5] * 8
    frames.free(store, m.block)
    if pooled:
        frames.check_returned(store)
    else:
        assert frames.leftover(prefix) == []
