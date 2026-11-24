"""Shared-memory block lifetime tests (PLAN.md "Large data")."""

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
