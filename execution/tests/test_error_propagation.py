"""Unit tests for ``execution.error_monitor``.

Verifies that a PipelineError on the shared queue is captured and the failed
Event is set; that the monitor exits on the None shutdown sentinel; and that
the first error survives any later errors.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from execution.error_monitor import ErrorMonitor
from execution.messages import PipelineError


def test_monitor_captures_first_error_and_sets_event():
    """A PipelineError sets the failed event and is stored as first_error."""
    error_q: mp.Queue = mp.Queue()
    monitor = ErrorMonitor(error_q)
    monitor.start()

    err = PipelineError(stage='classify', video='va00.mp4', traceback='boom')
    error_q.put(err)

    # Wait briefly for the monitor thread to react.
    assert monitor.failed.wait(timeout=2), "failed event was not set"
    assert monitor.first_error is not None
    assert monitor.first_error.stage == 'classify'
    assert monitor.first_error.video == 'va00.mp4'

    monitor.stop()


def test_monitor_retains_first_error_when_more_arrive():
    """Subsequent errors must not overwrite the first captured one."""
    error_q: mp.Queue = mp.Queue()
    monitor = ErrorMonitor(error_q)
    monitor.start()

    error_q.put(PipelineError(stage='decode', video=None, traceback='first'))
    # Give the monitor a moment to read the first message.
    monitor.failed.wait(timeout=2)
    error_q.put(PipelineError(stage='detect', video='va01.mp4', traceback='second'))
    # Allow the monitor to (potentially) read the second one too.
    time.sleep(0.1)

    assert monitor.first_error is not None
    assert monitor.first_error.stage == 'decode'
    assert monitor.first_error.traceback == 'first'

    monitor.stop()


def test_monitor_exits_on_none_sentinel():
    """Putting None on the error queue cleanly stops the monitor thread."""
    error_q: mp.Queue = mp.Queue()
    monitor = ErrorMonitor(error_q)
    monitor.start()

    monitor.stop()
    # If stop() returned, the monitor thread joined within the timeout.
    assert not monitor._thread.is_alive()
    # No error was reported.
    assert monitor.first_error is None
    assert not monitor.failed.is_set()
