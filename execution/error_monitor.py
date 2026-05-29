"""Error monitor thread.

A single thread in the main process that blocks on ``error_q.get()``.  On
any ``PipelineError`` message, sets a ``threading.Event`` so the main loop
can observe the failure on its next poll and initiate shutdown.

Stores the first error for the main loop to surface as the cause of the
shutdown.
"""

from __future__ import annotations

import threading
import multiprocessing as mp

from execution.messages import PipelineError


class ErrorMonitor:
    def __init__(self, error_q: mp.Queue):
        self.error_q = error_q
        self.failed = threading.Event()
        self.first_error: PipelineError | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='error-monitor')

    def start(self) -> None:
        """Begin monitoring."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the monitor thread to exit (on clean shutdown)."""
        # A literal None on the error queue is the monitor's shutdown sentinel.
        self.error_q.put(None)
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        """Block on error_q; capture the first PipelineError and set the failed event."""
        while True:
            item = self.error_q.get()
            if item is None:
                return
            if isinstance(item, PipelineError):
                with self._lock:
                    if self.first_error is None:
                        self.first_error = item
                # Set the event so the main loop sees the failure on next poll.
                self.failed.set()
            # Other message types on the error queue are ignored.
