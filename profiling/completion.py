"""
Count finished items per video across processes (profiling/PLAN.md 3'A).

Pack frees a video's frames block when the last of that video's pack inputs is done.  Pack runs several
worker processes and videos interleave, so the count lives in a Manager dict (video -> finished so far)
guarded by a Manager lock that every worker shares.  The call that brings a video to its expected count
gets True, exactly once, and the entry is removed.

The Manager must outlive every process that uses the counter (main creates it before the pipeline and
shuts it down after).
"""

from multiprocessing.managers import SyncManager


class VideoCompletion:
    """Picklable (its parts are Manager proxies), so it can be bound into a stage's setup or process."""

    def __init__(self, manager: SyncManager):
        self._counts = manager.dict()
        self._lock = manager.Lock()

    def finish(self, video: str, expected: int) -> bool:
        """Record one finished item of `video`; True only for the call that reaches `expected`."""
        if expected < 1:
            raise ValueError(f'expected must be >= 1, got {expected}')
        with self._lock:
            n = self._counts.get(video, 0) + 1
            if n > expected:
                raise RuntimeError(f'{video}: finished {n} items but only {expected} were expected')
            if n == expected:
                self._counts.pop(video, None)
                return True
            self._counts[video] = n
            return False

    def pending(self) -> dict[str, int]:
        """Videos with some but not all items finished, and their counts (for checks at shutdown)."""
        with self._lock:
            return dict(self._counts)
