"""Thread-safe, monotonic sequence number assignment."""

import itertools
import threading


class SequenceAssigner:
    """Assigns globally monotonic sequence numbers to captured buffers.

    Thread-safety is guaranteed, so it is safe to call from the real-time
    audio callback thread.
    """

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        """Return the next sequence number and advance the counter."""
        with self._lock:
            idx = self._next
            self._next += 1
            return idx

    @property
    def counter(self) -> int:
        with self._lock:
            return self._next
