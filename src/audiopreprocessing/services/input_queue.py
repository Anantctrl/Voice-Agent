"""A bounded, thread-safe, back-pressure aware input queue.

Wraps ``queue.Queue`` so the producer can signal dropped frames without
raising in the (real-time) audio callback thread.
"""

import queue as _queue
from typing import Any

from ..exceptions.pipeline_errors import QueueFullError


class InputQueue:
    """FIFO of raw ``(sequence, block)`` tuples awaiting processing."""

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._queue: _queue.Queue = _queue.Queue(maxsize=maxsize)

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def put(self, item: Any) -> None:
        """Blocking put (used by tests/offline feeds)."""
        self._queue.put(item)

    def put_nowait(self, item: Any) -> None:
        """Non-blocking put; raises QueueFullError if the queue is full."""
        try:
            self._queue.put_nowait(item)
        except _queue.Full as exc:
            raise QueueFullError("input queue is full") from exc

    def get(self) -> Any:
        """Blocking get of the next item."""
        return self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()
