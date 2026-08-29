"""Completed chunks queue and the final ordered ring buffer."""

import queue as _queue
from collections import deque
from typing import Any, Optional
import threading

from ..exceptions.pipeline_errors import PipelineStoppedError


class CompletedQueue:
    """Thread-safe results queue storing ``(sequence, processed_chunk)``."""

    def __init__(self) -> None:
        self._queue: _queue.Queue = _queue.Queue()

    def put(self, item: Any) -> None:
        self._queue.put(item)

    def get(self) -> Any:
        return self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()


class RingQueue:
    """The final bounded, ordered buffer consumed by the STT stage.

    ``deque(maxlen=RING_SIZE)`` silently discards the oldest item when full,
    matching the spec's behaviour (with a logged warning).
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("ring size must be a positive integer")
        self._size = size
        self._buffer: deque = deque(maxlen=size)
        self._lock = threading.Lock()
        self._closed = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self._size

    def push(self, item: Any) -> bool:
        """Append an item. Returns True if the oldest item was discarded."""
        with self._lock:
            if self._closed:
                raise PipelineStoppedError("ring queue is closed")
            overflowed = len(self._buffer) >= self._size
            self._buffer.append(item)
            return overflowed

    def pop(self) -> Optional[Any]:
        """Pop and return the oldest item (or None when empty)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
