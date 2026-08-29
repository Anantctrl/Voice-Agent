"""Reorder buffer: enforces ordered delivery of completed chunks.

Workers complete out of order, so this stage buffers arrivals and only emits
chunks in ascending sequence order (0, 1, 2, ...) onto the final ring queue.
"""

from typing import Optional

from ..models.audio_chunk import AudioChunk
from .ring_queue import CompletedQueue, RingQueue
from .producer import STOP


class ReorderBuffer:
    """Waits for the next expected index and forwards a strictly ordered stream."""

    def __init__(self, completed_queue: CompletedQueue, ring_queue: RingQueue) -> None:
        self._completed_queue = completed_queue
        self._ring_queue = ring_queue
        self._expected = 0
        self._pending: dict[int, AudioChunk] = {}

    def run(self) -> None:
        """Consume the completed queue until STOP, emitting in order."""
        while True:
            item = self._completed_queue.get()
            if item is STOP:
                self._ring_queue.push(STOP)
                self._completed_queue.task_done()
                break

            chunk: AudioChunk = item
            self._completed_queue.task_done()
            self._pending[chunk.sequence] = chunk

            while self._expected in self._pending:
                ordered = self._pending.pop(self._expected)
                overflowed = self._ring_queue.push(ordered)
                if overflowed:
                    print("Ring queue full - oldest chunk discarded")
                self._expected += 1
