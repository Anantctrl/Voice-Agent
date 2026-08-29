"""Consumer stage: reads ordered chunks from the ring queue and dispatches to STT.

Per the specification, this stage is a structural seam: the pipeline feeds
processed chunks to a speech-to-text backend via ``stt.feed(chunk)``. The STT
backend itself is intentionally not implemented yet.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..constants.audio import PipelineConfig
from ..models.audio_chunk import AudioChunk
from .ring_queue import RingQueue
from .producer import STOP


class SpeechToTextSink(ABC):
    """Contract for a consumer that accepts prepared audio chunks."""

    @abstractmethod
    def feed(self, chunk: AudioChunk) -> bool:
        """Accept a prepared chunk. Return True to continue, False to stop."""


class NoOpSpeechToTextSink(SpeechToTextSink):
    """Placeholder sink: consumes chunks without doing anything yet."""

    def feed(self, chunk: AudioChunk) -> bool:
        return True


class Consumer:
    """Polls the ring queue and forwards each ordered chunk to the STT sink."""

    def __init__(
        self,
        ring_queue: RingQueue,
        sink: SpeechToTextSink,
        *,
        poll_seconds: float = PipelineConfig.CONSUMER_POLL_SECONDS,
    ) -> None:
        self._ring_queue = ring_queue
        self._sink = sink
        self._poll_seconds = poll_seconds
        self._consumed = 0
        self._stopped = False

    @property
    def consumed_count(self) -> int:
        return self._consumed

    @property
    def stopped(self) -> bool:
        return self._stopped

    def run(self) -> None:
        """Consume the ring queue until STOP or the sink halts."""
        import threading

        while True:
            item = self._ring_queue.pop()
            if item is None:
                if self._stopped:
                    break
                threading.Event().wait(self._poll_seconds)
                continue
            if item is STOP:
                self._stopped = True
                break
            chunk: AudioChunk = item
            self._consumed += 1
            continue_ = self._sink.feed(chunk)
            if not continue_:
                self._stopped = True
                break
