"""Consumer stage: extracts features from ordered windows and dispatches to STT.

Per the specification, this stage is a structural seam: the pipeline feeds
log-Mel feature arrays to a speech-to-text backend via ``stt.feed(mel)``. The
STT backend itself is intentionally not implemented yet.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..constants.audio import PipelineConfig
from ..models.audio_window import AudioWindow
from .feature_extractor import FeatureExtractor, MelSpectrogramExtractor
from .producer import STOP


class SpeechToTextSink(ABC):
    """Contract for a consumer that accepts extracted feature arrays."""

    @abstractmethod
    def feed(self, mel: np.ndarray) -> bool:
        """Accept a prepared log-Mel feature array. Return True to continue, False to stop."""


class NoOpSpeechToTextSink(SpeechToTextSink):
    """Placeholder sink: consumes features without doing anything yet."""

    def feed(self, mel: np.ndarray) -> bool:
        return True


class Consumer:
    """Polls the window queue, extracts features, and forwards each mel to the STT sink."""

    def __init__(
        self,
        window_queue,
        sink: SpeechToTextSink,
        *,
        extractor: Optional[FeatureExtractor] = None,
        poll_seconds: float = PipelineConfig.CONSUMER_POLL_SECONDS,
    ) -> None:
        self._window_queue = window_queue
        self._sink = sink
        self._extractor = extractor or MelSpectrogramExtractor()
        self._poll_seconds = poll_seconds
        self._consumed = 0
        self._stopped = False

    @property
    def consumed_count(self) -> int:
        """Number of feature windows consumed (one per fixed-length window)."""
        return self._consumed

    @property
    def stopped(self) -> bool:
        return self._stopped

    def run(self) -> None:
        """Consume the window queue until STOP or the sink halts."""
        import threading

        while True:
            item = self._window_queue.get()
            if item is None:
                if self._stopped:
                    break
                threading.Event().wait(self._poll_seconds)
                continue
            if item is STOP:
                self._stopped = True
                break
            window: AudioWindow = item
            mel = self._extractor.extract(window)
            self._consumed += 1
            continue_ = self._sink.feed(mel)
            if not continue_:
                self._stopped = True
                break
