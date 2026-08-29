"""Accumulates ordered PCM16 chunks into fixed-length windows for feature extraction."""

import threading
from typing import Optional

import numpy as np

from ..constants.audio import AudioConfig, MelConfig, PipelineConfig
from ..models.audio_chunk import AudioChunk
from ..models.audio_window import AudioWindow
from .producer import STOP
from .ring_queue import RingQueue


class WindowAccumulator:
    """Buffers ordered PCM16 chunks until a full window (default 30 s) is ready.

    Converts PCM16 -> normalized float32 as chunks arrive, and emits a
    zero-padded final window on shutdown if a partial window remains.
    """

    def __init__(
        self,
        ring_queue: RingQueue,
        window_queue,
        *,
        window_samples: int = MelConfig.WINDOW_SAMPLES,
        poll_seconds: float = PipelineConfig.CONSUMER_POLL_SECONDS,
    ) -> None:
        self._ring_queue = ring_queue
        self._window_queue = window_queue
        self._window_samples = window_samples
        self._poll_seconds = poll_seconds

        self._buffer: list[np.ndarray] = []
        self._buffered_len = 0
        self._start_seq: Optional[int] = None
        self._last_seq: Optional[int] = None

    def _flush(self, pad: bool) -> None:
        samples = (
            np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
        )
        if pad and samples.size < self._window_samples:
            samples = np.pad(samples, (0, self._window_samples - samples.size))
        elif samples.size > self._window_samples:
            samples = samples[: self._window_samples]

        self._window_queue.put(AudioWindow(
            start_sequence=self._start_seq,
            end_sequence=self._last_seq,
            samples=samples,
            is_padded=pad,
        ))
        self._buffer, self._buffered_len = [], 0
        self._start_seq = None

    def run(self) -> None:
        """Consume the ring queue until STOP, emitting fixed-length windows."""
        while True:
            item = self._ring_queue.pop()

            if item is None:
                # Empty ring queue: back off instead of busy-spinning.
                threading.Event().wait(self._poll_seconds)
                continue

            if item is STOP:
                if self._buffer:
                    self._flush(pad=True)
                self._window_queue.put(STOP)
                break

            chunk: AudioChunk = item
            float_samples = chunk.samples.astype(np.float32) / AudioConfig.PCM16_SCALE
            if self._start_seq is None:
                self._start_seq = chunk.sequence
            self._last_seq = chunk.sequence
            self._buffer.append(float_samples)
            self._buffered_len += float_samples.size

            if self._buffered_len >= self._window_samples:
                self._flush(pad=False)
