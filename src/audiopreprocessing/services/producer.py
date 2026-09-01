"""Producer abstractions feeding the pipeline.

A producer's single responsibility is to capture raw audio buffers and push
``(sequence, block)`` tuples onto the input queue. Two concrete producers are
provided:
  * ``MicrophoneProducer`` - real-time capture via ``sounddevice.InputStream``.
  * ``FileProducer`` - offline replay of a WAV into the same pipeline.

On completion (notably offline mode) a producer may signal end-of-input by
calling ``finish()``, which enqueues the shared STOP sentinel.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..constants.audio import AudioConfig
from ..exceptions.pipeline_errors import QueueFullError
from .input_queue import InputQueue
from .sequence_assigner import SequenceAssigner

STOP: Any = object()
"""Singleton sentinel used to terminate the pipeline (poison pill)."""


class Producer(ABC):
    """Common contract for all pipeline producers."""

    def __init__(
        self,
        input_queue: InputQueue,
        sequence_assigner: SequenceAssigner,
    ) -> None:
        self._input_queue = input_queue
        self._sequences = sequence_assigner

    @abstractmethod
    def start(self) -> None:
        """Begin producing data onto the input queue (blocking)."""

    def finish(self) -> None:
        """Signal end-of-input by enqueueing the stop sentinel."""
        self._input_queue.put(STOP)

    def _emit(self, block: np.ndarray, *, blocking: bool = False) -> None:
        """Assign a sequence number and enqueue.

        When ``blocking`` is True, wait for space instead of dropping, so
        offline replay never loses frames. Live capture defaults to dropping
        on overflow so the microphone callback never blocks.
        """
        index = self._sequences.next()
        sample = np.asarray(block, dtype=np.float32).copy()
        if blocking:
            self._input_queue.put((index, sample))
            return
        try:
            self._input_queue.put_nowait((index, sample))
        except QueueFullError:
            print(f"Input queue full - dropping frame {index}")


class MicrophoneProducer(Producer):
    """Captures audio from a microphone via a sounddevice callback."""

    def __init__(
        self,
        input_queue: InputQueue,
        sequence_assigner: SequenceAssigner,
        *,
        sample_rate: int = AudioConfig.INPUT_SAMPLE_RATE,
        channels: int = AudioConfig.CHANNELS,
        block_size: int = AudioConfig.BLOCK_SIZE,
    ) -> None:
        super().__init__(input_queue, sequence_assigner)
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(status)
        self._emit(indata)

    def start(self) -> None:
        import threading

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._block_size,
                callback=self._callback,
            ):
                print("Listening... (Ctrl+C to stop)")
                threading.Event().wait()
        except KeyboardInterrupt:
            print("\nStopped.")


class FileProducer(Producer):
    """Replays a WAV file through the pipeline in offline mode.

    The WAV is decoded and fed as fixed-size blocks so that the downstream
    processing path is identical to live capture. It finishes by enqueueing
    the STOP sentinel so the pipeline can shut down cleanly.
    """

    def __init__(
        self,
        input_queue: InputQueue,
        sequence_assigner: SequenceAssigner,
        file_path: str,
        *,
        block_size: int = AudioConfig.BLOCK_SIZE,
    ) -> None:
        super().__init__(input_queue, sequence_assigner)
        self._file_path = file_path
        self._block_size = block_size
        self._source_rate: Optional[int] = None
        self._channels: Optional[int] = None

    @property
    def source_sample_rate(self) -> Optional[int]:
        return self._source_rate

    @property
    def channel_count(self) -> Optional[int]:
        return self._channels

    def start(self) -> None:
        with sf.SoundFile(self._file_path) as snd:
            self._source_rate = int(snd.samplerate)
            self._channels = snd.channels
            block = snd.read(frames=self._block_size, dtype="float32", always_2d=True)
            while block.shape[0] > 0:
                self._emit(block, blocking=True)
                block = snd.read(frames=self._block_size, dtype="float32", always_2d=True)

        print(
            f"Finished reading {self._file_path} "
            f"(sr={self._source_rate}, ch={self._channels}, "
            f"blocks={self._sequences.counter})"
        )
        self.finish()
