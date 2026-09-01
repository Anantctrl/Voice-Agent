"""GTRN noise-suppression stage.

Denoises ordered 16 kHz mono PCM16 chunks before they reach the window
accumulator. GTRN is treated as a noise suppressor (NS), not an ASR model: its
input and output are both audio.

The underlying WebRTC processor is stateful and not thread-safe, so exactly one
sequential ``GTRNWorker`` consumes the ordered chunks and forwards the clean
audio downstream.
"""

import logging
from typing import Callable, Optional

import numpy as np

from ..constants.audio import GTRNConfig
from ..models.audio_chunk import AudioChunk
from .producer import STOP

logger = logging.getLogger(__name__)


class _PassThroughProcessor:
    """Fallback when a real WebRTC noise-suppression backend is unavailable.

    Passes samples through unmodified so the pipeline still runs end-to-end
    (GTRN becomes a no-op NS stage) instead of failing at startup.
    """

    def process(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.int16)


# Best-effort import of the optional WebRTC backend. If it is missing (e.g. the
# installed pywebrtc-audio stub exposes no AudioProcessor), GTRN falls back to
# pass-through so the pipeline can still run.
def _resolve_backend():
    try:
        import pywebrtc_audio

        backend = getattr(pywebrtc_audio, "AudioProcessor", None)
        if backend is None:
            logger.warning(
                "pywebrtc_audio has no AudioProcessor; GTRN running in "
                "pass-through (no noise suppression)"
            )
            return None
        return backend
    except Exception as exc:  # noqa: BLE001 - any import/attr error disables NS
        logger.warning(
            "WebRTC backend unavailable (%s); GTRN running in pass-through "
            "(no noise suppression)",
            exc,
        )
        return None


class GTRNProcessor:
    """Encapsulates the WebRTC noise-suppression/AGC/HPF backend.

    A single instance owns one stateful ``AudioProcessor``; call :meth:`process`
    sequentially from one thread only. If the WebRTC backend is not installed,
    it degrades to a pass-through so the pipeline remains functional.
    """

    def __init__(
        self,
        *,
        sample_rate: int = GTRNConfig.SAMPLE_RATE,
        num_channels: int = GTRNConfig.NUM_CHANNELS,
        ns_level: int = GTRNConfig.NS_LEVEL,
        noise_suppression: bool = GTRNConfig.NOISE_SUPPRESSION,
        high_pass_filter: bool = GTRNConfig.HIGH_PASS_FILTER,
        auto_gain_control: bool = GTRNConfig.AUTO_GAIN_CONTROL,
        echo_cancellation: bool = GTRNConfig.ECHO_CANCELLATION,
        agc_gain_db: float = GTRNConfig.AGC_GAIN_DB,
        agc_max_gain_db: float = GTRNConfig.AGC_MAX_GAIN_DB,
        headroom_db: float = GTRNConfig.HEADROOM_DB,
        backend_factory: Optional[Callable] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if backend_factory is None:
            backend_factory = _resolve_backend()

        if backend_factory is None:
            self._processor = _PassThroughProcessor()
            return

        try:
            self._processor = backend_factory(
                sample_rate=sample_rate,
                num_channels=num_channels,
                noise_suppression=noise_suppression,
                high_pass_filter=high_pass_filter,
                auto_gain_control=auto_gain_control,
                echo_cancellation=echo_cancellation,
                ns_level=ns_level,
                agc_gain_db=agc_gain_db,
                agc_max_gain_db=agc_max_gain_db,
                headroom_db=headroom_db,
            )
        except TypeError:
            self._processor = backend_factory(
                sample_rate=sample_rate,
                num_channels=num_channels,
                noise_suppression=noise_suppression,
                high_pass_filter=high_pass_filter,
                auto_gain_control=auto_gain_control,
                echo_cancellation=echo_cancellation,
                ns_level=ns_level,
                agc_gain_db=agc_gain_db,
                agc_max_gain_db=agc_max_gain_db,
            )

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Denoise an int16 mono buffer, returning an int16 mono buffer of the
        same length."""
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        mono = samples.reshape(-1) if samples.ndim > 1 else samples
        return self._processor.process(mono).astype(np.int16)


class GTRNWorker:
    """Sequential consumer that denoises ordered chunks and forwards them.

    Reads from an input queue, runs each chunk through a single
    ``GTRNProcessor`` in arrival order (preserving sequence numbers), and pushes
    the clean ``AudioChunk`` onto an output queue. Propagates the STOP sentinel
    downstream and exits.
    """

    def __init__(self, input_queue, output_queue, processor: GTRNProcessor) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._processor = processor
        self._processed = 0

    @property
    def processed_count(self) -> int:
        return self._processed

    def run(self) -> None:
        while True:
            item = self._input_queue.get()
            self._input_queue.task_done()

            if item is STOP:
                self._output_queue.push(STOP)
                break

            chunk: AudioChunk = item
            clean = self._processor.process(chunk.samples)
            self._output_queue.push(
                AudioChunk(
                    sequence=chunk.sequence,
                    samples=clean,
                    input_sample_rate=chunk.input_sample_rate,
                    output_sample_rate=chunk.output_sample_rate,
                )
            )
            self._processed += 1
