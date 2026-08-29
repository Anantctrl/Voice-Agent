"""Chunk processing: transform raw buffers into mono 16 kHz PCM16 chunks."""

from abc import ABC, abstractmethod

import numpy as np

from ..constants.audio import AudioConfig
from ..models.audio_chunk import AudioChunk
from ..utils.audio_utils import resample, to_mono, to_pcm16


class ChunkProcessor(ABC):
    """Strategy for turning a raw audio block into a processed AudioChunk."""

    @abstractmethod
    def process(self, sequence: int, block: np.ndarray) -> AudioChunk:
        """Process ``block`` and return a mono 16 kHz PCM16 chunk."""


class MonoResamplerPcm16Processor(ChunkProcessor):
    """Default processor: stereo->mono, resample to 16 kHz, quantise to PCM16."""

    def __init__(
        self,
        *,
        input_sample_rate: int = AudioConfig.INPUT_SAMPLE_RATE,
        target_sample_rate: int = AudioConfig.TARGET_SAMPLE_RATE,
    ) -> None:
        self._input_rate = input_sample_rate
        self._target_rate = target_sample_rate

    @property
    def input_sample_rate(self) -> int:
        return self._input_rate

    @property
    def target_sample_rate(self) -> int:
        return self._target_rate

    def process(self, sequence: int, block: np.ndarray) -> AudioChunk:
        mono = to_mono(block)
        mono_16k = resample(mono, self._input_rate, self._target_rate)
        pcm16 = to_pcm16(mono_16k)
        return AudioChunk(
            sequence=sequence,
            samples=pcm16,
            input_sample_rate=self._input_rate,
            output_sample_rate=self._target_rate,
        )
