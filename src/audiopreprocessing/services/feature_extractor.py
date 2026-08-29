"""Feature extraction: convert audio windows into log-Mel spectrograms for STT."""

from abc import ABC, abstractmethod

import numpy as np
import librosa

from ..constants.audio import MelConfig
from ..models.audio_window import AudioWindow


class FeatureExtractor(ABC):
    """Strategy for turning an audio window into a feature representation."""

    @abstractmethod
    def extract(self, window: AudioWindow) -> np.ndarray:
        """Extract features from a window, returning a float32 feature array."""


class MelSpectrogramExtractor(FeatureExtractor):
    """Compute an 80-bin log-Mel spectrogram normalized for Whisper-style STT."""

    def __init__(
        self,
        *,
        n_fft: int = MelConfig.N_FFT,
        hop_length: int = MelConfig.HOP_LENGTH,
        n_mels: int = MelConfig.N_MELS,
        sample_rate: int = MelConfig.SAMPLE_RATE,
    ) -> None:
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_mels = n_mels
        self._mel_filters = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels
        )

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def n_mels(self) -> int:
        return self._n_mels

    def extract(self, window: AudioWindow) -> np.ndarray:
        stft = librosa.stft(
            window.samples,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            window="hann",
        )
        power = np.abs(stft) ** 2
        mel = self._mel_filters @ power
        log_mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        return ((log_mel + 4.0) / 4.0).astype(np.float32)
