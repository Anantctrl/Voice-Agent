"""Feature extraction: convert audio windows into log-Mel spectrograms for STT.

Pure NumPy/SciPy implementation (no librosa / numba) so that the pipeline
startup never loads the numba -> llvmlite native DLL, which some Windows
Application Control policies block.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.signal import get_window

from ..constants.audio import MelConfig
from ..models.audio_window import AudioWindow


# --- minimal Slaney mel-scale helpers (equivalent to a librosa.filters.mel) ---

def _hz_to_mel(freq: np.ndarray, htk: bool = False) -> np.ndarray:
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mels = np.where(
        freq >= min_log_hz,
        min_log_mel + np.log(freq / min_log_hz) / logstep,
        mels,
    )
    return mels


def _mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    idx = mels > min_log_mel
    freqs = np.where(
        idx, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs
    )
    return freqs


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: str = "slaney",
) -> np.ndarray:
    """Build an (n_mels, 1 + n_fft // 2) mel filterbank (Slaney-normalized)."""
    if fmax is None:
        fmax = float(sr) / 2
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mel_f = _hz_to_mel(fftfreqs, htk=htk)
    mel_points = np.linspace(
        _hz_to_mel(np.asarray(fmin, dtype=float), htk),
        _hz_to_mel(fmax, htk),
        n_mels + 2,
    )
    f_points = _mel_to_hz(mel_points, htk=htk)

    fdiff = np.diff(f_points)
    ramps = np.subtract.outer(f_points, fftfreqs)

    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (f_points[2 : n_mels + 2] - f_points[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights


def _stft(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Short-time Fourier transform with librosa-compatible framing.

    Uses a periodic hann window, center-pads the input by ``n_fft // 2`` on each
    side with reflection, and returns an ``(1 + n_fft // 2, n_frames)`` complex
    array as librosa.stft does.
    """
    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)

    # center=True: pad by n_fft//2 on each side with reflect
    pad = n_fft // 2
    padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop_length

    frames = np.lib.stride_tricks.sliding_window_view(
        padded, n_fft
    )[::hop_length][:n_frames]
    framed = frames * win
    return np.fft.rfft(framed, n=n_fft, axis=-1).T  # (freq_bins, n_frames)


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
        self._mel_filters = _mel_filterbank(
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
        stft = _stft(window.samples, self._n_fft, self._hop_length)
        power = np.abs(stft) ** 2
        mel = self._mel_filters @ power
        log_mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        return ((log_mel + 4.0) / 4.0).astype(np.float32)
