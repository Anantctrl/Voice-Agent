"""WebRTC AEC3 backend: Chrome-grade echo cancellation via pywebrtc-audio.

Same :class:`EchoCanceller` interface as the retired NumPy backend, so the
worker thread, reference tap, speaker classifier, and fallback chain are
untouched. Differences worth knowing:

- AEC3 manages its own double-talk detection internally; the ``adapt`` flag
  is accepted for interface compatibility and effectively always on.
- A high-pass filter is always applied to capture (matches Chrome; removes
  DC offset that would otherwise degrade cancellation).
- Any frame length is accepted (no 10 ms constraint).
- ERLE is measured here per frame so our telemetry reads it identically.
"""
from __future__ import annotations

import numpy as np


class WebRtcAec:
    """AEC3 echo canceller (needs ``pip install pywebrtc-audio``)."""

    def __init__(self, sample_rate: int = 16_000, stream_delay_ms: float = 0.0,
                 ref_silence_thr: float = 3e-4):
        try:
            from pywebrtc_audio import EchoCanceller as _Native
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install pywebrtc-audio for the webrtc backend") from e
        # pybind requires integral milliseconds; config stays float.
        self._ec = _Native(sample_rate=int(sample_rate), num_channels=1,
                           stream_delay_ms=int(round(stream_delay_ms)))
        self.ref_silence_thr = ref_silence_thr

    def reset(self) -> None:
        self._ec.reset() if hasattr(self._ec, "reset") else None

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)) / 32768.0)

    def process_frame(self, mic: np.ndarray, ref: np.ndarray,
                      adapt: bool = True) -> tuple[np.ndarray, dict]:
        mic_a = np.asarray(mic, dtype=np.int16)
        ref_a = np.asarray(ref, dtype=np.int16)
        if mic_a.shape != ref_a.shape:
            raise ValueError(f"mic/ref shape mismatch: {mic_a.shape} vs {ref_a.shape}")
        if mic_a.size == 0:
            return np.zeros(0, dtype=np.int16), {"erle_db": 0.0, "bypassed": True}
        if self._rms(ref_a) < self.ref_silence_thr:
            # Bit-clean bypass: nothing to cancel, never touch near-end.
            return mic_a.copy(), {"erle_db": 0.0, "bypassed": True}
        try:
            clean = np.asarray(self._ec.process(mic_a, ref_a), dtype=np.int16)
        except Exception:
            return mic_a.copy(), {"erle_db": 0.0, "bypassed": True, "failed": True}
        mic_p = float(np.mean(mic_a.astype(np.float64) ** 2)) + 1e-12
        err_p = float(np.mean(clean.astype(np.float64) ** 2)) + 1e-12
        erle_db = 10.0 * float(np.log10(mic_p / err_p))
        return clean, {"erle_db": erle_db, "bypassed": False}
