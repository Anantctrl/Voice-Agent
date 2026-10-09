"""Pass-through canceller (AEC disabled) + backend factory.

The NumPy NLMS backend was removed in favor of WebRTC AEC3
(:mod:`voice_agent.audio.aec.webrtc`): 18-24 dB measured vs ~12 dB, 0.75 ms
vs ~5 ms per frame, no hand-rolled convergence to babysit.
"""
from __future__ import annotations

import numpy as np


class NoOpAEC:
    """Pass-through: AEC disabled, today's workaround stack decides alone."""

    def reset(self) -> None:
        return None

    def process_frame(self, mic: np.ndarray, ref: np.ndarray,
                      adapt: bool = True) -> tuple[np.ndarray, dict]:
        return np.asarray(mic, dtype=np.int16), {"erle_db": 0.0, "bypassed": True}


def create_canceller(backend: str = "webrtc", sample_rate: int = 16_000,
                     stream_delay_ms: float = 0.0, **kwargs) -> object:
    """Factory: 'webrtc' (AEC3, needs pip install pywebrtc-audio) | 'noop'."""
    name = (backend or "webrtc").lower()
    if name == "webrtc":
        from voice_agent.audio.aec.webrtc import WebRtcAec
        return WebRtcAec(sample_rate=sample_rate, stream_delay_ms=stream_delay_ms,
                         **kwargs)
    if name == "noop":
        return NoOpAEC()
    raise ValueError(f"unknown AEC backend {backend!r} (expected 'webrtc' or 'noop')")
