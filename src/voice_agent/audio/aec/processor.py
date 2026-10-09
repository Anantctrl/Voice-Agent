"""Per-pair AEC frame processor: filter + classify + detect, one call.

Owns a single frame's journey (mic bytes, ref bytes) -> (clean bytes,
decision, info) so the audio device stays a thin transport and this unit is
testable without threads or streams.
"""
from __future__ import annotations

import numpy as np


class FrameProcessor:
    def __init__(self, canceller, classifier, double_talk=None, monitor=None,
                 frame_bytes: int = 1024):
        self.canceller = canceller
        self.classifier = classifier
        self.double_talk = double_talk
        self.monitor = monitor
        self.frame_bytes = frame_bytes
        self._last_label: str | None = None
        self._prev_res_rms: float | None = None

    def _fit(self, raw: bytes) -> bytes:
        n = self.frame_bytes
        if len(raw) < n:
            return raw + b"\x00" * (n - len(raw))
        return raw[:n]

    @staticmethod
    def _rms(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(a * a)) / 32768.0)

    def process(self, mic_bytes: bytes, ref_bytes: bytes):
        mic = np.frombuffer(self._fit(mic_bytes), dtype=np.int16)
        ref = np.frombuffer(self._fit(ref_bytes), dtype=np.int16)
        # Freeze on classifier DOUBLE, or on detector evidence (Geigel +
        # normalized mic-error correlation) from the previous frame's
        # residual. First frame has no residual yet: label rule only.
        adapt = self._last_label != "DOUBLE"
        dt_detail: str | None = None
        if adapt and self.double_talk is not None and self._prev_res_rms is not None:
            try:
                if self.double_talk.is_double_talk(
                        self._rms(mic), self._rms(ref), self._prev_res_rms,
                        self._last_label):
                    adapt = False
                    dt_detail = "geigel_or_corr"
            except Exception:
                pass
        clean, info = self.canceller.process_frame(mic, ref, adapt=adapt)
        # Feed this frame's (mic, error) into the correlation statistics for
        # the NEXT frame's decision.
        observe = getattr(self.double_talk, "observe", None)
        if callable(observe):
            try:
                err = mic.astype(np.float64) - clean.astype(np.float64)
                observe(mic, err)
            except Exception:
                pass
        decision = None
        if self.classifier is not None:
            try:
                decision = self.classifier.classify(mic, ref, clean)
                self._last_label = decision.label
            except Exception:
                decision = None
        self._prev_res_rms = self._rms(clean)
        info = {**info, "adapted": adapt, "dt_freeze": dt_detail is not None}
        if self.monitor is not None and not info.get("bypassed", False):
            try:
                if self.monitor.update(float(info.get("erle_db", 0.0))):
                    info = {**info, "diverged": True}
            except Exception:
                pass
        return clean.tobytes(), decision, info
