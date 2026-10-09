"""Echo-path monitoring: delay estimation + divergence latch.

The canceller needs the reference aligned with the mic (speaker + system +
acoustic flight time, 0-400 ms on laptops). This module owns delay
estimation (normalized cross-correlation) and watches for divergence:
sustained negative ERLE means the model is adding energy instead of removing
it, in which case the caller should reset coefficients and re-converge
rather than ride a corrupt model.
"""
from __future__ import annotations

import numpy as np


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sample_rate: int = 16_000,
                   max_ms: float = 400.0) -> int:
    """Integer mic-vs-reference lag in samples (normalized cross-correlation)."""
    m = np.asarray(mic, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if m.size == 0 or r.size == 0:
        return 0
    m = m - m.mean()
    r = r - r.mean()
    denom = float(np.sqrt(np.sum(m * m) * np.sum(r * r)))
    if denom < 1e-12:
        return 0
    max_lag = min(int(sample_rate * max_ms / 1000.0), m.size - 1, r.size - 1)
    best_lag, best_val = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = m[lag:], r[:r.size - lag]
        else:
            a, b = m[:m.size + lag], r[-lag:]
        if a.size == 0:
            continue
        v = float(np.dot(a, b)) / denom
        if v > best_val:
            best_val, best_lag = v, lag
    return int(best_lag)


class EchoPathMonitor:
    """Latch-style divergence detector over per-frame ERLE (dB).

    `update()` returns True on the frame where sustained negative ERLE trips
    the latch (caller resets + re-converges). `acknowledge()` clears it.
    """

    def __init__(self, trip_db: float = -3.0, trip_frames: int = 30):
        self.trip_db = trip_db
        self.trip_frames = trip_frames
        self._bad = 0
        self.diverged = False

    def reset(self) -> None:
        self._bad = 0
        self.diverged = False

    def update(self, erle_db: float | None) -> bool:
        if erle_db is None:
            return self.diverged
        if erle_db < self.trip_db:
            self._bad += 1
        else:
            self._bad = 0
        if self._bad >= self.trip_frames:
            self.diverged = True
        return self.diverged

    def acknowledge(self) -> None:
        self.reset()
