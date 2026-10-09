"""In-memory metrics: counters + latency samples with p50/p95.

Turn/drop/barge/TTFA accounting lives here instead of scattered prints, so a
future Prometheus exporter reads one snapshot. Thread-safe; cheap.
"""
from __future__ import annotations

import threading


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._samples: dict[str, list[float]] = {}

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + amount

    def sample(self, name: str, value: float, keep: int = 200) -> None:
        with self._lock:
            buf = self._samples.setdefault(name, [])
            buf.append(float(value))
            if len(buf) > keep:
                del buf[:len(buf) - keep]

    @staticmethod
    def _pct(sorted_vals: list[float], pct: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(pct / 100.0 * len(sorted_vals)))
        return sorted_vals[idx]

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            dists = {}
            for name, vals in self._samples.items():
                s = sorted(vals)
                dists[name] = {"n": len(s), "last": s[-1],
                               "p50": self._pct(s, 50), "p95": self._pct(s, 95)}
            return {"counts": counts, "distributions": dists}

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._samples.clear()
