"""Fixed-capacity circular audio buffer for playback (lock-free-ish).

The PortAudio output callback runs on a high-priority audio thread and must
never block, allocate heavily, or touch the asyncio event loop. This ring
gives that thread a plain lock-protected byte buffer with O(1) read/write
and an instant-flush path used for barge-in (<70 ms cutoff).
"""
from __future__ import annotations

import threading


class PlaybackRingBuffer:
    """Byte-oriented circular buffer with overwrite-oldest-on-overflow."""

    def __init__(self, capacity_bytes: int = 32000 * 10):  # ~10 s @ 16 kHz mono int16
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        self._buf = bytearray(capacity_bytes)
        self._cap = capacity_bytes
        self._r = 0
        self._w = 0
        self._size = 0
        self._lock = threading.Lock()
        self.dropped_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._cap

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """Write bytes; overwrites oldest data on saturation. Returns bytes kept."""
        n = len(data)
        if n == 0:
            return 0
        with self._lock:
            if n >= self._cap:
                # Keep only the tail that fits.
                data = data[-self._cap:]
                n = self._cap
                self.dropped_bytes += self._size
                self._r = 0
                self._w = 0
                self._size = 0
            free = self._cap - self._size
            if n > free:
                # Drop oldest to make room (prioritize fresh speech).
                drop = n - free
                self._r = (self._r + drop) % self._cap
                self._size -= drop
                self.dropped_bytes += drop
            first = min(n, self._cap - self._w)
            self._buf[self._w:self._w + first] = data[:first]
            rest = n - first
            if rest:
                self._buf[0:rest] = data[first:]
            self._w = (self._w + n) % self._cap
            self._size += n
            return n

    def read(self, n: int, out: bytearray | memoryview) -> int:
        """Read up to n bytes into `out`. Zero-fills remainder. Returns bytes read."""
        with self._lock:
            take = min(n, self._size)
            first = min(take, self._cap - self._r)
            out[:first] = self._buf[self._r:self._r + first]
            rest = take - first
            if rest:
                out[first:take] = self._buf[0:rest]
            self._r = (self._r + take) % self._cap
            self._size -= take
            if take < n:
                out[take:n] = b"\x00" * (n - take)
            return take

    def clear(self) -> None:
        """Instant flush on barge-in. O(1)."""
        with self._lock:
            self._r = 0
            self._w = 0
            self._size = 0
