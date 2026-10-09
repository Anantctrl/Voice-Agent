"""Pure SessionState dataclass with generation tokens (pipeline/state.py).

100% instance-scoped: no globals / singletons, so N concurrent sessions are
safe. The ``generation_id`` is the cancellation token from the diagram's
"Cancel Token (Gen ID)" box — every barge-in bumps it, and every stale
async continuation compares and exits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SessionStatus(StrEnum):
    """Explicit session states (str-valued: compares equal to plain strings,
    so logs, telemetry, and legacy checks keep working unchanged)."""

    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class SessionState:
    status: SessionStatus = SessionStatus.LISTENING
    generation_id: int = 0
    # T0..T6 latency telemetry (perf_counter seconds, monotonic per turn).
    t: dict[str, float] = field(default_factory=dict)
    last_user_text: str = ""
    last_assistant_text: str = ""
    current_spoken: str = ""  # text queued for TTS in the active turn
    turn_count: int = 0

    def new_generation(self) -> int:
        """Invalidate all in-flight work. Returns the fresh generation id."""
        self.generation_id += 1
        self.status = SessionStatus.LISTENING
        return self.generation_id

    def is_stale(self, gen_id: int) -> bool:
        return gen_id != self.generation_id

    def mark(self, label: str, ts: float) -> None:
        self.t[label] = ts
