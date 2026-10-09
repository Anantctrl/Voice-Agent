"""Turn tracing: session/turn IDs plus one structured record per turn.

Lets any failure be attributed to AEC -> VAD -> STT -> validation -> LLM ->
TTS instead of guessed from interleaved logs.
"""
from __future__ import annotations

import uuid


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def build_turn_record(session_id: str, turn_id: int, *,
                      committed: bool, stt_text: str = "",
                      stt_confidence: float | None = None,
                      validation: str = "", validation_reason: str = "",
                      barge_in: bool = False, barge_path: str | None = None,
                      tts_playing: bool = False, mic_rms: float = 0.0,
                      echo_reduction_db: float = 0.0,
                      ttfa_ms: float | None = None) -> dict:
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "committed": committed,
        "tts_playing": tts_playing,
        "mic_rms": round(mic_rms, 4),
        "echo_reduction_db": round(echo_reduction_db, 2),
        "stt_text": stt_text[:200],
        "stt_confidence": stt_confidence,
        "validation": validation,
        "validation_reason": validation_reason,
        "barge_in": barge_in,
        "barge_path": barge_path,
        "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
    }
