"""Validated runtime configuration (Component 1)."""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _lenient_bool(v) -> bool:
    """bool() with terminal-forgiving parsing: strips whitespace and accepts
    1/0, yes/no, on/off alongside true/false (case-insensitive)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"cannot interpret {v!r} as bool")


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("aec_enabled", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        return _lenient_bool(v)

    # ---- Audio (16 kHz mono int16 everywhere) ----
    sample_rate: int = 16_000
    frame_size: int = 512  # 32 ms per frame @ 16 kHz
    channels: int = 1
    input_device: int | None = None
    output_device: int | None = None

    # ---- VAD ----
    vad_model_path: str = "silero_vad.onnx"
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 80  # DO NOT set higher than 100 ms
    speech_pad_ms: int = 30
    barge_in_frames: int = 2  # 2 frames = 64 ms confirmation
    # Echo-resistant barge-in tuning (see VoiceSessionCoordinator). In hot
    # rooms (speaker echo rides >3x baseline) the sustained path needs a high
    # bar or echo fires it; energy then serves loud onsets while transcripts
    # decide the rest. If interrupts feel deaf: lower toward 2.2.
    barge_in_frames_speaking: int = 4
    barge_in_mic_floor: float = 0.008
    barge_in_rise_ratio: float = 3.5
    barge_in_fast_ratio: float = 3.5
    # Energy barge-in ignores this long after each playback episode starts
    # (echo onset is indistinguishable from a user onset).
    barge_in_blank_s: float = 0.4
    # Echo-overlap/prefix filters are skipped when the speaker has been
    # silent this long with no recent barge: echo is then physically
    # impossible, so topical follow-ups can't be mistaken for it.
    echo_recency_s: float = 2.0
    # "text": interim transcript with words we are not saying confirms a real
    # interrupt (robust without AEC). "energy": legacy RMS-rise heuristic.
    barge_in_mode: str = "text"
    barge_in_min_novel_words: int = 2
    barge_in_min_novel_ratio: float = 0.5

    # ---- Transcript validation (last gate before the LLM) ----
    validator_min_confidence: float = 0.90
    validator_barge_window_s: float = 0.2
    validator_multi_confidence: float = 0.65
    validator_soup_ratio: float = 0.2

    # ---- AEC (optional acoustic echo cancellation) ----
    # WebRTC AEC3 backend (needs pip install pywebrtc-audio); off by default
    # so today's workaround stack decides alone, with automatic fallback if
    # the worker ever fails. stream_delay_ms hints the mic/speaker loop
    # latency (0 = the library's internal estimator decides).
    aec_enabled: bool = False
    aec_backend: str = "webrtc"
    aec_delay_ms: float = 0.0

    # ---- STT (Deepgram persistent WS) ----
    deepgram_api_key: str = Field(alias="DEEPGRAM_API_KEY")
    deepgram_model: str = "nova-3"
    deepgram_endpointing_ms: int = 100

    # ---- LLM (Groq streaming) ----
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    groq_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 256  # covers low-effort reasoning + 1-2 sentences;
    # higher budgets only lengthen TTFT on reasoning models (measured)
    system_prompt: str = (
        "You are a helpful, lightning-fast voice assistant. "
        "Keep your answers concise, direct, and conversational (1-2 sentences). "
        "Never use bullet points, markdown tables, or emojis."
    )

    # ---- TTS ----
    tts_provider: str = "cartesia"  # "cartesia" | "elevenlabs"
    cartesia_api_key: str = Field(default="", alias="CARTESIA_API_KEY")
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    cartesia_model: str = "sonic-english"
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"
    # Flash wins first-byte latency (measured ~700ms vs ~1300ms v4 REST);
    # v4 available via ELEVENLABS_MODEL=eleven_v4 for quality-critical use.
    elevenlabs_model: str = "eleven_flash_v2_5"