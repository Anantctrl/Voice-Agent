"""
Speech-to-Text: raw 16kHz mono PCM16 bytes -> English text.

Uses faster-whisper (CTranslate2 port of Whisper) because it runs fully
offline on CPU and is ~4x faster than openai-whisper.
"""

# numpy is used to convert raw PCM bytes into the float array whisper expects.
import numpy as np

# faster-whisper's model wrapper handles tokenization, inference and decoding.
from faster_whisper import WhisperModel

# Global singleton so the ~75MB model is loaded exactly once per server lifetime,
# not on every utterance (model load takes several seconds).
_model = None


def _get_model() -> WhisperModel:
    """Lazy-load the whisper model on first use to keep server startup instant."""
    global _model
    if _model is None:
        # base.en: good speed/accuracy balance for English on CPU;
        # int8 quantization halves memory with negligible accuracy loss.
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def transcribe(pcm16_bytes: bytes) -> str:
    """Transcribe a complete user utterance. Returns '' when nothing intelligible."""

    # Whisper wants float32 samples normalized to [-1, 1], so convert int16 -> float32/32768.
    audio = (
        np.frombuffer(pcm16_bytes, dtype=np.int16)  # interpret bytes as int16 samples
        .astype(np.float32)                         # widen to float32 as required
        / 32768.0                                   # scale int16 range into [-1, 1]
    )

    # beam_size=1 (greedy) trades a little accuracy for lower latency,
    # vad_filter=True makes whisper itself skip silent stretches inside the clip.
    segments, _info = _get_model().transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False # to prevent form the hallucination in one chunk 
    )

    # Segments are generators of timed text pieces; join them into one sentence string.
    text = " ".join(s.text.strip() for s in segments).strip()
    print("STT")
    print(text)
    return text
