"""
Text-to-Speech: text -> WAV bytes.

Extracted from the original tts.py so the server can reuse the exact same
Coqui model, but return raw WAV data instead of writing/playing a file.
"""

# BytesIO lets us build the WAV entirely in memory (no temp files on disk).
import io

# numpy is needed to shape the waveform samples for soundfile.
import numpy as np

# soundfile encodes a float waveform into a proper WAV byte stream.
import soundfile as sf

# Coqui TTS - same import path as the original tts.py (coqui-tts fork keeps API).
from TTS.api import TTS

# Global singleton because loading tacotron2-DDC takes seconds and ~1GB RAM;
# we must do that once, not per reply.
_tts = None


def _get_tts() -> TTS:
    """Lazy-load the TTS model on first use to keep server startup instant."""
    global _tts
    if _tts is None:
        # Identical model to tts.py: English single-speaker, decent CPU speed.
        _tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    return _tts


def synthesize(text: str) -> bytes:
    """Convert text into complete WAV bytes (22050Hz mono PCM16)."""

    # Collapse all whitespace runs and strip the ends: stray spaces around words
    # are enough to send Tacotron2's attention into an endless-garble loop.
    text = " ".join(text.split())

    # First synthesis pass.
    wav = np.array(_get_tts().tts(text=text))

    # Ask the synthesizer itself for its output sample rate (22050 for this model)
    # instead of hardcoding it, in case the model is swapped later.
    sr = _get_tts().synthesizer.output_sample_rate

    # Runaway guard: natural speech takes roughly 90ms per character, so if the
    # model returned far more audio than that (known Tacotron2 failure mode on
    # short/unpunctuated text), retry once with a period appended, which
    # reliably stops the decoder.
    estimated = max(1.0, len(text) * 0.09) + 1.0
    if len(wav) / sr > max(estimated * 3.0, 8.0):
        safe = text if text.endswith((".", "!", "?", ",")) else text + "."
        wav = np.array(_get_tts().tts(text=safe))

    # Absolute cap at 30s so even a pathological reply can never flood memory
    # or make the user listen to minutes of garbage.
    wav = wav[: sr * 30]

    # Serialize waveform -> WAV container bytes in memory.
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=sr, format="WAV")

    # Position pointer at 0 isn't needed since getvalue() reads the whole buffer.
    return buf.getvalue()
