"""
Voice Activity Detection with segment extraction.

Two-phase approach matching the user's architecture diagram:
  1. WebRTC VAD: streaming real-time detection of speech start/end (30ms frames)
  2. Silero VAD: precise segment extraction on the collected utterance

Flow:
  Audio Input → WebRTC VAD (streaming) → Collect Utterance
  → Silero VAD (segment extraction) → [Seg1, Seg2, Seg3]
  → Whisper STT (per segment) → Merge Transcripts → Final Transcript
"""

# numpy converts raw PCM bytes into float tensors that Silero VAD expects.
import numpy as np

# torch is required by Silero VAD for model inference and tensor operations.
import torch

# webrtcvad is Google's lightweight VAD — fast enough for real-time 30ms frame
# processing, used for the streaming phase to detect speech start/end.
import webrtcvad

# silero_vad provides the neural-network-based VAD that runs on the complete
# utterance to extract precise speech segments with sample-level boundaries.
from silero_vad import load_silero_vad, get_speech_timestamps


# ---------- constants --------------------------------------------------------

# 30ms at 16kHz = 480 samples; each sample is 2 bytes in PCM16 -> 960 bytes per frame,
# because WebRTC VAD ONLY accepts 10/20/30ms frames and 960 bytes is the exact 30ms size.
FRAME_BYTES = 960

# Require 2 consecutive voiced frames (~60ms) before we call it speech,
# to avoid reacting to clicks/coughs/short noise bursts.
SPEECH_START_FRAMES = 2

# Require ~16 consecutive silent frames (~480ms of trailing silence) before we call
# the utterance finished, so natural mid-sentence pauses don't cut the user off.
SPEECH_END_FRAMES = 16

# Discard utterances shorter than 0.4s (12800 bytes) because they are almost always
# background noise or a lip-smack that survived the start filter.
MIN_UTTERANCE_BYTES = 12800

# Hard cap of ~15s (16000 samples/s * 2 bytes * 15) so a stuck-open mic can never
# exhaust memory or feed an unbounded blob to whisper.
MAX_UTTERANCE_BYTES = 16000 * 2 * 15

# Sample rate expected by both WebRTC VAD and Silero VAD.
SAMPLE_RATE = 16000

# Minimum segment duration in seconds — segments shorter than this are discarded
# because they are likely noise artifacts, not real speech.
MIN_SEGMENT_DURATION = 0.1


# ---------- lazy-loaded Silero VAD model -------------------------------------

_silero_model = None


def _get_silero():
    """Lazy-load the Silero VAD model once to avoid startup latency.

    Uses ONNX backend so the model is bundled inside the silero-vad package
    and never needs to download anything from the internet. This is faster
    than the JIT/PyTorch backend and works fully offline.
    """
    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad(onnx=True)
    return _silero_model


# ---------- streaming VAD (WebRTC) -------------------------------------------


class VADetector:
    """Streaming VAD: feed 30ms frames, get utterance bytes when speech ends.

    Uses WebRTC VAD for real-time frame-level detection. This is the first phase
    of the pipeline — it determines WHEN the user is speaking so we can collect
    the complete utterance audio for segment extraction.
    """

    def __init__(self):
        # Aggressiveness 3 (0-3): most aggressive filtering of non-speech,
        # chosen because browser mic streams already have echo/noise suppression.
        self._vad = webrtcvad.Vad(3)
        # Counter of consecutive voiced frames seen while idle.
        self._voiced_run = 0
        # Counter of consecutive silent frames seen while recording.
        self._silent_run = 0
        # True once we are inside a detected utterance.
        self.speaking = False
        # List of 30ms byte-chunks making up the current utterance.
        self._chunks = []

    def process(self, frame: bytes):
        """Feed one 30ms frame; return utterance bytes when speech ends, else None."""

        # Ask WebRTC VAD if this frame contains human speech.
        is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

        # --- IDLE state: waiting for speech to begin ---
        if not self.speaking:
            if is_speech:
                # Build up evidence that speech really started.
                self._voiced_run += 1
                if self._voiced_run >= SPEECH_START_FRAMES:
                    # Enough proof -> start recording from now on.
                    self.speaking = True
                    self._silent_run = 0
                    self._chunks = [frame]
            else:
                # A silent frame breaks any partial voiced streak.
                self._voiced_run = 0
            return None

        # --- SPEAKING state: collecting the utterance ---
        if is_speech:
            # Real speech continues -> reset the end-of-speech countdown.
            self._silent_run = 0
            self._chunks.append(frame)
        else:
            # Silence during speech might just be a pause -> count it.
            self._silent_run += 1

        # Stop growing the buffer if the user simply won't stop talking,
        # so memory stays bounded (utterance will be force-emitted).
        total = len(self._chunks) * FRAME_BYTES
        if total < MAX_UTTERANCE_BYTES:
            pass  # still under cap, keep buffering normally
        elif self._silent_run == 0:
            # Cap hit while still "speaking": drop oldest half to stay bounded.
            self._chunks = self._chunks[len(self._chunks) // 2:]

        # End of utterance: enough trailing silence has accumulated OR buffer was capped hard.
        if self._silent_run >= SPEECH_END_FRAMES or total >= MAX_UTTERANCE_BYTES:
            # Join all 30ms chunks into one continuous PCM blob.
            utterance = b"".join(self._chunks)
            # Reset every piece of state for the next utterance.
            self.speaking = False
            self._voiced_run = 0
            self._silent_run = 0
            self._chunks = []
            # Ignore blips too short to be real words.
            if len(utterance) >= MIN_UTTERANCE_BYTES:
                return utterance
        return None


# ---------- segment extraction (Silero VAD) ----------------------------------


def extract_segments(pcm16_bytes: bytes) -> list[dict]:
    """Run Silero VAD on a complete utterance to find speech sub-segments.

    This is the second phase of the pipeline. After WebRTC VAD collects the
    full utterance (which may contain multiple speech segments separated by
    pauses), Silero VAD precisely identifies where speech actually exists
    and returns sample-level timestamps for each segment.

    Args:
        pcm16_bytes: Complete utterance as 16kHz mono PCM16 bytes.

    Returns:
        List of dicts with 'start' and 'end' keys (sample indices) and
        'start_sec' / 'end_sec' for human-readable timestamps.
        Returns a single segment covering the full audio if Silero finds nothing.
    """
    # Convert raw PCM16 bytes to float32 tensor normalized to [-1, 1],
    # which is the format Silero VAD expects.
    audio = (
        np.frombuffer(pcm16_bytes, dtype=np.int16)
        .astype(np.float32) / 32768.0
    )
    audio_tensor = torch.from_numpy(audio)

    # Run Silero VAD to get speech timestamps (sample-level boundaries).
    model = _get_silero()
    segments = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,           # confidence threshold for speech detection
        min_speech_duration_ms=100,  # ignore very short blips
        min_silence_duration_ms=300, # merge segments separated by <300ms silence
    )

    # If Silero found nothing (pure noise somehow passed WebRTC), treat the
    # entire utterance as one segment so STT still gets a chance to process it.
    if not segments:
        total_samples = len(audio)
        segments = [{"start": 0, "end": total_samples}]

    # Enrich segments with human-readable timestamps and filter tiny fragments.
    result = []
    for seg in segments:
        duration = (seg["end"] - seg["start"]) / SAMPLE_RATE
        if duration < MIN_SEGMENT_DURATION:
            continue  # skip noise artifacts too short to be real words
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "start_sec": round(seg["start"] / SAMPLE_RATE, 3),
            "end_sec": round(seg["end"] / SAMPLE_RATE, 3),
        })

    # If all segments were filtered out, fall back to the full utterance.
    if not result:
        total_samples = len(audio)
        result = [{
            "start": 0,
            "end": total_samples,
            "start_sec": 0.0,
            "end_sec": round(total_samples / SAMPLE_RATE, 3),
        }]

    return result


def extract_segment_audio(pcm16_bytes: bytes, segment: dict) -> bytes:
    """Extract raw PCM16 bytes for a single segment from the full utterance.

    Args:
        pcm16_bytes: Complete utterance as 16kHz mono PCM16 bytes.
        segment: Dict with 'start' and 'end' sample indices.

    Returns:
        PCM16 bytes for just the segment's time range.
    """
    # Each sample is 2 bytes (int16), so multiply sample indices by 2.
    byte_start = segment["start"] * 2
    byte_end = segment["end"] * 2
    return pcm16_bytes[byte_start:byte_end]
