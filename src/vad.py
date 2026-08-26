"""
Voice Activity Detection state machine.

Consumes 30ms chunks of 16kHz mono PCM16 and decides when a user
utterance starts and ends, so we only send real speech to STT.
"""

# Import Google's WebRTC VAD implementation (installed via webrtcvad-wheels),
# because it is tiny, fast on CPU, and designed exactly for this frame-based use.
import webrtcvad

# 30ms at 16kHz = 480 samples; each sample is 2 bytes in PCM16 -> 960 bytes per frame,
# because WebRTC VAD ONLY accepts 10/20/30ms frames and 960 bytes is the exact 30ms size.
FRAME_BYTES = 960

# Require 3 consecutive voiced frames (~90ms) before we call it speech,
# to avoid reacting to clicks/coughs/short noise bursts.
SPEECH_START_FRAMES = 2

# Require ~23 consecutive silent frames (~700ms of trailing silence) before we call
# the utterance finished, so natural mid-sentence pauses don't cut the user off.
SPEECH_END_FRAMES = 16

# Discard utterances shorter than 0.4s (12800 bytes) because they are almost always
# background noise or a lip-smack that survived the start filter.
MIN_UTTERANCE_BYTES = 12800

# Hard cap of ~15s (16000 samples/s * 2 bytes * 15) so a stuck-open mic can never
# exhaust memory or feed an unbounded blob to whisper.
MAX_UTTERANCE_BYTES = 16000 * 2 * 15


class VADetector:
    """Per-connection VAD: feed it 30ms frames, get back a full utterance when done."""

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
        """Feed one 30ms frame; return the full utterance bytes when speech ends, else None."""

        # Ask WebRTC VAD if this frame contains human speech.
        is_speech = self._vad.is_speech(frame, 16000)

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
