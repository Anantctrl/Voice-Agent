"""Speaker attribution for VAD: whose speech is in this frame.

Silero answers "is this speech?" — never "whose?". This module answers the
second question from AEC signals, so barge-in can tell our speaker apart
from the human without loudness guessing games:

- FAR    = reference hot, residual cold   -> AI only (VAD forced non-speech)
- NEAR   = reference cold, residual hot   -> human only (Silero verdict stands)
- DOUBLE = both hot                       -> real interrupt (barge may fire)
- QUIET  = both cold                      -> silence

Fail-safe: uncertain or misaligned frames label NEAR (a missed cut is
recoverable via the suspicion window; a deaf session is not).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAR, NEAR, DOUBLE, QUIET = "FAR", "NEAR", "DOUBLE", "QUIET"


# Single home for delay estimation is audio.aec.echo_detector; re-exported
# here so existing imports keep working.
from voice_agent.audio.aec.echo_detector import estimate_delay  # noqa: F401


@dataclass
class FrameDecision:
    label: str
    mic_rms: float
    ref_rms: float
    res_rms: float


class SpeakerClassifier:
    """Reference-correlated frame labels (no absolute cross-domain comparison).

    - Digital silence is exact, so the reference threshold can be tiny: any
      real activity in `ref` is genuine playback, never noise.
    - The DOUBLE-vs-FAR line uses the suppression ratio mic/residual. Under
      a strong canceller, converged echo-only frames score high (~7-18
      measured) while double-talk varies widely: AEC3's hard suppression can
      push it anywhere, so the bar (default 5.0) separates confident echo
      from everything else rather than naming speakers. Labels are therefore
      presence evidence for telemetry, not control signals: a whisper under
      loud echo also scores high (FAR), and its backstop is the transcript
      layer (novel words), exactly as for suppressed double-talk.
    """

    def __init__(self, ref_active_thr: float = 1e-3, mic_floor: float = 0.004,
                 double_ratio: float = 2.0):
        self.ref_active_thr = ref_active_thr
        self.mic_floor = mic_floor
        self.double_ratio = double_ratio

    @staticmethod
    def rms(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)) / 32768.0)

    def classify(self, mic: np.ndarray, ref: np.ndarray, residual: np.ndarray) -> FrameDecision:
        mic_rms = self.rms(mic)
        ref_rms = self.rms(ref)
        res_rms = self.rms(residual)
        ref_active = ref_rms >= self.ref_active_thr
        mic_active = mic_rms >= self.mic_floor
        if not ref_active and not mic_active:
            label = QUIET
        elif not ref_active:
            label = NEAR
        elif not mic_active:
            label = FAR
        else:
            ratio = mic_rms / max(res_rms, 1e-9)
            label = DOUBLE if ratio < self.double_ratio else FAR
        return FrameDecision(label, mic_rms, ref_rms, res_rms)
