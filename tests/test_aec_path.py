"""Phase-1 proof: AEC path evidence, voiced-gated ERLE, divergence recovery,
and repaired turn traces. No hardware, no network."""
import asyncio

import numpy as np

from voice_agent.audio.aec.webrtc import WebRtcAec
from voice_agent.audio.device import AudioDeviceManager
from voice_agent.audio.speaker import SpeakerClassifier

SR, FRAME = 16_000, 512


def _tone(n: int, f0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f0 * t) * (0.4 + 0.6 * (rng.standard_normal(n) ** 2))
    return (sig / max(np.abs(sig).max(), 1e-9) * 15000).astype(np.int16)


def _echo_of(ref: np.ndarray, delay: int = 800, gain: float = 0.3) -> np.ndarray:
    y = np.zeros_like(ref, dtype=np.float64)
    if delay < len(ref):
        y[delay:] = ref[:len(ref) - delay].astype(np.float64) * gain
    return y.astype(np.int16)


def _mgr():
    mgr = AudioDeviceManager()
    mgr.enable_aec(WebRtcAec(), SpeakerClassifier())
    return mgr


def test_path_proof_accumulates_then_freezes():
    mgr = _mgr()
    assert mgr.aec_stats().get("path_db") is None  # no evidence yet
    ref = _tone(FRAME * 60, 140.0, seed=1)
    mic = _echo_of(ref)
    for i in range(0, len(ref), FRAME):
        mgr._process_aec_pair(mic[i:i + FRAME].tobytes(), ref[i:i + FRAME].tobytes())
    st = mgr.aec_stats()
    assert st["frames"] == 60
    assert st.get("path_db") is not None  # in/out energy evidence frozen
    n0 = mgr._proof_n
    mgr._process_aec_pair(mic[:FRAME].tobytes(), ref[:FRAME].tobytes())
    assert mgr._proof_n == n0  # frozen past 50


def test_erle_ignores_silence():
    mgr = _mgr()
    silent = b"\x00" * mgr.frame_bytes
    for _ in range(10):
        mgr._process_aec_pair(silent, silent)
    assert mgr.aec_stats()["erle_db"] == 0.0  # bypassed frames never feed EMA


def test_divergence_resets_and_recovers():
    mgr = _mgr()
    mon = mgr._processor.monitor
    for _ in range(40):
        mon.update(-10.0)
    assert mon.diverged is True
    ref = _tone(FRAME, 140.0, seed=2)
    mic = _echo_of(ref)
    mgr._process_aec_pair(mic.tobytes(), ref.tobytes())
    assert mon.diverged is False  # device reset + acknowledged
    st = mgr.aec_stats()
    assert st["ref_qsize"] == 0  # queue introspection present


def test_committed_trace_carries_mic_erle_and_final_ttfa():
    from voice_agent.pipeline.session import VoiceSessionCoordinator

    class _Audio:
        def __init__(self):
            self.input_queue = asyncio.Queue(maxsize=64)
            self.output_queue: asyncio.Queue = asyncio.Queue()
            self.playing = False

        def is_playing(self):
            return False

        def playback_rms(self):
            return 0.0

        def playback_episode_age(self):
            return float("inf")

        def input_stats(self):
            return {}

        def aec_stats(self):
            return {"enabled": True, "erle_db": 12.5}

        def clear_playback(self):
            pass

        def pop_label(self):
            return None

    class _STT:
        def __init__(self):
            self.sent = 0
            self.on_turn_complete = None
            self.turn_count = 0

        async def connect(self):
            pass

        async def close(self):
            pass

        async def send_audio(self, chunk):
            self.sent += 1

    class _LLM:
        def __init__(self):
            self.prompts = []
            self.committed = []

        async def stream_response(self, prompt):
            self.prompts.append(prompt)
            yield "Hi there. "

        def commit_assistant(self, text):
            self.committed.append(text)

    class _TTS:
        async def synthesize_stream(self, text):
            yield b"\x01\x00" * 256

    async def run():
        logs: list[str] = []
        audio = _Audio()
        s = VoiceSessionCoordinator(audio, None, _STT(), _LLM(), _TTS(),  # type: ignore
                                    log_debug=logs.append,
                                    on_telemetry=lambda t: None)
        s._running = True
        s._mic_live = 0.02
        await s.on_turn_complete("hello there friend")
        await asyncio.sleep(0.2)
        s._running = False
        opens = [m for m in logs if m.startswith("[trace] {")]
        assert opens, "commit must emit an open trace"
        assert "'mic_rms': 0.02" in opens[0]
        assert "'echo_reduction_db': 12.5" in opens[0]
        finals = [m for m in logs if m.startswith("[trace] final")]
        assert finals, "telemetry must close the trace with ttfa"
        assert "'ttfa_ms': None" not in finals[0]

    asyncio.run(run())


def test_dropped_trace_carries_reason_not_zeros():
    from voice_agent.pipeline.session import VoiceSessionCoordinator

    class _Audio2:
        def __init__(self):
            self.playing = True

        def is_playing(self):
            return self.playing

        def aec_stats(self):
            return {"enabled": False}

        def clear_playback(self):
            pass

        def pop_label(self):
            return None

    class _LLM2:
        def __init__(self):
            self.prompts = []
            self.committed = []

        async def stream_response(self, prompt):
            self.prompts.append(prompt)
            yield "x "

        def commit_assistant(self, text):
            self.committed.append(text)

    class _TTS2:
        async def synthesize_stream(self, text):
            yield b"\x00" * 512

    async def run():
        logs: list[str] = []
        audio, llm = _Audio2(), _LLM2()
        s = VoiceSessionCoordinator(audio, None, None, llm, _TTS2(),  # type: ignore
                                    log_debug=logs.append)
        s.state.current_spoken = "Should I book it for Friday?"
        await s.on_turn_complete("Should I book it for Friday?")
        assert llm.prompts == []
        assert any("[trace]" in m and "'committed': False" in m for m in logs)
    asyncio.run(run())
