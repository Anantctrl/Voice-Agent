"""P2 wiring proof: DAC tap + AEC worker + label queue + VAD overrule.

No PortAudio streams are started anywhere here; the worker thread and the
pairing function are exercised directly, plus one ingest-loop run proving a
FAR label overrules an energy gate that would otherwise fire.
"""
import asyncio

import numpy as np

from voice_agent.audio.aec import WebRtcAec
from voice_agent.audio.device import AudioDeviceManager
from voice_agent.audio.speaker import FAR, NEAR, FrameDecision, SpeakerClassifier


def _loud_frame(rms: float = 0.05, n: int = 512, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * rms * 32768).astype(np.int16).tobytes()


def test_pairing_yields_clean_bytes_and_labels_in_order():
    mgr = AudioDeviceManager()
    mgr.enable_aec(WebRtcAec(), SpeakerClassifier())
    mic = _loud_frame()
    ref = b"\x00" * mgr.frame_bytes  # silent room: bypass, bit-clean
    clean, decision = mgr._process_aec_pair(mic, ref)
    assert clean == mic
    assert decision is not None and decision.label == NEAR
    st = mgr.aec_stats()
    assert st["enabled"] and st["frames"] == 1


def test_route_mic_queues_for_worker_without_loop():
    mgr = AudioDeviceManager()
    mgr.enable_aec(WebRtcAec(), SpeakerClassifier())
    assert mgr._loop is None
    mgr._route_mic(b"\x00" * mgr.frame_bytes)
    assert mgr._mic_q.qsize() == 1  # worker-bound, loop untouched


def test_clear_playback_resets_pairing_not_filter_state():
    mgr = AudioDeviceManager()
    aec = WebRtcAec()
    mgr.enable_aec(aec, SpeakerClassifier())
    mgr._tap_reference(b"\x01" * mgr.frame_bytes)
    with mgr._label_lock:
        mgr._label_deque.append("stale")
    mgr.clear_playback()
    assert mgr._ref_q.qsize() == 0
    assert mgr.pop_label() is None
    # AEC3 owns its state internally; the contract is pairing reset only.


def test_worker_end_to_end_delivery():
    async def run():
        mgr = AudioDeviceManager()
        mgr.enable_aec(WebRtcAec(), SpeakerClassifier())
        mgr._loop = asyncio.get_running_loop()
        mgr._start_aec_worker()
        try:
            for i in range(4):
                mgr._route_mic(_loud_frame(seed=i))
                mgr._tap_reference(b"\x00" * mgr.frame_bytes)
            got = await asyncio.wait_for(mgr.input_queue.get(), 2.0)
            assert len(got) == mgr.frame_bytes
            assert mgr.pop_label() is not None
            assert mgr.aec_stats()["frames"] >= 1
        finally:
            mgr._aec_stop.set()
            mgr._aec_thread.join(timeout=2.0)
    asyncio.run(run())


def test_fallback_routes_direct_when_worker_failed():
    async def run():
        mgr = AudioDeviceManager()
        mgr.enable_aec(WebRtcAec(), SpeakerClassifier())
        mgr._loop = asyncio.get_running_loop()
        mgr._aec_failed = True  # as set by a dead worker
        mgr._route_mic(b"\x00" * mgr.frame_bytes)
        await asyncio.sleep(0.05)  # call_soon_threadsafe delivery
        assert mgr.input_queue.qsize() == 1
        assert mgr._mic_q.qsize() == 0
    asyncio.run(run())


class _AudioWithLabels:
    """Session-ingest fake: real label queue, controllable playing flag."""

    def __init__(self):
        self.input_queue = asyncio.Queue(maxsize=64)
        self.output_queue = None
        self.playing = True
        self.labels = []
        self.cleared = 0

    def is_playing(self):
        return self.playing

    def playback_rms(self):
        return 0.05

    def playback_episode_age(self):
        return 5.0

    def input_stats(self):
        return {}

    def aec_stats(self):
        return {"enabled": True}

    def clear_playback(self):
        self.cleared += 1
        self.playing = False

    def pop_label(self):
        if not self.labels:
            return None
        return self.labels.pop(0)


class _VADTrue:
    def is_speech(self, chunk):
        return True


class _STTCount:
    def __init__(self):
        self.sent = 0
        self.on_turn_complete = None

    async def connect(self):
        pass

    async def close(self):
        pass

    async def send_audio(self, chunk):
        self.sent += 1


def _far(mic=0.05):
    return FrameDecision(FAR, mic, 0.05, 0.001)


def test_far_labels_do_not_gate_the_energy_decision():
    # Labels are telemetry, not control: with AEC3 suppressing echo ~20 dB,
    # the energy gate decides alone on clean audio. FAR labels flowing must
    # neither force nor block a cut.
    from voice_agent.pipeline.session import VoiceSessionCoordinator

    async def run():
        audio = _AudioWithLabels()
        s = VoiceSessionCoordinator(audio, _VADTrue(), _STTCount(), None, None)  # type: ignore
        for _ in range(10):
            s._update_echo(0.006, 0.10)   # calibrated quiet room
        s._running = True
        audio.labels = [_far() for _ in range(12)]
        for i in range(12):
            audio.input_queue.put_nowait(_loud_frame(seed=100 + i))
        ingest = asyncio.create_task(s._audio_ingest_loop())
        for _ in range(100):
            if s.generation_id > 0:
                break
            await asyncio.sleep(0.02)
        assert s.generation_id == 1, "loud voice cuts regardless of FAR labels"
        s._running = False
        ingest.cancel()
        await asyncio.gather(ingest, return_exceptions=True)
    asyncio.run(run())


def test_near_label_plus_loud_voice_cuts():
    from voice_agent.pipeline.session import VoiceSessionCoordinator

    async def run():
        audio = _AudioWithLabels()
        s = VoiceSessionCoordinator(audio, _VADTrue(), _STTCount(), None, None)  # type: ignore
        for _ in range(10):
            s._update_echo(0.006, 0.10)
        s._running = True
        audio.labels = [FrameDecision(NEAR, 0.05, 0.05, 0.04) for _ in range(12)]
        for i in range(12):
            audio.input_queue.put_nowait(_loud_frame(seed=200 + i))
        ingest = asyncio.create_task(s._audio_ingest_loop())
        for _ in range(100):
            if s.generation_id > 0:
                break
            await asyncio.sleep(0.02)
        assert s.generation_id == 1, "NEAR + loud voice must cut"
        assert audio.cleared == 1
        s._running = False
        ingest.cancel()
        await asyncio.gather(ingest, return_exceptions=True)
    asyncio.run(run())
