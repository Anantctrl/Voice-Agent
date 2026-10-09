"""Echo-gate regression tests: TTS must not hear itself."""

from voice_agent.audio.device import AudioDeviceManager
from voice_agent.pipeline.session import VoiceSessionCoordinator


def test_is_playing_gate_transitions():
    mgr = AudioDeviceManager()
    assert not mgr.is_playing()
    mgr.output_queue.put_nowait(b"x" * 2048)
    assert mgr.is_playing()
    mgr.clear_playback()
    assert not mgr.is_playing()


def test_playout_tail_holds_gate_briefly():
    mgr = AudioDeviceManager()
    mgr.play_tail_s = 10.0  # widen tail for deterministic check
    mgr.output_queue.put_nowait(b"y" * 2048)
    out = bytearray(mgr.frame_bytes)
    mgr._drain_into(out)
    assert mgr.is_playing()


def test_echo_transcripts_dropped_real_speech_kept():
    f = VoiceSessionCoordinator._is_likely_echo
    assert f("Can I help you today?", "Hey there! How can I help you today?")
    assert f("On your mind today?", "Hi! What's on your mind today?")
    # Garbled echo used to score 0.545 (stopword inflation) and slip the idle
    # bar; content-word scoring puts it at 0.75 — correctly dropped anywhere.
    assert f("Morning block. Start at 8AM and finish at 12PM.",
             "Got it: morning block runs from 8 AM to 12 PM.")
    assert f("Morning block. Start at 8AM and finish at 12PM.",
             "Got it: morning block runs from 8 AM to 12 PM.", 0.4)
    assert not f("I need to plan a trip to Goa", "Hey there! How can I help you today?")
    assert not f("hello", "Hey there! How can I help you today?")


def test_energy_gate_holds_echo_fires_on_loud_interrupt():
    import numpy as np
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(512) * 6000).astype(np.int16).tobytes()
    quiet = (rng.standard_normal(512) * 200).astype(np.int16).tobytes()
    lr, qr = VoiceSessionCoordinator._rms16(loud), VoiceSessionCoordinator._rms16(quiet)
    play_rms, floor, ratio = 0.03, 0.02, 1.5
    assert not (qr >= max(floor, play_rms * ratio))  # echo held
    assert lr >= max(floor, play_rms * ratio)  # real voice fires


def test_strict_mid_playback_suppression():
    f = VoiceSessionCoordinator._is_likely_echo
    assert f("Can I help you today?", "Hey there! How can I help you today?", 0.4)
    assert not f("stop, I need something else", "Morning block runs from 8 AM to 12 PM.", 0.4)


def test_input_stats_counts_frames():
    mgr = AudioDeviceManager()
    mgr._safe_enqueue(b"\x00" * mgr.frame_bytes)
    mgr._safe_enqueue(b"\x00" * mgr.frame_bytes)
    st = mgr.input_stats()
    assert st["frames"] == 2
    assert st["queue_depth"] == 2


def test_input_queue_absorbs_burst_without_drops():
    mgr = AudioDeviceManager()
    assert mgr.input_queue.maxsize >= 64
    for _ in range(40):
        mgr._safe_enqueue(b"\x00" * mgr.frame_bytes)
    st = mgr.input_stats()
    assert st["frames"] == 40
    assert st["queue_depth"] == 40  # nothing dropped


def test_echo_drop_is_logged_not_silent():
    import asyncio as _aio

    class _Audio:
        def is_playing(self):
            return False

    async def _run():
        logs: list[str] = []
        s = VoiceSessionCoordinator(audio=_Audio(), vad=None, stt=None, llm=None, tts=None,  # type: ignore
                                    log_debug=logs.append)
        s.state.last_assistant_text = "Hey there! How can I help you today?"
        await s.on_turn_complete("Can I help you today?")
        assert s.state.turn_count == 0  # dropped, no turn created
        assert any("dropped-as-echo" in m for m in logs)

    _aio.run(_run())


def test_genuine_turn_still_flows_with_debug_on():
    import asyncio as _aio

    class _Audio:
        def is_playing(self):
            return False

    async def _run():
        s = VoiceSessionCoordinator(audio=_Audio(), vad=None, stt=None, llm=None, tts=None,  # type: ignore
                                    log_debug=lambda m: None)
        await s.on_turn_complete("I need to plan a trip to Goa")
        assert s.state.turn_count == 1
        for t in list(s.active_tasks):
            t.cancel()

    _aio.run(_run())


def test_telemetry_marks_reset_per_turn():
    import asyncio as _aio

    class _Audio:
        def is_playing(self):
            return False

    async def _run():
        s = VoiceSessionCoordinator(audio=_Audio(), vad=None, stt=None, llm=None, tts=None,  # type: ignore
                                    log_debug=lambda m: None)
        s.state.mark("T_turn_end", 100.0)
        s.state.mark("T_first_token", 100.5)
        s.state.mark("T_tts_start", 100.8)
        s.state.mark("T_llm_done", 101.0)
        s.state.mark("T_first_audio", 101.1)
        await s.on_turn_complete("hello there friend")
        # New turn must not inherit any timing marks (negatives gone).
        for k in ("T_first_token", "T_tts_start", "T_llm_done", "T_first_audio"):
            assert k not in s.state.t, k
        assert "T_turn_end" in s.state.t
        for t in list(s.active_tasks):
            t.cancel()

    _aio.run(_run())


def _make_session(**kw):
    class _Audio:
        def is_playing(self):
            return False
    args = {
        "audio": _Audio(), "vad": None, "stt": None, "llm": None, "tts": None,  # type: ignore
        "log_debug": lambda m: None,
    }
    args.update(kw)
    return VoiceSessionCoordinator(**args)


def test_relative_rise_holds_steady_echo_fires_on_onset():
    s = _make_session()
    # Steady echo adapts baseline; same-level frames must not reach thresholds.
    for _ in range(30):
        s._update_echo(0.006, 0.10)
    fast_thr, sust_thr = s._barge_thresholds(0.10)
    assert fast_thr > 0.006 and sust_thr > 0.006  # echo held
    # Abrupt user onset (4x baseline) clears both paths.
    assert fast_thr <= 0.025 and sust_thr <= 0.025
    # Coupling learned into a sane band, not the impossible 1.5x digital ratio.
    assert s._coupling is not None and 0.002 <= s._coupling <= 1.0


def test_fast_path_needs_fewer_frames_than_sustained():
    s = _make_session(barge_in_fast_frames=2, barge_in_frames_speaking=4)
    assert s.barge_in_fast_frames < s.barge_in_frames_speaking


def test_strip_echo_prefix_keeps_genuine_remainder():
    f = VoiceSessionCoordinator._strip_echo_prefix
    assert f("How can I help you today I need a taxi",
             "Hey there! How can I help you today?") == "I need a taxi"
    assert f("Can I help you today?", "Hey there! How can I help you today?") == ""
    assert f("I need to plan a trip to Goa",
             "Hey there! How can I help you today?") == "I need to plan a trip to Goa"
    assert f("stop", "Morning block runs from 8 AM to 12 PM.") == "stop"


def test_glued_turn_is_answered_not_dropped():
    import asyncio as _aio

    async def _run():
        logs: list[str] = []
        s = _make_session(log_debug=logs.append)
        s.state.last_assistant_text = "Hey there! How can I help you today?"
        await s.on_turn_complete("How can I help you today I need a taxi")
        assert s.state.turn_count == 1  # answered with de-echoed text
        assert s.state.last_user_text == "I need a taxi"
        assert any("de-echoed" in m for m in logs)
        for t in list(s.active_tasks):
            t.cancel()

    _aio.run(_run())
