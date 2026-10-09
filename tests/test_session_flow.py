"""Regression tests for the real coordinator, STT supervisor and ingest loop,
using fake providers (no network, no audio hardware)."""
import asyncio
import json

from voice_agent.pipeline.session import VoiceSessionCoordinator
from voice_agent.providers import stt_deepgram
from voice_agent.providers.stt_deepgram import DeepgramPersistentSTT


class FakeOut:
    def __init__(self):
        self.chunks = []

    async def put(self, chunk):
        self.chunks.append(chunk)


class FakeAudio:
    def __init__(self):
        self.output_queue = FakeOut()
        self.input_queue = asyncio.Queue(maxsize=64)
        self.playing = False
        self.cleared = 0

    def start(self, loop): pass
    def stop(self): pass
    def is_playing(self): return self.playing
    def playback_rms(self): return 0.0
    def input_stats(self): return {}

    def clear_playback(self):
        self.cleared += 1
        self.playing = False


class FakeVAD:
    def is_speech(self, chunk): return False


class FakeSTT:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.sent = 0
        self.on_turn_complete = None

    async def connect(self): pass
    async def close(self): pass

    async def send_audio(self, chunk):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.sent += 1


class FakeLLM:
    def __init__(self, tokens, delay=0.0):
        self.tokens, self.delay = tokens, delay
        self.committed = []
        self.prompts = []

    async def stream_response(self, prompt):
        self.prompts.append(prompt)
        for t in self.tokens:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield t

    def commit_assistant(self, text):
        self.committed.append(text)


class FakeTTS:
    def __init__(self, chunks=5, delay=0.01):
        self.chunks, self.delay = chunks, delay

    async def synthesize_stream(self, text):
        for _ in range(self.chunks):
            await asyncio.sleep(self.delay)
            yield b"\x01\x00" * 256


def make(tokens=("Booking ", "it ", "for ", "Friday. ", "Anything ", "else? "),
         llm_delay=0.0, tts_chunks=5, stt_delay=0.0, **kw):
    audio, llm = FakeAudio(), FakeLLM(tokens, llm_delay)
    s = VoiceSessionCoordinator(audio, FakeVAD(), FakeSTT(stt_delay), llm,
                                FakeTTS(tts_chunks), **kw)
    return s, audio, llm


async def settle(s, t=1.0):
    end = asyncio.get_running_loop().time() + t
    while s.active_tasks and asyncio.get_running_loop().time() < end:
        await asyncio.sleep(0.01)


def test_barge_in_fires_when_synthesis_done_but_audio_still_playing():
    async def run():
        s, audio, _ = make()
        audio.playing = True            # status is LISTENING, tail still audible
        gen = s.generation_id
        await s.on_user_speech_start()
        assert audio.cleared == 1 and s.generation_id == gen + 1
    asyncio.run(run())


def test_barge_in_noop_when_idle():
    async def run():
        s, audio, _ = make()
        await s.on_user_speech_start()
        assert audio.cleared == 0
    asyncio.run(run())


def test_short_reply_matching_assistant_text_is_not_dropped_when_idle():
    async def run():
        s, _audio, llm = make()
        s._recent_assistant.append("Should I book it for Friday?")
        await s.on_turn_complete("Friday")
        await settle(s)
        assert llm.prompts == ["Friday"]
    asyncio.run(run())


def test_echo_of_cancelled_partial_reply_is_dropped():
    async def run():
        s, audio, llm = make()
        # A reply was interrupted: only the live partial text exists.
        s.state.current_spoken = "Artificial intelligence is the field of computer science"
        audio.playing = True
        await s.on_turn_complete("Artificial intelligence is the field. Of computer science")
        assert llm.prompts == []        # echo must not become a user turn
    asyncio.run(run())


def test_interrupted_turn_commits_partial_text_to_history():
    async def run():
        s, audio, llm = make(llm_delay=0.02, tts_chunks=20)
        await s.on_turn_complete("tell me something")
        await asyncio.sleep(0.15)       # a phrase or two is queued
        audio.playing = True
        await s.on_user_speech_start()
        await asyncio.sleep(0.1)
        assert llm.committed, "partial assistant text must reach history"
        assert s.state.last_assistant_text == llm.committed[-1]
        assert llm.committed[-1].startswith("Booking")
    asyncio.run(run())


def test_second_turn_cancels_first_so_two_never_speak_at_once():
    async def run():
        s, _audio, _llm = make(llm_delay=0.02, tts_chunks=30)
        await s.on_turn_complete("first question please now")
        await asyncio.sleep(0.1)
        first = list(s.active_tasks)
        await s.on_turn_complete("second question please now")
        await asyncio.sleep(0.05)
        assert all(t.cancelled() or t.done() for t in first)
        assert len(s.active_tasks) == 1
    asyncio.run(run())


def test_text_confirmed_barge_in_ignores_echo_and_cuts_on_new_words():
    async def run():
        s, audio, _ = make()
        s.state.current_spoken = "Artificial intelligence is the field of computer science"
        audio.playing = True
        gen = s.generation_id
        await s.on_interim_text("Artificial intelligence is the")   # pure echo
        assert s.generation_id == gen
        await s.on_interim_text("Artificial intelligence is the uh")  # 1 filler word
        assert s.generation_id == gen                     # below the 2-word bar
        await s.on_interim_text("no stop wait cancel that")          # real interrupt
        assert s.generation_id == gen + 1 and audio.cleared == 1
    asyncio.run(run())


def test_empty_interim_is_noop_in_text_mode():
    async def run():
        s, audio, _ = make()
        assert s.barge_in_mode == "text"
        audio.playing = True
        await s.on_interim_text("")     # empty interim is a no-op
        assert audio.cleared == 0
    asyncio.run(run())


def test_slow_stt_socket_does_not_stall_the_ingest_loop():
    async def run():
        s, audio, _ = make(stt_delay=5.0)     # STT send hangs
        s._running = True
        sender = asyncio.create_task(s._stt_sender())
        ingest = asyncio.create_task(s._audio_ingest_loop())
        for _ in range(40):
            audio.input_queue.put_nowait(b"\x00\x00" * 512)
        await asyncio.sleep(0.3)
        assert audio.input_queue.qsize() == 0, "ingest loop must keep draining"
        assert s.stt_q_drops > 0            # backlog is dropped and counted
        s._running = False
        for t in (sender, ingest):
            t.cancel()
        await asyncio.gather(sender, ingest, return_exceptions=True)
    asyncio.run(run())


class FakeWS:
    def __init__(self, messages, then_block):
        self.messages, self.then_block = list(messages), then_block
        self.sent, self.closed = [], False
        self._hang = asyncio.Event()

    async def send(self, data): self.sent.append(data)
    async def close(self): self.closed = True; self._hang.set()

    def __aiter__(self): return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.then_block:
            await self._hang.wait()
        raise StopAsyncIteration


def _final(text):
    return json.dumps({"channel": {"alternatives": [{"transcript": text}]},
                       "is_final": True, "speech_final": True})


def test_stt_receives_again_after_socket_drops(monkeypatch):
    async def run():
        sockets = [FakeWS([], then_block=False),                 # drops immediately
                   FakeWS([_final("hello again")], then_block=True)]

        async def fake_connect(*a, **k):
            return sockets.pop(0)

        monkeypatch.setattr(stt_deepgram.websockets, "connect", fake_connect)
        got = []

        async def on_turn(t): got.append(t)

        stt = DeepgramPersistentSTT("key", on_turn)
        first = sockets[0]
        await stt.connect()
        for _ in range(100):
            if got:
                break
            await asyncio.sleep(0.02)
        await stt.close()
        assert got == ["hello again"], "receive loop must restart on the new socket"
        assert stt.reconnects == 1 and first.closed
    asyncio.run(run())


def test_send_on_closed_socket_does_not_spawn_reconnect_storm(monkeypatch):
    from websockets.exceptions import ConnectionClosed

    async def run():
        class Dead(FakeWS):
            async def send(self, data):
                raise ConnectionClosed(None, None)

        async def fake_connect(*a, **k):
            return Dead([], then_block=True)

        monkeypatch.setattr(stt_deepgram.websockets, "connect", fake_connect)
        stt = DeepgramPersistentSTT("key", lambda t: asyncio.sleep(0))
        await stt.connect()
        before = len(asyncio.all_tasks())
        for _ in range(200):
            await stt.send_audio(b"\x00" * 1024)
        assert len(asyncio.all_tasks()) <= before + 1
        await stt.close()
    asyncio.run(run())


def test_playback_queue_applies_backpressure_instead_of_dropping():
    from voice_agent.audio.device import HybridPlaybackQueue

    async def run():
        q = HybridPlaybackQueue(maxsize=2)
        await q.put(b"a"); await q.put(b"b")
        t = asyncio.create_task(q.put(b"c"))
        await asyncio.sleep(0.05)
        assert not t.done()                 # producer waits
        assert q.get_nowait() == b"a"       # nothing was dropped
        await asyncio.wait_for(t, 1.0)
        assert [q.get_nowait(), q.get_nowait()] == [b"b", b"c"]
    asyncio.run(run())


def test_stt_sender_skips_ahead_to_live_instead_of_lagging():
    async def run():
        s, _audio, _ = make()
        s._running = True
        for _ in range(24):                 # ~0.75 s backlog, instant socket
            s._stt_q.put_nowait(b"\x00\x00" * 512)
        sender = asyncio.create_task(s._stt_sender())
        for _ in range(200):
            if s._stt_q.qsize() <= 16:
                break
            await asyncio.sleep(0.01)
        assert s._stt_q.qsize() <= 16, "lag must stay bounded at ~0.5 s"
        assert s.stt_q_drops > 0, "stale audio is dropped and counted"
        s._running = False
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
    asyncio.run(run())


def test_llm_messages_never_duplicate_the_system_prompt():
    from voice_agent.providers.llm_groq import GroqStreamingLLM
    llm = GroqStreamingLLM(api_key="test-key")
    llm.history.append({"role": "user", "content": "hello"})
    msgs = llm._messages()
    assert sum(1 for m in msgs if m.get("role") == "system") == 1
    for _ in range(10):
        llm.history.append({"role": "user", "content": "x"})
        llm.history.append({"role": "assistant", "content": "y"})
    msgs = llm._messages()
    assert sum(1 for m in msgs if m.get("role") == "system") == 1
    assert len(msgs) <= 7


def test_energy_blanked_during_playback_onset():
    async def run():
        s, _audio, _ = make()
        for _ in range(30):
            s._update_echo(0.006, 0.10)     # calibrated echo level
        # Same loud frame: held inside the onset window, fires after it.
        assert s._energy_should_fire(0.05, 0.10, 10, episode_age_s=0.1) is None
        assert s._energy_should_fire(0.05, 0.10, 10, episode_age_s=5.0) == "fast"
    asyncio.run(run())


def test_fast_path_locked_until_calibrated_and_hysteresis_holds_boundary():
    async def run():
        s, _audio, _ = make()
        # Uncalibrated gate can never fire, however loud.
        assert s._energy_should_fire(0.09, 0.10, 10, float("inf")) is None
        # Boundary equality (float) holds: needs strict > with margin.
        s._echo_adapted_frames = 10
        assert s._energy_should_fire(0.016, 0.0, 2, float("inf")) is None
        assert s._energy_should_fire(0.017, 0.0, 2, float("inf")) == "fast"
    asyncio.run(run())


def test_post_barge_suspicion_window_drops_flushed_echo():
    async def run():
        s, audio, _ = make()
        s.state.last_assistant_text = "Hey there! How can I help you today?"
        s._cancel_turn()                    # false barge flushes playback
        assert not audio.is_playing()       # idle/flushed...
        await s.on_turn_complete("How can I help you today?")  # ...yet echo tail
        assert s.state.turn_count == 0      # ...still dropped as echo
    asyncio.run(run())


def test_single_word_interrupt_fires_with_energy_confirmation():
    async def run():
        logs = []
        s, audio, _ = make(log_debug=logs.append)
        s.state.current_spoken = "Artificial intelligence is the field of computer science"
        audio.playing = True
        for _ in range(10):
            s._update_echo(0.006, 0.10)
        gen = s.generation_id
        s._mic_live = 0.0                   # quiet mic: held + near-miss logged
        await s.on_interim_text("stop")
        assert s.generation_id == gen
        assert any("almost" in m for m in logs)
        s._mic_live = 0.03                  # hot mic: single word cuts
        await s.on_interim_text("stop it")  # cumulative restatement + new word
        assert s.generation_id == gen + 1 and audio.cleared == 1
    asyncio.run(run())


def test_speech_started_event_is_logged_without_creating_turn(monkeypatch):
    async def run():
        started = [FakeWS([json.dumps({"type": "SpeechStarted"})], then_block=True)]

        async def fake_connect(*a, **k):
            return started.pop(0)

        monkeypatch.setattr(stt_deepgram.websockets, "connect", fake_connect)
        logs: list[str] = []
        got: list[str] = []

        async def on_turn(t): got.append(t)

        stt = DeepgramPersistentSTT("key", on_turn, log_debug=logs.append)
        await stt.connect()
        await asyncio.sleep(0.2)
        await stt.close()
        assert got == []                      # event, not a transcript
        assert any("SpeechStarted" in m for m in logs)
        assert "vad_events=true" in stt.ws_url
    asyncio.run(run())


def test_validator_allowlists_and_rejects_by_rule():
    from voice_agent.pipeline.validator import TranscriptValidator
    v = TranscriptValidator()
    assert v.validate("stop") == (True, "valid_single_word")
    assert v.validate("sit") == (False, "unknown_single_word")
    assert v.validate("sit", confidence=0.95) == (True, "high_confidence_single_word")
    assert v.validate("sit", confidence=0.42) == (False, "unknown_single_word")
    assert v.validate("ok buddy", time_since_barge=0.1) == (False, "short_post_barge_fragment")
    assert v.validate("ok buddy")[0] is True
    assert v.validate("aaaaaaa") == (False, "unknown_single_word")
    assert v.validate("123 456") == (False, "incoherent")
    assert v.validate("tell me about water") == (True, "valid")
    assert v.validate("") == (False, "empty")


def test_fragment_sit_rejected_after_barge_no_llm_call():
    async def run():
        logs = []
        s, _audio, llm = make(log_debug=logs.append)
        s.state.last_assistant_text = "Sure thing just let me know"
        s._cancel_turn()                        # recent barge: suspicion active
        await s.on_turn_complete("sit", confidence=0.42)
        assert s.state.turn_count == 0           # rejected, never a turn
        assert llm.prompts == []
        assert any("rejected" in m for m in logs)
    asyncio.run(run())


def test_high_confidence_single_word_flows():
    async def run():
        s, audio, llm = make()
        audio.playing = True                    # force the validator path
        s.state.current_spoken = "hello there friend"
        await s.on_turn_complete("sit", confidence=0.95)
        await settle(s)
        assert llm.prompts == ["sit"]           # high confidence wins
    asyncio.run(run())


def test_metrics_counts_and_percentiles():
    from voice_agent.observability.metrics import Metrics
    m = Metrics()
    m.count("turn_committed")
    m.count("turn_committed")
    m.count("turn_dropped:echo-overlap")
    for v in (100.0, 200.0, 300.0, 400.0):
        m.sample("ttfa_ms", v)
    snap = m.snapshot()
    assert snap["counts"] == {"turn_committed": 2, "turn_dropped:echo-overlap": 1}
    assert snap["distributions"]["ttfa_ms"]["n"] == 4
    assert snap["distributions"]["ttfa_ms"]["p50"] == 300.0
    m.reset()
    assert m.snapshot() == {"counts": {}, "distributions": {}}


def test_tracing_ids_unique_and_record_shaped():
    from voice_agent.observability.tracing import build_turn_record, new_session_id
    a, b = new_session_id(), new_session_id()
    assert a != b and len(a) == 8
    rec = build_turn_record(a, 42, committed=True, stt_text="hello",
                            stt_confidence=0.97, validation="accepted",
                            tts_playing=False, ttfa_ms=1381.2)
    assert rec["session_id"] == a and rec["turn_id"] == 42
    assert rec["committed"] is True and rec["ttfa_ms"] == 1381.2
    assert set(rec) >= {"mic_rms", "echo_reduction_db", "validation_reason",
                        "barge_in", "barge_path"}


def test_logging_configure_is_safe():
    from voice_agent.observability.logging import configure_logging
    configure_logging(debug=False)
    configure_logging(debug=True)


def test_session_status_is_explicit_enum():
    from voice_agent.pipeline.state import SessionState, SessionStatus
    s = SessionState()
    assert s.status == SessionStatus.LISTENING == "LISTENING"
    s.status = SessionStatus.SPEAKING
    assert s.status in (SessionStatus.THINKING, SessionStatus.SPEAKING)
    s.new_generation()
    assert s.status is SessionStatus.LISTENING


def test_turn_commit_and_drop_feed_metrics_and_trace():
    async def run():
        logs = []
        s, _audio, _llm = make(log_debug=logs.append)
        await s.on_turn_complete("tell me about water please")
        await settle(s)
        snap = s.metrics.snapshot()
        assert snap["counts"].get("turn_committed") == 1
        assert any("[trace]" in m and "'committed': True" in m for m in logs)
        s._recent_assistant.append("tell me about water please")
        await s.on_turn_complete("tell me about water please")
        snap = s.metrics.snapshot()
        dropped = [k for k in snap["counts"] if k.startswith("turn_dropped:")]
        assert dropped, "echo repeat must be counted as dropped"
    asyncio.run(run())


def test_stem_fragments_are_echo_not_novelty():
    f = VoiceSessionCoordinator._is_echo_word
    spoken = {"buildings", "are", "constructed"}
    assert f("build", spoken) is True      # echo fragment, not an interrupt
    assert f("built", spoken) is True
    assert f("stop", spoken) is False      # genuinely new word
    assert f("a", {"a", "b"}) is True     # short words: exact only
    assert f("ax", {"a", "b"}) is False


def test_echo_stem_overlap_drops():
    f = VoiceSessionCoordinator._is_likely_echo
    assert f("built built", "buildings are constructed", 0.6) is True
    assert f("stop right now please", "buildings are constructed", 0.6) is False


def test_idle_fragment_without_question_is_validated_not_free():
    async def run():
        s, _audio, llm = make()
        await s.on_turn_complete("sit")   # idle, no question asked
        assert llm.prompts == []          # unknown fragment: no LLM call
    asyncio.run(run())


def test_question_context_still_accepts_short_answers():
    async def run():
        s, _audio, llm = make()
        s._recent_assistant.append("Should I book it for Friday?")
        s.state.last_assistant_text = "Should I book it for Friday?"
        await s.on_turn_complete("Friday")
        await settle(s)
        assert llm.prompts == ["Friday"]
    asyncio.run(run())


def test_speech_started_attribution_logs_verdict():
    async def run():
        logs = []
        s, _audio, _ = make(log_debug=logs.append)
        s._echo_baseline, s._mic_live = 0.01, 0.05
        s._mic_instant = 0.05
        s.note_speech_started()
        assert any("attributed user" in m for m in logs)
        s._mic_live = 0.005
        s._mic_instant = 0.005
        s.note_speech_started()
        assert any("uncertain" in m or "echo" in m for m in logs)
    asyncio.run(run())


def test_validator_rejects_low_confidence_soup():
    from voice_agent.pipeline.validator import TranscriptValidator
    v = TranscriptValidator()
    assert v.validate("Any AI is the I also include about the l l n",
                      confidence=0.607) == (False, "low_confidence_fragment")
    assert v.validate("could you repeat that please",
                      confidence=0.60) == (True, "valid")
    assert v.validate("TVs in the US and UK", confidence=0.95)[0] is True


def test_telemetry_line_names_segments():
    import io
    from contextlib import redirect_stdout

    from voice_agent.main import print_telemetry
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_telemetry({"T_turn_end": 1.0, "T_first_token": 1.5,
                         "T_tts_start": 1.51, "T_llm_done": 1.9,
                         "T_first_audio": 2.1})
    out = buf.getvalue()
    assert "CHUNK" in out and "LLM " in out and "TTFA" in out


def test_startup_flush_drops_stale_input():
    async def run():
        s, audio, _ = make()
        for _ in range(5):
            audio.input_queue.put_nowait(b"\x00" * 512)
        assert s._flush_input("test") == 5
        assert audio.input_queue.qsize() == 0
    asyncio.run(run())


def test_delta_novelty_ignores_restated_echo():
    async def run():
        logs = []
        s, audio, _ = make(log_debug=logs.append)
        s.state.current_spoken = "we should talk about building design today"
        audio.playing = True
        gen = s.generation_id
        await s.on_interim_text("we should talk")          # echo part 1
        await s.on_interim_text("we should talk about building")  # restatement
        assert s.generation_id == gen                     # no cut on restatement
        await s.on_interim_text("we should talk about building stop that now")
        assert s.generation_id == gen + 1                 # new words cut
        assert audio.cleared == 1
    asyncio.run(run())


def test_glued_tail_rescues_genuine_question():
    async def run():
        s, _audio, llm = make()
        s._recent_assistant.append("Hey there! How can I help you today?")
        s.state.last_assistant_text = "Hey there! How can I help you today?"
        await s.on_turn_complete("How can I help you today? Tell me about buildings.")
        await settle(s)
        assert llm.prompts == ["Tell me about buildings."]
    asyncio.run(run())


def test_pure_echo_tail_has_nothing_to_rescue():
    from voice_agent.pipeline.session import VoiceSessionCoordinator as _S
    ref = "Hey there! How can I help you today?"
    assert _S._rescue_tail("How can I help you today?", ref) == ""
    assert _S._rescue_tail("Tell me about buildings.", ref) == "Tell me about buildings."
    assert _S._rescue_tail("Hi", ref) == ""  # too short to trust


def test_committed_confidence_is_sampled():
    async def run():
        s, _audio, _llm = make()
        await s.on_turn_complete("tell me about water please", confidence=0.82)
        await settle(s)
        snap = s.metrics.snapshot()
        assert snap["distributions"]["stt_conf"]["n"] == 1
        assert snap["distributions"]["stt_conf"]["last"] == 0.82
    asyncio.run(run())


def test_attribution_needs_real_signal():
    async def run():
        logs = []
        s, audio, _ = make(log_debug=logs.append)
        s._echo_baseline, s._mic_live = 0.0, 0.0
        s.note_speech_started()   # silence, idle: uncertain, never "user"
        assert any("uncertain" in m for m in logs)
        s._echo_baseline = 0.01
        audio.playing = True
        s.note_speech_started()   # silence during playback: echo, not user
        assert any("attributed echo" in m for m in logs)
    asyncio.run(run())


def test_drop_traces_name_the_would_be_turn():
    async def run():
        logs = []
        s, audio, _llm = make(log_debug=logs.append)
        assert s.state.turn_count == 0
        s.state.current_spoken = "Should I book it for Friday?"
        audio.playing = True
        await s.on_turn_complete("Should I book it for Friday?")
        assert s.state.turn_count == 0
        drops = [m for m in logs if "'committed': False" in m]
        assert drops and "'turn_id': 1" in drops[0]
    asyncio.run(run())


def test_attribution_reads_fresh_level_not_stale_peak():
    async def run():
        logs = []
        s, _audio, _ = make(log_debug=logs.append)
        s._echo_baseline = 0.01
        s._mic_live = 0.06      # stale peak from minutes-old speech...
        s._mic_instant = 0.0001  # ...while the room is actually silent
        s.note_speech_started()
        assert not any("attributed user" in m for m in logs), \
            "stale peak must never read as a live user"
        s._mic_instant = 0.06   # genuinely hot right now
        s.note_speech_started()
        assert any("attributed user" in m for m in logs)
    asyncio.run(run())


def test_recency_gate_accepts_topical_followup_after_silence():
    async def run():
        logs = []
        s, _audio, llm = make(log_debug=logs.append)
        s.state.last_assistant_text = (
            "Artificial intelligence is the field of computer science.")
        # Speaker silent for minutes, no barge: echo physically impossible.
        s.audio = _SilentLongAgo()
        await s.on_turn_complete("include the LLM")
        await settle(s)
        assert llm.prompts == ["include the LLM"]
        assert any("recency gate" in m for m in logs)
    asyncio.run(run())


class _SilentLongAgo:
    """Audio fake: nothing playing, last sound ages ago, no labels."""

    def is_playing(self):
        return False

    def seconds_since_playback(self):
        return 30.0

    def input_stats(self):
        return {}

    def aec_stats(self):
        return {"enabled": False}

    def clear_playback(self):
        pass

    def pop_label(self):
        return None


def test_stopwords_dont_inflate_echo_overlap():
    f = VoiceSessionCoordinator._is_likely_echo
    ref = "Artificial intelligence is the field of computer science"
    # One shared article + one shared noun must not condemn a 3-word turn.
    assert f("include the LLM", ref, 0.4) is False
    assert f("the the the", ref, 0.4) is False  # pure function words: unjudgeable
    assert f("artificial intelligence field", ref, 0.4) is True


def test_repeated_word_is_incoherent():
    from voice_agent.pipeline.validator import TranscriptValidator
    v = TranscriptValidator()
    assert v.validate("built built") == (False, "incoherent")
    assert v.validate("very very good")[0] is True
    assert v.validate("stop") == (True, "valid_single_word")
