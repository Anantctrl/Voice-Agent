"""Zero-lag orchestrator (Component 8 + diagram boxes 3 & 4).

Wiring (matches the architecture diagram):
  Microphone (16 kHz PCM16) -> Capture Ring (async Queue, lock-free handoff)
      |-Raw PCM stream--> Persistent Deepgram WebSocket -\
      |-32 ms frames---> Silero VAD (ONNX) --------------> Session State FSM
          Speech Start/End -> Cancel Token (Gen ID) -> Cancellation Engine:
             Drop active generation + Flush socket buffers + Clear audio
             buffer instantly. Committed Turn -> Groq Streaming LLM ->
             Token Stream -> Adaptive Clause Chunker -> First 4 words or
             Sentence -> Streaming TTS WebSocket -> Decoded PCM16 ->
             Playback Ring Buffer -> Headset/Speaker.

Rules enforced here:
- No sleep()/timer between VAD end and LLM start.
- Deepgram WS stays open across turns.
- gen_id checked at EVERY yield/await boundary (no ghost voice).
- Zero queues between LLM->chunker->TTS (straight generator pass-through).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

from voice_agent.audio.device import AudioDeviceManager
from voice_agent.observability.metrics import Metrics
from voice_agent.observability.tracing import build_turn_record, new_session_id
from voice_agent.pipeline.chunker import AdaptiveClauseChunker
from voice_agent.pipeline.state import SessionState, SessionStatus
from voice_agent.vad.onnx_vad import SileroVADONNX

log = logging.getLogger("voice_agent.session")

# Function words carry no speaker identity ("the" appears in virtually every
# utterance): counting them in echo overlap lets one shared article plus one
# shared noun kill a 3-word follow-up ("include the LLM"). Excluded from both
# sides of the overlap score; the strip matcher keeps them for alignment.
STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "is", "it", "in", "and", "that", "this",
    "for", "on", "are", "was", "be", "as", "at", "or", "by", "with", "from",
    "you", "we", "they", "he", "she", "i", "me", "my", "your", "our",
})


class VoiceSessionCoordinator:
    def __init__(self, audio: AudioDeviceManager, vad: SileroVADONNX,
                 stt, llm, tts,
                 barge_in_frames: int = 2,
                 barge_in_cooldown_s: float = 0.4,
                 barge_in_frames_speaking: int = 4,
                 barge_in_mic_floor: float = 0.008,
                 barge_in_echo_ratio: float = 1.8,
                 barge_in_rise_ratio: float = 3.5,
                 barge_in_fast_ratio: float = 3.5,
                 barge_in_fast_frames: int = 2,
                 barge_in_blank_s: float = 0.4,
                 echo_recency_s: float = 2.0,
                 barge_in_mode: str = "text",
                 barge_in_min_novel_words: int = 2,
                 barge_in_min_novel_ratio: float = 0.5,
                 validator_min_confidence: float = 0.90,
                 validator_barge_window_s: float = 0.2,
                 validator_multi_confidence: float = 0.65,
                 validator_soup_ratio: float = 0.2,
                 log_debug: Callable[[str], None] | None = None,
                 on_assistant_text: Callable[[str], Awaitable[None] | None] | None = None,
                 on_telemetry: Callable[[dict], None] | None = None):
        self.audio = audio
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.barge_in_frames = max(1, barge_in_frames)
        # Echo-resistant barge-in: the mic hears our speaker, and digital
        # playback RMS lives in a different domain than acoustic mic RMS, so
        # an absolute comparison can never fire. Instead we track a slow
        # echo baseline (+ digital coupling) and fire on ABRUPT RISES above
        # it: fast path for loud onsets (~70 ms), sustained path (~130 ms).
        self.barge_in_cooldown_s = max(0.0, barge_in_cooldown_s)
        self.barge_in_frames_speaking = max(self.barge_in_frames, barge_in_frames_speaking)
        self.barge_in_mic_floor = barge_in_mic_floor
        self.barge_in_echo_ratio = barge_in_echo_ratio  # coupling margin
        self.barge_in_rise_ratio = barge_in_rise_ratio
        self.barge_in_fast_ratio = barge_in_fast_ratio
        self.barge_in_fast_frames = max(1, barge_in_fast_frames)
        # Onset blanking: echo onset is acoustically identical to a user
        # onset, so no energy trigger may fire in the first seconds-fraction
        # of each playback episode. Text confirmation is unaffected.
        self.barge_in_blank_s = max(0.0, barge_in_blank_s)
        self.echo_recency_s = max(0.0, echo_recency_s)
        # Post-barge suspicion: for a while after any cut, transcripts use
        # the strict echo bar even when idle/flushed (echo tails outlive it).
        self.barge_in_suspicion_s = 1.5
        self._last_barge_ts: float | None = None
        # Live mic peak (fast attack, slow release) for the 1-word rule.
        self._mic_live: float = 0.0
        # Instantaneous frame RMS for attribution: the peak above decays too
        # slowly to describe *now* (it reported minutes-old peaks as current
        # speech). Attribution reads this; interruption keeps the peak.
        self._mic_instant: float = 0.0
        # "text": energy never cuts playback by itself; an interim transcript
        # with words we are NOT currently saying confirms a real interrupt.
        # "energy": legacy RMS-rise heuristic (echo-prone without AEC).
        self.barge_in_mode = barge_in_mode
        self.barge_in_min_novel_words = barge_in_min_novel_words
        self.barge_in_min_novel_ratio = barge_in_min_novel_ratio
        from voice_agent.pipeline.validator import TranscriptValidator
        self.validator = TranscriptValidator(
            min_single_confidence=validator_min_confidence,
            barge_window_s=validator_barge_window_s,
            min_multi_confidence=validator_multi_confidence,
            soup_ratio=validator_soup_ratio)
        self._recent_assistant: deque[str] = deque(maxlen=3)
        self._last_interim_words: list[str] = []
        self._stt_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self._stt_task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        self.stt_send_timeouts = 0
        self.stt_q_drops = 0
        self._echo_baseline: float | None = None
        self._coupling: float | None = None
        self._echo_adapted_frames: int = 0
        self._streak_start_ts: float | None = None
        self.log_debug = log_debug
        self._speech_started_ts: float | None = None
        # Mic-health window: aggregated per ~1 s for --debug diagnostics.
        self._mic_rms_sum = 0.0
        self._mic_rms_peak = 0.0
        self._mic_window_n = 0
        self._last_mic_log_ts = time.monotonic()
        self._last_chunk_ts = time.monotonic()
        self._watchdog_fired = False
        self.state = SessionState()
        self.session_id = new_session_id()
        self.metrics = Metrics()
        self._pending_trace: dict | None = None
        self.active_tasks: set[asyncio.Task] = set()
        self._ingest_task: asyncio.Task | None = None
        self._running = False
        self.on_assistant_text = on_assistant_text
        self.on_telemetry = on_telemetry

    @staticmethod
    def _rms16(chunk: bytes) -> float:
        """RMS loudness of a 16-bit mono PCM frame, 0..1."""
        if not chunk:
            return 0.0
        try:
            import numpy as _np
            s = _np.frombuffer(chunk, dtype=_np.int16).astype(_np.float32)
            if s.size == 0:
                return 0.0
            return float(_np.sqrt(_np.mean(s * s)) / 32768.0)
        except Exception:
            return 0.0

    def _update_echo(self, mic_rms: float, play_rms: float) -> None:
        """Adapt echo baseline + digital coupling on non-triggered frames."""
        base = self._echo_baseline
        if base is None:
            self._echo_baseline = mic_rms
        else:
            nb = base + (mic_rms - base) * 0.08
            self._echo_baseline = max(0.001, min(0.05, nb))
        if play_rms > 1e-4:
            inst = mic_rms / play_rms
            inst = max(0.002, min(1.0, inst))
            if self._coupling is None:
                self._coupling = inst
            else:
                self._coupling += (inst - self._coupling) * 0.05
        self._echo_adapted_frames = min(1000, self._echo_adapted_frames + 1)

    def _barge_thresholds(self, play_rms: float) -> tuple[float, float]:
        """(fast_thr, sustained_thr) for the current echo estimate."""
        floor = self.barge_in_mic_floor
        base = self._echo_baseline
        fast_thr = floor * 2.0
        sust_thr = floor
        if base is not None:
            fast_thr = max(fast_thr, base * self.barge_in_fast_ratio)
            sust_thr = max(sust_thr, base * self.barge_in_rise_ratio)
        if self._coupling is not None and play_rms > 1e-4:
            sust_thr = max(sust_thr, play_rms * self._coupling * self.barge_in_echo_ratio)
        return fast_thr, sust_thr

    def _energy_should_fire(self, mic_rms: float, play_rms: float,
                            streak: int, episode_age_s: float) -> str | None:
        """Guarded energy trigger. Returns 'fast', 'sustained', or None.

        Guards (each one exists because its absence caused a real failure):
        - onset blanking: nothing fires in the first barge_in_blank_s of a
          playback episode (echo onset == user onset acoustically);
        - calibration lock: the fast path needs >= 8 adapted echo frames;
        - hysteresis: strict > with x1.05 (float-equality self-triggers).
        """
        if episode_age_s < self.barge_in_blank_s:
            return None
        fast_thr, sust_thr = self._barge_thresholds(play_rms)
        if (streak >= self.barge_in_fast_frames
                and self._echo_adapted_frames >= 8
                and mic_rms > fast_thr * 1.05):
            return "fast"
        if (streak >= self.barge_in_frames_speaking
                and self._echo_baseline is not None
                and mic_rms > sust_thr * 1.05):
            return "sustained"
        return None

    def _dbg(self, msg: str) -> None:
        # Exactly one sink: the caller-provided hook when present (main wires
        # it to logging), else the module logger. Never both (double lines).
        if self.log_debug is not None:
            with contextlib.suppress(Exception):
                self.log_debug(msg)
        else:
            log.debug(msg)

    @property
    def status(self) -> str:
        return self.state.status

    @property
    def generation_id(self) -> int:
        return self.state.generation_id

    # ---------------- lifecycle ----------------
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        # Audio streams started on the running loop for call_soon_threadsafe.
        try:
            self.audio.start(loop)
        except TypeError:
            # Back-compat if a device-bound start_with_devices is needed.
            self.audio.start(loop)
        # Wire STT callback BEFORE connect so no turn is lost.
        with contextlib.suppress(Exception):
            self.stt.on_turn_complete = self.on_turn_complete  # type: ignore
        self._running = True
        self._last_chunk_ts = time.monotonic()
        # Consumers first: frames arriving during connect must be drained at
        # line rate, never accumulate into a stale backlog for turn 1.
        self._ingest_task = asyncio.create_task(self._audio_ingest_loop(), name="mic-ingest")
        self._stt_task = asyncio.create_task(self._stt_sender(), name="stt-sender")
        await self.stt.connect()
        self._flush_input("post-connect")  # drop pre-run mic muttering
        # First-turn warmup (background): pay Groq TLS + TTS handshake now so
        # turn #1 doesn't. Behavior-neutral; failures only logged.
        self._warm_task = asyncio.create_task(self._warmup_providers(), name="provider-warmup")

    def _flush_input(self, reason: str = "") -> int:
        """Drop queued mic frames (stale audio must never become turn 1)."""
        n = 0
        while True:
            try:
                self.audio.input_queue.get_nowait()
                n += 1
            except Exception:
                break
        if n:
            self._dbg(f"[mic] flushed {n} stale frames ({reason})")
        self._last_chunk_ts = time.monotonic()
        return n

    async def _warmup_providers(self) -> None:
        # Deferred past loop stabilization: provider TLS handshakes contending
        # the event loop in second 0 starved ingest (n=88 backlog) and ate a
        # first-turn "hello". Behavior identical after second ~2.
        try:
            await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            return
        try:
            warm = getattr(self.llm, "warmup", None)
            if callable(warm):
                t0 = time.perf_counter()
                ok = await warm()
                dt = (time.perf_counter() - t0) * 1000.0
                self._dbg(f"[warmup] llm {'ready' if ok else 'skipped/failed'} in {dt:.0f}ms")
        except Exception as e:
            self._dbg(f"[warmup] llm failed: {type(e).__name__}")
        try:
            warm = getattr(self.tts, "warmup", None)
            if callable(warm):
                t0 = time.perf_counter()
                ok = await warm()
                dt = (time.perf_counter() - t0) * 1000.0
                self._dbg(f"[warmup] tts {'ready' if ok else 'skipped/failed'} in {dt:.0f}ms")
        except Exception as e:
            self._dbg(f"[warmup] tts failed: {type(e).__name__}")

    async def stop(self) -> None:
        self._running = False
        if self._ingest_task:
            self._ingest_task.cancel()
        for t in (self._stt_task, self._warm_task):
            if t:
                t.cancel()
        for t in list(self.active_tasks):
            t.cancel()
        self.active_tasks.clear()
        try:
            await asyncio.wait_for(self.stt.close(), 2.0)
        except Exception as e:
            log.warning("stt close failed: %r", e)
        aclose = getattr(self.tts, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(aclose(), 1.0)
        with contextlib.suppress(Exception):
            self.audio.stop()

    # ---------------- barge-in (Cancellation Engine) ----------------
    def _cancel_turn(self) -> None:
        """Invalidate the active generation, flush playback, cancel turn tasks."""
        self.state.new_generation()  # 1. invalidate token
        self._speech_started_ts = None
        self._last_barge_ts = time.monotonic()
        self._last_interim_words = []
        self.audio.clear_playback()  # 2. flush hardware buffer instantly
        for task in list(self.active_tasks):  # 3. cancel network tasks
            task.cancel()
        self.active_tasks.clear()

    def _is_busy(self) -> bool:
        busy = self.state.status in ("THINKING", "SPEAKING")
        with contextlib.suppress(Exception):
            busy = busy or self.audio.is_playing()
        return busy

    async def on_user_speech_start(self) -> None:
        """Instant interruption: drop generation + flush playback.

        Also fires when synthesis already finished but audio is still playing
        (status is LISTENING then, yet the user is hearing the reply).
        """
        if self._is_busy():
            self._cancel_turn()

    def _energy_confirms(self) -> bool:
        """Is the mic currently hot enough that speech is a real voice?"""
        base = self._echo_baseline
        live = self._mic_live
        if base is not None and live >= base * self.barge_in_rise_ratio:
            return True
        return live >= self.barge_in_mic_floor * 2.0

    def note_speech_started(self) -> None:
        """Server heard speech: attribute it (echo vs user) from live levels.

        Called on Deepgram SpeechStarted events. Clearly above the echo
        baseline means the user; near it while we play means the room
        answering itself. Below the absolute floor there is no signal at all
        (the old code read 0 >= 0 as "user"); uncertain cases say so instead
        of guessing.
        """
        try:
            base = self._echo_baseline or 0.0
            live = self._mic_instant
            playing = self._is_busy()
            if live < 0.005:
                verdict = "uncertain" if not playing else "echo"
            elif live >= (base * 3.0 if playing
                          else max(base * 2.0, self.barge_in_mic_floor * 2.0)):
                verdict = "user"
            else:
                verdict = "echo" if playing else "uncertain"
            self._dbg(f"[stt] SpeechStarted attributed {verdict} "
                      f"(live={live:.4f} base={base:.4f} playing={playing})")
        except Exception:
            pass

    async def on_interim_text(self, text: str) -> None:
        """Text-confirmed barge-in: cut on novel words, incl. sharp 1-word cuts.

        Novelty is judged on the DELTA since the last interim, not the whole
        cumulative transcript: echo restatements otherwise dilute a real
        interrupt's new words below the ratio bar (observed 0.43 on 6 genuinely
        new words). Pure restatements carry no delta and never fire.
        """
        if self.barge_in_mode != "text" or not self._is_busy():
            return
        words = self._norm_words(text)
        if not words:
            return
        prev = self._last_interim_words
        # Restated echo carries no new information; revised transcripts are
        # judged whole since word positions may have shifted.
        delta = words[len(prev):] if prev and words[:len(prev)] == prev else words
        self._last_interim_words = words
        if not delta:
            return
        spoken = set(self._norm_words(self._echo_reference()))
        novel = [w for w in delta if not self._is_echo_word(w, spoken)]
        ratio = len(novel) / len(delta)
        if (len(novel) >= self.barge_in_min_novel_words
                and ratio >= self.barge_in_min_novel_ratio):
            self._dbg(f"[barge] text-confirmed: novel={novel[:6]} interim={text[:60]!r}")
            self.metrics.count("barge:text")
            self._cancel_turn()
            return
        if len(novel) == 1 and ratio >= 1.0 and self._energy_confirms():
            # "Stop!" — single novel word plus a hot mic: real interrupt.
            self._dbg(f"[barge] text-confirmed (1-word): novel={novel} interim={text[:60]!r}")
            self.metrics.count("barge:text-1word")
            self._cancel_turn()
            return
        if novel:
            self._dbg(f"[barge] almost: novel={novel[:6]} ratio={ratio:.2f} "
                      f"live={self._mic_live:.4f} base={self._echo_baseline or 0:.4f}")

    def _expects_answer(self) -> bool:
        """Did we just ask something? A trailing '?' means short replies are
        answers ("Friday"), not echo — even when the word appears in the
        question itself."""
        for src in (self.state.last_assistant_text, self._echo_reference()):
            if src.strip().endswith("?"):
                return True
        return False

    def _echo_reference(self) -> str:
        """Everything we said or are saying: current partial + recent replies."""
        return " ".join([*self._recent_assistant, self.state.current_spoken]).strip()

    # ---------------- turn execution (Generation & Streaming Playback) ----------------
    @staticmethod
    def _stem_match(a: str, b: str) -> bool:
        """Same word up to STT inflection/noise? ('build'/'built' vs
        'buildings'). Longest-common-substring over 80% of the shorter word;
        short words match exactly only."""
        if a == b:
            return True
        if len(a) < 4 or len(b) < 4:
            return False
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        best, ls = 0, len(short)
        for i in range(ls):
            for j in range(i + 1, ls + 1):
                if short[i:j] in long and j - i > best:
                    best = j - i
                    if best == ls:
                        return True
        return best / ls >= 0.8

    @staticmethod
    def _is_echo_word(word: str, spoken: set[str]) -> bool:
        """Echo-word test with stem awareness: 'build'/'built' vs 'buildings'
        is echo, not a novel word."""
        return any(VoiceSessionCoordinator._stem_match(word, s) for s in spoken)

    @staticmethod
    def _norm_words(s: str) -> list[str]:
        import re
        s = s.lower()
        # Split attached numbers/letters ("8am" -> "8 am") and strip
        # punctuation ("block." -> "block") so STT/TTS formatting
        # differences don't hide the overlap.
        s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        return " ".join(s.split()).split()

    @staticmethod
    def _strip_echo_prefix(transcript: str, last_assistant: str) -> str:
        """Remove a leading echo run glued to genuine user speech.

        Deepgram often commits "...<own TTS words> <user words>" as one turn
        when the user talks over playback. Strip the longest leading run
        that appears VERBATIM (in order) in what we just said, keeping the
        real remainder. Bag-of-words matching is deliberately NOT used here:
        common words ("I", "a") would eat genuine speech. Returns "" when
        nothing but echo remains.
        """
        orig = transcript.split()
        tw = VoiceSessionCoordinator._norm_words(transcript)
        aw = VoiceSessionCoordinator._norm_words(last_assistant)
        if not tw:
            return ""
        if not aw:
            return transcript.strip()
        tj, aj = " ".join(tw), " ".join(aw)
        if tj == aj or tj in aj:
            return ""
        best = 0
        for p in range(min(len(tw) - 1, len(aw)), 1, -1):
            seq = tw[:p]
            if any(aw[i:i + p] == seq for i in range(len(aw) - p + 1)):
                best = p
                break
        if best == 0:
            return transcript.strip()
        # Map norm-token cut back to original words (digit-splits like "8am"
        # can make norm longer than the original split).
        counts = [len(VoiceSessionCoordinator._norm_words(w)) for w in orig]
        cut, used = len(orig) - 1, 0
        for i, c in enumerate(counts):
            used += c
            if used >= best:
                cut = min(len(orig) - 1, i + 1)
                break
        return " ".join(orig[cut:]).strip() if cut < len(orig) else ""

    @staticmethod
    def _rescue_tail(transcript: str, reference: str, min_words: int = 3) -> str:
        """Recover a genuine request glued inside echo (turn #7 class).

        After de-echo stripping, echo words can still be *interleaved* through
        the remainder, failing overlap as a whole. The user's own words
        survive as the longest trailing run absent (stem-aware) from what we
        said — e.g. '...Explain artificial intelligence. Please.' Return "" if
        no such run reaches min_words.
        """
        tw = VoiceSessionCoordinator._norm_words(transcript)
        aw = VoiceSessionCoordinator._norm_words(reference)
        if not tw or not aw:
            return ""
        aw_set = set(aw)

        def _novel(w: str) -> bool:
            return not any(VoiceSessionCoordinator._stem_match(w, a) for a in aw_set)

        best, run = 0, 0
        for w in reversed(tw):
            if _novel(w):
                run += 1
                best = max(best, run)
            else:
                break
        else:
            # Ran off the front: every word is novel (genuine speech, no echo
            # at all) — return whole, unless too short to trust on its own.
            return transcript.strip() if len(tw) >= min_words else ""
        if best < min_words:
            return ""
        tail = tw[len(tw) - best:]
        orig = transcript.split()
        if len(orig) == len(tw):
            return " ".join(orig[len(orig) - best:])
        return " ".join(tail)  # digit-splits misalign counts; norm text still asks it

    @staticmethod
    def _is_likely_echo(transcript: str, last_assistant: str, threshold: float = 0.6) -> bool:
        """Belt-and-suspenders: drop STT results that are our own TTS echo.

        The mic gate (ingest loop) stops most feedback, but echo tails that
        arrive just after the gate releases still produce speech_final. Those
        transcripts heavily overlap the text we just spoke.
        """
        tw = VoiceSessionCoordinator._norm_words(transcript)
        aw = VoiceSessionCoordinator._norm_words(last_assistant)
        t, a = " ".join(tw), " ".join(aw)
        if not t or not a:
            return False
        if t == a or t in a or a in t:
            return True
        if not tw:
            return False
        # Stem-aware overlap on content words only: fragments ("built") of
        # spoken words ("buildings") count as echo, not novelty — but bare
        # function words ("the") prove nothing either way and are excluded
        # from both sides, so short follow-ups ("include the LLM") are judged
        # on their content words alone.
        tw_c = [w for w in tw if w not in STOPWORDS]
        aw_c = [w for w in aw if w not in STOPWORDS]
        if not tw_c:
            return False
        hits = sum(1 for w in tw_c
                   if any(VoiceSessionCoordinator._stem_match(w, a) for a in aw_c))
        overlap = hits / len(tw_c)
        return overlap >= threshold and len(tw_c) <= max(12, len(aw_c))

    def _trace(self, turn_id: int | None = None, **fields) -> None:
        try:
            fields.setdefault("mic_rms", round(self._mic_live, 4))
            if "echo_reduction_db" not in fields:
                try:
                    stats = self.audio.aec_stats()
                    fields["echo_reduction_db"] = float(stats.get("erle_db", 0.0)) \
                        if stats.get("enabled") else 0.0
                except Exception:
                    fields.setdefault("echo_reduction_db", 0.0)
            rec = build_turn_record(self.session_id,
                                    self.state.turn_count if turn_id is None else turn_id,
                                    **fields)
            self._pending_trace = rec
            self._dbg(f"[trace] {rec}")
        except Exception:
            pass

    async def on_turn_complete(self, user_transcript: str,
                               confidence: float | None = None) -> None:
        """Dispatched the moment STT confirms speech_final. No artificial lag."""
        text = (user_transcript or "").strip()
        if not text:
            return
        playing = self.state.status == "SPEAKING"
        with contextlib.suppress(Exception):
            playing = playing or self.audio.is_playing()
        if not playing and self._last_barge_ts is not None:
            # Suspicion window: flushed echo tails outlive playback state.
            # Judge them strictly or they commit as ghost turns and loop.
            playing = (time.monotonic() - self._last_barge_ts) < self.barge_in_suspicion_s
        reference = self._echo_reference() or self.state.last_assistant_text
        short = len(self._norm_words(text)) < 4
        tsb = None
        if self._last_barge_ts is not None:
            tsb = time.monotonic() - self._last_barge_ts
        try:
            since_play = self.audio.seconds_since_playback()
        except Exception:
            since_play = 0.0
        # Recency gate: echo is physically impossible when the speaker has
        # been silent for a while with no recent barge — topical follow-ups
        # ("include the LLM") must not die matching stale assistant text.
        echo_impossible = ((not playing) and since_play > self.echo_recency_s
                           and (tsb is None or tsb > self.echo_recency_s))
        if echo_impossible:
            self._dbg(f"[filter] recency gate: silent {since_play:.1f}s, echo impossible")
        if (not playing) and short and self._expects_answer():
            # Answering our question ("Friday?" <- "Friday"): accept without
            # echo/validator screening, which cannot tell answers from echo.
            self._dbg(f"[filter] answer-to-question: {text[:80]!r}")
        else:
            # Transcript validation judges the text on its own merits
            # (fragments, noise, post-barge shards) — including short idle
            # replies without a question, so bare fragments ("sit") cannot
            # hijack the conversation; echo matching follows unless the
            # recency gate already ruled echo out.
            ok, reason = self.validator.validate(text, confidence, tsb)
            if not ok:
                self.metrics.count(f"turn_dropped:validator-{reason}")
                self._dbg(f"[filter] rejected reason={reason} text={text[:80]!r}")
                self._trace(committed=False, stt_text=text, stt_confidence=confidence,
                            validation="rejected", validation_reason=reason,
                            tts_playing=playing, turn_id=self.state.turn_count + 1)
                return
            if echo_impossible:
                # Skip echo-text matching only; validator above still ran.
                # Falls through to the shared commit below (single counting).
                pass
            else:
                # First peel echo glued to real speech ("...today? I need X" -> "I need X").
                stripped = self._strip_echo_prefix(text, reference)
                if not stripped:
                    self.metrics.count("turn_dropped:echo-prefix")
                    self._dbg(f"[filter] dropped-as-echo: {text[:80]!r}")
                    self._trace(committed=False, stt_text=text, stt_confidence=confidence,
                                validation="dropped-echo-prefix", tts_playing=playing,
                                turn_id=self.state.turn_count + 1)
                    return
                if stripped != text:
                    self._dbg(f"[filter] de-echoed {text[:60]!r} -> {stripped[:60]!r}")
                    text = stripped
            if (not echo_impossible) and self._is_likely_echo(
                    text, reference, threshold=0.4 if playing else 0.6):
                rescued = self._rescue_tail(text, reference)
                if rescued:
                    self.metrics.count("turn_rescued:echo-tail")
                    self._dbg(f"[filter] rescued {rescued[:60]!r} from {text[:60]!r}")
                    text = rescued
                else:
                    self.metrics.count("turn_dropped:echo-overlap")
                    self._dbg(f"[filter] dropped-as-echo while {'SPEAKING' if playing else 'idle'}: {text[:80]!r}")
                    self._trace(committed=False, stt_text=text, stt_confidence=confidence,
                                validation="dropped-echo-overlap", tts_playing=playing,
                                turn_id=self.state.turn_count + 1)
                    return
        if self.active_tasks:
            self._cancel_turn()  # never let two turns speak at once
        self.state.current_spoken = ""
        self.state.status = SessionStatus.THINKING
        self.state.last_user_text = text
        self.state.turn_count += 1
        self.metrics.count("turn_committed")
        if confidence is not None:
            self.metrics.sample("stt_conf", confidence)
        self._trace(committed=True, stt_text=text, stt_confidence=confidence,
                    validation="accepted", tts_playing=playing)
        # Per-turn telemetry reset: stale marks from the previous turn
        # produced negative deltas in the log. All five keys are per-turn.
        for _k in ("T_first_token", "T_tts_start", "T_llm_done", "T_first_audio"):
            self.state.t.pop(_k, None)
        self._last_interim_words = []
        self.state.mark("T_turn_end", time.perf_counter())
        this_gen = self.state.generation_id
        task = asyncio.create_task(self._execute_turn(text, this_gen), name=f"turn-{self.state.turn_count}")
        self.active_tasks.add(task)
        task.add_done_callback(lambda t: self.active_tasks.discard(t))

    async def _execute_turn(self, prompt: str, gen_id: int) -> None:
        chunker = AdaptiveClauseChunker()
        assistant_parts: list[str] = []
        t_first_token: float | None = None
        t_first_audio: float | None = None
        try:
            async for token in self.llm.stream_response(prompt):
                if gen_id != self.state.generation_id:
                    return  # cancelled by barge-in
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    self.state.mark("T_first_token", t_first_token)
                phrases = chunker.push(token)
                for phrase in phrases:
                    assistant_parts.append(phrase)
                    self.state.current_spoken = " ".join(assistant_parts)
                    if "T_tts_start" not in self.state.t:
                        self.state.mark("T_tts_start", time.perf_counter())
                    played = await self._synthesize_and_play(phrase, gen_id)
                    if t_first_audio is None and played:
                        t_first_audio = time.perf_counter()
                        self.state.mark("T_first_audio", t_first_audio)
                    if gen_id != self.state.generation_id:
                        return
            if "T_llm_done" not in self.state.t:
                self.state.mark("T_llm_done", time.perf_counter())
            for phrase in chunker.flush():
                if gen_id != self.state.generation_id:
                    return
                assistant_parts.append(phrase)
                self.state.current_spoken = " ".join(assistant_parts)
                if "T_tts_start" not in self.state.t:
                    self.state.mark("T_tts_start", time.perf_counter())
                played = await self._synthesize_and_play(phrase, gen_id)
                if t_first_audio is None and played:
                    t_first_audio = time.perf_counter()
                    self.state.mark("T_first_audio", t_first_audio)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("turn failed: %s: %s", type(e).__name__, str(e)[:160], exc_info=True)
        finally:
            # Remember what was said even when the turn was interrupted: the
            # echo filter and the LLM history both need the partial text.
            spoken = " ".join(assistant_parts).strip()
            if spoken:
                self._recent_assistant.append(spoken)
                self.state.last_assistant_text = spoken
                commit = getattr(self.llm, "commit_assistant", None)
                if callable(commit):
                    try:
                        commit(spoken)
                    except Exception:
                        log.exception("history commit failed")
            if gen_id == self.state.generation_id:
                self.state.status = SessionStatus.LISTENING
                self._speech_started_ts = None
                full = spoken
                if full and self.on_assistant_text is not None:
                    try:
                        r = self.on_assistant_text(full)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        pass
                if self.on_telemetry is not None and self._running:
                    with contextlib.suppress(Exception):
                        self.on_telemetry(dict(self.state.t))
                    end, aud = self.state.t.get("T_turn_end"), self.state.t.get("T_first_audio")
                    if end and aud and aud >= end:
                        ttfa_ms = (aud - end) * 1000.0
                        self.metrics.sample("ttfa_ms", ttfa_ms)
                        if (self._pending_trace is not None
                                and self._pending_trace.get("turn_id") == self.state.turn_count):
                            self._pending_trace["ttfa_ms"] = round(ttfa_ms, 1)
                            self._dbg(f"[trace] final {self._pending_trace}")
                            self._pending_trace = None

    async def _synthesize_and_play(self, phrase: str, gen_id: int) -> bool:
        self.state.status = SessionStatus.SPEAKING
        if self._speech_started_ts is None:
            self._speech_started_ts = time.perf_counter()
        got_audio = False
        try:
            async for pcm in self.tts.synthesize_stream(phrase):
                if gen_id != self.state.generation_id:
                    return got_audio  # stale chunk — drop silently
                if pcm:
                    await self.audio.output_queue.put(pcm)
                    got_audio = True
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("tts synthesize failed: %s: %s", type(e).__name__, str(e)[:160])
        if not got_audio:
            if gen_id == self.state.generation_id:
                log.warning("tts produced no audio for phrase %r", phrase[:40])
            else:
                self._dbg(f"[tts] cut by barge: {phrase[:40]!r}")
        return got_audio

    def _enqueue_stt(self, chunk: bytes) -> None:
        """Never block the ingest loop on the network: drop oldest and count it."""
        if self._stt_q.full():
            try:
                self._stt_q.get_nowait()
                self.stt_q_drops += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self._stt_q.put_nowait(chunk)
        except asyncio.QueueFull:
            self.stt_q_drops += 1

    async def _stt_sender(self) -> None:
        # Never cancel an in-flight send: cancelling ws.send() poisons the
        # websocket (death spiral: slower socket -> more timeouts -> more
        # cancels). Lag is bounded instead by skip-ahead below (<=0.5 s).
        try:
            while self._running:
                chunk = await self._stt_q.get()
                # Skip-ahead: if we fell behind realtime, drop stale audio
                # down to ~0.5 s so Deepgram hears *now*, not 30 s ago.
                skipped = 0
                while self._stt_q.qsize() > 16:
                    try:
                        self._stt_q.get_nowait()
                        skipped += 1
                    except asyncio.QueueEmpty:
                        break
                if skipped:
                    self.stt_q_drops += skipped
                try:
                    await self.stt.send_audio(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.warning("stt send failed: %r", e)
        except asyncio.CancelledError:
            pass

    async def _vad_one(self, chunk: bytes) -> bool:
        try:
            return bool(await asyncio.to_thread(self.vad.is_speech, chunk))
        except Exception:
            return False

    # ---------------- mic ingest (Detection & Hearing) ----------------
    async def _audio_ingest_loop(self) -> None:
        speech_streak = 0
        pending: asyncio.Task | None = None
        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(self.audio.input_queue.get(), timeout=2.0)
                except TimeoutError:
                    # Watchdog: starved mic frames mean a dead/default-device
                    # stream — the classic "no transcribe at all" with zero
                    # logs. Fail loud with a device hint instead of silence.
                    self._dbg("[mic] WARNING: no mic frames for 2s — wrong/default device? "
                              "run --list-devices and check mic gain/mute")
                    continue
                now = time.monotonic()
                self._last_chunk_ts = now
                # Always stream to STT first: VAD must never serialize behind
                # Deepgram, and Deepgram must never wait on VAD. Echo is
                # handled by the energy gate below + transcript suppression.
                self._enqueue_stt(chunk)
                # Mic-health window for --debug (RMS avg/peak, VAD in loop).
                try:
                    _rms = self._rms16(chunk)
                    self._mic_instant = _rms  # fresh level for attribution
                    self._mic_rms_sum += _rms
                    self._mic_rms_peak = max(self._mic_rms_peak, _rms)
                    self._mic_window_n += 1
                    if now - self._last_mic_log_ts >= 1.0:
                        n = max(1, self._mic_window_n)
                        avg = self._mic_rms_sum / n
                        peak = self._mic_rms_peak
                        try:
                            ist = self.audio.input_stats()
                        except Exception:
                            ist = {}
                        try:
                            _as = self.audio.aec_stats()
                            _ae = (f" erle={_as.get('erle_db', '?')}dB "
                                   f"spk={_as.get('classes', '?')}") if _as.get("enabled") else ""
                        except Exception:
                            _ae = ""
                        self._dbg(f"[mic] rms avg={avg:.4f} peak={peak:.4f} n={n} "
                                  f"q={ist.get('queue_depth', '?')} frames={ist.get('frames', '?')} "
                                  f"ovf={ist.get('overflows', '?')} qdrop={ist.get('queue_drops', '?')} "
                                  f"stt_q={self._stt_q.qsize()} stt_drop={self.stt_q_drops} "
                                  f"stt_to={self.stt_send_timeouts} stt_turns={getattr(self.stt, 'turn_count', '?')}{_ae}")
                        self._mic_rms_sum = 0.0
                        self._mic_rms_peak = 0.0
                        self._mic_window_n = 0
                        self._last_mic_log_ts = now
                except Exception:
                    pass
                # Speaker labels are telemetry here, not control: with AEC3
                # suppressing echo ~20 dB, residue cannot sustain an energy
                # streak (onsets stay blanked), so VAD runs ungated on clean
                # audio. Consume labels to keep the queue fresh for stats.
                with contextlib.suppress(Exception):
                    self.audio.pop_label()
                # Pipelined VAD: inference for frame N runs concurrently with
                # the NEXT frame's send_audio, so the ingest loop feeds
                # Deepgram at line rate even when the event loop is busy with
                # LLM/TTS streaming. Decision trails by one frame (32 ms).
                if pending is None:
                    pending = asyncio.create_task(self._vad_one(chunk))
                    continue
                vad_task, pending = pending, asyncio.create_task(self._vad_one(chunk))
                try:
                    is_speech = await vad_task
                except asyncio.CancelledError:
                    pending.cancel()
                    raise
                except Exception:
                    is_speech = False
                if not is_speech:
                    speech_streak = 0
                    self._streak_start_ts = None
                    continue
                speech_streak += 1
                if speech_streak == 1:
                    self._streak_start_ts = time.monotonic()
                try:
                    playing = self.state.status == "SPEAKING" or self.audio.is_playing()
                except Exception:
                    playing = self.state.status == "SPEAKING"
                if not playing:
                    if speech_streak >= self.barge_in_frames:
                        await self.on_user_speech_start()
                        speech_streak = 0
                    continue
                # While our TTS plays the mic hears the speaker. The guarded
                # energy gate fires on abrupt rises above the adapted echo
                # baseline (blanked at onset, locked until calibrated, with
                # hysteresis). In text mode the gate is a fallback alongside
                # interim confirmation; in energy mode it is the decider.
                # Steady echo only adapts the baseline and never fires.
                mic_rms = self._rms16(chunk)
                self._mic_live = max(mic_rms, self._mic_live * 0.95)
                try:
                    play_rms = self.audio.playback_rms()
                except Exception:
                    play_rms = 0.0
                try:
                    episode_age = self.audio.playback_episode_age()
                except Exception:
                    episode_age = float("inf")
                path = self._energy_should_fire(mic_rms, play_rms, speech_streak, episode_age)
                if path is not None:
                    # Fires in every mode: text mode additionally confirms via
                    # interims, energy mode relies on this gate alone.
                    self.metrics.count(f"barge:energy-{path}")
                    if self._streak_start_ts is not None:
                        lat_ms = (time.monotonic() - self._streak_start_ts) * 1000.0
                        self.state.mark("T_barge", lat_ms)
                        self._dbg(f"[barge] cut in {lat_ms:.0f}ms ({path} "
                                  f"mic={mic_rms:.4f} base={self._echo_baseline or 0:.4f} "
                                  f"play={play_rms:.4f})")
                    else:
                        self._dbg(f"[barge] cut ({path} mic={mic_rms:.4f})")
                    await self.on_user_speech_start()
                    speech_streak = 0
                    self._streak_start_ts = None
                else:
                    # Probable echo — adapt, hold streak, keep listening.
                    self._update_echo(mic_rms, play_rms)
        except asyncio.CancelledError:
            pass
        finally:
            if pending is not None:
                pending.cancel()
