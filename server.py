"""
Voice AI realtime server — production streaming pipeline.

Architecture:
  Browser mic → WebRTC → Wake Word (OpenWakeWord) → VAD
  → STT (faster-whisper) → LLM (Groq streaming)
  → sentence buffer (sequence-numbered)
  → async queue (maxsize=10, backpressure)
  → 3 parallel TTS workers (Coqui Tacotron2-DDC)
  → ordered audio buffer (play in sequence order)
  → PlayerTrack → WebRTC → browser speaker

Every stage is logged with monotonic timestamps for latency analysis.
"""

# asyncio powers the event loop that aiortc and FastAPI both run on,
# and to_thread offloads the blocking CPU work (whisper/TTS) so audio keeps flowing.
import asyncio

# collections.deque is an O(1) FIFO queue used to buffer outgoing audio frames.
import collections

# dataclass gives us lightweight Sentence/AudioChunk containers without boilerplate.
from dataclasses import dataclass, field

# fractions.Fraction is required for AV frame timestamps (PyAV 17 no longer
# re-exports it as av.Fraction).
from fractions import Fraction

# io wraps bytes in BytesIO so we can build WAV containers entirely in memory.
import io

# re is used for sentence-boundary detection in the LLM stream.
import re

# queue provides Queue (thread-safe FIFO) for bridging sync generators to async.
import queue as _queue

# time gives us a monotonic clock for pacing outbound audio and performance metrics.
import time

# wave is used to build a tiny silent WAV placeholder when TTS permanently
# fails for a sentence, so the ordered playback buffer never stalls on a gap.
import wave

# typing.Dict for the ordered audio buffer's type hints.
from typing import Dict, Optional

# PyAV ships with aiortc; AudioResampler converts between sample rates/layouts,
# and av.open lets us decode the TTS WAV bytes into raw frames in memory.
import av

# numpy handles fast slicing/reshaping of interleaved PCM samples.
import numpy as np

# FastAPI is the web framework serving the page, SDP signaling and WS events.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# StaticFiles mounts the static/ folder; FileResponse serves index.html at /.
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# pydantic BaseModel validates the JSON body of POST /offer.
from pydantic import BaseModel

# aiortc is the actual WebRTC implementation (peer connections, media tracks).
from aiortc import RTCPeerConnection, RTCSessionDescription

# MediaStreamTrack is the base class we subclass to build our audio "player".
# MediaStreamError is raised by inbound tracks when the browser disconnects.
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError

from src.vad import VADetector, FRAME_BYTES          # speech start/end detection
from src.stt import transcribe                        # PCM -> text
from src.llm_client import chat_stream                 # streaming LLM tokens
from src.tts_engine import synthesize                 # text -> WAV bytes

# OpenWakeWord detects a specific wake phrase ("hey jarvis") in the mic stream
# before we start full STT pipeline, so the assistant only activates on command.
from openwakeword.model import Model as OWWModel

# Sentence-ending characters used to split the LLM stream into TTS-able chunks.
# Semicolons and colons are included too: long explanatory text (lists of
# concepts, definitions) leans on them as natural pause points, and splitting
# there keeps each TTS chunk shorter and grammatically whole instead of one
# giant run-on clause, which is what makes autoregressive TTS rush/garble.
_SENTENCE_END = re.compile(r'[.!?;:]\s|\n')

# Maximum characters accumulated before we force-flush a sentence even without
# a proper ending punctuation — prevents the buffer from growing unbounded
# while waiting for the LLM to finish a long clause.
_MAX_CHARS_NO_END = 300

# Natural breathing gap inserted between consecutive sentences during playback.
# Without this, back-to-back TTS chunks have zero gap and play as one rushed,
# breathless stream instead of sounding like normal human speech.
_INTER_SENTENCE_PAUSE_S = 0.22

# Number of parallel TTS workers running concurrently in the producer-consumer pool.
_TTS_WORKERS = 3

# Maximum number of sentences the queue can hold before the LLM producer blocks
# (backpressure) so memory stays bounded even if TTS falls behind.
_QUEUE_MAXSIZE = 10

# Number of retry attempts per sentence before we skip it entirely.
_TTS_RETRIES = 2

# Seconds of grace after the player queue drains — covers the last buffered frame
# actually reaching the browser speaker.
_DRAIN_GRACE = 0.3


# Create the app instance uvicorn will serve on http://localhost:8000.
app = FastAPI(title="VoiceAI")

# Keep every peer connection in a set so we can close them cleanly on shutdown.
pcs = set()

# Registry of connected WebSockets so transcripts/status can be broadcast.
clients = set()

# Global "busy" flag: while True the mic stream is ignored so the AI isn't
# interrupted by its own voice coming through the user's speakers.
busy = False

# Lazy-loaded OpenWakeWord model: scans every 16kHz mono chunk for "hey jarvis"
# and only returns True when the wake phrase is detected. Loaded once on first
# connection so the ~1.3MB ONNX model stays in memory thereafter.
_oww_model = None


def get_oww():
    """Lazy-load the OpenWakeWord model so server startup stays instant."""
    global _oww_model
    if _oww_model is None:
        _oww_model = OWWModel(
            wakeword_models=[r"D:\Project\voiceai\models\hey_jarvis_v0.1.onnx"],
            inference_framework="onnx",
        )
    return _oww_model


# ---------- silent-audio placeholder for permanently-failed TTS -------------


def _make_silence_wav(duration_s: float = 0.3, rate: int = 22050) -> bytes:
    """Build a short silent WAV in memory.

    Used as a stand-in when a sentence's TTS call fails every retry. Without
    this, push_wav_ordered() for that seq never happens, and PlayerTrack's
    _drain_pending() waits forever on that exact sequence number — silently
    stalling playback of every sentence that came after it, even ones that
    synthesized fine. A short blip of silence keeps the sequence unbroken and
    the rest of the reply audible.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(duration_s * rate))
    return buf.getvalue()


_SILENCE_WAV = _make_silence_wav()


# ---------- performance metrics ---------------------------------------------


class TurnMetrics:
    """Collects monotonic timestamps at every pipeline stage for one turn.

    Logs a single summary line when the turn completes so we can track
    latency regression over time without external monitoring tools.
    """

    def __init__(self):
        self.stt_start = time.monotonic()
        self.stt_end: Optional[float] = None
        self.llm_start: Optional[float] = None
        self.llm_first_token: Optional[float] = None
        self.llm_end: Optional[float] = None
        self.first_sentence_time: Optional[float] = None
        self.tts_start: Dict[int, float] = {}
        self.tts_end: Dict[int, float] = {}
        self.playback_end: Optional[float] = None

    def log(self):
        """Print a one-line summary of every stage's latency."""
        parts = []
        if self.stt_end:
            parts.append(f"STT={self.stt_end - self.stt_start:.2f}s")
        if self.llm_first_token and self.llm_start:
            parts.append(f"TTFA={self.llm_first_token - self.llm_start:.2f}s")
        if self.llm_end and self.llm_start:
            parts.append(f"LLM={self.llm_end - self.llm_start:.2f}s")
        if self.first_sentence_time and self.llm_start:
            parts.append(f"first_audio={self.first_sentence_time - self.llm_start:.2f}s")
        for seq in sorted(self.tts_end.keys()):
            if seq in self.tts_start:
                parts.append(f"TTS[{seq}]={self.tts_end[seq] - self.tts_start[seq]:.2f}s")
        if self.playback_end and self.stt_start:
            parts.append(f"total={self.playback_end - self.stt_start:.2f}s")
        print(f"[metrics] {' '.join(parts)}")


# ---------- sentence / audio chunk containers -------------------------------


@dataclass
class Sentence:
    """A sentence extracted from the LLM stream with its sequence number."""
    seq: int
    text: str


@dataclass
class AudioChunk:
    """A decoded WAV chunk with its sequence number for ordered playback."""
    seq: int
    wav: bytes


# ---------- ordered audio player track --------------------------------------


class PlayerTrack(MediaStreamTrack):
    """A push-based audio track with sequence-ordered playback.

    The old PlayerTrack used a plain deque: if TTS worker 2 finished before
    worker 0, audio played out of order. This version maintains a pending
    buffer keyed by sequence number so chunks always play in the order they
    were produced by the LLM, regardless of TTS completion order.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        # Ready-to-send 48kHz stereo s16 numpy chunks in exact playback order.
        self._queue = collections.deque()
        # Pending audio keyed by sequence number — held until its turn arrives.
        self._pending: Dict[int, bytes] = {}
        # The next sequence number we are allowed to push into _queue.
        self._next_seq: int = 0
        # Wall-clock reference point when playback of a burst begins.
        self._start = None
        # Next absolute deadline for sending a frame (keeps real-time pacing).
        self._next_time = None
        # Monotonically increasing RTP timestamp measured in samples @48kHz.
        self._pts = 0
        # Frames are 1024 samples/channel (~21ms) — small enough for low jitter,
        # large enough to keep per-frame overhead negligible.
        self.samples_per_frame = 1024
        self.rate = 48000

    def reset_for_new_turn(self):
        """Reset ordered-playback state at the start of every turn.

        seq_counter in run_turn() restarts at 0 for every new utterance, but
        this PlayerTrack instance is reused for the whole WebRTC session (it's
        created once in /offer). Without this reset, _next_seq is left wherever
        the previous turn ended (e.g. 3), so _drain_pending()'s
        `while self._next_seq in self._pending` never matches turn 2's fresh
        seq 0/1/2 — and nothing ever plays again for the rest of the session.
        Any leftover pending audio from a turn that was cut off is also
        discarded so it can't leak into the next turn's playback.
        """
        self._next_seq = 0
        self._pending.clear()

    async def recv(self):
        """Called by aiortc whenever it needs the next frame; we pace to real time."""

        now = time.monotonic()

        if self._start is None:
            # First frame ever: anchor the pacing clock to right now.
            self._start = now
            self._next_time = now
        else:
            # If we're ahead of schedule, sleep so audio plays at natural speed
            # instead of blasting the whole reply in a fraction of a second.
            wait = self._next_time - now
            if wait > 0:
                await asyncio.sleep(wait)

        # Schedule the next frame exactly one frame-duration later.
        self._next_time += self.samples_per_frame / self.rate

        # Check if the next-in-sequence chunk has arrived yet.
        self._drain_pending()

        if len(self._queue) > 0:
            # Real audio available -> pop it.
            data = self._queue.popleft()
        else:
            # Nothing queued -> send digital silence, which keeps the RTP stream
            # alive and lets the browser's jitter buffer stay stable.
            # Shape must be (1, N) because PyAV packed s16 expects one row of
            # interleaved samples: N = channels(2) * samples_per_frame.
            data = np.zeros((1, self.samples_per_frame * 2), dtype=np.int16)

        # Build an AV audio frame from the int16 array of shape (1, interleaved).
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(data), format="s16", layout="stereo"
        )
        frame.sample_rate = self.rate
        # Timestamp every frame so receivers can order/jitter-buffer correctly.
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.rate)
        self._pts += self.samples_per_frame
        return frame

    def _drain_pending(self):
        """Move consecutive ready chunks from _pending into _queue.

        This is the core of ordered playback: we only advance _next_seq
        when the chunk for that exact sequence is available, guaranteeing
        natural sentence order even if TTS workers finish out of order.
        """
        while self._next_seq in self._pending:
            wav_bytes = self._pending.pop(self._next_seq)
            self._decode_wav_into_queue(wav_bytes)
            self._next_seq += 1

    def push_wav_ordered(self, seq: int, wav_bytes: bytes):
        """Store a TTS result keyed by sequence number for ordered playback."""
        self._pending[seq] = wav_bytes

    def _decode_wav_into_queue(self, wav_bytes: bytes):
        """Decode WAV bytes into 48kHz stereo frames and append to _queue.

        This is the same logic as the old push_wav() but writes to _queue
        directly — called by _drain_pending() when it's this chunk's turn.
        """
        container = av.open(io.BytesIO(wav_bytes))
        # Resampler converts 22050Hz mono -> 48kHz stereo s16 in one call;
        # AV automatically upmixes mono->stereo during resampling.
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=48000)

        flat = []
        for packet in container.decode(audio=0):
            # resample() returns a list of converted AudioFrames per input packet.
            for f in resampler.resample(packet):
                # Packed s16 -> shape (1, N*2); flatten to one interleaved list.
                flat.append(f.to_ndarray().reshape(-1))

        # Concatenate everything into a single interleaved [L,R,L,R,...] buffer.
        pcm = np.concatenate(flat).astype(np.int16)

        # Slice into fixed frames of 2*1024 interleaved samples (stereo!),
        # skipping any trailing partial frame — a few ms cut at the very
        # end is inaudible.
        n = self.samples_per_frame * 2
        count = len(pcm) // n
        for i in range(count):
            chunk = pcm[i * n:(i + 1) * n]
            self._queue.append(chunk.reshape(1, -1))

        # Natural breathing gap after this sentence. Without it, the next
        # sentence's audio (once _drain_pending() reaches it) plays with
        # zero gap right after this one — which is what makes a multi-
        # sentence reply sound like one rushed, breathless stream instead
        # of normal human speech.
        self._append_silence(_INTER_SENTENCE_PAUSE_S)

    def _append_silence(self, duration_s: float):
        """Queue `duration_s` seconds of digital silence as playback frames."""
        n_frames = int(duration_s * self.rate / self.samples_per_frame)
        silence = np.zeros((1, self.samples_per_frame * 2), dtype=np.int16)
        for _ in range(n_frames):
            self._queue.append(silence)

    def pending_duration(self) -> float:
        """Seconds of unplayed audio still queued — used to know when playback ends."""
        return len(self._queue) * self.samples_per_frame / self.rate


# ---------- transcript broadcast helpers ------------------------------------


async def broadcast(payload: dict):
    """Send a JSON status/transcript message to every connected browser tab."""
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            # A dead websocket must never break the audio pipeline.
            clients.discard(ws)


# ---------- streaming LLM producer -----------------------------------------


async def _llm_producer(text: str, queue: asyncio.Queue, metrics: TurnMetrics,
                        seq_counter: list):
    """Stream LLM tokens, detect sentence boundaries, push Sentence objects.

    Runs chat_stream() in a worker thread (it's a blocking HTTP generator),
    bridges tokens to async via a thread-safe queue, then splits on sentence
    boundaries on-the-fly. Each sentence gets a monotonically increasing
    sequence number so the ordered audio buffer can play them back correctly.

    This is truly streaming: TTS starts on the first sentence while the LLM
    is still generating the rest.

    Args:
        text: The user's transcribed utterance.
        queue: asyncio.Queue(maxsize=10) for backpressure — blocks when full.
        metrics: TurnMetrics collector for latency logging.
        seq_counter: Single-element list [0] used as a mutable counter across
                     producer and workers (avoids passing a shared int).
    """
    metrics.llm_start = time.monotonic()

    # Thread-safe queue to bridge the sync generator into the async event loop.
    # Use Python's queue.Queue (not asyncio.Queue) because we write from a thread.
    token_q: _queue.Queue = _queue.Queue()

    def _sync_pump():
        """Run the blocking chat_stream() in a worker thread, push tokens."""
        try:
            for token in chat_stream(text):
                token_q.put(token)
        finally:
            token_q.put(None)  # sentinel: stream ended

    # Start the blocking generator in a worker thread; run_in_executor returns
    # a Future that resolves when the thread finishes (all tokens pumped).
    pump_future = asyncio.get_event_loop().run_in_executor(None, _sync_pump)

    try:
        # Accumulate tokens into a buffer; we split on sentence boundaries.
        buf = ""
        first_token = True

        while True:
            # Read tokens from the thread-safe queue (blocks the thread, but we
            # wrap it in to_thread so the event loop stays responsive).
            token = await asyncio.to_thread(token_q.get)

            # None sentinel: LLM stream ended.
            if token is None:
                break

            if first_token:
                metrics.llm_first_token = time.monotonic()
                first_token = False

            buf += token

            # Check if we have a complete sentence to push.
            while True:
                # Look for a sentence boundary: ., !, ? followed by space, or newline.
                match = _SENTENCE_END.search(buf)
                if match is None:
                    # No boundary yet, but force-flush if buffer is too long
                    # (prevents unbounded growth during a very long clause).
                    if len(buf) > _MAX_CHARS_NO_END:
                        # Don't hard-cut at a raw character index — that can
                        # slice straight through a word (e.g. "...gets bet" |
                        # "ter the more...") and hand the TTS engine a
                        # fragment it wasn't trained to pronounce cleanly,
                        # which is exactly what shows up as rushed/garbled
                        # audio. Back up to the nearest comma, or failing
                        # that whitespace, so the chunk stays a grammatical
                        # unit even when force-flushed.
                        cut = buf.rfind(',', 0, _MAX_CHARS_NO_END)
                        if cut == -1:
                            cut = buf.rfind(' ', 0, _MAX_CHARS_NO_END)
                        split_at = cut + 1 if cut != -1 else _MAX_CHARS_NO_END
                    else:
                        break
                else:
                    split_at = match.end()

                sent_text = buf[:split_at].strip()
                buf = buf[split_at:]

                if not sent_text:
                    continue

                seq = seq_counter[0]
                seq_counter[0] += 1
                sentence = Sentence(seq=seq, text=sent_text)
                await queue.put(sentence)  # blocks if queue is full (backpressure)

                # Broadcast the sentence text to the frontend incrementally.
                await broadcast({"type": "ai_stream", "text": sent_text})

        # Flush any remaining buffer as the final sentence.
        if buf.strip():
            seq = seq_counter[0]
            seq_counter[0] += 1
            sentence = Sentence(seq=seq, text=buf.strip())
            await queue.put(sentence)
            await broadcast({"type": "ai_stream", "text": buf.strip()})

        metrics.llm_end = time.monotonic()

        # Wait for the pump thread to finish (it should already be done).
        # If chat_stream() raised inside the thread, this re-raises it here —
        # the `finally` below still guarantees workers get their stop signal.
        await pump_future

    finally:
        # Signal all workers that no more sentences are coming — this must
        # run even if the try block above raised (e.g. the LLM call failed),
        # otherwise every _tts_worker blocks on queue.get() forever and the
        # tasks leak for the lifetime of the process.
        for _ in range(_TTS_WORKERS):
            await queue.put(None)


# ---------- parallel TTS workers -------------------------------------------


async def _tts_worker(worker_id: int, queue: asyncio.Queue,
                      player: PlayerTrack, metrics: TurnMetrics):
    """Pull sentences from the queue, synthesize with retry, push to ordered buffer.

    Each worker runs independently; 3 workers means 3 sentences can be
    synthesized simultaneously, reducing total TTS time by ~66% for long replies.

    Args:
        worker_id: Numeric ID for logging which worker did what.
        queue: The shared sentence queue (with backpressure from maxsize).
        player: The PlayerTrack to push decoded audio into.
        metrics: TurnMetrics collector for per-sentence latency logging.
    """
    while True:
        sentence = await queue.get()

        # None sentinel means the producer is done — exit this worker.
        if sentence is None:
            break

        # Retry loop: synthesize once, retry once on failure, skip on second failure.
        wav = None
        for attempt in range(_TTS_RETRIES):
            try:
                metrics.tts_start[sentence.seq] = time.monotonic()
                # synthesize() is CPU-bound (Tacotron2 inference) — run in worker thread.
                wav = await asyncio.to_thread(synthesize, sentence.text)
                metrics.tts_end[sentence.seq] = time.monotonic()
                break
            except Exception as e:
                print(f"[tts-worker-{worker_id}] attempt {attempt+1} failed "
                      f"for seq={sentence.seq}: {e}")
                if attempt == _TTS_RETRIES - 1:
                    print(f"[tts-worker-{worker_id}] seq={sentence.seq} failed "
                          f"after {_TTS_RETRIES} attempts — substituting silence "
                          f"so playback order isn't blocked")

        # Always push something for this seq, even on permanent failure.
        # PlayerTrack._drain_pending() requires strictly contiguous sequence
        # numbers — if a seq is simply never pushed, every later sentence
        # (which may have synthesized fine) is stuck in _pending forever and
        # the rest of the reply goes silent. A short silent placeholder keeps
        # the sequence unbroken.
        player.push_wav_ordered(sentence.seq, wav if wav is not None else _SILENCE_WAV)

        queue.task_done()


# ---------- the per-utterance pipeline --------------------------------------


async def run_turn(utterance: bytes, player: PlayerTrack):
    """Full streaming pipeline for one detected utterance.

    Architecture:
      1. STT transcribes the utterance (blocking, threaded).
      2. LLM producer streams tokens, detects sentences, pushes to queue.
      3. 3 TTS workers pull from queue in parallel, push to ordered buffer.
      4. PlayerTrack plays chunks in sequence order via _drain_pending().
      5. We wait for the player queue to drain, then log metrics.

    This replaces the old "collect all -> chunk -> parallel gather" approach
    with a streaming producer-consumer that starts TTS on the first sentence
    while the LLM is still generating the rest.
    """
    global busy

    metrics = TurnMetrics()

    try:
        await broadcast({"type": "status", "state": "transcribing"})

        # whisper is pure CPU blocking work -> run it in a worker thread
        # so the asyncio loop keeps serving WebRTC frames meanwhile.
        text = await asyncio.to_thread(transcribe, utterance)
        metrics.stt_end = time.monotonic()

        if not text:
            return

        await broadcast({"type": "user", "text": text})
        await broadcast({"type": "status", "state": "thinking"})
        await broadcast({"type": "status", "state": "speaking"})

        # Reset the player's ordered-playback state for this turn. seq_counter
        # below restarts at 0 for every utterance, and the player instance is
        # shared across the whole WebRTC session, so without this reset the
        # player would still be waiting on whatever seq the previous turn left
        # off at — and this turn's audio would never play.
        player.reset_for_new_turn()

        # The sentence queue: maxsize=10 creates backpressure so the LLM
        # producer blocks if TTS workers fall behind, bounding memory.
        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

        # Shared sequence counter: producer increments, workers read.
        seq_counter = [0]

        # Launch the LLM producer (streams tokens -> sentences -> queue).
        producer_task = asyncio.ensure_future(
            _llm_producer(text, sentence_queue, metrics, seq_counter)
        )

        # Launch 3 parallel TTS workers (queue -> synthesize -> ordered buffer).
        worker_tasks = [
            asyncio.ensure_future(
                _tts_worker(i, sentence_queue, player, metrics)
            )
            for i in range(_TTS_WORKERS)
        ]

        # Wait for producer to finish pushing all sentences.
        await producer_task

        # Wait for all workers to finish synthesizing their assigned sentences.
        await asyncio.gather(*worker_tasks)

        # Record the time the first sentence became available for playback.
        if 0 in metrics.tts_end:
            metrics.first_sentence_time = metrics.tts_end[0]

        # Wait until the player queue drains so 'busy' stays True until
        # the user has actually heard the full answer.
        while player.pending_duration() > 0:
            await asyncio.sleep(0.05)

        # Small grace period covering the last buffered frame reaching the speaker.
        await asyncio.sleep(_DRAIN_GRACE)

        metrics.playback_end = time.monotonic()
        metrics.log()

    except Exception as e:
        # Never let one bad turn kill the connection — report and move on.
        await broadcast({"type": "status", "state": f"error: {e}"})

    finally:
        busy = False
        await broadcast({"type": "status", "state": "waiting_for_wake_word"})


# ---------- inbound mic consumer ---------------------------------------------


async def consume_mic(track, player: PlayerTrack):
    """Pull decoded mic frames, run wake word -> VAD -> turns."""

    global busy

    # Resamples whatever the browser sends (Opus-decoded 48k stereo)
    # down to exactly what wake word/VAD+whisper need: 16kHz mono signed-16bit.
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    # Raw byte accumulator: resampled chunks arrive in odd sizes, but wake word
    # model + VAD both require exact 30ms (960-byte) slices, so we buffer and slice.
    acc = bytearray()
    vad = VADetector()
    activated = False          # True once wake word has been detected this turn

    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return

        if busy:
            continue

        for f in resampler.resample(frame):
            acc.extend(f.to_ndarray().reshape(-1).tobytes())

        # Process every complete 30ms window that has accumulated.
        while len(acc) >= FRAME_BYTES:
            chunk = bytes(acc[:FRAME_BYTES])
            del acc[:FRAME_BYTES]

            if not activated:
                # Run wake word model on each 30ms chunk until it fires.
                pcm16 = np.frombuffer(chunk, dtype=np.int16)
                scores = get_oww().predict(pcm16)
                # The model returns a dict; any score > threshold = wake word heard.
                if any(v > 0.5 for v in scores.values()):
                    activated = True
                    await broadcast({"type": "status", "state": "listening"})
                continue           # skip VAD entirely until activated

            # Wake word detected -> now feed audio through VAD as before.
            utterance = vad.process(chunk)

            if utterance is not None and not busy:
                busy = True
                activated = False   # reset after this turn so wake word must be said again
                asyncio.ensure_future(run_turn(utterance, player))


# ---------- HTTP endpoints ----------------------------------------------------


# Request schema for POST /offer: standard trickle-free SDP exchange body.
class Offer(BaseModel):
    sdp: str      # session description payload from the browser
    type: str     # always "offer"


@app.post("/offer")
async def offer(params: Offer):
    """WebRTC signaling: turn the browser's SDP offer into our SDP answer."""

    # One RTCPeerConnection per browser tab/session.
    pc = RTCPeerConnection()

    # Track it globally for cleanup on server shutdown.
    pcs.add(pc)

    # The player must be added BEFORE createAnswer() so the negotiated SDP
    # includes a send-direction audio m-line for the AI voice.
    player = PlayerTrack()
    pc.addTrack(player)

    @pc.on("track")
    def on_track(track):
        # Fires when the remote (browser) starts sending its mic track.
        if track.kind == "audio":
            # Start the async consumer that does resampling + VAD + turns.
            asyncio.ensure_future(consume_mic(track, player))

    @pc.on("connectionstatechange")
    async def on_state():
        # Clean up fully once the browser leaves or the link fails.
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    # Load the remote description received from the browser.
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params.sdp, type=params.type))

    # Generate our side of the negotiation (answer with player track).
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # localDescription reflects the actual SDP after ICE gathering defaults,
    # which is what the browser must apply as its remote description.
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.websocket("/events")
async def events(ws: WebSocket):
    """WebSocket channel pushing transcripts/status updates to the UI."""

    # Complete the HTTP->WS upgrade handshake.
    await ws.accept()
    clients.add(ws)

    # Tell the new tab immediately where things stand.
    await ws.send_json({"type": "status", "state": "waiting_for_wake_word"})

    try:
        # We never expect client messages; just hold the socket open,
        # receiving (and ignoring) anything until the client disconnects.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.get("/")
async def index():
    # Serve the single-page client at the root URL.
    return FileResponse("static/index.html")


@app.on_event("shutdown")
async def shutdown():
    """Close all peer connections when Ctrl-C stops the server."""
    for pc in list(pcs):
        await pc.close()