"""Audio hardware layer (Component 2) — non-blocking sounddevice I/O.

Concurrency contract (matches blueprint, fixed for thread-safety):
- Mic callback runs on the PortAudio thread: NEVER allocates large arrays,
  blocks, or touches disk. It pushes raw bytes into an asyncio.Queue via
  ``loop.call_soon_threadsafe``.
- Playback pulls from a thread-safe ``queue.Queue`` (drained with
  ``get_nowait`` inside the output callback) plus a ``PlaybackRingBuffer``
  used as the instant-flush target on barge-in.

Why not ``asyncio.Queue`` for output? ``asyncio.Queue.get_nowait`` is not
safe to call from a non-event-loop thread. Using a plain ``queue.Queue``
for the DAC callback is the correct zero-polling primitive; async producers
call the (non-blocking, thread-safe) ``enqueue_playback`` / ``put_nowait``
helpers, and ``HybridPlaybackQueue`` also exposes ``await put()`` so the
orchestrator code in the blueprint keeps working unchanged.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import math
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

log = logging.getLogger("voice_agent.audio")

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - import-time guard for CI without PortAudio
    sd = None  # type: ignore

from voice_agent.audio.ring_buffer import PlaybackRingBuffer

if TYPE_CHECKING:  # avoids importing AEC (numpy) until enable_aec
    from voice_agent.audio.aec.processor import FrameProcessor


class HybridPlaybackQueue:
    """Async-compatible facade over a thread-safe ``queue.Queue``.

    - Async producers: ``await q.put(chunk)`` (never blocks; drops oldest on
      saturation to prioritize fresh speech, mirroring the blueprint).
    - Audio callback thread: ``q.get_nowait_for_callback(outdata)``.
    """

    def __init__(self, maxsize: int = 64):
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)

    # -- async side (event loop thread) --
    async def put(self, chunk: bytes) -> None:
        """Backpressure: wait for room instead of dropping the start of speech."""
        while self._q.full():
            await asyncio.sleep(0.005)
        self.put_nowait(chunk)

    def put_nowait(self, chunk: bytes) -> None:
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._q.get_nowait()  # drop oldest
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(chunk)

    # -- callback side (audio thread) --
    def get_nowait(self) -> bytes | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> int:
        n = 0
        while True:
            try:
                self._q.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


class AudioDeviceManager:
    def __init__(self, sample_rate: int = 16_000, frame_size: int = 512,
                 input_device: int | None = None, output_device: int | None = None):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_bytes = frame_size * 2  # int16 mono
        self._loop: asyncio.AbstractEventLoop | None = None

        # 64-deep: absorbs event-loop bursts while LLM/TTS stream so the
        # mic callback never overflows into drop-oldest gaps for Deepgram.
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        self.output_queue = HybridPlaybackQueue(maxsize=256)
        self.ring = PlaybackRingBuffer()

        # Spillover buffer: TTS chunks (e.g. 33 KB) are much larger than one
        # DAC frame (1024 B). Without this, only the first frame per chunk
        # would play and the rest would be silently dropped (-> "cuts off").
        self._pending = bytearray()
        self._pending_lock = threading.Lock()

        self._in_stream: Any = None
        self._out_stream: Any = None
        self._lock = threading.Lock()
        self.input_overflows = 0
        self.input_queue_drops = 0
        self.input_frames = 0
        self.played_frames = 0
        self.dropped_surplus_bytes = 0
        # Last time a non-silent frame was drained (monotonic). Used by
        # is_playing() so the mic gate holds through short queue gaps and
        # releases ~300 ms after audio truly ends (tail, not head, clipping).
        self._last_play_ts: float = 0.0
        self.play_tail_s: float = 0.3
        # Playback episode start: silence (>1 s) -> audio restarts mark a new
        # episode. The barge-in gate blanks the first ~400 ms of each episode
        # because echo onset is indistinguishable from a user onset.
        self._episode_start_ts: float = 0.0
        # Playback loudness estimate (0..1 RMS) for the energy-gated
        # barge-in: real user interrupts are louder than speaker echo.
        # Fast attack, slow release; guarded by _pending_lock.
        self._play_rms: float = 0.0
        # ---- AEC path (disabled by default; see enable_aec) ----
        # Worker thread: mic frames + DAC reference frames rendezvous here,
        # clean frames + speaker labels flow to the event loop. Callbacks stay
        # allocation-light; pairing is 1:1 at frame cadence (taps absorb drift).
        self._aec: Any = None
        self._aec_classifier = None
        self._aec_thread: threading.Thread | None = None
        self._aec_stop = threading.Event()
        self._aec_failed = False
        self._mic_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._ref_q: queue.Queue[bytes] = queue.Queue(maxsize=8)
        self._label_lock = threading.Lock()
        self._label_deque: collections.deque = collections.deque(maxlen=256)
        self._aec_frames = 0
        self._aec_erle = 0.0
        self._aec_classes: dict[str, int] = {}
        self._aec_ref_drops = 0
        self._aec_mic_drops = 0
        self._processor: FrameProcessor | None = None
        # First-50-frames path proof: proves STT-bound audio actually passed
        # through the canceller (in/out energy), then freezes as evidence.
        self._proof_in = 0.0
        self._proof_out = 0.0
        self._proof_n = 0
        self._diverge_logged = False

    def _drain_into(self, outdata) -> None:
        """Fill one DAC frame from pending spillover + queued chunks.

        Runs on the high-priority audio thread: never blocks.
        """
        need = len(outdata)
        with self._pending_lock:
            buf = self._pending
            # Pull queued chunks until we can fill the frame.
            while len(buf) < need:
                chunk = self.output_queue.get_nowait()
                if chunk is None:
                    break
                buf.extend(chunk)
            take = min(len(buf), need)
            tapped: bytes | None = None
            if take > 0:
                frame = bytes(buf[:take])
                outdata[:take] = frame
                del buf[:take]
                self.played_frames += 1
                # Any real (possibly silent-padded) playout counts, but the
                # tail timer only extends while we still hold audio: pending
                # or queued data existed this callback.
                _now = time.monotonic()
                if _now - self._last_play_ts > 1.0:
                    self._episode_start_ts = _now  # fresh onset after silence
                self._last_play_ts = _now
                # Loudness estimate for echo-resistant barge-in. Tiny
                # (512-sample) int16 RMS; cheap enough for the audio thread.
                try:
                    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0)
                except Exception:
                    rms = 0.0
                if rms > self._play_rms:
                    self._play_rms = rms
                else:
                    self._play_rms += (rms - self._play_rms) * 0.02
                if self._aec is not None:
                    # Full-frame copy (zero-padded) for constant tap cadence.
                    tapped = frame + (b"\x00" * (need - take)) if take < need else frame
            if take < need:
                outdata[take:] = b"\x00" * (need - take)
                # Decay the loudness estimate during underruns so the gate
                # re-opens promptly after speech ends (no laggy tail).
                self._play_rms *= 0.94
        # Queue the reference outside the pending lock (short critical section).
        if tapped is not None:
            self._tap_reference(tapped)
        # NOTE: tap content is captured above; queueing happens outside the
        # pending lock to keep the critical section short.
        if self._aec is not None and take > 0:
            try:
                frame_out = bytearray(need)
                frame_out[:take] = outdata[:take]
                self._tap_reference(bytes(frame_out))
            except Exception:
                pass

    def playback_rms(self) -> float:
        with self._pending_lock:
            return self._play_rms

    def playback_episode_age(self) -> float:
        """Seconds since the current playback episode started (inf if idle)."""
        with self._pending_lock:
            start = self._episode_start_ts
        if start <= 0.0:
            return float("inf")
        return time.monotonic() - start

    def seconds_since_playback(self) -> float:
        """Seconds since any audio reached the DAC (inf if never). While
        playing this is 0.0. Used by the recency gate: echo is physically
        impossible when the speaker has been silent for seconds."""
        with self._pending_lock:
            last = self._last_play_ts
        if last <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - last)

    # ---------------- AEC (optional, off unless enable_aec was called) ------
    def enable_aec(self, canceller, classifier=None, double_talk=None,
                   monitor=None) -> None:
        """Attach an echo canceller + speaker classifier. Call before start().

        Falls back automatically: any worker failure flips to pass-through
        and today's workaround stack decides alone, exactly as without AEC.
        """
        if self._in_stream is not None:
            raise RuntimeError("enable_aec must be called before start()")
        self._aec = canceller
        self._aec_classifier = classifier
        if double_talk is None or monitor is None:
            from voice_agent.audio.aec.double_talk import DoubleTalkDetector
            from voice_agent.audio.aec.echo_detector import EchoPathMonitor
            if double_talk is None:
                double_talk = DoubleTalkDetector()
            if monitor is None:
                monitor = EchoPathMonitor()
        from voice_agent.audio.aec.processor import FrameProcessor
        self._processor = FrameProcessor(canceller, classifier, double_talk,
                                         monitor, frame_bytes=self.frame_bytes)

    def _tap_reference(self, frame: bytes) -> None:
        try:
            self._ref_q.put_nowait(frame)
        except queue.Full:
            try:
                self._ref_q.get_nowait()  # drop oldest: live pairing only
                self._aec_ref_drops += 1
            except queue.Empty:
                pass
            with contextlib.suppress(queue.Full):
                self._ref_q.put_nowait(frame)

    def _route_mic(self, raw: bytes) -> None:
        """Audio-thread entry: AEC worker when healthy, direct loop handoff."""
        # No loop requirement here: the worker delivers via call_soon_threadsafe
        # only when a loop exists (guarded at delivery); queuing itself is
        # plain thread-safe and bounded. In production start() always sets the
        # loop before any callback fires, so behavior is identical.
        if self._aec is not None and not self._aec_failed:
            try:
                self._mic_q.put_nowait(raw)
                return
            except queue.Full:
                try:
                    self._mic_q.get_nowait()
                    self._aec_mic_drops += 1
                except queue.Empty:
                    pass
                try:
                    self._mic_q.put_nowait(raw)
                    return
                except queue.Full:
                    pass
        try:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._safe_enqueue, raw)
        except RuntimeError:
            pass  # loop closed

    def _process_aec_pair(self, mic_bytes: bytes, ref_bytes: bytes):
        """One mic/reference frame -> (clean bytes, FrameDecision|None).

        Delegates to the shared FrameProcessor (filter + classify + detect);
        this method only maintains device-level stats.
        """
        assert self._processor is not None, "enable_aec() before pairing"
        clean, decision, info = self._processor.process(mic_bytes, ref_bytes)
        if not info.get("bypassed", False):
            mic_a = np.frombuffer(mic_bytes, dtype=np.int16)
            mic_rms = (float(np.sqrt(np.mean(mic_a.astype(np.float64) ** 2)) / 32768.0)
                       if mic_a.size else 0.0)
            if mic_rms >= 0.004:
                # Voiced gate: silence frames carry no echo information, and
                # folding them into ERLE is what made the metric swing wildly.
                erle = float(info.get("erle_db", 0.0))
                self._aec_erle += (erle - self._aec_erle) * 0.05
            if self._proof_n < 50:
                # Path proof: in/out energy of the first 50 non-bypassed
                # frames, frozen as evidence that STT-bound audio passed
                # through the canceller (restarts after a divergence reset).
                out_a = np.frombuffer(clean, dtype=np.int16).astype(np.float64)
                in_a = np.frombuffer(mic_bytes[:len(clean)], dtype=np.int16).astype(np.float64)
                self._proof_in += float(np.sum(in_a * in_a))
                self._proof_out += float(np.sum(out_a * out_a))
                self._proof_n += 1
        self._aec_frames += 1
        if decision is not None:
            self._aec_classes[decision.label] = self._aec_classes.get(decision.label, 0) + 1
        if info.get("diverged", False) and not self._diverge_logged:
            self._diverge_logged = True
            log.warning("aec diverged (sustained negative ERLE): resetting coefficients")
            with contextlib.suppress(Exception):
                self._aec.reset()
            with contextlib.suppress(Exception):
                self._processor.monitor.acknowledge()
            self._proof_in = self._proof_out = 0.0
            self._proof_n = 0
        elif not info.get("diverged", False):
            self._diverge_logged = False
        return clean, decision

    def _deliver_clean(self, clean: bytes, decision) -> None:
        # Runs on the event loop thread.
        self._safe_enqueue(clean)
        if decision is not None:
            with self._label_lock:
                self._label_deque.append(decision)

    def pop_label(self):
        """Newest pending speaker decision, or None (AEC off / not yet paired)."""
        with self._label_lock:
            if not self._label_deque:
                return None
            out = self._label_deque[-1]
            self._label_deque.clear()
            return out

    def aec_stats(self) -> dict:
        with self._label_lock:
            classes = dict(self._aec_classes)
        try:
            diverged = bool(self._processor is not None
                            and self._processor.monitor is not None
                            and self._processor.monitor.diverged)
        except Exception:
            diverged = False
        try:
            ref_qsize = self._ref_q.qsize()
        except Exception:
            ref_qsize = -1
        if self._proof_n >= 50 and self._proof_out > 0:
            path_db = round(10.0 * math.log10(self._proof_in / self._proof_out), 1)
        else:
            path_db = None
        return {"enabled": self._aec is not None, "failed": self._aec_failed,
                "frames": self._aec_frames, "erle_db": round(self._aec_erle, 1),
                "classes": classes, "ref_drops": self._aec_ref_drops,
                "mic_drops": self._aec_mic_drops, "diverged": diverged,
                "ref_qsize": ref_qsize, "path_db": path_db,
                "worker_alive": self._aec_thread.is_alive() if self._aec_thread else False}

    def _aec_worker_loop(self) -> None:
        try:
            zeros = b"\x00" * self.frame_bytes
            while not self._aec_stop.is_set():
                try:
                    mic = self._mic_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    while self._ref_q.qsize() > 1:
                        self._ref_q.get_nowait()  # resync: freshest reference
                    ref = self._ref_q.get_nowait()
                except queue.Empty:
                    ref = zeros  # silence -> bit-clean bypass inside
                try:
                    clean, decision = self._process_aec_pair(mic, ref)
                except Exception:
                    clean, decision = mic, None
                try:
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(self._deliver_clean, clean, decision)
                except RuntimeError:
                    return  # loop closed
        except Exception:
            self._aec_failed = True  # pass-through takes over automatically

    def _start_aec_worker(self) -> None:
        if self._aec is None:
            return
        self._aec_stop.clear()
        self._aec_failed = False
        self._aec_thread = threading.Thread(target=self._aec_worker_loop,
                                            name="aec-worker", daemon=True)
        self._aec_thread.start()

    def is_playing(self) -> bool:
        """True while TTS audio isqueued, buffered, or just finished.

        Thread-safe; callable from the event loop. Covers queue + spillover
        + a short release tail so echo decay isn't mistaken for user speech.
        """
        with self._pending_lock:
            buffered = len(self._pending) > 0 or not self.output_queue.empty()
            last = self._last_play_ts
        if buffered:
            return True
        return (time.monotonic() - last) < self.play_tail_s

    # ------------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not available (PortAudio missing).")
        self._loop = loop

        def _in_callback(indata, frames, time_info, status) -> None:
            if status is not None and getattr(status, "input_overflow", False):
                self.input_overflows += 1
            try:
                raw = bytes(indata)  # small fixed copy (1024 B) — allowed
            except Exception:
                return
            self._route_mic(raw)

        def _out_callback(outdata, frames, time_info, status) -> None:
            # Frame-accurate playout: spillover across callbacks, zero-fill
            # only on true underrun.
            self._drain_into(outdata)

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
            callback=_in_callback,
        )
        # RawOutputStream delivers raw bytes to the callback.
        self._out_stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
            callback=_out_callback,
        )
        self._in_stream.start()
        self._out_stream.start()
        self._start_aec_worker()

    def start_with_devices(self, loop: asyncio.AbstractEventLoop,
                           input_device: int | None, output_device: int | None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not available (PortAudio missing).")
        self._loop = loop

        def _in_callback(indata, frames, time_info, status) -> None:
            if status is not None and getattr(status, "input_overflow", False):
                self.input_overflows += 1
            try:
                raw = bytes(indata)
            except Exception:
                return
            self._route_mic(raw)

        def _out_callback(outdata, frames, time_info, status) -> None:
            self._drain_into(outdata)

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            blocksize=self.frame_size, callback=_in_callback, device=input_device,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            blocksize=self.frame_size, callback=_out_callback, device=output_device,
        )
        self._in_stream.start()
        self._out_stream.start()
        self._start_aec_worker()

    # ------------------------------------------------------------------
    def _safe_enqueue(self, chunk: bytes) -> None:
        # Runs on the event loop thread (via call_soon_threadsafe).
        self.input_frames += 1
        if not self.input_queue.full():
            self.input_queue.put_nowait(chunk)
        else:
            try:
                self.input_queue.get_nowait()  # drop oldest
                self.input_queue_drops += 1
            except asyncio.QueueEmpty:
                pass
            with contextlib.suppress(asyncio.QueueFull):
                self.input_queue.put_nowait(chunk)

    def input_stats(self) -> dict:
        """Mic health snapshot (cheap, any thread)."""
        try:
            depth = self.input_queue.qsize()
        except Exception:
            depth = -1
        return {"frames": self.input_frames, "queue_depth": depth,
                "overflows": self.input_overflows,
                "queue_drops": self.input_queue_drops}

    async def play_pcm(self, chunk: bytes) -> None:
        """Async producer helper used by the orchestrator."""
        await self.output_queue.put(chunk)

    def clear_playback(self) -> None:
        """Instant flush on barge-in. Callable from any thread."""
        self.output_queue.clear()
        self.ring.clear()
        with self._pending_lock:
            self._pending.clear()
            self._last_play_ts = 0.0
            self._episode_start_ts = 0.0
            self._play_rms = 0.0
        # Fresh pairing after a flush; labels belong to dropped audio.
        # Coefficients survive (the room didn't change); mic frames in flight
        # are live speech and must not be dropped.
        while True:
            try:
                self._ref_q.get_nowait()
            except queue.Empty:
                break
        with self._label_lock:
            self._label_deque.clear()

    def stop(self) -> None:
        self._aec_stop.set()
        if self._aec_thread is not None:
            self._aec_thread.join(timeout=1.0)
            self._aec_thread = None
        with self._lock:
            if self._in_stream is not None:
                try:
                    self._in_stream.stop()
                    self._in_stream.close()
                except Exception:
                    pass
                self._in_stream = None
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None

    @staticmethod
    def list_devices() -> str:
        if sd is None:
            return "<sounddevice unavailable>"
        return str(sd.query_devices())
