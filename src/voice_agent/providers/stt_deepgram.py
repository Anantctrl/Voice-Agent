"""Persistent Deepgram WebSocket STT (Component 4).

Crucial rules (from blueprint):
- Connect ONCE in start/connect. Keep open across turns (0 ms overhead/turn).
- Send a KeepAlive JSON frame every ~5 s when idle.
- Stream raw PCM16 frames continuously.
- ``speech_final == True`` triggers turn completion IMMEDIATELY.

Compatible with websockets>=12 (extra_headers) and >=13.1
(additional_headers). Reconnects with backoff on drops.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except Exception as e:  # pragma: no cover
    raise RuntimeError("websockets is required: pip install websockets") from e


class DeepgramPersistentSTT:
    def __init__(self, api_key: str,
                 on_turn_complete: Callable[..., Awaitable[None]],
                 model: str = "nova-3", endpointing_ms: int = 100,
                 sample_rate: int = 16_000,
                 log_debug: Callable[[str], None] | None = None):
        self.api_key = api_key
        self.on_turn_complete = on_turn_complete
        self.model = model
        self.endpointing_ms = endpointing_ms
        self.sample_rate = sample_rate
        self.ws = None
        self._running = False
        self._recv_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._current_parts: list[str] = []
        self._current_confs: list[float | None] = []
        self._last_send_ts = 0.0
        self.on_interim = None  # optional sync/async hook(text)
        self.on_speech_started = None  # optional sync/async hook() for SpeechStarted
        self.log_debug = log_debug
        self.log_debug = log_debug
        # Health counters: interim vs committed turns + last error. If the
        # user sees interim but no replies, the stall is downstream (LLM);
        # if neither appears, it's mic capture or this socket.
        self.interim_count = 0
        self.final_count = 0
        self.turn_count = 0
        self.last_error: str | None = None
        self.reconnects = 0

    def _dbg(self, msg: str) -> None:
        try:
            if self.log_debug is not None:
                self.log_debug(msg)
        except Exception:
            pass

    @property
    def ws_url(self) -> str:
        return (
            "wss://api.deepgram.com/v1/listen"
            f"?encoding=linear16&sample_rate={self.sample_rate}&channels=1"
            f"&model={self.model}&endpointing={self.endpointing_ms}"
            "&interim_results=true&smart_format=true"
            "&vad_events=true"  # SpeechStarted diagnostics (no behavior change)
        )

    async def connect(self) -> None:
        try:
            await self._connect_once()
        except Exception as e:
            self.last_error = f"connect failed: {type(e).__name__}: {str(e)[:160]}"
            self._dbg(f"[stt] {self.last_error}")
            raise
        self._running = True
        self._dbg(f"[stt] connected (model={self.model} endpointing={self.endpointing_ms}ms)")
        self._recv_task = asyncio.create_task(self._supervisor(), name="dg-supervisor")
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="dg-keepalive")

    async def _connect_once(self) -> None:
        # websockets 12.x: extra_headers; 13.1+: additional_headers (both
        # accepted in 14.x via legacy shim, but we probe safely).
        try:
            self.ws = await websockets.connect(
                self.ws_url, extra_headers={"Authorization": f"Token {self.api_key}"}  # type: ignore
            )
        except TypeError:
            self.ws = await websockets.connect(
                self.ws_url, additional_headers={"Authorization": f"Token {self.api_key}"}  # type: ignore
            )

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.ws or not self._running:
            return
        try:
            await self.ws.send(pcm16)
            self._last_send_ts = time.monotonic()
        except ConnectionClosed as e:
            self.last_error = f"send on closed connection: {str(e)[:120]}"
            self._dbg(f"[stt] {self.last_error} — supervisor will reconnect")
            self.ws = None  # frames are dropped while disconnected (live audio)
        except Exception as e:
            self.last_error = f"send failed: {type(e).__name__}: {str(e)[:120]}"
            self._dbg(f"[stt] {self.last_error}")

    async def _keepalive_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(5)
                if not self._running:
                    continue
                if not self.ws:
                    continue  # supervisor owns reconnects; keep looping
                if time.monotonic() - self._last_send_ts >= 4.5:
                    with contextlib.suppress(Exception):
                        await self.ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        assert self.ws is not None, "supervisor connects before receiving"
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message) if isinstance(message, (str, bytes)) else {}
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "SpeechStarted":
                    # Server-side VAD fired: proves audio arrives intelligibly
                    # even before any transcript exists. Absence + mic motion
                    # below means capture/device; presence + no commit below
                    # means endpointing/single-word behavior.
                    self._dbg("[stt] server heard speech (SpeechStarted)")
                    if self.on_speech_started is not None:
                        try:
                            r = self.on_speech_started()
                            if asyncio.iscoroutine(r):
                                await r
                        except Exception as e:
                            self._dbg(f"[stt] speech-started hook failed: {e}")
                    continue
                if not isinstance(data, dict) or "channel" not in data:
                    continue
                try:
                    alt = data["channel"]["alternatives"][0]
                except Exception:
                    continue
                text = (alt.get("transcript") or "").strip()
                try:
                    conf = alt.get("confidence", None)
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                speech_final = bool(data.get("speech_final", False))
                is_final = bool(data.get("is_final", False))
                if text and is_final:
                    self._current_parts.append(text)
                    self._current_confs.append(conf)
                    self.final_count += 1
                if text and not speech_final:
                    self.interim_count += 1
                if self.on_interim is not None and text and not speech_final:
                    try:
                        r = self.on_interim(text)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception as e:
                        self._dbg(f"[stt] interim hook failed: {e}")
                if speech_final and self._current_parts:
                    full = " ".join(self._current_parts).strip()
                    confs = [c for c in self._current_confs if c is not None]
                    conf = (sum(confs) / len(confs)) if confs else None
                    self._current_parts.clear()
                    self._current_confs.clear()
                    if full:
                        self.turn_count += 1
                        self._dbg(f"[stt] speech_final turn #{self.turn_count}: {full[:100]!r}")
                        try:
                            await self.on_turn_complete(full, conf)
                        except TypeError:
                            # Legacy single-arg handlers (tests, early wiring).
                            await self.on_turn_complete(full)
                        except Exception as e:
                            self.last_error = f"turn handler failed: {type(e).__name__}: {str(e)[:120]}"
                            self._dbg(f"[stt] {self.last_error}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = f"receive loop died: {type(e).__name__}: {str(e)[:160]}"
            self._dbg(f"[stt] {self.last_error}")

    async def _supervisor(self) -> None:
        """Own the socket lifecycle: receive, and on any end/failure reconnect
        with backoff and RESTART receiving on the new socket."""
        attempt = 0
        while self._running:
            if self.ws is None:
                try:
                    await self._connect_once()
                    attempt = 0
                    self.reconnects += 1
                    self._dbg(f"[stt] reconnected (#{self.reconnects})")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.last_error = f"reconnect failed: {type(e).__name__}: {str(e)[:120]}"
                    self._dbg(f"[stt] {self.last_error}")
                    await asyncio.sleep(min(0.25 * 2 ** attempt, 5))
                    attempt += 1
                    continue
            await self._receive_loop()  # returns when the socket ends or errors
            old, self.ws = self.ws, None
            self._current_parts.clear()
            self._current_confs.clear()
            if old is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(old.close(), 1.0)
            if self._running:
                await asyncio.sleep(0.1)

    async def close(self) -> None:
        self._running = False
        for t in (self._recv_task, self._keepalive_task):
            if t:
                t.cancel()
        if self.ws:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.ws.close(), 1.0)
            self.ws = None
