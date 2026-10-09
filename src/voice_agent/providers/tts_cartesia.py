"""Streaming TTS providers (Component 7).

- Cartesia Sonic over WebSocket: first audio in ~90 ms, PCM16 @ 16 kHz.
- ElevenLabs Turbo v2.5 over WebSocket (stream-input).
- ElevenLabs v4 (`eleven_v4`): REST chunked streaming — v4 is NOT accepted
  on the TTS WebSocket (HTTP 400); realtime v4-turbo lives on a different
  Text-to-Dialogue socket. REST gives one POST per phrase, PCM16 @ 16 kHz.
- ``create_tts_provider(cfg)`` factory picks from AgentConfig.tts_provider.

All paths yield raw PCM16 mono bytes ready for the DAC queue.
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import json
from collections.abc import AsyncIterator

import httpx
import websockets


def _decode_audio_payload(payload: str) -> bytes:
    """Cartesia sends base64 (current) — older docs showed hex. Accept both."""
    s = payload.strip().strip('"')
    # Heuristic: hex is [0-9a-fA-F] with even length; base64 has +/ or = padding.
    try:
        if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s[:64]):
            try:
                return bytes.fromhex(s)
            except (ValueError, binascii.Error):
                pass
        # Pad base64 if needed.
        pad = "=" * (-len(s) % 4)
        return base64.b64decode(s + pad)
    except Exception:
        return b""


class CartesiaStreamingTTS:
    WS_URL = "wss://api.cartesia.ai/tts/websocket/v1"

    def __init__(self, api_key: str, voice_id: str,
                 model_id: str = "sonic-english", sample_rate: int = 16_000):
        if not api_key:
            raise ValueError("CARTESIA_API_KEY is required for cartesia TTS.")
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate

    async def warmup(self) -> bool:
        """Background TLS/DNS warmup: handshake then close, no audio."""
        try:
            url = f"{self.WS_URL}?api_key={self.api_key}&cartesia_version=2024-06-10"
            async with websockets.connect(url, max_size=1024):
                return True
        except Exception:
            return False

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        url = f"{self.WS_URL}?api_key={self.api_key}&cartesia_version=2024-06-10"
        async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "model_id": self.model_id,
                "transcript": text,
                "voice": {"mode": "id", "id": self.voice_id},
                "output_format": {"container": "raw", "encoding": "pcm_s16le",
                                  "sample_rate": self.sample_rate},
                "language": "en",
            }))
            async for message in ws:
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                chunk_b64 = data.get("data") or data.get("audio") or ""
                if chunk_b64:
                    raw = _decode_audio_payload(chunk_b64)
                    if raw:
                        yield raw
                if data.get("done", False) or data.get("status") == "done":
                    break


class ElevenLabsStreamingTTS:
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM",
                 model_id: str = "eleven_turbo_v2_5", sample_rate: int = 16_000):
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for elevenlabs TTS.")
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        """One pooled client per session: keep-alive avoids a TLS handshake per phrase."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _connect(self, url: str):
        try:
            return websockets.connect(url, max_size=8 * 1024 * 1024,
                                      additional_headers={"xi-api-key": self.api_key})
        except TypeError:
            return websockets.connect(url, max_size=8 * 1024 * 1024,
                                      extra_headers={"xi-api-key": self.api_key})  # type: ignore

    def _is_v4(self) -> bool:
        return (self.model_id or "").lower().startswith("eleven_v4")

    async def _synthesize_rest(self, text: str) -> AsyncIterator[bytes]:
        """Eleven v4 path: chunked-REST streaming (no TTS WebSocket)."""
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
               f"?output_format=pcm_16000")
        async with self._http().stream(
            "POST", url,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": self.model_id,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:200]
                raise RuntimeError(f"elevenlabs HTTP {resp.status_code}: {body!r}")
            async for chunk in resp.aiter_bytes(4096):
                if chunk:
                    yield bytes(chunk)

    async def warmup(self) -> bool:
        """Background TLS/DNS warmup: handshake then close, no audio."""
        if self._is_v4():
            try:
                # Any HTTP response proves DNS+TLS are warm in the pooled client;
                # restricted keys get 401 on /v1/user, which must not fail warmup.
                await self._http().get("https://api.elevenlabs.io/v1/models",
                                       headers={"xi-api-key": self.api_key}, timeout=10.0)
                return True
            except Exception:
                return False
        try:
            url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
                   f"?model_id={self.model_id}&output_format=pcm_16000")
            async with self._connect(url):
                return True
        except Exception:
            return False

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        if self._is_v4():
            async for chunk in self._synthesize_rest(text):
                yield chunk
            return
        url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
               f"?model_id={self.model_id}&output_format=pcm_16000")
        async with self._connect(url) as ws:
            await ws.send(json.dumps({"text": text, "voice_settings": {"stability": 0.5}}))
            await ws.send(json.dumps({"text": ""}))  # flush / EOS marker
            async for message in ws:
                if isinstance(message, bytes):
                    yield message
                else:
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    audio_b64 = data.get("audio") or ""
                    if audio_b64:
                        with contextlib.suppress(Exception):
                            yield base64.b64decode(audio_b64)
                    if data.get("isFinal") or data.get("is_final"):
                        break


def create_tts_provider(cfg):  # -> CartesiaStreamingTTS | ElevenLabsStreamingTTS
    provider = (cfg.tts_provider or "cartesia").lower()
    if provider == "elevenlabs":
        return ElevenLabsStreamingTTS(api_key=cfg.elevenlabs_api_key,
                                      voice_id=cfg.elevenlabs_voice_id,
                                      model_id=cfg.elevenlabs_model)
    return CartesiaStreamingTTS(api_key=cfg.cartesia_api_key,
                                voice_id=cfg.cartesia_voice_id,
                                model_id=cfg.cartesia_model)
