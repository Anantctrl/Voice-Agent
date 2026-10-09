"""Abstract provider protocols (Component contracts)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol


class STTProvider(Protocol):
    async def connect(self) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    async def close(self) -> None: ...


class LLMProvider(Protocol):
    def stream_response(self, user_text: str) -> AsyncIterator[str]: ...


class TTSProvider(Protocol):
    def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...


TurnCallback = Callable[[str], Awaitable[None]]
