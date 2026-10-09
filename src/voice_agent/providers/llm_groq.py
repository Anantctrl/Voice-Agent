"""Fast Groq streaming LLM client (Component 6).

Defaults to openai/gpt-oss-20b with reasoning_effort="low" for minimal
TTFT. Bounded history (system + last 6 messages) keeps voice replies
snappy. max_tokens covers reasoning + answer (reasoning models consume
tokens invisibly; 256 fits low-effort reasoning plus 1-2 sentences, and
larger budgets only lengthen worst-case TTFT).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

try:
    from groq import AsyncGroq
except Exception as e:  # pragma: no cover
    raise RuntimeError("groq is required: pip install groq") from e


class GroqStreamingLLM:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b",
                 system_prompt: str = "", temperature: float = 0.6,
                 max_tokens: int = 256, reasoning_effort: str = "low"):
        # 256 covers low-effort reasoning + 1-2 sentence replies; larger
        # budgets lengthen worst-case TTFT without helping short answers.
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    async def _create_stream(self, messages: list[dict]):
        kwargs: dict = {"model": self.model, "messages": messages, "stream": True,
                        "temperature": self.temperature, "max_tokens": self.max_tokens}
        # Reasoning models (gpt-oss, qwen3) accept reasoning_effort; plain
        # chat models reject it — retry without on TypeError/400.
        if self.reasoning_effort:
            try:
                return await self.client.chat.completions.create(
                    **kwargs, reasoning_effort=self.reasoning_effort)  # type: ignore
            except Exception:
                pass
        return await self.client.chat.completions.create(**kwargs)

    async def warmup(self) -> bool:
        """Background TLS/handshake warmup. No history pollution."""
        try:
            kwargs: dict = {"model": self.model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1, "temperature": self.temperature}
            try:
                await self.client.chat.completions.create(
                    **kwargs, reasoning_effort="low")  # type: ignore
            except TypeError:
                await self.client.chat.completions.create(**kwargs)
            return True
        except Exception:
            return False

    def _messages(self) -> list[dict]:
        # Never duplicate the system prompt: [0] + tail double-counts it
        # while history is short (extra tokens on every early turn).
        tail = self.history[-6:]
        if tail and tail[0].get("role") == "system":
            return tail
        return [self.history[0]] + tail

    async def stream_response(self, user_text: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": user_text})
        messages = self._messages()
        stream = await self._create_stream(messages)
        try:
            async for chunk in stream:
                try:
                    token = chunk.choices[0].delta.content or ""
                except Exception:
                    token = ""
                if token:
                    yield token
        finally:
            # Runs on barge-in too (generator closed or cancelled): release
            # the HTTP stream instead of leaking it until GC.
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    r = close()
                    if hasattr(r, "__await__"):
                        await r
                except Exception:
                    pass

    def commit_assistant(self, text: str) -> None:
        """Record what the assistant actually said (partial text on barge-in)."""
        if text:
            self.history.append({"role": "assistant", "content": text})
        # Bound memory: keep system + last 12 exchanges.
        if len(self.history) > 25:
            self.history = [self.history[0]] + self.history[-24:]
