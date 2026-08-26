import os
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt so the model identifies as Jarvis instead of ChatGPT,
# and keeps replies short since this is a voice-only interface.
_SYSTEM = (
    "You are Jarvis, a voice assistant. You are NOT ChatGPT. "
    "Never say you are ChatGPT. Never say you are an AI language model. "
    "Keep responses short and conversational — one or two sentences max "
    "unless the user explicitly asks for detail. "
    "Never use special characters like *, #, [], _, ~, or emojis. "
    "Only use plain text and basic punctuation (period, comma, question mark, exclamation)."
    "Your reply must be in natural, human-like English. With slighly umm , hn text"
)


def chat(message: str) -> str:
    """Non-streaming: return the full reply as a single string."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


def chat_stream(message: str) -> Generator[str, None, None]:
    """Streaming: yield tokens as they arrive from Groq for low latency."""
    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": message},
        ],
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


if __name__ == "__main__":
    print("=== chat() ===")
    print(chat("hello"))
    print("\n=== chat_stream() ===")
    for token in chat_stream("Say hi in one sentence"):
        print(token, end="", flush=True)
    print()
