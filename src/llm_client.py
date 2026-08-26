"""
Thin wrapper so the server imports the LLM through a stable, import-safe path.
"""

# Reuse YOUR existing Groq chat functions unchanged - same model, same client.
from llm import chat, chat_stream

# Re-export under this module's namespace for server.py.
__all__ = ["chat", "chat_stream"]
