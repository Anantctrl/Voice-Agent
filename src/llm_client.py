"""
Thin wrapper so the server imports the LLM through a stable, import-safe path.

llm.py now guards its demo code with __main__, so importing it here is side-effect free.
"""

# Reuse YOUR existing Groq chat function unchanged - same model, same client.
from llm import chat

# Re-export under this module's namespace for server.py.
__all__ = ["chat"]
