"""2-Tier Adaptive Clause Chunker (Component 5) — the sub-400 ms TTFA secret.

Tier 1 (first chunk fast-path): emit as soon as 4-6 words OR any clause
punctuation (,, ;, :, -, —) is seen. This gets the first phrase to TTS
within ~120 ms of LLM start instead of waiting for a full sentence.
Tier 2 (subsequent): wait for full sentence terminators (., !, ?, newline)
to preserve natural prosody.
"""
from __future__ import annotations


class AdaptiveClauseChunker:
    CLAUSE_PUNCTUATION = (",", ";", ":", "-", "—")
    SENTENCE_PUNCTUATION = (".", "!", "?", "\n")
    FIRST_CHUNK_WORDS = 5
    FIRST_CHUNK_MIN_WORDS_WITH_CLAUSE = 2

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.is_first_chunk = True
        self.buffer = ""

    def push(self, token: str) -> list[str]:
        self.buffer += token
        chunks: list[str] = []
        if self.is_first_chunk:
            words = self.buffer.strip().split()
            has_clause = any(p in self.buffer for p in self.CLAUSE_PUNCTUATION + self.SENTENCE_PUNCTUATION)
            if (has_clause and len(words) >= self.FIRST_CHUNK_MIN_WORDS_WITH_CLAUSE) or \
               len(words) >= self.FIRST_CHUNK_WORDS:
                text = self.buffer.strip()
                if text:
                    chunks.append(text)
                self.buffer = ""
                self.is_first_chunk = False
                return chunks
        if any(p in self.buffer for p in self.SENTENCE_PUNCTUATION):
            text = self.buffer.strip()
            if text:
                chunks.append(text)
            self.buffer = ""
        return chunks

    def flush(self) -> list[str]:
        rem = self.buffer.strip()
        self.buffer = ""
        self.is_first_chunk = False
        return [rem] if rem else []
