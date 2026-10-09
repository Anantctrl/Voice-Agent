"""Transcript validation: last gate before a transcript may reach the LLM.

AEC removes predictable echo from audio; this removes *unpredictable*
garbage from transcripts: single-word STT fragments ("sit"), repeated-char
noise, numeric junk, and short shards arriving right after a barge. Layered
with (not replacing) the echo-text filters: those match against what we
said, this judges the transcript on its own merits.

The allowlist is a safety mechanism, not a language model: real answers
("Friday", "Programming") flow through the short-idle path and confidence
scores, never through list membership alone.
"""
from __future__ import annotations

import re


class TranscriptValidator:
    VALID_SINGLE_WORDS = frozenset({
        "yes", "no", "yeah", "nope", "stop", "cancel", "skip", "pause",
        "help", "hello", "hi", "bye", "ok", "okay", "sure", "thanks",
    })

    def __init__(self, min_single_confidence: float = 0.90,
                 barge_window_s: float = 0.2,
                 min_multi_confidence: float = 0.65,
                 soup_ratio: float = 0.2):
        self.min_single_confidence = min_single_confidence
        self.barge_window_s = barge_window_s
        # Low-confidence AND fragmentary transcripts are almost always STT
        # garbage ("Any AI is the... l l n" at 0.607). Confidence alone never
        # rejects (noisy rooms are quiet but real); soup alone never rejects
        # (abbreviations exist). Both together mean garbage with evidence.
        self.min_multi_confidence = min_multi_confidence
        self.soup_ratio = soup_ratio

    @staticmethod
    def norm_words(text: str) -> list[str]:
        s = text.lower()
        s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        return " ".join(s.split()).split()

    @staticmethod
    def _consonant_soup_ratio(words: list[str]) -> float:
        """Fraction of single-letter consonant tokens ('l l n'). Vowels ('I',
        'a') are normal words and never count — only consonant spray does."""
        if not words:
            return 0.0
        vowels = set("aeiou")
        soup = sum(1 for w in words if len(w) == 1 and w not in vowels)
        return soup / len(words)

    @staticmethod
    def _is_coherent(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        for char in set(lowered):
            if lowered.count(char) > len(text) * 0.7:
                return False  # repeated-character noise ("aaaa...")
        words = lowered.split()
        if len(words) > 1 and len(set(words)) == 1:
            return False  # repeated-word stutter ("built built"): STT echo
        # Numeric garbage.
        return not all(char.isdigit() or char.isspace() for char in text)

    def validate(self, text: str, confidence: float | None = None,
                 time_since_barge: float | None = None) -> tuple[bool, str]:
        text = (text or "").strip()
        if not text:
            return False, "empty"
        words = self.norm_words(text)
        if not words:
            return False, "empty"
        if len(words) == 1:
            word = words[0]
            if word in self.VALID_SINGLE_WORDS:
                return True, "valid_single_word"
            if confidence is not None and confidence >= self.min_single_confidence:
                return True, "high_confidence_single_word"
            return False, "unknown_single_word"
        if (time_since_barge is not None
                and time_since_barge < self.barge_window_s
                and len(words) < 3):
            return False, "short_post_barge_fragment"
        if not self._is_coherent(text):
            return False, "incoherent"
        if (confidence is not None and confidence < self.min_multi_confidence
                and self._consonant_soup_ratio(words) >= self.soup_ratio):
            # Low confidence AND fragmentary ("...about the l l n" at 0.607):
            # either alone is survivable (noisy rooms, abbreviations), both
            # together mean STT garbage with evidence.
            return False, "low_confidence_fragment"
        return True, "valid"
