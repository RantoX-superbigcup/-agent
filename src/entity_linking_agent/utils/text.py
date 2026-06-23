"""Text helpers for normalization and lightweight matching."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Optional

_NORMALIZE_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    return _NORMALIZE_PATTERN.sub("", text.lower())


def sequence_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def extract_context(
    text: str,
    start: Optional[int],
    end: Optional[int],
    fallback: str = "",
    window: int = 48,
) -> str:
    if start is None or end is None or start < 0 or end < start or end > len(text):
        return fallback or text
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def keyword_hits(context: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in context]
