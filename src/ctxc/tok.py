"""Naive token counting (whitespace / word approx)."""

from __future__ import annotations


def tok_cnt(txt: str) -> int:
    """Approx token count: non-empty whitespace-separated pieces."""
    if not txt or not txt.strip():
        return 0
    return len(txt.split())
