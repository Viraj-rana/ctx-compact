"""Regex / fragment judges + extractive stub answer (no live LLM required)."""

from __future__ import annotations

import re

from ctxc.mly import CompactOut


def judge_ctx(out: CompactOut, *, must_frags: list[str], answer_re: str) -> bool:
    """Pass if compacted context supports the probe.

    Primary: answer_re matches stub_answer(ctx).
    Fallback: all must_frags appear in ctx (case-insensitive).
    """
    ctx = out.ctx
    ans = stub_answer(ctx, must_frags=must_frags)
    if re.search(answer_re, ans):
        return True
    if re.search(answer_re, ctx):
        return True
    low = ctx.lower()
    if must_frags and all(f.lower() in low for f in must_frags):
        return True
    return False


def stub_answer(ctx: str, *, must_frags: list[str] | None = None) -> str:
    """Extractive stand-in for a completion model.

    Returns lines from ctx that hit must_frags / digits; else 'unknown'.
    """
    must_frags = must_frags or []
    hits: list[str] = []
    for line in (ctx or "").splitlines():
        low = line.lower()
        if must_frags and any(f.lower() in low for f in must_frags):
            hits.append(line.strip())
        elif any(ch.isdigit() for ch in line) and (
            "limit" in low
            or "retry" in low
            or "ceiling" in low
            or "utc" in low
            or "day" in low
            or "never" in low
            or "constraint" in low
            or "rule" in low
            or "budget" in low
        ):
            hits.append(line.strip())
    return "\n".join(hits) if hits else "unknown"
