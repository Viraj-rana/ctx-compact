"""Prefill / TTFT proxies (offline, no serving stack required)."""

from __future__ import annotations


def est_prefill_ms(n_tok: int, *, ms_per_tok: float = 0.05) -> float:
    """Linear prefill-time proxy: tokens * ms_per_tok.

    Swap for real vLLM/SGLang TTFT when available; CI uses this estimate.
    """
    return max(n_tok, 0) * ms_per_tok
