"""Baselines: last-N, similarity eviction, ovlp+recency, extractive drop-summary."""

from __future__ import annotations

from ctxc.mly import CompactOut, mly_select
from ctxc.score import emb_sim, ensure_emb
from ctxc.tok import tok_cnt
from ctxc.trn import Trn


def last_n(
    trns: list[Trn],
    tok_bud: int,
    *,
    pin: str = "",
) -> CompactOut:
    """Keep newest turns that fit after charging the pinned prefix."""
    pin = pin or ""
    pin_cost = tok_cnt(pin)
    rem = max(tok_bud - pin_cost, 0)
    kept: list[Trn] = []
    used = 0
    for trn in sorted(trns, key=lambda t: t.idx, reverse=True):
        if used + trn.tok > rem:
            continue
        kept.append(trn)
        used += trn.tok
    kept.sort(key=lambda t: t.idx)
    return CompactOut(
        pin=pin,
        kept=kept,
        used_tok=pin_cost + used,
        tok_bud=tok_bud,
        method="last_n",
    )


def sim_evict(
    trns: list[Trn],
    tok_bud: int,
    qry: str,
    *,
    pin: str = "",
    thr: float = 0.92,
) -> CompactOut:
    """Walk newest-first; drop a turn if emb_sim to an already-kept turn >= thr."""
    pin = pin or ""
    pin_cost = tok_cnt(pin)
    rem = max(tok_bud - pin_cost, 0)
    kept: list[Trn] = []
    kept_embs = []
    used = 0
    for trn in sorted(trns, key=lambda t: t.idx, reverse=True):
        emb = ensure_emb(trn)
        if kept_embs and max(emb_sim(emb, ke) for ke in kept_embs) >= thr:
            continue
        if used + trn.tok > rem:
            continue
        kept.append(trn)
        kept_embs.append(emb)
        used += trn.tok
    kept.sort(key=lambda t: t.idx)
    return CompactOut(
        pin=pin,
        kept=kept,
        used_tok=pin_cost + used,
        tok_bud=tok_bud,
        method="sim_evict",
    )


def ovlp_rec(
    trns: list[Trn],
    tok_bud: int,
    qry: str,
    *,
    pin: str = "",
) -> CompactOut:
    """Greedy by query overlap + recency; no dependency repair."""
    return mly_select(
        trns,
        tok_bud,
        qry,
        pin=pin,
        use_dy=False,
        collapse_parents=False,
        score_mode="ovlp_rec",
        method="ovlp_rec",
    )


def sum_drop(
    trns: list[Trn],
    tok_bud: int,
    *,
    pin: str = "",
    max_sum_tok: int = 40,
) -> CompactOut:
    """Last-N for recent turns; extractive one-liners for older dropped turns.

    Older turns become short summary pins (first ~8 words each) until max_sum_tok.
    """
    pin = pin or ""
    pin_cost = tok_cnt(pin)
    rem = max(tok_bud - pin_cost, 0)
    # reserve summary budget from rem
    sum_bud = min(max_sum_tok, rem // 3)
    turn_bud = rem - sum_bud

    kept: list[Trn] = []
    used = 0
    for trn in sorted(trns, key=lambda t: t.idx, reverse=True):
        if used + trn.tok > turn_bud:
            continue
        kept.append(trn)
        used += trn.tok
    kept_ids = {t.id for t in kept}
    kept.sort(key=lambda t: t.idx)

    pins_extra: list[str] = []
    sum_used = 0
    for trn in sorted(trns, key=lambda t: t.idx):
        if trn.id in kept_ids:
            continue
        # prefer high-signal / early constraint-like lines
        words = trn.txt.split()
        line = " ".join(words[:8])
        if len(words) > 8:
            line += "…"
        c = tok_cnt(line)
        if sum_used + c > sum_bud:
            continue
        # rank preference: lines with digits or "limit"/"rule"/"constraint"
        low = trn.txt.lower()
        if not (
            any(ch.isdigit() for ch in trn.txt)
            or "limit" in low
            or "rule" in low
            or "constraint" in low
            or "budget" in low
            or "never" in low
            or "must" in low
        ):
            continue
        pins_extra.append(line)
        sum_used += c

    return CompactOut(
        pin=pin,
        kept=kept,
        pins_extra=pins_extra,
        used_tok=pin_cost + used + sum_used,
        tok_bud=tok_bud,
        method="sum_drop",
    )
