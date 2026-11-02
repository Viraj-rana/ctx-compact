"""Benchmark runner: golden convo through mly vs baselines."""

from __future__ import annotations

from dataclasses import dataclass

from ctxc.base import last_n, sim_evict
from ctxc.dy import build_dy
from ctxc.mly import CompactOut, mly_select
from ctxc.trn import Trn

# Constraint phrase that must survive for the probe to pass.
CONSTRAINT = "hard limit of 12 tokens per tool call"


def golden_convo() -> tuple[list[Trn], str, str]:
    """Long dialogue: early pin_fact constraint, late back-ref probe need.

    Returns (turns, pin_prefix, query).
    """
    pin = "You are a careful assistant. Obey all user constraints."
    rows = [
        ("t0", "Hi, I need help designing an agent loop.", False),
        ("t1", f"Important: use a {CONSTRAINT}. Never exceed it.", True),
        ("t2", "Also log every tool name to stdout.", False),
        ("t3", "We will call Search, Fetch, and Summarize tools.", False),
        ("t4", "Search should look over docs first.", False),
        ("t5", "Fetch pulls the top three urls.", False),
        ("t6", "Summarize must be short.", False),
        ("t7", "Please add retries on network errors.", False),
        ("t8", "Retries: three attempts with backoff.", False),
        ("t9", "Ok sketch the full loop with tools.", False),
        ("t10", "Include error handling and logging.", False),
        (
            "t11",
            "As I said earlier about that limit — what was the hard cap again?",
            False,
        ),
    ]
    trns: list[Trn] = []
    for i, (tid, txt, pin_fact) in enumerate(rows):
        ent: set[str] = set()
        if pin_fact:
            ent.add("limit")
            ent.add("tokens")
            ent.add("tool")
        if "limit" in txt.lower() or "as i said" in txt.lower():
            ent |= {"limit", "tokens"}
        if "Search" in txt or "Fetch" in txt or "Summarize" in txt:
            ent |= {w.lower() for w in ("Search", "Fetch", "Summarize") if w in txt}
        trns.append(Trn(id=tid, txt=txt, idx=i, ent=ent, pin_fact=pin_fact))
    qry = "what was the hard cap again?"
    return trns, pin, qry


def constraint_kept(out: CompactOut, phrase: str = CONSTRAINT) -> bool:
    """Stub judge: constraint string (or clear pin fragment) present in context."""
    ctx = out.ctx.lower()
    if phrase.lower() in ctx:
        return True
    # collapsed pin may truncate; accept key fragments together
    return "12" in ctx and "limit" in ctx and "token" in ctx


@dataclass
class BenchRow:
    method: str
    kept_ok: bool
    used_tok: int
    tok_bud: int
    n_kept: int


def run_bench(tok_bud: int = 55) -> list[BenchRow]:
    """Run mly / last_n / sim_evict on the golden convo."""
    trns, pin, qry = golden_convo()
    dy = build_dy(trns)
    outs = [
        mly_select(trns, tok_bud, qry, pin=pin, dy=dy),
        last_n(trns, tok_bud, pin=pin),
        sim_evict(trns, tok_bud, qry, pin=pin, thr=0.5),
    ]
    rows: list[BenchRow] = []
    for out in outs:
        rows.append(
            BenchRow(
                method=out.method,
                kept_ok=constraint_kept(out),
                used_tok=out.used_tok,
                tok_bud=tok_bud,
                n_kept=len(out.kept),
            )
        )
    return rows
