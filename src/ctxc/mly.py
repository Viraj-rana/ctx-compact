"""Main selector: pinned budget + greedy score/token with parent repair."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ctxc.dy import build_dy, collapse_pin, parent_closure
from ctxc.score import ensure_emb, score_per_tok
from ctxc.tok import tok_cnt
from ctxc.trn import Trn


@dataclass
class CompactOut:
    """Result of one compaction pass."""

    pin: str
    kept: list[Trn]
    pins_extra: list[str] = field(default_factory=list)
    used_tok: int = 0
    tok_bud: int = 0
    method: str = "mly"

    @property
    def ctx(self) -> str:
        parts: list[str] = []
        if self.pin.strip():
            parts.append(self.pin.strip())
        for p in self.pins_extra:
            if p.strip():
                parts.append(p.strip())
        for trn in sorted(self.kept, key=lambda t: t.idx):
            parts.append(trn.txt.strip())
        return "\n".join(parts)


def mly_select(
    trns: list[Trn],
    tok_bud: int,
    qry: str,
    *,
    pin: str = "",
    dy: dict[str, set[str]] | None = None,
    collapse_parents: bool = True,
    max_pin_tok: int = 12,
    use_dy: bool = True,
    score_mode: str = "full",
    method: str = "mly",
) -> CompactOut:
    """Fill remaining budget by score/token; keep parent closure of selected turns.

    Similarity only demotes via score_per_tok; parents of a kept child are never
    silently dropped — they are kept whole if they fit, else collapsed to a pin line.

    Ablations: use_dy=False, collapse_parents=False, score_mode='ovlp_rec'.
    """
    if tok_bud < 0:
        raise ValueError("tok_bud must be >= 0")

    pin = pin or ""
    pin_cost = tok_cnt(pin)
    if pin_cost > tok_bud:
        return CompactOut(pin=pin, kept=[], used_tok=pin_cost, tok_bud=tok_bud, method=method)

    rem = tok_bud - pin_cost
    if not use_dy:
        dy_map: dict[str, set[str]] = {t.id: set() for t in trns}
    elif dy is not None:
        dy_map = dy
    else:
        dy_map = build_dy(trns)
    by_id = {t.id: t for t in trns}
    n = len(trns)

    chosen: set[str] = set()
    collapsed: set[str] = set()
    pins_extra: list[str] = []
    kept_embs: list[np.ndarray] = []
    used = 0
    remaining = {t.id for t in trns}

    def whole_cost(ids: set[str]) -> int:
        return sum(
            by_id[i].tok
            for i in ids
            if i in by_id and i not in chosen and i not in collapsed
        )

    while remaining and used < rem:
        prio_q = sorted(
            remaining,
            key=lambda tid: (
                score_per_tok(
                    by_id[tid], qry, kept_embs=kept_embs, n_trns=n, mode=score_mode
                ),
                by_id[tid].idx,
            ),
            reverse=True,
        )
        tid = prio_q[0]
        trn = by_id[tid]
        need = parent_closure(dy_map, {tid}) if use_dy else {tid}
        new_ids = {i for i in need if i not in chosen and i not in collapsed}
        add_cost = whole_cost(new_ids)

        if used + add_cost <= rem:
            for i in new_ids:
                chosen.add(i)
                remaining.discard(i)
                ensure_emb(by_id[i])
                kept_embs.append(by_id[i].emb)  # type: ignore[arg-type]
            used += add_cost
            continue

        if collapse_parents and use_dy:
            parent_ids = new_ids - {tid}
            pin_lines = [
                collapse_pin(by_id[pid], max_tok=max_pin_tok)
                for pid in sorted(parent_ids, key=lambda x: by_id[x].idx)
            ]
            pin_cost_extra = sum(tok_cnt(line) for line in pin_lines)
            child_cost = 0 if tid in chosen else trn.tok
            if used + child_cost + pin_cost_extra <= rem and tid not in collapsed:
                for pid in parent_ids:
                    collapsed.add(pid)
                    remaining.discard(pid)
                if tid not in chosen:
                    chosen.add(tid)
                    remaining.discard(tid)
                    ensure_emb(trn)
                    kept_embs.append(trn.emb)  # type: ignore[arg-type]
                    used += child_cost
                for line in pin_lines:
                    pins_extra.append(line)
                used += pin_cost_extra
                continue

        remaining.discard(tid)

    kept = [by_id[i] for i in chosen if i in by_id]
    return CompactOut(
        pin=pin,
        kept=kept,
        pins_extra=pins_extra,
        used_tok=pin_cost + used,
        tok_bud=tok_bud,
        method=method,
    )
