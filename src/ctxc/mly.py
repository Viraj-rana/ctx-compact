"""Main selector: pinned budget + greedy score/token with parent repair."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ctxc.dy import build_dy, collapse_pin, parent_closure
from ctxc.score import bm25_scores, emb_mat, ovlp
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
    max_depth: int | None = None,
    score_mode: str = "full",
    method: str = "mly",
) -> CompactOut:
    """Fill remaining budget by score/token; keep parent closure of selected turns.

    Similarity only demotes; parents of a kept child are never silently dropped —
    they are kept whole if they fit, else collapsed to a pin line.

    score_mode: 'full' (overlap + recency + redundancy, per token), 'ovlp_rec'
    (overlap + recency, per token) or 'bm25' (BM25 rank, same order as the bm25
    baseline, so the only difference is the parent repair).
    """
    if tok_bud < 0:
        raise ValueError("tok_bud must be >= 0")

    pin = pin or ""
    pin_cost = tok_cnt(pin)
    if pin_cost > tok_bud or not trns:
        return CompactOut(pin=pin, kept=[], used_tok=pin_cost, tok_bud=tok_bud, method=method)

    rem = tok_bud - pin_cost
    if not use_dy:
        dy_map: dict[str, set[str]] = {}
    elif dy is not None:
        dy_map = dy
    else:
        dy_map = build_dy(trns)

    n = len(trns)
    pos = {t.id: i for i, t in enumerate(trns)}
    tok = np.array([t.tok for t in trns])
    cost = np.maximum(tok, 1).astype(np.float64)
    idx = np.array([t.idx for t in trns], dtype=np.float64)
    full = score_mode == "full"
    if score_mode == "bm25":
        base = bm25_scores(trns, qry)
        cost = np.ones(n)
    else:
        rec = idx / max(n - 1, 1) if n > 1 else np.ones(n)
        base = 4.0 * np.array([ovlp(qry, t.txt) for t in trns]) + rec
    if full:
        base = base + np.array([1.5 if t.pin_fact else 0.0 for t in trns])
        emb = emb_mat(trns)
        max_sim = np.full(n, -np.inf)

    alive = np.ones(n, dtype=bool)
    chosen = np.zeros(n, dtype=bool)
    collapsed = np.zeros(n, dtype=bool)
    pins_extra: list[str] = []
    used = 0

    def keep(ids: list[int]) -> None:
        nonlocal max_sim
        chosen[ids] = True
        alive[ids] = False
        if full:
            max_sim = np.maximum(max_sim, (emb[ids] @ emb.T).max(axis=0))

    while alive.any() and used < rem:
        if full:
            red = np.where(np.isfinite(max_sim), max_sim, 0.0)
            sc = (base - 0.85 * red) / cost
        else:
            sc = base / cost
        sc = np.where(alive, sc, -np.inf)
        ties = np.flatnonzero(sc == sc.max())
        i = int(ties[np.argmax(idx[ties])])
        tid = trns[i].id

        need = parent_closure(dy_map, {tid}, max_depth) if use_dy else {tid}
        new = [pos[x] for x in need if x in pos and not chosen[pos[x]] and not collapsed[pos[x]]]
        add_cost = int(tok[new].sum()) if new else 0

        if used + add_cost <= rem:
            keep(new)
            used += add_cost
            continue

        if collapse_parents and use_dy:
            parents = sorted((p for p in new if p != i), key=lambda p: idx[p])
            lines = [collapse_pin(trns[p], max_tok=max_pin_tok) for p in parents]
            extra = sum(tok_cnt(line) for line in lines)
            child_cost = 0 if chosen[i] else int(tok[i])
            if used + child_cost + extra <= rem:
                collapsed[parents] = True
                alive[parents] = False
                if not chosen[i]:
                    keep([i])
                    used += child_cost
                pins_extra.extend(lines)
                used += extra
                continue

        alive[i] = False

    return CompactOut(
        pin=pin,
        kept=[trns[i] for i in np.flatnonzero(chosen)],
        pins_extra=pins_extra,
        used_tok=pin_cost + used,
        tok_bud=tok_bud,
        method=method,
    )
