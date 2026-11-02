"""Score-per-token and embedding similarity helpers."""

from __future__ import annotations

import re

import numpy as np

from ctxc.trn import Trn

_WORD = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _toks(txt: str) -> set[str]:
    return set(_WORD.findall((txt or "").lower()))


def ovlp(qry: str, txt: str) -> float:
    """Query token overlap into content (intersection / content size)."""
    c = _toks(txt)
    if not c:
        return 0.0
    q = _toks(qry)
    if not q:
        return 0.0
    return len(c & q) / len(c)


def bow_emb(txt: str, dim: int = 64) -> np.ndarray:
    import hashlib

    v = np.zeros(dim, dtype=np.float64)
    for w in _toks(txt):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
        v[h] += 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def ensure_emb(trn: Trn, dim: int = 64) -> np.ndarray:
    if trn.emb is None:
        trn.emb = bow_emb(trn.txt, dim=dim)
    return trn.emb


def emb_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; 0 if either vector is empty."""
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_per_tok(
    trn: Trn,
    qry: str,
    kept_embs: list[np.ndarray],
    *,
    n_trns: int,
    red_pen: float = 0.85,
    mode: str = "full",
) -> float:
    """Value density for greedy fill.

    mode:
      full — ovlp + recency + pin_fact - emb_sim redundancy
      ovlp_rec — ovlp + recency only (ablation / baseline scorer)
    """
    cost = max(trn.tok, 1)
    rel = ovlp(qry, trn.txt)
    rec = trn.idx / max(n_trns - 1, 1) if n_trns > 1 else 1.0
    if mode == "ovlp_rec":
        return (4.0 * rel + 1.0 * rec) / cost
    pin_b = 1.5 if trn.pin_fact else 0.0
    emb = ensure_emb(trn)
    red = 0.0
    if kept_embs:
        red = max(emb_sim(emb, ke) for ke in kept_embs)
    raw = 4.0 * rel + 1.0 * rec + pin_b - red_pen * red
    return raw / cost
