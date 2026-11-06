"""Score-per-token and embedding similarity helpers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from ctxc.trn import Trn

_WORD = re.compile(r"[a-z0-9_]+", re.IGNORECASE)

STOP = frozenset(
    """a an the and or but if of to in on at for with from by is are was were be been being
    do does did have has had i you he she it we they me him her us them my your his its our their
    this that these those what which who whom when where why how not no so as about into than then
    there here just very can could would should will shall may might must also too up out over
    again more most some any all each both few other such only own same s t don now ever""".split()
)
_SUF = ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "s", "ly")


def stem(w: str) -> str:
    for suf in _SUF:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


@lru_cache(maxsize=None)
def tok_list(txt: str) -> tuple[str, ...]:
    return tuple(_WORD.findall((txt or "").lower()))


@lru_cache(maxsize=None)
def stem_list(txt: str) -> tuple[str, ...]:
    return tuple(stem(w) for w in tok_list(txt) if w not in STOP)


@lru_cache(maxsize=None)
def _toks(txt: str) -> frozenset[str]:
    return frozenset(tok_list(txt))


def ovlp(qry: str, txt: str) -> float:
    """Query token overlap into content (intersection / content size)."""
    c = _toks(txt)
    if not c:
        return 0.0
    q = _toks(qry)
    if not q:
        return 0.0
    return len(c & q) / len(c)


def bm25_scores(
    trns: list[Trn], qry: str, k1: float = 1.5, b: float = 0.75, *, stemmed: bool = False
) -> np.ndarray:
    tl = stem_list if stemmed else tok_list
    docs = [tl(t.txt) for t in trns]
    n = len(docs)
    if not n:
        return np.zeros(0)
    avg = sum(len(d) for d in docs) / n or 1.0
    q = set(tl(qry))
    df = Counter(w for d in docs for w in set(d) if w in q)
    out = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(w for w in d if w in q)
        for w, c in tf.items():
            idf = math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
            out[i] += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * len(d) / avg))
    return out


def minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def _sess(t: Trn) -> str:
    return t.id.split(":")[0] if ":" in t.id else ""


def nb_smooth(trns: list[Trn], s: np.ndarray, a: float) -> np.ndarray:
    """Add a * the best adjacent score within the same session.

    An answer usually sits next to the turn that matches the question
    (question -> reply), so a turn inherits part of its neighbours' relevance.
    """
    if a == 0 or len(s) < 2:
        return s
    ss = [_sess(t) for t in trns]
    prv = np.r_[0.0, s[:-1]] * np.r_[False, [ss[i] == ss[i - 1] for i in range(1, len(s))]]
    nxt = np.r_[s[1:], 0.0] * np.r_[[ss[i] == ss[i + 1] for i in range(len(s) - 1)], False]
    return s + a * np.maximum(prv, nxt)


def bow_emb(txt: str, dim: int = 64) -> np.ndarray:
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


def emb_mat(trns: list[Trn]) -> np.ndarray:
    """Row-normalized embedding matrix, so E @ E.T is cosine similarity."""
    m = np.stack([ensure_emb(t) for t in trns]).astype(np.float64)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


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
