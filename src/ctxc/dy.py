"""Dependency graph: entity reuse + back-ref edges, parent closure."""

from __future__ import annotations

import re
from collections import defaultdict

from ctxc.trn import Trn

# Later turn mentioning these cues depends on earlier constraint / statement turns.
_BACK_REF = re.compile(
    r"\b("
    r"as i said|as we said|that limit|the limit|that constraint|"
    r"earlier|previously|as mentioned|the second option|that budget|"
    r"hard cap|how many|which timezone|are we allowed|how often|"
    r"ceiling number|remind me"
    r")\b",
    re.IGNORECASE,
)

_CAP_WORD = re.compile(r"\b([A-Z][a-zA-Z0-9_]{1,})\b")
_QUOTED = re.compile(r"['\"]([^'\"]{2,})['\"]")
_NUM = re.compile(r"\b(\d+)\b")
_KEY = re.compile(
    r"\b(limit|token|tokens|retry|retries|budget|ceiling|constraint|rule|"
    r"password|timezone|utc|delete|production|tool|call|days?|units?)\b",
    re.IGNORECASE,
)

_ANCHOR = re.compile(
    r"\b(important|rule|constraint|budget|never|must|maximum|fixed|only)\b",
    re.IGNORECASE,
)


def extract_ent(txt: str) -> set[str]:
    """Entity harvest: quotes, Capitalized tokens, numbers, constraint keywords."""
    ent: set[str] = set()
    for m in _QUOTED.finditer(txt or ""):
        ent.add(m.group(1).strip().lower())
    for m in _CAP_WORD.finditer(txt or ""):
        w = m.group(1)
        if w.lower() in {"i", "the", "a", "an"}:
            continue
        ent.add(w.lower())
    for m in _NUM.finditer(txt or ""):
        ent.add(m.group(1))
    for m in _KEY.finditer(txt or ""):
        ent.add(m.group(1).lower())
    return ent


def is_anchor(trn: Trn) -> bool:
    """Constraint-like turn: labeled pin_fact or cue words + a number/never."""
    if trn.pin_fact:
        return True
    txt = trn.txt or ""
    if not _ANCHOR.search(txt):
        return False
    return bool(_NUM.search(txt) or re.search(r"\bnever\b", txt, re.I))


def enrich_ent(trns: list[Trn]) -> None:
    """Fill empty ent sets from text."""
    for trn in trns:
        if not trn.ent:
            trn.ent = extract_ent(trn.txt)


def build_dy(trns: list[Trn]) -> dict[str, set[str]]:
    """Build child -> parents map.

    Edge parent -> child when a later turn reuses an entity first introduced
    in an earlier turn, or when the later turn matches a back-ref / probe cue
    (then every earlier anchor turn is a parent).
    """
    enrich_ent(trns)
    intro: dict[str, str] = {}
    dy: dict[str, set[str]] = defaultdict(set)
    anchors: list[str] = []

    for trn in sorted(trns, key=lambda t: t.idx):
        dy.setdefault(trn.id, set())
        if _BACK_REF.search(trn.txt or "") and anchors:
            for pid in anchors:
                if pid != trn.id:
                    dy[trn.id].add(pid)
        for e in trn.ent:
            if e not in intro:
                intro[e] = trn.id
            else:
                parent = intro[e]
                if parent != trn.id:
                    dy[trn.id].add(parent)
        if is_anchor(trn):
            anchors.append(trn.id)

    return dict(dy)


def parent_closure(dy: dict[str, set[str]], ids: set[str]) -> set[str]:
    """All ancestors of ids (transitive parents), including ids themselves."""
    out = set(ids)
    stack = list(ids)
    while stack:
        cur = stack.pop()
        for p in dy.get(cur, ()):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def collapse_pin(trn: Trn, max_tok: int = 12) -> str:
    """One-line pin from a parent turn; truncated by word budget."""
    words = (trn.txt or "").split()
    if len(words) <= max_tok:
        return trn.txt.strip()
    return " ".join(words[:max_tok]).strip() + "…"
