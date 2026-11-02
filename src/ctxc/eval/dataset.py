"""JSONL dialogue loader. Entities extracted automatically (no hand-stuffed ent)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ctxc.dy import extract_ent
from ctxc.trn import Trn


@dataclass
class BenchEx:
    id: str
    pin: str
    trns: list[Trn]
    qry: str
    constraint: str
    must_frags: list[str]
    answer_re: str
    meta: dict = field(default_factory=dict)


def repo_data_path(name: str = "bench_test.jsonl") -> Path:
    """Resolve data/<name> from the ctx-compact repo root."""
    # src/ctxc/eval/dataset.py -> parents[3] = repo root
    root = Path(__file__).resolve().parents[3]
    return root / "data" / name


def load_jsonl(path: str | Path) -> list[BenchEx]:
    path = Path(path)
    out: list[BenchEx] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            trns: list[Trn] = []
            for i, row in enumerate(raw["turns"]):
                txt = row["txt"]
                # auto entities only; do not honor labeled pin_fact in scoring
                ent = extract_ent(txt)
                trns.append(
                    Trn(
                        id=f"{raw['id']}_t{i}",
                        txt=txt,
                        idx=i,
                        ent=ent,
                        pin_fact=False,
                    )
                )
            out.append(
                BenchEx(
                    id=raw["id"],
                    pin=raw.get("pin", ""),
                    trns=trns,
                    qry=raw["qry"],
                    constraint=raw.get("constraint", ""),
                    must_frags=list(raw.get("must_frags", [])),
                    answer_re=raw["answer_re"],
                    meta=dict(raw.get("split_meta", {})),
                )
            )
    return out
