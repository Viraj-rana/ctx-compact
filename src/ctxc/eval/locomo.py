"""LoCoMo (Maharana et al., 2024) evidence-recall eval.

Each question lists the dialogue turns that hold its answer. A compactor is scored
by how many of those turns survive compaction — no model in the loop.
Categories follow the LoCoMo release: 1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop. Category 5 (adversarial, unanswerable) is left out.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ctxc.base import bm25, last_n, ovlp_rec, sim_evict, sum_drop
from ctxc.dy import build_dy
from ctxc.mly import CompactOut, mly_select
from ctxc.ollama import Emb
from ctxc.score import bm25_scores, minmax, nb_smooth
from ctxc.trn import Trn

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

_DIA = re.compile(r"D(\d+)\s*:\s*(\d+)")


@dataclass
class LocoQ:
    qry: str
    ev: set[str]
    cat: int
    ans: str = ""


@dataclass
class LocoConv:
    id: str
    trns: list[Trn]
    speakers: set[str]
    qs: list[LocoQ] = field(default_factory=list)
    dropped: int = 0
    dates: dict[str, str] = field(default_factory=dict)


def default_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "locomo" / "locomo10.json"


def _dia(s: str) -> list[str]:
    return [f"D{int(a)}:{int(b)}" for a, b in _DIA.findall(str(s))]


def load_locomo(path: str | Path | None = None) -> list[LocoConv]:
    raw = json.loads(Path(path or default_path()).read_text(encoding="utf-8"))
    out: list[LocoConv] = []
    for s in raw:
        c = s["conversation"]
        nums = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
        trns: list[Trn] = []
        speakers: set[str] = set()
        for num in nums:
            for turn in c[f"session_{num}"]:
                ids = _dia(turn.get("dia_id", ""))
                if not ids:
                    continue
                txt = f"{turn['speaker']}: {turn.get('text', '')}"
                if turn.get("blip_caption"):
                    txt += f" [photo: {turn['blip_caption']}]"
                speakers.add(turn["speaker"])
                trns.append(Trn(id=ids[0], txt=txt, idx=len(trns)))

        known = {t.id for t in trns}
        dates = {f"D{n}": c.get(f"session_{n}_date_time", "") for n in nums}
        conv = LocoConv(
            id=s.get("sample_id", str(len(out))), trns=trns, speakers=speakers, dates=dates
        )
        for q in s["qa"]:
            cat = int(q.get("category", 0))
            if cat not in CATS:
                continue
            ev = {d for e in q.get("evidence", []) for d in _dia(e)} & known
            if not ev:
                conv.dropped += 1
                continue
            conv.qs.append(LocoQ(qry=q["question"], ev=ev, cat=cat, ans=str(q.get("answer", ""))))
        out.append(conv)
    return out


def recall(out: CompactOut, ev: set[str]) -> float:
    kept = {t.id for t in out.kept}
    return len(ev & kept) / len(ev)


def _dy(conv: LocoConv, max_df: float) -> dict[str, set[str]]:
    return build_dy(
        conv.trns,
        back_ref=False,
        max_df=max_df,
        skip_ent={s.lower() for s in conv.speakers},
    )


HYB_W = 0.3
HYB_A = 0.5
DENSE_A = 0.25


def dense_methods(conv: LocoConv, emb: Emb, dy_ctx: dict[str, set[str]]):
    """Dense (nomic-embed-text via Ollama) and BM25+dense hybrids.

    HYB_W (BM25 share of the fused score), HYB_A and DENSE_A (neighbour weights)
    were picked on the dev split.
    """
    trns = conv.trns
    d = emb([f"search_document: {t.txt}" for t in trns])
    qv = dict(zip([q.qry for q in conv.qs], emb([f"search_query: {q.qry}" for q in conv.qs])))

    def cos(q: str) -> np.ndarray:
        return d @ (qv[q] if q in qv else emb([f"search_query: {q}"])[0])

    def hyb(q: str) -> np.ndarray:
        return HYB_W * minmax(bm25_scores(trns, q, stemmed=True)) + (1 - HYB_W) * minmax(cos(q))

    def sel(sc: np.ndarray, b: int, dy=None) -> CompactOut:
        return mly_select(trns, b, "", dy=dy, use_dy=dy is not None, max_depth=1,
                          score_mode="ext", ext_sc=sc)

    return {
        "dense": lambda q, b: sel(cos(q), b),
        "dense_nb": lambda q, b: sel(nb_smooth(trns, minmax(cos(q)), DENSE_A), b),
        "hybrid": lambda q, b: sel(hyb(q), b),
        "hybrid_nb": lambda q, b: sel(nb_smooth(trns, hyb(q), HYB_A), b),
        "mly_hybrid": lambda q, b: sel(nb_smooth(trns, hyb(q), HYB_A), b, dy_ctx),
    }


def methods(conv: LocoConv, *, max_df: float, max_depth: int | None, emb: Emb | None = None):
    """max_df/max_depth were picked on the dev split (best mean recall over budgets)."""
    trns = conv.trns
    dy = _dy(conv, max_df)
    dy_wide = _dy(conv, 0.02)
    dy_ctx = _dy(conv, 0.002)
    extra = dense_methods(conv, emb, dy_ctx) if emb is not None else {}
    return extra | {
        "mly_ctx": lambda q, b: mly_select(
            trns, b, q, dy=dy_ctx, max_depth=1, score_mode="ctx"
        ),
        "ctx_no_dy": lambda q, b: mly_select(trns, b, q, use_dy=False, score_mode="ctx"),
        "mly_bm25": lambda q, b: mly_select(
            trns, b, q, dy=dy, max_depth=max_depth, score_mode="bm25"
        ),
        "mly_bm25_wide": lambda q, b: mly_select(
            trns, b, q, dy=dy_wide, max_depth=2, score_mode="bm25"
        ),
        "mly": lambda q, b: mly_select(trns, b, q, dy=dy, max_depth=max_depth),
        "mly_no_dy": lambda q, b: mly_select(trns, b, q, use_dy=False, collapse_parents=False),
        "bm25": lambda q, b: bm25(trns, b, q),
        "bm25_stem": lambda q, b: bm25(trns, b, q, stemmed=True),
        "ovlp_rec": lambda q, b: ovlp_rec(trns, b, q),
        "last_n": lambda q, b: last_n(trns, b),
        "sim_evict": lambda q, b: sim_evict(trns, b, q, thr=0.85),
        "sum_drop": lambda q, b: sum_drop(trns, b),
    }


@dataclass
class Row:
    split: str
    method: str
    tok_bud: int
    cat: str
    n: int
    recall: float
    all_hit: float
    mean_tok: float
    mean_ms: float


def run(
    convs: list[LocoConv],
    budgets: list[int],
    *,
    split: str,
    max_df: float = 0.005,
    max_depth: int | None = 1,
    only: list[str] | None = None,
    emb: Emb | None = None,
) -> list[Row]:
    acc: dict[tuple[str, int, str], list[tuple[float, int, float]]] = defaultdict(list)
    for conv in convs:
        ms = methods(conv, max_df=max_df, max_depth=max_depth, emb=emb)
        for name, fn in ms.items():
            if only and name not in only:
                continue
            for b in budgets:
                for q in conv.qs:
                    t0 = time.perf_counter()
                    out = fn(q.qry, b)
                    dt = (time.perf_counter() - t0) * 1000.0
                    r = recall(out, q.ev)
                    for cat in (CATS[q.cat], "all"):
                        acc[(name, b, cat)].append((r, out.used_tok, dt))

    rows: list[Row] = []
    for (name, b, cat), xs in sorted(acc.items()):
        k = len(xs)
        rows.append(
            Row(
                split=split,
                method=name,
                tok_bud=b,
                cat=cat,
                n=k,
                recall=sum(x[0] for x in xs) / k,
                all_hit=sum(1 for x in xs if x[0] >= 1.0) / k,
                mean_tok=sum(x[1] for x in xs) / k,
                mean_ms=sum(x[2] for x in xs) / k,
            )
        )
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "tok_bud", "cat", "n", "recall", "all_hit", "mean_tok", "mean_ms"])
        for r in rows:
            w.writerow(
                [r.split, r.method, r.tok_bud, r.cat, r.n, f"{r.recall:.4f}",
                 f"{r.all_hit:.4f}", f"{r.mean_tok:.1f}", f"{r.mean_ms:.2f}"]
            )


def table(rows: list[Row], cat: str = "all") -> str:
    hdr = f"{'method':12} {'bud':>5} {'recall':>7} {'all_hit':>8} {'tok':>7} {'ms':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.cat == cat:
            lines.append(
                f"{r.method:12} {r.tok_bud:5d} {r.recall:7.3f} {r.all_hit:8.3f} "
                f"{r.mean_tok:7.0f} {r.mean_ms:7.2f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoCoMo evidence-recall sweep")
    p.add_argument("--path", default=None)
    p.add_argument("--budgets", default="500,1000,2000,4000")
    p.add_argument("--max-df", type=float, default=0.005)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--dense", action="store_true", help="add dense/hybrid methods (needs Ollama)")
    args = p.parse_args(argv)

    path = Path(args.path) if args.path else default_path()
    if not path.exists():
        raise SystemExit(f"missing {path} — run: python scripts/get_locomo.py")
    convs = load_locomo(path)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    splits = {"dev": convs[:5], "test": convs[5:]}
    emb = Emb() if args.dense else None

    rows: list[Row] = []
    for name, cs in splits.items():
        rs = run(cs, budgets, split=name, max_df=args.max_df, max_depth=args.max_depth, emb=emb)
        if emb is not None:
            emb.save()
        nq = sum(len(c.qs) for c in cs)
        print(f"\n=== {name}: {len(cs)} conversations, {nq} questions ===")
        print(table(rs))
        rows.extend(rs)

    if args.out:
        write_csv(rows, Path(args.out))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
