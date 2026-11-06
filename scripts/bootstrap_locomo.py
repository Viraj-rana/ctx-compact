"""Paired bootstrap CIs for recall differences on the LoCoMo held-out split.

Resamples questions with replacement (10,000 draws, seed 0) and reports the mean
difference in evidence recall between two methods with a 95% percentile interval.
Writes results/locomo_bootstrap.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ctxc.eval.locomo import Emb, default_path, load_locomo, methods, recall

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "locomo_bootstrap.csv"
BUDGETS = [500, 1000, 2000, 4000]
PAIRS = [
    ("hybrid_nb", "hybrid"),
    ("hybrid", "dense"),
    ("ctx_no_dy", "bm25_stem"),
    ("dense_nb", "dense"),
    ("hybrid_nb", "bm25"),
]
DRAWS = 10_000


def per_question(names: set[str]) -> dict[tuple[str, int], np.ndarray]:
    convs = load_locomo(default_path())[5:]
    emb = Emb()
    out: dict[tuple[str, int], list[float]] = {(m, b): [] for m in names for b in BUDGETS}
    for conv in convs:
        ms = methods(conv, max_df=0.005, max_depth=1, emb=emb)
        for m in names:
            for b in BUDGETS:
                out[(m, b)].extend(recall(ms[m](q.qry, b), q.ev) for q in conv.qs)
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    rec = per_question({m for p in PAIRS for m in p})
    rng = np.random.default_rng(0)
    n = len(next(iter(rec.values())))
    idx = rng.integers(0, n, size=(DRAWS, n))
    rows = []
    for a, b in PAIRS:
        for bud in BUDGETS:
            d = rec[(a, bud)] - rec[(b, bud)]
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append([a, b, bud, n, f"{d.mean() * 100:.2f}", f"{lo * 100:.2f}", f"{hi * 100:.2f}"])
            print(f"{a:>10} - {b:<10} @ {bud:4d}: {d.mean() * 100:+.2f}  [{lo * 100:+.2f}, {hi * 100:+.2f}]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method_a", "method_b", "tok_bud", "n", "diff", "ci_lo", "ci_hi"])
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
