"""Paired bootstrap 95% intervals for judge-accuracy differences on the cached answers."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from ctxc.ollama import CACHE

PAIRS = [("mly_hybrid", "hybrid"), ("mly_hybrid", "bm25"), ("mly_ctx", "bm25"),
         ("hybrid", "bm25"), ("mly_hybrid", "last_n")]


def main(model: str = "qwen2.5:7b", tok_bud: int = 1000, n_boot: int = 10000) -> None:
    by: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for line in (CACHE / f"qa_{model.replace(':', '_')}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["tok_bud"] == tok_bud:
            by[r["method"]][(r["conv"], r["qi"])] = float(r["j"])
    rng = np.random.default_rng(0)
    for a, b in PAIRS:
        keys = sorted(set(by[a]) & set(by[b]))
        d = np.array([by[a][k] - by[b][k] for k in keys])
        bs = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
        print(f"{a:11} - {b:7} n={len(d)}  {d.mean()*100:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]")


if __name__ == "__main__":
    main()
