"""Render results/locomo.png from results/locomo_sweep.csv (held-out test split)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "locomo_sweep.csv"
OUT = ROOT / "results" / "locomo.png"

LINES = ["mly_hybrid", "hybrid", "dense", "mly_ctx", "bm25_stem", "bm25", "last_n", "sum_drop"]
CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
COLORS = {
    "mly_hybrid": "#d62728",
    "hybrid": "#ff9896",
    "hybrid_nb": "#e377c2",
    "dense": "#9467bd",
    "dense_nb": "#c5b0d5",
    "mly_ctx": "#ff7f0e",
    "bm25_stem": "#2e7d32",
    "bm25": "#1565c0",
    "mly_bm25": "#f48fb1",
    "mly_bm25_wide": "#ef6c00",
    "mly": "#7b1fa2",
    "last_n": "#546e7a",
    "sum_drop": "#00897b",
    "sim_evict": "#6d4c41",
    "ovlp_rec": "#5c6bc0",
    "mly_no_dy": "#9e9e9e",
}


def load(split: str = "test"):
    d: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    with SRC.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                d[(r["method"], r["cat"])][int(r["tok_bud"])] = r
    return d


def style(ax, title):
    ax.set_facecolor("#eeeeee")
    ax.grid(True, color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, pad=8)


def main() -> None:
    d = load()
    buds = sorted(next(iter(d.values())).keys())
    n = int(d[("bm25", "all")][buds[0]]["n"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("#fafafa")

    ax = axes[0]
    style(ax, "Evidence recall vs budget")
    for m in LINES:
        ys = [float(d[(m, "all")][b]["recall"]) * 100 for b in buds]
        lw = 2.8 if m == "mly_hybrid" else 1.6
        ax.plot(buds, ys, marker="o", ms=4.5, lw=lw, color=COLORS[m], label=m, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(buds)
    ax.set_xticklabels([str(b) for b in buds])
    ax.set_xlabel("Budget (words)")
    ax.set_ylabel("Evidence turns kept (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="center right", ncol=2)

    ax = axes[1]
    b = buds[1]
    style(ax, f"By question type @ {b} words")
    ms = ["mly_hybrid", "hybrid", "dense", "mly_ctx", "bm25", "last_n"]
    xs = np.arange(len(CATS))
    w = 0.8 / len(ms)
    for k, m in enumerate(ms):
        ys = [float(d[(m, c)][b]["recall"]) * 100 for c in CATS]
        ax.bar(xs + (k - (len(ms) - 1) / 2) * w, ys, width=w, color=COLORS[m], label=m, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(CATS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Evidence turns kept (%)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[2]
    style(ax, "Neighbour smoothing gain, per ranker")
    for with_nb, base, col in (
        ("mly_ctx", "bm25_stem", COLORS["mly_ctx"]),
        ("dense_nb", "dense", COLORS["dense"]),
        ("mly_hybrid", "hybrid", COLORS["mly_hybrid"]),
    ):
        ys = [
            (float(d[(with_nb, "all")][bb]["recall"]) - float(d[(base, "all")][bb]["recall"])) * 100
            for bb in buds
        ]
        ax.plot(buds, ys, marker="o", ms=4, lw=1.8, color=col,
                label=f"{with_nb} − {base}", zorder=3)
    ax.axhline(0, color="#222", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(buds)
    ax.set_xticklabels([str(bb) for bb in buds])
    ax.set_xlabel("Budget (words)")
    ax.set_ylabel("Recall gain (points)")
    ax.legend(fontsize=8, frameon=False, loc="center right")

    fig.suptitle(
        f"LoCoMo held-out conversations (5 conversations, {n} questions) — evidence recall",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
