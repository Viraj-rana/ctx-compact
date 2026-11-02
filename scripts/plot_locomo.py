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

LINES = ["bm25", "mly_bm25", "mly_bm25_wide", "mly", "last_n", "sum_drop"]
CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
COLORS = {
    "bm25": "#1565c0",
    "mly_bm25": "#c62828",
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
        ax.plot(buds, ys, marker="o", ms=4.5, lw=1.8, color=COLORS[m], label=m, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(buds)
    ax.set_xticklabels([str(b) for b in buds])
    ax.set_xlabel("Budget (words)")
    ax.set_ylabel("Evidence turns kept (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    b = buds[1]
    style(ax, f"By question type @ {b} words")
    ms = ["bm25", "mly_bm25", "mly_bm25_wide", "last_n"]
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
    style(ax, "Dependency repair minus BM25")
    for c, ls in (("multi-hop", "-"), ("all", "--")):
        for m in ("mly_bm25", "mly_bm25_wide"):
            ys = [
                (float(d[(m, c)][bb]["recall"]) - float(d[("bm25", c)][bb]["recall"])) * 100
                for bb in buds
            ]
            ax.plot(buds, ys, ls=ls, marker="o", ms=4, lw=1.8, color=COLORS[m],
                    label=f"{m} · {c}", zorder=3)
    ax.axhline(0, color="#222", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(buds)
    ax.set_xticklabels([str(bb) for bb in buds])
    ax.set_xlabel("Budget (words)")
    ax.set_ylabel("Recall difference (points)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

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
