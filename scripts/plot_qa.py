"""Render results/locomo_qa.png from results/locomo_qa.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "locomo_qa.csv"
OUT = ROOT / "results" / "locomo_qa.png"

ORDER = ["oracle", "mly_hybrid", "hybrid", "mly_ctx", "bm25", "last_n"]
CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
COLORS = {
    "oracle": "#9e9e9e",
    "mly_hybrid": "#d62728",
    "hybrid": "#ff9896",
    "mly_ctx": "#ff7f0e",
    "bm25": "#1565c0",
    "last_n": "#546e7a",
}


def style(ax, title):
    ax.set_facecolor("#eeeeee")
    ax.grid(True, axis="y", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, pad=8)


def main() -> None:
    d: dict[tuple[str, str], dict] = {}
    with SRC.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["method"], r["cat"])] = r
    ms = [m for m in ORDER if (m, "all") in d]
    n = int(d[(ms[0], "all")]["n"])
    bud = d[(ms[0], "all")]["tok_bud"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.patch.set_facecolor("#fafafa")

    ax = axes[0]
    style(ax, "Overall")
    xs = np.arange(len(ms))
    w = 0.38
    for k, (key, lab) in enumerate((("j", "judge accuracy"), ("f1", "token F1"))):
        ys = [float(d[(m, "all")][key]) * 100 for m in ms]
        bars = ax.bar(xs + (k - 0.5) * w, ys, width=w, zorder=3,
                      color=[COLORS[m] for m in ms], alpha=1.0 if k == 0 else 0.55,
                      label=lab)
        for x, y in zip(xs + (k - 0.5) * w, ys):
            ax.text(x, y + 1, f"{y:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ms, fontsize=8.5, rotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              handles=[plt.Rectangle((0, 0), 1, 1, color="#555"),
                       plt.Rectangle((0, 0), 1, 1, color="#555", alpha=0.55)],
              labels=["judge accuracy", "token F1"])

    ax = axes[1]
    style(ax, "Judge accuracy by question type")
    xs = np.arange(len(CATS))
    w = 0.8 / len(ms)
    for k, m in enumerate(ms):
        ys = [float(d[(m, c)]["j"]) * 100 for c in CATS]
        ax.bar(xs + (k - (len(ms) - 1) / 2) * w, ys, width=w, color=COLORS[m], label=m, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(CATS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)

    fig.suptitle(
        f"LoCoMo held-out answers, qwen2.5:7b on the compacted context ({bud} words, {n} questions)",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
