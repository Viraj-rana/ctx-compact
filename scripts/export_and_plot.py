"""Run held-out eval, write CSV, render README figures from that data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ctxc.dy import build_dy, is_anchor
from ctxc.eval.dataset import load_jsonl, repo_data_path
from ctxc.eval.run import sweep_budgets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BUDGETS = [40, 55, 80, 120]

MAIN = ["mly", "last_n", "sim_evict", "ovlp_rec", "sum_drop"]
ABL = ["mly", "mly_no_dy", "mly_no_collapse", "mly_ovlp_score"]
COLORS = {
    "mly": "#c62828",
    "last_n": "#546e7a",
    "sim_evict": "#6d4c41",
    "ovlp_rec": "#00897b",
    "sum_drop": "#1565c0",
    "mly_no_dy": "#ef6c00",
    "mly_no_collapse": "#7b1fa2",
    "mly_ovlp_score": "#5c6bc0",
}


def _rows_to_csv(rows, path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "method",
                "tok_bud",
                "n",
                "acc",
                "mean_used_tok",
                "mean_compact_ms",
                "mean_est_prefill_ms",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "split": split,
                    "method": r.method,
                    "tok_bud": r.tok_bud,
                    "n": r.n,
                    "acc": f"{r.acc:.4f}",
                    "mean_used_tok": f"{r.mean_used_tok:.2f}",
                    "mean_compact_ms": f"{r.mean_compact_ms:.3f}",
                    "mean_est_prefill_ms": f"{r.mean_est_prefill_ms:.3f}",
                }
            )


def _by_method(rows):
    d = {}
    for r in rows:
        d.setdefault(r.method, []).append(r)
    for m in d:
        d[m] = sorted(d[m], key=lambda x: x.tok_bud)
    return d


def plot_acc_grid(rows, path: Path) -> None:
    # 2x3 panels — same spirit as multi-task accuracy grids, our metrics
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True)
    fig.patch.set_facecolor("#fafafa")
    panels = [
        (axes[0, 0], MAIN, "acc", "Accuracy", True),
        (axes[0, 1], ABL, "acc", "Ablation accuracy", True),
        (axes[0, 2], MAIN, "mean_used_tok", "Mean tokens kept", False),
        (axes[1, 0], MAIN, "mean_compact_ms", "Compact time (ms)", False),
        (axes[1, 1], MAIN, "mean_est_prefill_ms", "Est. prefill (ms)", False),
        (axes[1, 2], None, "bar55", "Accuracy @ budget 55", True),
    ]
    grouped = _by_method(rows)

    for ax, methods, key, title, is_acc in panels:
        ax.set_facecolor("#eeeeee")
        ax.grid(True, color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, pad=8)

        if key == "bar55":
            methods = MAIN + ["mly_no_dy"]
            xs = np.arange(len(methods))
            vals = []
            for m in methods:
                hit = [r for r in grouped.get(m, []) if r.tok_bud == 55]
                vals.append(hit[0].acc * 100 if hit else 0.0)
            bars = ax.bar(
                xs,
                vals,
                color=[COLORS.get(m, "#333") for m in methods],
                zorder=3,
                width=0.72,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.axhline(50, color="#222", ls="--", lw=0.9, alpha=0.7)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 2,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            continue

        for m in methods:
            series = grouped.get(m, [])
            if not series:
                continue
            xs = [r.tok_bud for r in series]
            ys = [getattr(r, key) * (100 if is_acc else 1) for r in series]
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=4.5,
                lw=1.8,
                color=COLORS.get(m, "#333"),
                label=m,
                zorder=3,
            )
        if is_acc:
            ax.set_ylim(0, 105)
            ax.axhline(50, color="#222", ls="--", lw=0.9, alpha=0.7)
            ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("Token budget")
        ax.legend(fontsize=7, frameon=False, loc="best")

    fig.suptitle(
        "ctxc held-out test (n=12) — real sweep metrics",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_dy_example(path: Path) -> None:
    # one real dialogue: turns as nodes, dy edges as links
    exs = load_jsonl(repo_data_path("bench_test.jsonl"))
    ex = exs[0]
    dy = build_dy(ex.trns)
    n = len(ex.trns)

    # layout: early turns left, late right; slight vertical jitter by hash
    pos = {}
    for t in ex.trns:
        y = ((hash(t.id) % 1000) / 1000.0 - 0.5) * 2.2
        pos[t.id] = (t.idx / max(n - 1, 1) * 10.0, y)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_title(
        f"Dependency graph dy — example `{ex.id}` (entities + back-refs)",
        fontsize=12,
        pad=10,
    )

    # edges first
    for child, parents in dy.items():
        if child not in pos:
            continue
        for p in parents:
            if p not in pos:
                continue
            x0, y0 = pos[p]
            x1, y1 = pos[child]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#ef9a9a",
                lw=0.9,
                alpha=0.75,
                zorder=1,
            )

    for t in ex.trns:
        x, y = pos[t.id]
        if is_anchor(t):
            c, s = "#c62828", 120
        elif t.idx == n - 1:
            c, s = "#00897b", 110
        else:
            c, s = "#90a4ae", 70
        ax.scatter([x], [y], s=s, c=c, zorder=3, edgecolors="#37474f", linewidths=0.6)
        ax.text(x, y + 0.18, f"t{t.idx}", ha="center", va="bottom", fontsize=7, color="#37474f")

    ax.scatter([], [], c="#c62828", s=80, label="anchor / constraint-like")
    ax.scatter([], [], c="#00897b", s=80, label="probe (last turn)")
    ax.scatter([], [], c="#90a4ae", s=60, label="filler")
    ax.plot([], [], color="#ef9a9a", lw=1.2, label="depends-on edge")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("turn order →")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = load_jsonl(repo_data_path("bench_test.jsonl"))
    rows = sweep_budgets(ex, BUDGETS)
    csv_path = OUT / "bench_test_sweep.csv"
    _rows_to_csv(rows, csv_path, "test")
    plot_acc_grid(rows, OUT / "acc_vs_budget.png")
    plot_dy_example(OUT / "dy_example.png")
    print(f"wrote {csv_path}")
    print(f"wrote {OUT / 'acc_vs_budget.png'}")
    print(f"wrote {OUT / 'dy_example.png'}")


if __name__ == "__main__":
    main()
