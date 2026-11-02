"""Eval runner: methods x examples x budgets, with compaction timing."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ctxc.base import last_n, ovlp_rec, sim_evict, sum_drop
from ctxc.eval.dataset import BenchEx, load_jsonl, repo_data_path
from ctxc.eval.judge import judge_ctx
from ctxc.eval.ttft import est_prefill_ms
from ctxc.mly import CompactOut, mly_select


CompactFn = Callable[[BenchEx, int], CompactOut]


@dataclass
class MethodSpec:
    name: str
    fn: CompactFn


@dataclass
class RunRow:
    method: str
    tok_bud: int
    n: int
    acc: float
    mean_used_tok: float
    mean_compact_ms: float
    mean_est_prefill_ms: float


def _default_methods() -> list[MethodSpec]:
    def _mly(ex: BenchEx, bud: int) -> CompactOut:
        return mly_select(ex.trns, bud, ex.qry, pin=ex.pin, method="mly")

    def _mly_no_dy(ex: BenchEx, bud: int) -> CompactOut:
        return mly_select(
            ex.trns,
            bud,
            ex.qry,
            pin=ex.pin,
            use_dy=False,
            collapse_parents=False,
            method="mly_no_dy",
        )

    def _mly_no_collapse(ex: BenchEx, bud: int) -> CompactOut:
        return mly_select(
            ex.trns,
            bud,
            ex.qry,
            pin=ex.pin,
            collapse_parents=False,
            method="mly_no_collapse",
        )

    def _mly_ovlp(ex: BenchEx, bud: int) -> CompactOut:
        return mly_select(
            ex.trns,
            bud,
            ex.qry,
            pin=ex.pin,
            score_mode="ovlp_rec",
            method="mly_ovlp_score",
        )

    def _last(ex: BenchEx, bud: int) -> CompactOut:
        return last_n(ex.trns, bud, pin=ex.pin)

    def _sim(ex: BenchEx, bud: int) -> CompactOut:
        return sim_evict(ex.trns, bud, ex.qry, pin=ex.pin, thr=0.85)

    def _ovlp(ex: BenchEx, bud: int) -> CompactOut:
        return ovlp_rec(ex.trns, bud, ex.qry, pin=ex.pin)

    def _sum(ex: BenchEx, bud: int) -> CompactOut:
        return sum_drop(ex.trns, bud, pin=ex.pin)

    return [
        MethodSpec("mly", _mly),
        MethodSpec("mly_no_dy", _mly_no_dy),
        MethodSpec("mly_no_collapse", _mly_no_collapse),
        MethodSpec("mly_ovlp_score", _mly_ovlp),
        MethodSpec("last_n", _last),
        MethodSpec("sim_evict", _sim),
        MethodSpec("ovlp_rec", _ovlp),
        MethodSpec("sum_drop", _sum),
    ]


def run_eval(
    examples: list[BenchEx],
    tok_bud: int,
    *,
    methods: list[MethodSpec] | None = None,
) -> list[RunRow]:
    methods = methods or _default_methods()
    # accumulate
    ok: dict[str, int] = defaultdict(int)
    used: dict[str, list[int]] = defaultdict(list)
    cms: dict[str, list[float]] = defaultdict(list)
    pms: dict[str, list[float]] = defaultdict(list)

    for ex in examples:
        for spec in methods:
            t0 = time.perf_counter()
            out = spec.fn(ex, tok_bud)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if judge_ctx(out, must_frags=ex.must_frags, answer_re=ex.answer_re):
                ok[spec.name] += 1
            used[spec.name].append(out.used_tok)
            cms[spec.name].append(dt_ms)
            pms[spec.name].append(est_prefill_ms(out.used_tok))

    n = len(examples) or 1
    rows: list[RunRow] = []
    for spec in methods:
        u = used[spec.name]
        rows.append(
            RunRow(
                method=spec.name,
                tok_bud=tok_bud,
                n=len(examples),
                acc=ok[spec.name] / n,
                mean_used_tok=(sum(u) / len(u)) if u else 0.0,
                mean_compact_ms=(sum(cms[spec.name]) / len(cms[spec.name]))
                if cms[spec.name]
                else 0.0,
                mean_est_prefill_ms=(sum(pms[spec.name]) / len(pms[spec.name]))
                if pms[spec.name]
                else 0.0,
            )
        )
    return rows


def sweep_budgets(
    examples: list[BenchEx],
    budgets: list[int] | None = None,
    *,
    methods: list[MethodSpec] | None = None,
) -> list[RunRow]:
    budgets = budgets or [40, 55, 80, 120]
    rows: list[RunRow] = []
    for b in budgets:
        rows.extend(run_eval(examples, b, methods=methods))
    return rows


def format_table(rows: list[RunRow]) -> str:
    hdr = (
        f"{'method':18} {'bud':>4} {'acc':>6} {'tok':>7} "
        f"{'cmp_ms':>8} {'pre_ms':>8}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r.method:18} {r.tok_bud:4d} {r.acc:6.2f} {r.mean_used_tok:7.1f} "
            f"{r.mean_compact_ms:8.2f} {r.mean_est_prefill_ms:8.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run ctxc multi-dialogue eval")
    p.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="test",
    )
    p.add_argument("--budgets", default="40,55,80,120")
    args = p.parse_args(argv)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    splits = []
    if args.split in ("dev", "both"):
        splits.append(("dev", repo_data_path("bench_dev.jsonl")))
    if args.split in ("test", "both"):
        splits.append(("test", repo_data_path("bench_test.jsonl")))

    for name, path in splits:
        ex = load_jsonl(path)
        print(f"\n=== split={name} n={len(ex)} path={path} ===")
        rows = sweep_budgets(ex, budgets)
        print(format_table(rows))


if __name__ == "__main__":
    main()
