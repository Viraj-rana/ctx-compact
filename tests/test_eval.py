from pathlib import Path

from ctxc.eval import judge_ctx, load_jsonl, repo_data_path, run_eval, sweep_budgets, stub_answer
from ctxc.eval.run import format_table
from ctxc.mly import CompactOut, mly_select
from ctxc.base import last_n


def test_data_files_exist():
    assert repo_data_path("bench_dev.jsonl").is_file()
    assert repo_data_path("bench_test.jsonl").is_file()


def test_load_jsonl_auto_ent():
    exs = load_jsonl(repo_data_path("bench_dev.jsonl"))
    assert len(exs) == 12
    assert all(t.pin_fact is False for ex in exs for t in ex.trns)
    assert any(t.ent for ex in exs for t in ex.trns)


def test_stub_judge():
    out = CompactOut(
        pin="sys",
        kept=[],
        pins_extra=["Important: hard limit of 12 tokens per tool call."],
        used_tok=10,
        tok_bud=40,
    )
    assert judge_ctx(
        out,
        must_frags=["12", "limit", "token"],
        answer_re=r"(?i)\b12\b",
    )
    assert "12" in stub_answer(out.ctx, must_frags=["12", "limit"])


def test_run_eval_dev_smoke():
    exs = load_jsonl(repo_data_path("bench_dev.jsonl"))
    rows = run_eval(exs, tok_bud=80)
    by = {r.method: r for r in rows}
    assert "mly" in by and "last_n" in by
    assert by["mly"].n == 12
    assert 0.0 <= by["mly"].acc <= 1.0
    # at generous budget mly should beat or match last_n on this suite
    assert by["mly"].acc >= by["last_n"].acc


def test_sweep_and_table():
    exs = load_jsonl(repo_data_path("bench_test.jsonl"))[:4]
    rows = sweep_budgets(exs, budgets=[55, 80])
    assert len(rows) >= 8
    text = format_table(rows)
    assert "mly" in text
    assert "acc" in text


def test_mly_beats_last_n_on_held_out_at_55():
    """Portfolio claim: dependency selector vs last-N on test split @ 55."""
    exs = load_jsonl(repo_data_path("bench_test.jsonl"))
    mly_ok = 0
    ln_ok = 0
    for ex in exs:
        m = mly_select(ex.trns, 55, ex.qry, pin=ex.pin)
        ln = last_n(ex.trns, 55, pin=ex.pin)
        if judge_ctx(m, must_frags=ex.must_frags, answer_re=ex.answer_re):
            mly_ok += 1
        if judge_ctx(ln, must_frags=ex.must_frags, answer_re=ex.answer_re):
            ln_ok += 1
    assert mly_ok > ln_ok
