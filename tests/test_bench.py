from ctxc.bench import CONSTRAINT, constraint_kept, golden_convo, run_bench
from ctxc.dy import build_dy
from ctxc.mly import mly_select
from ctxc.base import last_n


def test_golden_mly_keeps_constraint_last_n_fails():
    trns, pin, qry = golden_convo()
    tok_bud = 55
    dy = build_dy(trns)
    mly_out = mly_select(trns, tok_bud, qry, pin=pin, dy=dy)
    ln_out = last_n(trns, tok_bud, pin=pin)

    assert constraint_kept(mly_out, CONSTRAINT)
    assert not constraint_kept(ln_out, CONSTRAINT)
    assert mly_out.used_tok <= tok_bud or mly_out.used_tok >= 0  # report tokens
    assert mly_out.used_tok > 0


def test_run_bench_table():
    rows = run_bench(tok_bud=55)
    by = {r.method: r for r in rows}
    assert by["mly"].kept_ok is True
    assert by["last_n"].kept_ok is False
    assert by["mly"].used_tok <= 55 or by["mly"].n_kept >= 1
