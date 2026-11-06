import json

import numpy as np
import pytest

from ctxc.eval.locomo import load_locomo
from ctxc.eval.qa import f1, fmt_ctx, oracle, summarize
from ctxc.mly import mly_select
from ctxc.score import minmax

from test_locomo import SAMPLE


def _conv(tmp_path):
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return load_locomo(p)[0]


def test_f1():
    assert f1("a corgi", "Corgi.") == 1.0
    assert f1("unknown", "corgi") == 0.0
    assert f1("Biscuit the corgi", "Biscuit, corgi") == 1.0
    assert 0 < f1("pottery classes", "pottery") < 1


def test_gold_and_dates(tmp_path):
    conv = _conv(tmp_path)
    assert conv.qs[0].ans == "corgi"
    assert conv.dates["D2"] == "2:00 pm on 9 May, 2023"


def test_fmt_ctx_groups_by_session(tmp_path):
    conv = _conv(tmp_path)
    ctx = fmt_ctx(conv, oracle(conv, {"D1:3", "D2:2"}))
    assert ctx.splitlines()[0] == "[1:00 pm on 1 May, 2023]"
    assert "[2:00 pm on 9 May, 2023]" in ctx
    assert ctx.index("corgi") < ctx.index("pottery")


def test_ext_scores(tmp_path):
    conv = _conv(tmp_path)
    sc = np.zeros(len(conv.trns))
    sc[4] = 1.0
    out = mly_select(conv.trns, conv.trns[4].tok, "", use_dy=False, score_mode="ext", ext_sc=sc)
    assert [t.id for t in out.kept] == ["D2:2"]
    with pytest.raises(ValueError):
        mly_select(conv.trns, 10, "", score_mode="ext")
    assert minmax(np.array([2.0, 4.0])).tolist() == [0.0, 1.0]


def test_summarize():
    rows = [
        {"method": "m", "tok_bud": 10, "cat": "single-hop", "f1": 1.0, "j": True, "ctx_tok": 5, "ms": 1},
        {"method": "m", "tok_bud": 10, "cat": "multi-hop", "f1": 0.0, "j": False, "ctx_tok": 7, "ms": 3},
    ]
    s = {r["cat"]: r for r in summarize(rows)}
    assert s["all"]["f1"] == 0.5 and s["all"]["j"] == 0.5 and s["all"]["n"] == 2
