import json

import numpy as np

from ctxc.base import bm25
from ctxc.eval.locomo import load_locomo, recall, run
from ctxc.mly import mly_select
from ctxc.score import nb_smooth, stem, stem_list
from ctxc.trn import Trn

SAMPLE = [
    {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Ana",
            "speaker_b": "Ben",
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [
                {"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a dog named Biscuit."},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "Nice! What breed?"},
                {"speaker": "Ana", "dia_id": "D1:3", "text": "Biscuit is a corgi.",
                 "blip_caption": "a corgi on grass"},
            ],
            "session_2_date_time": "2:00 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Ben", "dia_id": "D2:1", "text": "How was your week?"},
                {"speaker": "Ana", "dia_id": "D2:2", "text": "Busy, I started pottery classes."},
            ],
        },
        "qa": [
            {"question": "What breed is Ana's dog?", "answer": "corgi",
             "evidence": ["D1:3"], "category": 4},
            {"question": "What is the dog's name and breed?", "answer": "Biscuit, corgi",
             "evidence": ["D1:1", "D1:3"], "category": 1},
            {"question": "What hobby did Ana start?", "answer": "pottery",
             "evidence": ["D2:2"], "category": 4},
            {"question": "Unanswerable", "adversarial_answer": "x",
             "evidence": ["D1:2"], "category": 5},
            {"question": "Bad evidence", "answer": "x", "evidence": ["D9:9"], "category": 4},
        ],
    }
]


def _load(tmp_path):
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return load_locomo(p)


def test_load_locomo(tmp_path):
    (conv,) = _load(tmp_path)
    assert [t.id for t in conv.trns] == ["D1:1", "D1:2", "D1:3", "D2:1", "D2:2"]
    assert conv.trns[0].txt.startswith("Ana: ")
    assert "[photo: a corgi on grass]" in conv.trns[2].txt
    assert conv.speakers == {"Ana", "Ben"}
    assert len(conv.qs) == 3
    assert conv.dropped == 1


def test_recall_partial(tmp_path):
    (conv,) = _load(tmp_path)
    q = conv.qs[1]
    out = bm25(conv.trns, conv.trns[2].tok, "corgi breed")
    assert recall(out, q.ev) == 0.5


def test_bm25_mode_matches_baseline(tmp_path):
    (conv,) = _load(tmp_path)
    for q in conv.qs:
        for b in (6, 12, 30):
            a = {t.id for t in bm25(conv.trns, b, q.qry).kept}
            m = mly_select(conv.trns, b, q.qry, score_mode="bm25", use_dy=False,
                           collapse_parents=False)
            assert a == {t.id for t in m.kept}


def test_stem_and_smooth():
    assert stem("painting") == "paint"
    assert stem("hobbies") == "hobby"
    assert "the" not in stem_list("the paintings")
    trns = [Trn(id=f"D{s}:{k}", txt="x", idx=i)
            for i, (s, k) in enumerate([(1, 1), (1, 2), (2, 1)])]
    out = nb_smooth(trns, np.array([2.0, 0.0, 4.0]), 0.5)
    assert out.tolist() == [2.0, 1.0, 4.0]


def test_ctx_keeps_reply(tmp_path):
    (conv,) = _load(tmp_path)
    b = conv.trns[1].tok + conv.trns[2].tok
    out = mly_select(conv.trns, b, "What breed?", use_dy=False, score_mode="ctx")
    assert {"D1:2", "D1:3"} == {t.id for t in out.kept}


def test_run_rows(tmp_path):
    convs = _load(tmp_path)
    rows = run(convs, [10, 40], split="t", only=["bm25", "mly_bm25", "mly_ctx", "last_n"])
    cats = {r.cat for r in rows}
    assert {"all", "single-hop", "multi-hop"} <= cats
    for r in rows:
        assert 0.0 <= r.recall <= 1.0
        assert r.mean_tok <= r.tok_bud
    full = [r for r in rows if r.tok_bud == 40 and r.cat == "all" and r.method == "bm25"]
    assert full[0].recall == 1.0
