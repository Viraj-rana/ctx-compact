from ctxc.dy import build_dy
from ctxc.mly import mly_select
from ctxc.tok import tok_cnt
from ctxc.trn import Trn


def test_pin_never_dropped():
    pin = "SYS RULES ALWAYS"
    trns = [Trn(id="u1", txt="hello there friend", idx=0)]
    out = mly_select(trns, tok_bud=3, qry="hello", pin=pin)
    assert pin in out.ctx
    assert out.used_tok >= tok_cnt(pin)


def test_budget_not_exceeded_when_pin_fits():
    pin = "sys"
    trns = [
        Trn(id="a", txt="alpha beta gamma delta", idx=0),
        Trn(id="b", txt="epsilon zeta eta theta", idx=1),
    ]
    bud = 6
    out = mly_select(trns, tok_bud=bud, qry="alpha", pin=pin)
    assert out.used_tok <= bud or tok_cnt(pin) > bud


def test_keeping_child_forces_parent():
    trns = [
        Trn(
            id="p",
            txt="Important: hard limit of 12 tokens per tool call forever.",
            idx=0,
            pin_fact=True,
            ent={"limit", "tokens"},
        ),
        Trn(id="noise", txt="Talk about weather sunny clouds rain.", idx=1),
        Trn(
            id="c",
            txt="As I said earlier about that limit — what was the hard cap again?",
            idx=2,
            ent={"limit"},
        ),
    ]
    dy = build_dy(trns)
    assert "p" in dy["c"]
    # budget fits child+parent+pin
    pin = "sys"
    need = tok_cnt(pin) + trns[0].tok + trns[2].tok
    out = mly_select(trns, tok_bud=need + 2, qry="what was the hard cap again?", pin=pin, dy=dy)
    kept_ids = {t.id for t in out.kept}
    assert "c" in kept_ids
    # parent whole or collapsed pin
    assert "p" in kept_ids or any("12" in p and "limit" in p.lower() for p in out.pins_extra) or (
        "12" in out.ctx and "limit" in out.ctx.lower()
    )
