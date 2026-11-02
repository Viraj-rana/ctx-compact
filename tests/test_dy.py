from ctxc.dy import build_dy, extract_ent, parent_closure
from ctxc.trn import Trn


def test_extract_ent_quoted_and_cap():
    ent = extract_ent('Use "AlphaBot" with Search tool')
    assert "alphabot" in ent
    assert "search" in ent


def test_entity_reuse_edge():
    trns = [
        Trn(id="a", txt="Introducing WidgetX today.", idx=0, ent={"widgetx"}),
        Trn(id="b", txt="Configure WidgetX next.", idx=1, ent={"widgetx"}),
    ]
    dy = build_dy(trns)
    assert "a" in dy["b"]
    assert dy["a"] == set()


def test_back_ref_to_pin_fact():
    trns = [
        Trn(id="a", txt="hard limit of 12 tokens", idx=0, pin_fact=True, ent={"limit"}),
        Trn(id="b", txt="As I said earlier about that limit — remind me?", idx=1, ent={"limit"}),
    ]
    dy = build_dy(trns)
    assert "a" in dy["b"]


def test_parent_closure_transitive():
    dy = {"c": {"b"}, "b": {"a"}, "a": set()}
    assert parent_closure(dy, {"c"}) == {"a", "b", "c"}
