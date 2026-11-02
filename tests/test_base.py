from ctxc.base import last_n, sim_evict
from ctxc.bench import CONSTRAINT, golden_convo
from ctxc.score import bow_emb
from ctxc.trn import Trn


def test_last_n_drops_early_constraint():
    trns, pin, _qry = golden_convo()
    # tight budget: only newest turns fit
    out = last_n(trns, tok_bud=40, pin=pin)
    ctx = out.ctx.lower()
    assert CONSTRAINT.lower() not in ctx
    assert all(t.idx >= 8 for t in out.kept) or len(out.kept) < len(trns)


def test_sim_evict_drops_near_paraphrase():
    """Near-duplicate of a needed fact is evicted by similarity threshold."""
    pin = "sys"
    needed = Trn(
        id="need",
        txt="budget ceiling is forty two units maximum",
        idx=0,
        pin_fact=True,
        emb=bow_emb("budget ceiling is forty two units maximum"),
    )
    para = Trn(
        id="para",
        txt="budget ceiling is forty two units maximum today",
        idx=1,
        emb=bow_emb("budget ceiling is forty two units maximum today"),
    )
    late = Trn(id="late", txt="what is the ceiling number please", idx=2)
    # generous budget so eviction is from similarity, not space
    out = sim_evict([needed, para, late], tok_bud=200, qry="ceiling", pin=pin, thr=0.85)
    ids = {t.id for t in out.kept}
    # newest first: late kept, then para; needed should be skipped as similar to para
    assert "late" in ids
    assert "para" in ids
    assert "need" not in ids
