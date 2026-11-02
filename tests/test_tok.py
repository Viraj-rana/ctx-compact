from ctxc.tok import tok_cnt


def test_tok_cnt_basic():
    assert tok_cnt("a b c") == 3
    assert tok_cnt("") == 0
    assert tok_cnt("  ") == 0
