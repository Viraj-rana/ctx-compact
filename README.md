# ctxc

Dependency-aware conversation context compaction under a token budget.

`dy` builds turn dependencies (entity reuse + back-refs). `mly` fills the remaining budget by score-per-token and keeps parents of selected turns. Offline held-out eval — no API key.

[![ci](https://github.com/REPLACE_ME/ctx-compact/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_ME/ctx-compact/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/REPLACE_ME/ctx-compact/blob/main/notebooks/demo.ipynb)

```bash
pip install -e ".[dev]"
python -m pytest -v
python -m ctxc.eval.run --split test --budgets 40,55,80,120
python scripts/export_and_plot.py
```

Live walkthrough (pytest + sweep + plots): [`notebooks/demo.ipynb`](notebooks/demo.ipynb) — use the Colab badge above after the repo is on GitHub (set `REPO` in the first cell if needed).

## Results (held-out `bench_test.jsonl`, n=12)

Generated from `scripts/export_and_plot.py` → [`results/bench_test_sweep.csv`](results/bench_test_sweep.csv).

![held-out sweep](results/acc_vs_budget.png)

**Accuracy @ budget 55** (same CSV):

| method | acc | mean tok |
|--------|-----|----------|
| mly | 1.00 | 54.6 |
| mly_no_collapse | 1.00 | 54.6 |
| mly_ovlp_score | 1.00 | 54.5 |
| sum_drop | 1.00 | 49.1 |
| ovlp_rec | 0.33 | 54.1 |
| mly_no_dy | 0.33 | 53.8 |
| last_n | 0.17 | 52.8 |
| sim_evict | 0.17 | 52.8 |

## Dependency graph (real example `d012`)

![dy example](results/dy_example.png)

## Method

1. Charge pinned system text first.
2. Build `dy` from entities + probe/back-ref cues into earlier anchors.
3. Greedy fill; keep parent closure (or collapse parents to short pins).
4. Judge with extractive stub + `answer_re` / `must_frags`.

## Layout

```
data/                  bench_dev / bench_test
results/               csv + png
notebooks/demo.ipynb
scripts/
src/ctxc/
tests/
.github/workflows/ci.yml
```
ctxc compacts long chat histories to fit a fixed token budget without losing earlier constraints that later turns rely on. Keeping only the last N messages drops an early rule like "hard limit of 12 tokens per tool call" once the conversation gets long, so a later question about that limit can no longer be answered.

The selector builds a dependency graph between turns from reused entities and back-references ("as I said", "that limit"). It always keeps the system prompt, then fills the remaining budget by value per token. When it keeps a turn, it also keeps the earlier turns that turn depends on, or shortens them to a one-line pin.

It's evaluated on 24 synthetic dialogues (12 tuned on, 12 held out) at budgets from 40 to 120 tokens. At 55 tokens on the held-out set, ctxc kept the constraint in 100% of dialogues, against 17% for keeping the last N messages and 33% with dependency tracking switched off. An extractive summary baseline also scored 100% on this suite. Everything runs offline with no API key, and CI runs the tests and the evaluation on every push.
