# ctxc

Dependency-aware context compaction for long conversations under a token budget.

When a chat history no longer fits, something has to go. `ctxc` ranks turns against the current question, lets each turn inherit part of the relevance of the turn next to it (a reply depends on what it answers), and keeps the turns a selected turn depends on (`dy`: a later turn depends on the turn that introduced an entity it mentions), folding them into one-line pins when they do not fit. On the held-out half of LoCoMo it keeps 70% of the answer-bearing turns at a 500-word budget, against 55% for BM25 and 2% for keeping the last N messages.

[![ci](https://github.com/Viraj-rana/ctx-compact/actions/workflows/ci.yml/badge.svg)](https://github.com/Viraj-rana/ctx-compact/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viraj-rana/ctx-compact/blob/main/notebooks/demo.ipynb)

```bash
pip install -e ".[dev]"
python -m pytest -v

python scripts/get_locomo.py          # LoCoMo, CC BY-NC 4.0, not vendored
python -m ctxc.eval.locomo --out results/locomo_sweep.csv
python scripts/plot_locomo.py

python -m ctxc.eval.run --split test --budgets 40,55,80,120
python scripts/export_and_plot.py
```

## Real conversations: LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) has 10 long two-person conversations (369–689 turns, 9k–19k words) and ~2,000 questions, each labelled with the turns that hold its answer. A compactor is scored by **evidence recall**: the share of those turns still present after compacting the conversation to the budget. No model in the loop, so the number is about what the context keeps, not how well some LLM reads it.

Conversations 1–5 are the dev split, used to pick every setting (neighbour weight, entity document-frequency cap, parent depth; best mean recall over budgets). The figure and table are conversations 6–10, scored afterwards with those settings frozen. Adversarial questions (category 5) are excluded.

![LoCoMo held-out results](results/locomo.png)

Held-out, 776 questions ([`results/locomo_sweep.csv`](results/locomo_sweep.csv)):

| method | 500 words | 1000 | 2000 | 4000 | multi-hop @ 4000 |
|--------|-----------|------|------|------|------------------|
| **mly_ctx** (stemmed BM25 + neighbour + repair) | **70.4** | **77.1** | **83.0** | **87.2** | **69.1** |
| bm25_stem (same tokenizer, no neighbour) | 63.4 | 70.0 | 76.3 | 82.3 | 62.9 |
| bm25 | 55.0 | 62.2 | 70.3 | 77.4 | 57.0 |
| mly_bm25 (plain BM25 + repair) | 54.1 | 61.6 | 70.8 | 78.0 | 59.4 |
| mly_bm25_wide (2-level repair) | 50.1 | 59.3 | 68.6 | 77.5 | 61.6 |
| mly (original overlap scorer) | 5.6 | 9.8 | 19.3 | 37.5 | — |
| last_n | 2.0 | 5.2 | 10.9 | 24.7 | — |
| sum_drop | 1.6 | 5.0 | 10.6 | 24.2 | — |

What this says:

- **`mly_ctx` is best at every budget.** +15.4 points over BM25 at 500 words and +9.8 at 4,000; +12.1 on multi-hop at 4,000. Against `bm25_stem`, which shares its tokenizer, it is still +4.9 to +7.1, so the gain is not just stemming. Dev showed the same ordering (mean over budgets 81.5 vs 74.7 vs 68.9).
- **Adjacency is the dependency that pays off.** In LoCoMo the answer is often the reply to the turn that matches the question ("What breed?" → "Biscuit is a corgi"). Adding half of the best same-session neighbour's score to each turn captures that link directly.
- **Entity repair adds nothing on top.** With neighbour smoothing in place, `mly_ctx` and the same scorer with repair off give identical recall on the held-out split. On plain BM25 repair is a trade-off: within a point overall, +4.6 on multi-hop at 4,000 words with the 2-level setting, about −5 at 500.
- **Where it does not win:** three of the 16 type × budget cells, all narrowly against `bm25_stem`: open-domain at 500 and 1,000 words (32.0 vs 32.7, 40.5 vs 41.7; 48 questions) and temporal at 500 (67.1 vs 67.9). Spending budget on a neighbour costs a little when the answer is a single isolated turn.
- **Truncation policies fail on real chat.** Anything that keeps recent or keyword-flagged turns finds under 25% of the evidence even at 4,000 words. The original overlap-plus-recency scorer, tuned on the synthetic suite, does not transfer either.

## Controlled suite: early constraint, late question

24 template dialogues (12 dev, 12 held out) where a rule stated early ("hard limit of 12 tokens per tool call") is asked about many turns later. Entities are extracted automatically; nothing is hand-labelled on the eval path. Scored by whether the constraint survives compaction (regex on an extractive stub answer).

![synthetic held-out sweep](results/acc_vs_budget.png)

At 55 tokens on the held-out set ([`results/bench_test_sweep.csv`](results/bench_test_sweep.csv)): `mly` 1.00, `mly_no_dy` 0.33, `last_n` 0.17, `sim_evict` 0.17. `sum_drop` also reaches 1.00 here because its keyword filter happens to match these rules, and it fails on LoCoMo for the same reason. This suite is built around the failure the method targets, so it shows the mechanism works, not that it generalises; LoCoMo is the test of that.

Dependency graph for one held-out dialogue (red: constraint-like anchor, teal: the late question):

![dy example](results/dy_example.png)

## Method

1. Charge the pinned system text against the budget first.
2. Build `dy`: an edge from the turn that first introduced an entity to each later turn that reuses it, skipping speaker names and entities in more than `max_df` of turns. On the synthetic suite, back-reference cues ("as I said", "that limit") also link to earlier constraint-like turns.
3. Score turns. On LoCoMo (`score_mode="ctx"`): BM25 over stemmed, stopword-free tokens, plus 0.5 × the best adjacent score in the same session. On the synthetic suite: query overlap + recency + pin bonus − redundancy, per token.
4. Pick the best remaining turn, add its parents up to `max_depth` levels; if they do not fit whole, fold them into one-line pins.
5. Repeat until the budget is spent.

## Limits

- Evidence recall, not answer accuracy. The next step is feeding the compacted context to a fixed open model and scoring its answers against the LoCoMo gold answers.
- Budgets are in words; a real deployment should count with the serving model's tokenizer.
- Prefill time in the synthetic sweep is a linear estimate, not measured on a server.

## Layout

```
data/                  synthetic bench_dev / bench_test (LoCoMo downloads to data/locomo/)
results/               csv + png for both evals
notebooks/demo.ipynb   runs the tests and the synthetic sweep
scripts/               data generation, download, plotting
src/ctxc/              dy, mly, baselines, score
src/ctxc/eval/         synthetic and LoCoMo evals
tests/
```

## About

ctxc compacts long chat histories to fit a fixed token budget without losing earlier constraints that later turns rely on. Keeping only the last N messages drops an early rule like "hard limit of 12 tokens per tool call" once the conversation gets long, so a later question about that limit can no longer be answered.

The selector builds a dependency graph between turns from reused entities and back-references ("as I said", "that limit"). It always keeps the system prompt, then fills the remaining budget by value per token. When it keeps a turn, it also keeps the earlier turns that turn depends on, or shortens them to a one-line pin.

It's evaluated on 24 synthetic dialogues (12 tuned on, 12 held out) at budgets from 40 to 120 tokens. At 55 tokens on the held-out set, ctxc kept the constraint in 100% of dialogues, against 17% for keeping the last N messages and 33% with dependency tracking switched off. An extractive summary baseline also scored 100% on this suite. On real conversations (LoCoMo, held out) it keeps 70% of the answer turns at 500 words, against 55% for BM25. Everything runs offline with no API key, and CI runs the tests and the evaluation on every push.
