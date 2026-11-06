# ctxc

Dependency-aware context compaction for long conversations under a token budget.

When a chat history no longer fits, something has to go. `ctxc` ranks turns against the current question, lets each turn inherit part of the relevance of the turn next to it (a reply depends on what it answers), and keeps the turns a selected turn depends on (`dy`: a later turn depends on the turn that introduced an entity it mentions), folding them into one-line pins when they do not fit. On the held-out half of LoCoMo the best configuration (BM25 + dense hybrid with neighbour smoothing) keeps 78% of the answer-bearing turns at a 500-word budget, against 76% for the same hybrid ranker alone, 55% for BM25 and 2% for keeping the last N messages. On LoCoMo the gain comes from neighbour smoothing; the entity dependency graph only helps on the synthetic suite.

[![ci](https://github.com/Viraj-rana/ctx-compact/actions/workflows/ci.yml/badge.svg)](https://github.com/Viraj-rana/ctx-compact/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viraj-rana/ctx-compact/blob/main/notebooks/demo.ipynb)

```bash
pip install -e ".[dev]"
python -m pytest -v

python scripts/get_locomo.py          # LoCoMo, CC BY-NC 4.0, not vendored
python -m ctxc.eval.locomo --out results/locomo_sweep.csv          # lexical methods only
python -m ctxc.eval.locomo --dense --out results/locomo_sweep.csv  # + dense/hybrid (needs Ollama, nomic-embed-text)
python scripts/plot_locomo.py
python scripts/bootstrap_locomo.py                                  # 95% CIs for key differences (needs Ollama)
python -m ctxc.eval.qa --budget 1000                                # answer accuracy (needs Ollama, qwen2.5:7b)

python -m ctxc.eval.run --split test --budgets 40,55,80,120
python scripts/export_and_plot.py
```

## Real conversations: LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) has 10 long two-person conversations (369–689 turns, 9k–19k words) and ~2,000 questions, each labelled with the turns that hold its answer. A compactor is scored by **evidence recall**: the share of those turns still present after compacting the conversation to the budget. No model in the loop, so the number is about what the context keeps, not how well some LLM reads it.

Conversations 1–5 are the dev split, used to pick every setting (neighbour weight, BM25 share of the hybrid score, entity document-frequency cap, parent depth; best mean recall over budgets). The figure and table are conversations 6–10, scored afterwards with those settings frozen. Adversarial questions (category 5) are excluded.

![LoCoMo held-out results](results/locomo.png)

Held-out, 776 questions ([`results/locomo_sweep.csv`](results/locomo_sweep.csv)):

| method | 500 words | 1000 | 2000 | 4000 | multi-hop @ 4000 |
|--------|-----------|------|------|------|------------------|
| **mly_hybrid** (BM25 + dense + neighbour + repair\*) | **77.9** | **84.0** | **88.7** | **94.1** | **85.3** |
| hybrid_nb (same, repair off) | 77.9 | 84.0 | 88.7 | 94.1 | 85.3 |
| hybrid (0.3 × BM25 + 0.7 × dense, no neighbour) | 75.9 | 82.8 | 88.6 | 93.2 | 85.3 |
| dense_nb (dense + neighbour) | 71.7 | 79.7 | 86.9 | 92.1 | 82.5 |
| dense (nomic-embed-text) | 71.4 | 78.6 | 85.3 | 91.3 | 82.6 |
| mly_ctx (stemmed BM25 + neighbour + repair\*) | 70.4 | 77.1 | 83.0 | 87.2 | 69.1 |
| ctx_no_dy (same, repair off) | 70.4 | 77.1 | 83.0 | 87.2 | 69.1 |
| bm25_stem (same tokenizer, no neighbour) | 63.4 | 70.0 | 76.3 | 82.3 | 62.9 |
| bm25 | 55.0 | 62.2 | 70.3 | 77.4 | 57.0 |
| mly_bm25 (plain BM25 + repair) | 54.1 | 61.6 | 70.8 | 78.0 | 59.4 |
| mly_bm25_wide (2-level repair) | 50.1 | 59.3 | 68.6 | 77.5 | 61.6 |
| mly (original overlap scorer) | 5.6 | 9.8 | 19.3 | 37.5 | — |
| last_n | 2.0 | 5.2 | 10.9 | 24.7 | — |
| sum_drop | 1.6 | 5.0 | 10.6 | 24.2 | — |

\* At the entity document-frequency cap these two use (`max_df=0.002`, picked on dev), the dependency graph has no edges in any of the 10 conversations: with 369–689 turns an entity may appear in at most one turn, so no turn can link to another. Repair is a no-op here, which is why both rows match their repair-off ablations exactly.

What this says:

- **Hybrid ranking plus neighbour smoothing is best overall at every budget,** but by a small margin over the hybrid ranker alone: +2.1 points at 500 words, +1.2 at 1,000, +0.1 at 2,000, +0.9 at 4,000. Dev showed the same ordering (mean over budgets 88.2 vs 86.6 vs 84.4 for dense alone). All of that gain is neighbour smoothing (see the footnote), and only the 500-word gain is clearly real: its 95% paired-bootstrap interval is +0.3 to +3.9, while at 1,000 to 4,000 words the intervals include zero ([`results/locomo_bootstrap.csv`](results/locomo_bootstrap.csv)).
- **Dense retrieval beats every lexical method.** Plain `dense` is above `mly_ctx` at every budget (+1.0 at 500, +4.1 at 4,000) and far above it on multi-hop (82.6 vs 69.1 at 4,000). Most of the gap to BM25 is the ranker, not the compaction policy.
- **Adjacency is the dependency that pays off, mainly for lexical ranking.** In LoCoMo the answer is often the reply to the turn that matches the question ("What breed?" → "Biscuit is a corgi"). Neighbour smoothing adds +4.9 to +7.2 points on stemmed BM25, +0.1 to +2.1 on hybrid, and +0.3 to +1.6 on dense: the better the ranker, the less a neighbour adds.
- **Entity repair does not help on LoCoMo.** Wherever neighbour smoothing is used, dev tuning picked a cap so strict that the dependency graph is empty, i.e. it chose to turn repair off. Where repair is active (plain BM25, `max_df` 0.005 and 0.02, 0 to 543 edges per conversation) it is a trade-off: within a point overall, +4.6 on multi-hop at 4,000 words with the 2-level setting, about −5 at 500. Entity reuse in casual chat ("work", "family", "trip") is too common to mark a dependency.
- **Where `mly_hybrid` does not win:** its gain over `hybrid` comes from single-hop questions (+6.2 at 500 words, 423 questions). It loses 7 of the 16 type × budget cells: multi-hop at 500, 1,000 and 2,000 words (48.2 vs 52.5, 60.8 vs 64.2, 74.4 vs 74.8; 140 questions), open-domain at 500 and 1,000 (45.8 vs 50.1, 57.9 vs 60.5; 48 questions) and temporal at 500 and 2,000 (78.7 vs 79.9, 87.1 vs 89.4). Spending budget on a neighbour costs recall when the evidence is spread over several distant turns.
- **Truncation policies fail on real chat.** Anything that keeps recent or keyword-flagged turns finds under 25% of the evidence even at 4,000 words. The original overlap-plus-recency scorer, tuned on the synthetic suite, does not transfer either.

### Answer accuracy

Each method's 1,000-word context goes to Qwen2.5-7B (Ollama, temperature 0), which answers in a few words; the same model judges each answer CORRECT/WRONG against the gold answer (J), next to SQuAD-style token F1. `oracle` sees only the gold evidence turns. Held-out, 776 questions ([`results/locomo_qa.csv`](results/locomo_qa.csv)):

| method | recall | F1 | J | J multi-hop | J temporal |
|--------|--------|----|---|-------------|------------|
| oracle (gold turns, 63 words) | 100.0 | 57.0 | 70.9 | 75.0 | 49.1 |
| **mly_hybrid** | 84.0 | **52.5** | **62.4** | **55.0** | **35.8** |
| hybrid | 82.8 | 52.0 | **62.4** | **55.0** | **35.8** |
| mly_ctx | 77.1 | 50.3 | 59.0 | 49.3 | 31.5 |
| bm25 | 62.2 | 41.9 | 50.9 | 40.7 | 27.3 |
| last_n | 5.2 | 5.3 | 6.4 | 10.0 | 4.2 |

The recall ordering carries over to answers: hybrid ranking is +11.5 J over BM25 (95% paired bootstrap +8.1 to +14.8). Neighbour smoothing on top of the hybrid changes nothing at this budget (+0.6 F1, −1.2 to +2.3; identical J, 53 questions won each way). The best compactor still trails the oracle by 8.5 J (+5.0 to +12.0). The judge is lenient on list answers ("rock climbing" passes for "rock climbing, fishing, camping").

## Controlled suite: early constraint, late question

24 template dialogues (12 dev, 12 held out) where a rule stated early ("hard limit of 12 tokens per tool call") is asked about many turns later. Entities are extracted automatically; nothing is hand-labelled on the eval path. Scored by whether the constraint survives compaction (regex on an extractive stub answer).

![synthetic held-out sweep](results/acc_vs_budget.png)

At 55 tokens on the held-out set ([`results/bench_test_sweep.csv`](results/bench_test_sweep.csv)): `mly` 1.00, `mly_no_dy` 0.33, `last_n` 0.17, `sim_evict` 0.17. `sum_drop` also reaches 1.00 here because its keyword filter happens to match these rules, and it fails on LoCoMo for the same reason. This suite is built around the failure the method targets, so it shows the mechanism works, not that it generalises; LoCoMo is the test of that.

Dependency graph for one held-out dialogue (red: constraint-like anchor, teal: the late question):

![dy example](results/dy_example.png)

## Method

1. Charge the pinned system text against the budget first.
2. Build `dy`: an edge from the turn that first introduced an entity to each later turn that reuses it, skipping speaker names and entities in more than `max_df` of turns. On the synthetic suite, back-reference cues ("as I said", "that limit") also link to earlier constraint-like turns.
3. Score turns. On LoCoMo, `mly_hybrid`: 0.3 × min-max BM25 over stemmed, stopword-free tokens plus 0.7 × min-max cosine similarity from `nomic-embed-text`, then plus 0.5 × the best adjacent score in the same session; `mly_ctx` is the same without the dense term. On the synthetic suite: query overlap + recency + pin bonus − redundancy, per token.
4. Pick the best remaining turn, add its parents up to `max_depth` levels; if they do not fit whole, fold them into one-line pins.
5. Repeat until the budget is spent.

## Limits

- Answer accuracy is measured with one 7B model at one budget (1,000 words), and the same model is the judge.
- The gain of neighbour smoothing over `hybrid` is small (0.1 to 2.1 points) and comes from single-hop questions.
- The dependency graph, the part of the method this repo is named after, is only shown to matter on the synthetic suite, which was built around the failure it targets. On LoCoMo the tuned settings switch it off.
- The dense methods need a local embedding model; the lexical ones run with no model at all.
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

It's evaluated on 24 synthetic dialogues (12 tuned on, 12 held out) at budgets from 40 to 120 tokens. At 55 tokens on the held-out set, ctxc kept the constraint in 100% of dialogues, against 17% for keeping the last N messages and 33% with dependency tracking switched off. An extractive summary baseline also scored 100% on this suite. On real conversations (LoCoMo, held out) the hybrid configuration keeps 78% of the answer turns at 500 words, against 76% for hybrid retrieval alone and 55% for BM25. Everything runs offline with no API key (the dense methods use a local Ollama embedding model), and CI runs the tests and the synthetic evaluation on every push.
