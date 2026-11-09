"""Generate multi-dialogue JSONL (dev/test). Run: python scripts/gen_bench_data.py"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

PIN = "You are a careful assistant. Obey all user constraints."

FACTS = [
    (
        "Important: use a hard limit of 12 tokens per tool call. Never exceed it.",
        "what was the hard cap for tool calls again?",
        r"(?i)\b12\b",
        ["12", "limit", "token"],
    ),
    (
        "Rule: max retry count is 3 with exponential backoff.",
        "how many retries did we agree on?",
        r"(?i)\b3\b",
        ["3", "retry"],
    ),
    (
        "Constraint: never call the Delete API in production mode.",
        "are we allowed to call Delete in production?",
        r"(?i)never|not allowed|no\b|forbidden",
        ["delete", "production"],
    ),
    (
        "Budget ceiling is 42 units maximum for this job.",
        "what is the ceiling number please?",
        r"(?i)\b42\b",
        ["42", "ceiling"],
    ),
    (
        "Timezone for all reports must be UTC+3 only.",
        "which timezone should the reports use?",
        r"(?i)utc\+?3|utc\+3",
        ["utc", "3"],
    ),
    (
        "Password rotation interval is 90 days fixed.",
        "how often do we rotate passwords?",
        r"(?i)\b90\b",
        ["90", "day"],
    ),
]

FILLER = [
    "Also log every tool name to stdout.",
    "We will call Search, Fetch, and Summarize tools.",
    "Search should look over docs first.",
    "Fetch pulls the top three urls.",
    "Summarize must be short.",
    "Please add retries on network errors.",
    "Include error handling and logging.",
    "Sketch the full loop with tools.",
    "Cache intermediate results in memory.",
    "Prefer JSON over XML for payloads.",
    "Add metrics for latency and errors.",
    "Document the public interface clearly.",
    "Keep functions under fifty lines if possible.",
    "Write unit tests for edge cases.",
    "Avoid global mutable state.",
]


def _mk_turns(
    fact: str,
    n_fill: int,
    probe: str,
    *,
    paraphrase_probe: bool,
    rng: random.Random,
) -> list[dict]:
    turns: list[dict] = [{"txt": "Hi, I need help with this workflow.", "role": "user"}]
    turns.append({"txt": fact, "role": "user", "pin_fact": True})
    fill = FILLER[:]
    rng.shuffle(fill)
    for line in fill[:n_fill]:
        turns.append({"txt": line, "role": "user"})
    if paraphrase_probe:
        # no explicit "as I said" — harder for back-ref patterns
        turns.append({"txt": probe, "role": "user"})
    else:
        turns.append(
            {
                "txt": f"As I said earlier about that limit — {probe}",
                "role": "user",
            }
        )
    return turns


def _ex(
    eid: str,
    fact_i: int,
    n_fill: int,
    *,
    paraphrase: bool,
    rng: random.Random,
) -> dict:
    fact, qry, ans_re, frags = FACTS[fact_i % len(FACTS)]
    return {
        "id": eid,
        "pin": PIN,
        "turns": _mk_turns(fact, n_fill, qry, paraphrase_probe=paraphrase, rng=rng),
        "qry": qry,
        "constraint": fact,
        "must_frags": frags,
        "answer_re": ans_re,
        "split_meta": {
            "n_fill": n_fill,
            "paraphrase_probe": paraphrase,
            "fact_i": fact_i % len(FACTS),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    # 12 dev + 12 test; vary length and probe style
    dev: list[dict] = []
    test: list[dict] = []
    for i in range(24):
        n_fill = 6 + (i % 5) * 3  # 6..18 fillers
        paraphrase = i % 2 == 1
        ex = _ex(f"d{i:03d}", i, n_fill, paraphrase=paraphrase, rng=rng)
        (test if i >= 12 else dev).append(ex)

    for name, rows in (("bench_dev.jsonl", dev), ("bench_test.jsonl", test)):
        path = OUT / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} n={len(rows)}")


if __name__ == "__main__":
    main()
