"""LoCoMo answer accuracy: compact, ask a local model, score against the gold answer.

Two scores per answer: SQuAD-style token F1 (the metric in the LoCoMo paper) and a
CORRECT/WRONG verdict from the same model acting as judge (the Mem0-style "J" score).
Every answer is cached, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

from ctxc.eval.locomo import CATS, LocoConv, default_path, load_locomo, methods
from ctxc.mly import CompactOut
from ctxc.ollama import CACHE, Emb, chat

# Same context size for answer and judge calls; a different one makes Ollama reload the model.
NUM_CTX = 3072

SYS = (
    "You answer questions about a long conversation between two people, using only the "
    "excerpts provided. Each block of excerpts starts with the date of that session. "
    "Reply with a short answer of a few words, no explanation. For questions about time, "
    "never answer with a relative word: convert it using the date of the session where it "
    "was said, e.g. 'yesterday' in a session dated 8 May 2023 means 7 May 2023, and "
    "'last week' means 'the week before 8 May 2023'. If the excerpts do not contain the "
    "answer, reply: unknown."
)

JUDGE = (
    "You grade answers to questions about a conversation. Given the question, the gold "
    "answer and a generated answer, reply CORRECT if the generated answer refers to the "
    "same thing as the gold answer, even if it is worded or formatted differently or "
    "includes extra detail; for dates, the same date or period counts. Otherwise reply "
    "WRONG. Reply with one word."
)


def fmt_ctx(conv: LocoConv, out: CompactOut) -> str:
    lines: list[str] = []
    cur = None
    for t in sorted(out.kept, key=lambda t: t.idx):
        s = t.id.split(":")[0]
        if s != cur:
            cur = s
            lines.append(f"\n[{conv.dates.get(s, '') or s}]")
        lines.append(t.txt)
    return "\n".join(lines).strip()


def _norm(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def oracle(conv: LocoConv, ev: set[str]) -> CompactOut:
    kept = [t for t in conv.trns if t.id in ev]
    return CompactOut(pin="", kept=kept, used_tok=sum(t.tok for t in kept), method="oracle")


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.d: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["key"]] = r

    def get(self, key: str) -> dict | None:
        return self.d.get(key)

    def put(self, r: dict) -> None:
        self.d[r["key"]] = r
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")


def answer(model: str, conv: LocoConv, qry: str, out: CompactOut) -> tuple[str, float]:
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Excerpts:\n{fmt_ctx(conv, out)}\n\nQuestion: {qry}\nShort answer:"},
    ]
    t0 = time.perf_counter()
    a = chat(model, msgs, num_ctx=NUM_CTX, num_predict=48)
    return a, (time.perf_counter() - t0) * 1000.0


def judge(model: str, qry: str, gold: str, pred: str) -> bool:
    msgs = [
        {"role": "system", "content": JUDGE},
        {"role": "user", "content": f"Question: {qry}\nGold answer: {gold}\nGenerated answer: {pred}"},
    ]
    return chat(model, msgs, num_ctx=NUM_CTX, num_predict=4).upper().startswith("CORRECT")


def run(
    convs: list[LocoConv],
    names: list[str],
    tok_bud: int,
    *,
    model: str,
    emb: Emb | None,
    cache: Cache,
    log_every: int = 50,
) -> list[dict]:
    rows: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    for conv in convs:
        ms = methods(conv, max_df=0.005, max_depth=1, emb=emb)
        for qi, q in enumerate(conv.qs):
            for name in names:
                key = f"{conv.id}|{qi}|{name}|{tok_bud}|{model}"
                r = cache.get(key)
                if r is None:
                    out = oracle(conv, q.ev) if name == "oracle" else ms[name](q.qry, tok_bud)
                    pred, ms_ans = answer(model, conv, q.qry, out)
                    r = {
                        "key": key, "conv": conv.id, "qi": qi, "method": name, "tok_bud": tok_bud,
                        "cat": CATS[q.cat], "qry": q.qry, "gold": q.ans, "pred": pred,
                        "f1": f1(pred, q.ans), "j": judge(model, q.qry, q.ans, pred),
                        "ctx_tok": out.used_tok, "ms": ms_ans,
                    }
                    cache.put(r)
                    done += 1
                    if done % log_every == 0:
                        el = time.perf_counter() - t0
                        print(f"  {done} new answers, {el / done:.2f}s each", flush=True)
                rows.append(r)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    acc: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for r in rows:
        for cat in (r["cat"], "all"):
            acc[(r["method"], r["tok_bud"], cat)].append(r)
    out = []
    for (m, b, c), xs in sorted(acc.items()):
        k = len(xs)
        out.append({
            "method": m, "tok_bud": b, "cat": c, "n": k,
            "f1": sum(x["f1"] for x in xs) / k,
            "j": sum(x["j"] for x in xs) / k,
            "ctx_tok": sum(x["ctx_tok"] for x in xs) / k,
            "ms": sum(x["ms"] for x in xs) / k,
        })
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoCoMo answer accuracy with a local Ollama model")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--methods", default="oracle,mly_ctx,mly_hybrid,hybrid,bm25,last_n")
    p.add_argument("--out", default="results/locomo_qa.csv")
    args = p.parse_args(argv)

    convs = load_locomo(default_path())
    convs = convs[:5] if args.split == "dev" else convs[5:]
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    emb = Emb() if any(n in ("dense", "dense_nb", "hybrid", "mly_hybrid") for n in names) else None
    cache = Cache(CACHE / f"qa_{args.model.replace(':', '_')}.jsonl")

    rows = run(convs, names, args.budget, model=args.model, emb=emb, cache=cache)
    if emb is not None:
        emb.save()
    summ = summarize(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "tok_bud", "cat", "n", "f1", "j", "ctx_tok", "ms"])
        for s in summ:
            w.writerow([args.split, s["method"], s["tok_bud"], s["cat"], s["n"], f"{s['f1']:.4f}",
                        f"{s['j']:.4f}", f"{s['ctx_tok']:.1f}", f"{s['ms']:.1f}"])

    print(f"{'method':12} {'F1':>6} {'J':>6} {'tok':>6}")
    for s in summ:
        if s["cat"] == "all":
            print(f"{s['method']:12} {s['f1']*100:6.1f} {s['j']*100:6.1f} {s['ctx_tok']:6.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
