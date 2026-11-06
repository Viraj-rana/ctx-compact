"""Minimal Ollama client (stdlib only) with an on-disk embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def _post(path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(
    model: str, msgs: list[dict], *, num_ctx: int = 4096, num_predict: int = 64, tries: int = 3
) -> str:
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    # Ollama unloads the model after a CUDA OOM and serves the next request fine.
    for k in range(tries):
        try:
            return _post("/api/chat", body)["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code != 500 or k == tries - 1:
                raise
            time.sleep(5)
    raise AssertionError("unreachable")


class Emb:
    """Embeds texts through Ollama, caching vectors by md5 of model + text."""

    def __init__(self, model: str = "nomic-embed-text", batch: int = 64):
        self.model = model
        self.batch = batch
        self.path = CACHE / f"emb_{model.replace(':', '_').replace('/', '_')}.npz"
        self.vecs: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path)
            self.vecs = dict(zip(z["k"].tolist(), z["v"]))
        self.dirty = False

    def _key(self, txt: str) -> str:
        return hashlib.md5(f"{self.model}\x00{txt}".encode()).hexdigest()

    def __call__(self, txts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in txts]
        todo = [(k, t) for k, t in dict(zip(keys, txts)).items() if k not in self.vecs]
        for i in range(0, len(todo), self.batch):
            part = todo[i : i + self.batch]
            got = _post("/api/embed", {"model": self.model, "input": [t for _, t in part]})
            for (k, _), v in zip(part, got["embeddings"]):
                self.vecs[k] = np.asarray(v, dtype=np.float32)
            self.dirty = True
        m = np.stack([self.vecs[k] for k in keys]).astype(np.float64)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return np.divide(m, n, out=np.zeros_like(m), where=n > 0)

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ks = list(self.vecs)
        np.savez(self.path, k=np.array(ks), v=np.stack([self.vecs[k] for k in ks]))
        self.dirty = False
