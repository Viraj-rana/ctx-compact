"""Turn record used across the selector stack."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ctxc.tok import tok_cnt


@dataclass
class Trn:
    """One conversation turn."""

    id: str
    txt: str
    idx: int = 0
    ent: set[str] = field(default_factory=set)
    emb: np.ndarray | None = None
    pin_fact: bool = False
    tok: int = 0

    def __post_init__(self) -> None:
        if self.tok <= 0:
            self.tok = tok_cnt(self.txt)
        if self.ent and not isinstance(self.ent, set):
            self.ent = set(self.ent)
