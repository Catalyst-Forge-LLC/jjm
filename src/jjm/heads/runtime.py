"""Encoder plus cache: what fitting and serving both call to turn states into features."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from jjm.compile import CompiledCriterion
from jjm.heads.cache import FeatureCache, cache_key
from jjm.heads.encoders import EncodeStats, Encoder, encoder_digest, encoder_input


@dataclass
class FeatureRun:
    features: np.ndarray
    cache_hits: int
    encoded: int
    encode: Optional[EncodeStats]
    elapsed_ms: float


class Featurizer:
    """Looks up the in-process memo, then the disk cache, then runs the encoder on what is left.

    Both caches are exact-input reuse: the key is the encoder identity digest plus the exact text.
    """

    MEMO_LIMIT = 50_000

    def __init__(self, encoder: Encoder, cache_dir: Optional[Path] = None):
        self.encoder = encoder
        self.digest = encoder_digest(encoder)
        self.cache = FeatureCache(Path(cache_dir)) if cache_dir else None
        self.memo: dict[str, np.ndarray] = {}

    def inputs(self, state_texts: Sequence[str], compiled: CompiledCriterion) -> list[tuple[str, str]]:
        template = self.encoder.identity()["prefix_template"]
        return [encoder_input(t, compiled, template) for t in state_texts]

    def features(self, items: Sequence[tuple[str, str]]) -> FeatureRun:
        t0 = time.perf_counter()
        vecs: list[Any] = [None] * len(items)
        keys = [cache_key(self.digest, p, b) for p, b in items]
        vecs = [self.memo.get(k) for k in keys]
        missing = [i for i, v in enumerate(vecs) if v is None]
        if self.cache is not None and missing:
            for i, v in zip(missing, self.cache.get_many([keys[i] for i in missing])):
                vecs[i] = v
        todo = [i for i, v in enumerate(vecs) if v is None]
        stats = None
        if todo:
            # identical inputs inside one call are encoded once
            unique: dict[str, int] = {}
            for i in todo:
                unique.setdefault(keys[i], i)
            order = list(unique.values())
            fresh, stats = self.encoder.encode([items[i] for i in order])
            by_key = {keys[i]: fresh[j] for j, i in enumerate(order)}
            for i in todo:
                vecs[i] = by_key[keys[i]]
            if self.cache is not None:
                self.cache.put_many([keys[i] for i in order], fresh)
        if len(self.memo) + len(keys) > self.MEMO_LIMIT:
            self.memo.clear()
        for k, v in zip(keys, vecs):
            self.memo[k] = v
        return FeatureRun(np.stack(vecs), len(items) - len(todo), len(todo), stats,
                          (time.perf_counter() - t0) * 1000.0)
