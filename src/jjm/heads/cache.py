"""Feature cache: an identical input to an identical encoder is read once.

Keys are SHA-256 over the encoder identity digest and the exact (prefix, body) text, so a model,
template, depth, pooling, or truncation change can never return an old vector. One SQLite file
per cache directory; vectors are stored as float32 bytes.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SCHEMA = "CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"


def cache_key(encoder_digest: str, prefix: str, body: str) -> str:
    h = hashlib.sha256()
    for part in (encoder_digest, prefix, body):
        data = part.encode("utf-8")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


class FeatureCache:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "features.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute(SCHEMA)
        self.hits = 0
        self.misses = 0

    def get_many(self, keys: Sequence[str]) -> list[Optional[np.ndarray]]:
        found: dict[str, np.ndarray] = {}
        for s in range(0, len(keys), 500):
            part = list(keys[s: s + 500])
            marks = ",".join("?" * len(part))
            for key, dim, blob in self.db.execute(f"SELECT key, dim, vec FROM features WHERE key IN ({marks})", part):
                found[key] = np.frombuffer(blob, dtype=np.float32, count=dim)
        out = [found.get(k) for k in keys]
        hit = sum(v is not None for v in out)
        self.hits += hit
        self.misses += len(keys) - hit
        return out

    def put_many(self, keys: Sequence[str], vecs: np.ndarray) -> None:
        rows = [(k, int(v.shape[0]), np.ascontiguousarray(v, dtype=np.float32).tobytes()) for k, v in zip(keys, vecs)]
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO features (key, dim, vec) VALUES (?, ?, ?)", rows)

    def close(self) -> None:
        self.db.close()
