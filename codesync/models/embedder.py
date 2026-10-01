"""Qwen3-Embedding-8B in the Modal engine.

Qwen3-Embedding is asymmetric: a *query* carries a one-line task
instruction, a *document* is embedded bare. Requirement matching runs both
directions (A as query over B, and B as query over A), so every requirement
is embedded once in each role.

Vectors from one embedding model are not comparable with another's, so
`name` is part of every embedding cache key: changing the model re-embeds
the library instead of silently comparing across spaces.
"""
from __future__ import annotations

import numpy as np

from .. import config
from .chat import engine


def _query_text(text: str) -> str:
    return f"Instruct: {config.EMBED_TASK}\nQuery:{text}"


class Embedder:
    name = f"modal:{config.EMBED_MODEL}:bf16"

    def __init__(self):
        self._engine = None
        self.stats = {"calls": 0, "texts": 0, "elapsed_s": 0.0}

    def embed(self, texts: list[str], as_query: bool) -> np.ndarray:
        """L2-normalised float32 vectors, shape (len(texts), EMBED_DIM)."""
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        if self._engine is None:
            self._engine = engine()
        inputs = [_query_text(t) if as_query else t for t in texts]
        parts = []
        for i in range(0, len(inputs), config.EMBED_BATCH):
            res = self._engine.embed.remote(inputs[i:i + config.EMBED_BATCH])
            parts.append(np.frombuffer(res["data"], dtype=np.float32)
                         .reshape(res["shape"]))
            self.stats["calls"] += 1
            self.stats["texts"] += res["shape"][0]
            self.stats["elapsed_s"] += res["elapsed_s"]
        arr = np.concatenate(parts).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr
