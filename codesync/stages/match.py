"""Embedding and candidate matching.

Judging every A-requirement against every B-requirement would cost
len(A) x len(B) LLM calls. Embeddings cut that down to the handful of
pairs per requirement that plausibly address the same thing; the LLM then
decides what the relation actually is.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .. import config, library
from ..models.embedder import Embedder
from ..schema import Requirement


def requirement_text(r: Requirement) -> str:
    """What gets embedded: subject and parameter lead, so two requirements
    about the same thing land close even when worded differently."""
    head = " - ".join(p for p in (r.subject, r.parameter) if p)
    return f"{head}: {r.statement}" if head else r.statement


def embedding_key(reqs: list[Requirement]) -> str:
    h = hashlib.sha256()
    for r in reqs:
        h.update(requirement_text(r).encode("utf-8") + b"\0")
    return f"{Embedder.name}|{config.EMBED_TASK}|{h.hexdigest()[:16]}"


def embed_requirements(doc_id: str, reqs: list[Requirement],
                       embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
    """(document vectors, query vectors), cached on disk per document."""
    key = embedding_key(reqs)
    cached = library.load_embeddings(doc_id, key)
    if cached is not None:
        return cached[1], cached[2]
    texts = [requirement_text(r) for r in reqs]
    doc_vecs = embedder.embed(texts, as_query=False)
    query_vecs = embedder.embed(texts, as_query=True)
    library.save_embeddings(doc_id, [r.id for r in reqs], doc_vecs,
                            query_vecs, key)
    return doc_vecs, query_vecs


def candidate_pairs(emb_a: tuple[np.ndarray, np.ndarray],
                    emb_b: tuple[np.ndarray, np.ndarray],
                    top_k: int = config.TOP_K,
                    min_sim: float = config.MIN_SIMILARITY
                    ) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """[(index in A, index in B, similarity)] plus the full score matrix.

    The score is the mean of both retrieval directions (A as query over B,
    B as query over A). A pair is kept if it is in the top-k of either
    side, so a requirement that is nobody's best match still gets its own
    best candidates judged."""
    doc_a, query_a = emb_a
    doc_b, query_b = emb_b
    if len(doc_a) == 0 or len(doc_b) == 0:
        return [], np.zeros((len(doc_a), len(doc_b)), dtype=np.float32)
    sim = 0.5 * (query_a @ doc_b.T + (query_b @ doc_a.T).T)
    keep: set[tuple[int, int]] = set()
    k_b = min(top_k, sim.shape[1])
    for i in range(sim.shape[0]):
        for j in np.argpartition(-sim[i], k_b - 1)[:k_b]:
            if sim[i, j] >= min_sim:
                keep.add((i, int(j)))
    k_a = min(top_k, sim.shape[0])
    for j in range(sim.shape[1]):
        for i in np.argpartition(-sim[:, j], k_a - 1)[:k_a]:
            if sim[i, j] >= min_sim:
                keep.add((int(i), j))
    pairs = sorted(((i, j, float(sim[i, j])) for i, j in keep),
                   key=lambda p: (p[0], -p[2]))
    return pairs, sim
