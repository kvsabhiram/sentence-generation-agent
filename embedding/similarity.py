"""Near-duplicate detection via cosine similarity of sentence embeddings.

This module is the single authority on duplicate/near-duplicate rejection.
The judge prompt also asks the LLM to flag duplicates as a secondary,
non-authoritative signal, but pipeline/validation.py only trusts this module's
verdict when deciding whether to reject a sentence as a near-duplicate.

A word's accepted-sentence embeddings persist across generation rounds via
pipeline/state.py, so a sentence generated in a backfill retry is checked
against everything already accepted for that word, not just the current batch.
"""

from __future__ import annotations

import numpy as np

from embedding.model import embed_texts


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_near_duplicate(
    candidate_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float,
) -> int | None:
    """Return the index of the first existing embedding that is a near-duplicate
    of the candidate, or None if the candidate is sufficiently novel."""
    for idx, existing in enumerate(existing_embeddings):
        if cosine_similarity(candidate_embedding, existing) >= threshold:
            return idx
    return None


def filter_near_duplicates(
    candidates: list[dict],
    existing_embeddings: list[list[float]],
    threshold: float,
    embedding_model: str,
    text_key: str = "sentence",
) -> tuple[list[dict], list[dict]]:
    """Split candidates into (kept, rejected) based on near-duplicate similarity
    against `existing_embeddings` and against each other (within-batch dedup).

    Each kept candidate dict is returned with an added "_embedding" key so the
    caller can persist it to state without re-embedding.
    """
    if not candidates:
        return [], []

    texts = [c[text_key] for c in candidates]
    embeddings = embed_texts(texts, model=embedding_model)

    kept: list[dict] = []
    rejected: list[dict] = []
    seen_embeddings = list(existing_embeddings)

    for candidate, embedding in zip(candidates, embeddings):
        dup_idx = find_near_duplicate(embedding, seen_embeddings, threshold)
        if dup_idx is not None:
            rejected.append({**candidate, "reason": "near-duplicate of an existing/earlier sentence"})
            continue
        kept.append({**candidate, "_embedding": embedding})
        seen_embeddings.append(embedding)

    return kept, rejected
