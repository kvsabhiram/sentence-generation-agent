"""Local embedding model used for near-duplicate detection.

Runs on-GPU via sentence-transformers instead of calling a paid embedding API
-- this module is called very frequently (every accepted-sentence history
re-embed, every batch's within-round dedup check), so keeping it local and
free matters at 15k+ word scale.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[list[float]]:
    """Embed a batch of strings. Returns one vector per input, same order."""
    if not texts:
        return []
    m = _get_model(model)
    vectors = m.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
    return vectors.tolist()


def embed_text(text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[float]:
    return embed_texts([text], model=model)[0]
