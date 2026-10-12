"""Embedding models. SentenceTransformer for production, Hashing for offline tests."""
from __future__ import annotations

import hashlib
import re
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    """Wraps a Hugging Face sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Used in unit tests only:
    needs no model download and no network."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")
