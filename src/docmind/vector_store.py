"""Vector store: FAISS (inner product on normalised vectors = cosine similarity),
with a pure-NumPy fallback so the code and tests run where FAISS isn't installed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.records: List[Dict] = []  # chunk metadata, aligned with vector rows
        self._vectors = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim) if HAS_FAISS else None

    def __len__(self) -> int:
        return len(self.records)

    def add(self, vectors: np.ndarray, records: List[Dict]) -> None:
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have the same length")
        if len(vectors) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self._index is not None:
            self._index.add(vectors)
        self._vectors = np.vstack([self._vectors, vectors])
        self.records.extend(records)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[Tuple[Dict, float]]:
        if not self.records:
            return []
        k = min(k, len(self.records))
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            pairs = zip(idx[0], scores[0])
        else:
            sims = (self._vectors @ q.T).ravel()
            top = np.argsort(-sims)[:k]
            pairs = ((i, sims[i]) for i in top)
        return [(self.records[i], float(s)) for i, s in pairs if i != -1]

    def remove_source(self, source: str) -> None:
        """Drop all chunks of a source (used to re-ingest an updated file)."""
        keep = [i for i, r in enumerate(self.records) if r["source"] != source]
        if len(keep) == len(self.records):
            return
        vecs = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype="float32")
        recs = [self.records[i] for i in keep]
        self.__init__(self.dim)
        self.add(vecs, recs)

    # ---- persistence ----
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._vectors)
        (d / "records.json").write_text(json.dumps(self.records, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        vecs = np.load(d / "vectors.npy")
        records = json.loads((d / "records.json").read_text())
        store = cls(vecs.shape[1] if vecs.ndim == 2 and vecs.shape[0] else 384)
        store.add(vecs, records)
        return store

    @staticmethod
    def exists(directory: str | Path) -> bool:
        d = Path(directory)
        return (d / "vectors.npy").exists() and (d / "records.json").exists()
