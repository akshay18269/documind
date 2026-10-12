import numpy as np

from docmind.embeddings import HashingEmbedder
from docmind.vector_store import VectorStore

DOCS = [
    ("cats purr and sleep all day", "a.txt"),
    ("python is a programming language", "b.txt"),
    ("the stock market fell sharply today", "c.txt"),
]


def build():
    emb = HashingEmbedder()
    vecs = emb.embed([t for t, _ in DOCS])
    store = VectorStore(vecs.shape[1])
    store.add(vecs, [{"text": t, "source": s, "page": 1, "chunk_id": i} for i, (t, s) in enumerate(DOCS)])
    return emb, store


def test_search_returns_most_similar_first():
    emb, store = build()
    hits = store.search(emb.embed(["programming language python"])[0], k=2)
    assert hits[0][0]["source"] == "b.txt"
    assert hits[0][1] >= hits[1][1]


def test_k_larger_than_store():
    emb, store = build()
    assert len(store.search(emb.embed(["cats"])[0], k=50)) == 3


def test_save_and_load_roundtrip(tmp_path):
    emb, store = build()
    store.save(tmp_path)
    assert VectorStore.exists(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 3
    hit = loaded.search(emb.embed(["stock market"])[0], k=1)[0][0]
    assert hit["source"] == "c.txt"


def test_remove_source():
    emb, store = build()
    store.remove_source("a.txt")
    assert len(store) == 2
    assert all(r["source"] != "a.txt" for r in store.records)


def test_empty_store_search():
    assert VectorStore(8).search(np.ones(8, dtype="float32"), k=3) == []
