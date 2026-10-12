import pytest

from docmind.chunking import chunk_pages, split_text


def test_chunks_respect_size_limit():
    text = ("Sentence number one is here. " * 200).strip()
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", 800, 100) == ["Hello world."]


def test_overlap_shares_text_between_neighbours():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(words, chunk_size=200, overlap=60)
    shared = set(chunks[0].split()) & set(chunks[1].split())
    assert shared, "adjacent chunks should overlap"


def test_no_content_lost():
    words = [f"w{i}" for i in range(300)]
    chunks = split_text(" ".join(words), chunk_size=150, overlap=30)
    joined = " ".join(chunks)
    assert all(w in joined.split() for w in words)


def test_invalid_args():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_chunk_pages_keeps_metadata():
    out = chunk_pages([(1, "Page one text."), (2, "Page two text.")], "doc.pdf")
    assert [c.page for c in out] == [1, 2]
    assert all(c.source == "doc.pdf" for c in out)
    assert [c.chunk_id for c in out] == [0, 1]
