from pathlib import Path

from docmind.config import Settings
from docmind.embeddings import HashingEmbedder
from docmind.rag import NO_ANSWER, RAGPipeline

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


class FakeLLM:
    def __init__(self, reply="Answer [1]."):
        self.reply, self.calls = reply, []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.reply


def make(llm=None, **kw):
    s = Settings(min_score=kw.pop("min_score", 0.15), **kw)
    p = RAGPipeline(HashingEmbedder(), llm, settings=s)
    p.ingest_directory(SAMPLE)
    return p


def test_ingest_creates_chunks_with_metadata():
    p = make()
    assert len(p.store) > 3
    assert set(p.documents()) == {"northwind_handbook.md"}


def test_retrieval_finds_relevant_chunk():
    p = make()
    hits = p.retrieve("How many days of paid annual leave do employees get?", top_k=3)
    assert any("24 days of paid annual leave" in r["text"] for r, _ in hits)


def test_prompt_contains_numbered_context_and_question():
    llm = FakeLLM()
    p = make(llm)
    out = p.ask("What is the notice period during probation?")
    system, user = llm.calls[0]
    assert "[1]" in user and "Question: What is the notice period during probation?" in user
    assert "ONLY" in system
    assert out["answer"] == "Answer [1]." and out["sources"][0]["id"] == 1


def test_irrelevant_question_refused_without_llm_call():
    llm = FakeLLM()
    p = make(llm, min_score=0.5)
    out = p.ask("zxqv wlmn plkj")
    assert out["refused"] and out["answer"] == NO_ANSWER
    assert llm.calls == []


def test_model_refusal_is_detected():
    p = make(FakeLLM(NO_ANSWER))
    assert p.ask("How many days of paid annual leave do employees get?")["refused"]


def test_reingest_replaces_old_chunks():
    p = make()
    before = len(p.store)
    p.ingest_file(SAMPLE / "northwind_handbook.md")
    assert len(p.store) == before


def test_save_and_load(tmp_path):
    p = make(index_dir=str(tmp_path))
    p.save()
    q = RAGPipeline(HashingEmbedder(), settings=Settings(index_dir=str(tmp_path), min_score=0.15))
    assert q.load() and len(q.store) == len(p.store)
