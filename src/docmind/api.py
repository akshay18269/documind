"""FastAPI service exposing the RAG pipeline.

Run:  uvicorn docmind.api:app --reload   (from the project root with PYTHONPATH=src)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import __version__
from .config import get_settings
from .embeddings import SentenceTransformerEmbedder
from .llm import build_llm
from .loader import SUPPORTED
from .rag import RAGPipeline
from .schemas import AskRequest, AskResponse, IngestResponse

settings = get_settings()
app = FastAPI(title="DocuMind RAG API", version=__version__)
_lock = threading.Lock()  # protects index mutation
UPLOAD_DIR = Path("data/uploads")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Lazily build the pipeline (loads the embedding model + any saved index)."""
    global _pipeline
    if _pipeline is None:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        llm = build_llm(settings)
        _pipeline = RAGPipeline(embedder, llm, settings=settings)
        _pipeline.load()
    return _pipeline


@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "documents": len(p.documents()),
        "chunks": len(p.store) if p.store else 0,
    }


@app.get("/documents")
def documents():
    return get_pipeline().documents()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use {sorted(SUPPORTED)}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(file.filename).name  # strip any path components
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    p = get_pipeline()
    with _lock:
        added = p.ingest_file(dest)
        p.save()
    if added == 0:
        raise HTTPException(422, "No extractable text found (scanned PDF?).")
    return IngestResponse(file=dest.name, chunks_added=added, documents=p.documents())


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    p = get_pipeline()
    if not p.store or len(p.store) == 0:
        raise HTTPException(409, "No documents indexed yet. POST a file to /ingest first.")
    return p.ask(req.question, req.top_k)
