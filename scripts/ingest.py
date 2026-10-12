"""Index every document in a folder.   python scripts/ingest.py data/sample_docs"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmind.config import get_settings
from docmind.embeddings import SentenceTransformerEmbedder
from docmind.rag import RAGPipeline


def main(folder: str) -> None:
    s = get_settings()
    pipe = RAGPipeline(SentenceTransformerEmbedder(s.embedding_model), settings=s)
    pipe.load()
    for name, n in pipe.ingest_directory(folder).items():
        print(f"indexed {name}: {n} chunks")
    pipe.save()
    print(f"Saved index to {s.index_dir} ({len(pipe.store)} chunks total)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs")
