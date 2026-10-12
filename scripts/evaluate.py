"""Evaluate the RAG pipeline on eval/eval_set.json.

  python scripts/evaluate.py                    # full eval (retrieval + generation)
  python scripts/evaluate.py --retrieval-only   # no API key needed

Metrics
  Retrieval hit@k     : does a retrieved chunk contain the gold evidence string?
  Answer accuracy     : do all gold keywords appear in the generated answer?
  Refusal accuracy    : are unanswerable questions correctly refused?
  Latency             : mean and p95 end-to-end time per question
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docmind.config import get_settings
from docmind.embeddings import SentenceTransformerEmbedder
from docmind.llm import build_llm
from docmind.rag import RAGPipeline


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def run(docs_dir: str, eval_path: str, retrieval_only: bool, k: int) -> dict:
    s = get_settings()
    llm = None if retrieval_only else build_llm(s)
    pipe = RAGPipeline(SentenceTransformerEmbedder(s.embedding_model), llm, settings=s)
    pipe.ingest_directory(docs_dir)  # fresh in-memory index for reproducibility
    items = json.loads(Path(eval_path).read_text())

    rows, hits, acc, refusals = [], [], [], []
    for it in items:
        q = it["question"]
        retrieved = pipe.retrieve(q, k)
        row = {"id": it["id"], "question": q, "type": it["type"]}
        if it["type"] == "answerable":
            hit = any(norm(it["evidence"]) in norm(r["text"]) for r, _ in retrieved)
            row["retrieval_hit"] = hit
            hits.append(hit)
        else:
            top = retrieved[0][1] if retrieved else 0.0
            row["top_score"] = round(top, 3)
        if not retrieval_only:
            out = pipe.ask(q, k)
            row.update(answer=out["answer"], latency_ms=out["latency_ms"], refused=out["refused"])
            if it["type"] == "answerable":
                ok = (not out["refused"]) and all(norm(kw) in norm(out["answer"]) for kw in it["answer_keywords"])
                row["answer_correct"] = ok
                acc.append(ok)
            else:
                row["correctly_refused"] = out["refused"]
                refusals.append(out["refused"])
        rows.append(row)

    summary = {"n_questions": len(items), "top_k": k, "retrieval_hit_at_k": round(sum(hits) / len(hits), 3)}
    if not retrieval_only:
        lat = sorted(r["latency_ms"] for r in rows)
        summary.update(
            answer_accuracy=round(sum(acc) / len(acc), 3),
            refusal_accuracy=round(sum(refusals) / len(refusals), 3),
            latency_mean_ms=round(statistics.mean(lat), 1),
            latency_p95_ms=lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            llm_provider=s.llm_provider,
            llm_model=s.llm_model,
        )
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("-k", type=int, default=4)
    a = ap.parse_args()
    result = run(a.docs, a.eval_set, a.retrieval_only, a.k)
    Path("eval").mkdir(exist_ok=True)
    Path("eval/results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    for r in result["rows"]:
        flag = r.get("answer_correct", r.get("correctly_refused", r.get("retrieval_hit")))
        print(f"{'PASS' if flag else 'FAIL'}  #{r['id']:>2} {r['question']}")
