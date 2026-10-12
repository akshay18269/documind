# 📄 DocuMind: RAG Document Q&A Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application. Upload PDF/TXT/Markdown
files, ask questions in natural language, and get **grounded answers with citations**. If the
answer isn't in your documents, it says so instead of making something up.

**Stack:** Python · Ollama / Groq / Gemini / Anthropic (pluggable LLM) · Sentence-Transformers (Hugging Face) embeddings · FAISS ·
FastAPI · Streamlit · Docker · GitHub Actions · pytest

## Architecture

```
            ┌──────────── Ingestion ────────────┐
 PDF/TXT/MD │ load → chunk (800 chars, 150      │
 ──────────▶│ overlap) → embed (MiniLM) → FAISS │──┐
            └───────────────────────────────────┘  │ persisted to data/index
                                                   ▼
 Question ──▶ embed ──▶ FAISS top-k ──▶ relevance threshold ──▶ prompt (numbered
                         (cosine)        (refuse if too low)     context + rules)
                                                                     │
                                       cited answer  ◀──── Claude (temperature 0)
```

Key design choices
- **Grounded prompting:** the model must answer only from numbered passages, cite them as `[1]`, and
  reply with a fixed refusal sentence when the context lacks the answer.
- **Two-layer hallucination guard:** a similarity threshold (`MIN_SCORE`) refuses *before* calling the
  LLM when nothing relevant is retrieved, which saves cost and latency. The prompt handles the rest.
- **Prompt-injection hygiene:** retrieved text is explicitly treated as data, not instructions.
- **Provider-agnostic:** `LLM` and `Embedder` are small protocols, so swapping models is one class.
- **Testable:** a deterministic hashing embedder and a fake LLM let the whole pipeline be unit-tested
  with no network, API key or model download.

## Choosing an LLM (free options included)

Set `LLM_PROVIDER` in `.env`:

| Provider | Cost | Setup |
|---|---|---|
| `ollama` (default) | Free, runs locally | Install Ollama, then `ollama pull llama3.2` |
| `groq` / `gemini` | Free tier (limits apply) | Create a key, set `LLM_API_KEY` and `LLM_MODEL` |
| `anthropic` | Paid | Set `ANTHROPIC_API_KEY` |

All but Anthropic go through one OpenAI-compatible client (`OpenAICompatLLM`).

## Quick start

```bash
git clone <your-repo-url> && cd documind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # choose an LLM provider (default: free local Ollama)
export PYTHONPATH=src

# 1) Start the API (indexes are loaded/saved in data/index)
uvicorn docmind.api:app --reload
# 2) In another terminal, start the UI
streamlit run app/streamlit_app.py
```

Open http://localhost:8501, upload `data/sample_docs/northwind_handbook.md` (a fictional company
handbook included for demos), and ask, for example, *"How many days of paid leave do I get?"*

**Or with Docker:** `docker compose up --build` → UI at :8501, API docs at :8000/docs

### API

| Method | Path         | Description                              |
|--------|--------------|------------------------------------------|
| GET    | `/health`    | Status, model name, index size           |
| GET    | `/documents` | Indexed documents and chunk counts       |
| POST   | `/ingest`    | Upload a PDF/TXT/MD file and index it    |
| POST   | `/ask`       | `{"question": "...", "top_k": 4}` → answer + cited sources |

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the notice period during probation?"}'
```

## Evaluation

`eval/eval_set.json` has 16 questions: 12 answerable (each with gold evidence and keywords) and
4 unanswerable (should be refused).

```bash
python scripts/evaluate.py                   # retrieval + generation (needs API key)
python scripts/evaluate.py --retrieval-only  # retrieval only, free
```

| Metric | What it measures |
|---|---|
| Retrieval hit@k | A retrieved chunk contains the gold evidence |
| Answer accuracy | All gold keywords appear in the answer and it was not refused |
| Refusal accuracy | Unanswerable questions are correctly refused |
| Latency (mean / p95) | End-to-end time per question |

### Results

> 🔧 **Fill this in after you run it yourself:**
>
> | Model | Embeddings | hit@4 | Answer acc. | Refusal acc. | Mean latency |
> |---|---|---|---|---|---|
> | `llama3.2` (Ollama) | all-MiniLM-L6-v2 | _x_ | _x_ | _x_ | _x_ ms |

Experiments worth running and reporting: vary `CHUNK_SIZE` (400/800/1200), `TOP_K` (2/4/6) and
`MIN_SCORE` (0.15–0.4), and note how hit@k and refusal accuracy trade off.

## Tests

```bash
pytest -q     # 21 tests: chunking, vector store, RAG pipeline (fake LLM, no network)
```
CI runs the same suite on every push (`.github/workflows/ci.yml`).

## Limitations & future work
- Scanned PDFs need OCR (not included).
- Single-user, in-process index; for multi-user use a managed vector DB (Pinecone, pgvector).
- Keyword-based evaluation is a proxy; an LLM-as-judge or human review would be more robust.
- Ideas: hybrid BM25 + dense retrieval, a cross-encoder re-ranker, conversation memory, streaming responses.

## Project structure
```
src/docmind/   config · loader · chunking · embeddings · vector_store · llm · rag · api · schemas
app/           Streamlit UI
scripts/       ingest.py · evaluate.py
eval/          eval_set.json
tests/         pytest suite
```

## License
MIT
