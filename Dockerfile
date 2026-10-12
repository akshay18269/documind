FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container starts fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY data/sample_docs ./data/sample_docs

EXPOSE 8000
CMD ["uvicorn", "docmind.api:app", "--host", "0.0.0.0", "--port", "8000"]
