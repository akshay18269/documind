"""Central configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # load variables from a local .env file (ANTHROPIC_API_KEY, etc.)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional (e.g. in CI, Docker uses env_file)
    pass


# provider -> (default model, OpenAI-compatible base URL). Model names change over
# time; override with LLM_MODEL / LLM_BASE_URL in .env if a default is outdated.
PROVIDER_DEFAULTS = {
    "ollama": ("llama3.2", "http://localhost:11434/v1"),
    "groq": ("llama-3.1-8b-instant", "https://api.groq.com/openai/v1"),
    "gemini": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "anthropic": ("claude-haiku-5-5", ""),
    "openai_compatible": ("", ""),
}
_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
_DEF_MODEL, _DEF_URL = PROVIDER_DEFAULTS.get(_PROVIDER, ("", ""))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _PROVIDER
    llm_model: str = os.getenv("LLM_MODEL") or _DEF_MODEL
    llm_base_url: str = os.getenv("LLM_BASE_URL") or _DEF_URL
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))        # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Minimum cosine similarity for a chunk to count as relevant. If the best
    # chunk is below this, we refuse to answer without calling the LLM.
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))
    index_dir: str = os.getenv("INDEX_DIR", "data/index")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "600"))


def get_settings() -> Settings:
    return Settings()
