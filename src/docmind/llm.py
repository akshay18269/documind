"""LLM client wrappers. Swap providers by implementing `generate`."""
from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str, max_tokens: int = 600, api_key: str | None = None):
        import anthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,  # deterministic answers -> reproducible evaluation
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


class OpenAICompatLLM:
    """Works with any server that speaks the OpenAI chat-completions format:
    Ollama (local, free), Groq, Google Gemini, OpenAI, vLLM, etc. Uses plain `requests`."""

    def __init__(self, model: str, base_url: str, api_key: str = "", max_tokens: int = 600, timeout: int = 180):
        if not model:
            raise RuntimeError("LLM_MODEL is not set.")
        if not base_url:
            raise RuntimeError("LLM_BASE_URL is not set.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key, self.max_tokens, self.timeout = api_key, max_tokens, timeout

    def generate(self, system: str, user: str) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach the LLM server at {self.base_url}. "
                "If you use Ollama, make sure the Ollama app is running."
            ) from e
        if r.status_code != 200:
            raise RuntimeError(f"LLM request failed ({r.status_code}): {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()


def build_llm(settings) -> LLM:
    """Create the LLM client chosen by LLM_PROVIDER (default: ollama, which is free)."""
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.max_tokens)
    return OpenAICompatLLM(
        settings.llm_model, settings.llm_base_url, settings.llm_api_key, settings.max_tokens
    )
