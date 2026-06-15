from typing import Protocol

import httpx

from app.config import settings

_BATCH_SIZE = 32  # max texts per API call


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...


class OllamaEmbedder:
    """Calls the local Ollama /api/embed endpoint (nomic-embed-text or any pulled model)."""

    def __init__(self, base_url: str, model: str) -> None:
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i : i + _BATCH_SIZE]
                resp = await client.post(self.url, json={"model": self.model, "input": batch})
                resp.raise_for_status()
                embeddings.extend(resp.json()["embeddings"])
        return embeddings


class OpenAIEmbedder:
    """Calls the OpenAI embeddings API (text-embedding-3-small or any model)."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            resp = await self._client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in resp.data)
        return embeddings


def _build_embedder() -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    return OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)


# Module-level singleton — swap provider via EMBEDDING_PROVIDER env var.
embedder: Embedder = _build_embedder()
