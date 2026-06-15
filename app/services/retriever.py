from dataclasses import dataclass
from typing import Protocol

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings


@dataclass
class ChunkResult:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def _vec_literal(embedding: list[float]) -> str:
    """
    Inlined directly into SQL (not a bind parameter) because asyncpg's binary
    wire protocol conflicts with SQLAlchemy's text-based Vector bind_processor.
    Safe: values are floats produced by our own embedder, never user-supplied strings.
    """
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def search_chunks(
    embedding: list[float],
    db: AsyncSession,
    top_k: int | None = None,
) -> list[ChunkResult]:
    k = top_k or settings.top_k_chunks
    vec = _vec_literal(embedding)

    rows = (
        await db.execute(
            text(f"""
                SELECT
                    c.id          AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.content,
                    1 - (c.embedding <=> '{vec}'::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> '{vec}'::vector
                LIMIT :k
            """),
            {"k": k},
        )
    ).all()

    return [
        ChunkResult(
            chunk_id=str(row.chunk_id),
            document_id=str(row.document_id),
            filename=row.filename,
            chunk_index=row.chunk_index,
            content=row.content,
            score=float(row.score),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class Generator(Protocol):
    async def generate(self, prompt: str) -> str: ...


class OllamaChatGenerator:
    def __init__(self, base_url: str, model: str) -> None:
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]


class OpenAIChatGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


def _build_generator() -> Generator:
    if settings.chat_provider == "openai":
        return OpenAIChatGenerator(settings.openai_api_key, settings.chat_model)
    return OllamaChatGenerator(settings.ollama_base_url, settings.chat_model)


generator: Generator = _build_generator()


def _rag_prompt(question: str, chunks: list[ChunkResult]) -> str:
    context = "\n\n".join(
        f"[{i}] ({c.filename}, chunk {c.chunk_index}, score {c.score:.2f}):\n{c.content.strip()}"
        for i, c in enumerate(chunks, 1)
    )
    return (
        "You are a document assistant. Answer the question using ONLY the information "
        "in the numbered context sections below. Cite sources inline as [1], [2], etc. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def generate_answer(question: str, chunks: list[ChunkResult]) -> str:
    if not chunks:
        return "No relevant documents were found to answer this question."
    return await generator.generate(_rag_prompt(question, chunks))
