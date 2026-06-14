from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Chunk


async def search_chunks(embedding: list[float], db: AsyncSession, top_k: int | None = None) -> list[Chunk]:
    # TODO: SELECT chunks ORDER BY embedding <=> :query_vec LIMIT top_k
    raise NotImplementedError


async def generate_answer(question: str, chunks: list[Chunk]) -> str:
    # TODO: build prompt with retrieved chunks as context, call chat model, return answer text
    raise NotImplementedError
