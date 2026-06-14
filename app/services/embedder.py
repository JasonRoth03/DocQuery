from app.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    # TODO: call OpenAI embeddings API (or swap provider) and return one vector per text
    raise NotImplementedError


async def embed_query(text: str) -> list[float]:
    results = await embed_texts([text])
    return results[0]
