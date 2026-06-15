import math
import pytest
import httpx
from app.config import settings
from app.services.embedder import OllamaEmbedder


def _ollama_available() -> bool:
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


_OLLAMA_UP = _ollama_available()
skip_if_no_ollama = pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running")


@pytest.fixture(scope="module")
def emb():
    return OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)


@skip_if_no_ollama
@pytest.mark.integration
async def test_returns_one_vector_per_text(emb):
    vectors = await emb.embed(["Hello world", "Second sentence"])
    assert len(vectors) == 2


@skip_if_no_ollama
@pytest.mark.integration
async def test_vector_dimension_matches_config(emb):
    vectors = await emb.embed(["test"])
    assert len(vectors[0]) == settings.embedding_dimensions


@skip_if_no_ollama
@pytest.mark.integration
async def test_vectors_are_floats(emb):
    vectors = await emb.embed(["test"])
    assert all(isinstance(v, float) for v in vectors[0])


@skip_if_no_ollama
@pytest.mark.integration
async def test_batch_larger_than_batch_size(emb):
    # _BATCH_SIZE is 32 — send 50 to exercise batching logic
    texts = [f"sentence number {i}" for i in range(50)]
    vectors = await emb.embed(texts)
    assert len(vectors) == 50
    assert all(len(v) == settings.embedding_dimensions for v in vectors)


@skip_if_no_ollama
@pytest.mark.integration
async def test_similar_texts_score_higher_than_unrelated(emb):
    vectors = await emb.embed([
        "The cat sat on the mat",
        "A cat resting on a mat",   # semantically similar to first
        "PostgreSQL is a relational database",  # unrelated
    ])

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

    sim_related = cosine(vectors[0], vectors[1])
    sim_unrelated = cosine(vectors[0], vectors[2])
    assert sim_related > sim_unrelated, (
        f"expected similar texts to score higher: {sim_related:.3f} vs {sim_unrelated:.3f}"
    )
