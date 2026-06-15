import pytest
from sqlalchemy import text

from app.database import engine, AsyncSessionLocal


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires external services (Ollama, PostgreSQL)")


@pytest.fixture(autouse=True)
async def clean_db():
    """Delete all documents (cascades to chunks) before each test so stale rows
    from prior runs never pollute results."""
    async with AsyncSessionLocal() as session:
        await session.execute(text("DELETE FROM documents"))
        await session.commit()
    yield
    await engine.dispose()
