"""
Integration test for POST /query.

Requires:
  - PostgreSQL + pgvector running  (docker compose up -d)
  - Ollama running with nomic-embed-text pulled  (ollama serve)

Run:
  pytest tests/test_query_endpoint.py -v -m integration -s
"""
import io
import pytest
from httpx import AsyncClient, ASGITransport

from main import app

pytestmark = pytest.mark.integration

SAMPLE_DOCUMENT = """\
DocQuery Refund Policy

Customers may request a full refund within 30 days of purchase.
Refunds are processed within 5-7 business days after approval.
To initiate a refund, contact support@docquery.io with your order number.

DocQuery Subscription Plans

The Basic plan costs $9 per month and includes up to 10 documents.
The Pro plan costs $29 per month and includes unlimited documents and priority support.
Annual billing provides a 20% discount on all plans.

DocQuery Security

All documents are encrypted at rest using AES-256.
Data is transmitted over TLS 1.3.
DocQuery is SOC 2 Type II certified.
"""


async def upload_sample(client: AsyncClient) -> str:
    response = await client.post(
        "/documents/upload",
        files={"file": ("policy.txt", io.BytesIO(SAMPLE_DOCUMENT.encode()), "text/plain")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def delete_doc(client: AsyncClient, doc_id: str) -> None:
    await client.delete(f"/documents/{doc_id}")


async def test_query_returns_refund_chunk():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        doc_id = await upload_sample(client)
        try:
            response = await client.post("/query/", json={"question": "How do I get a refund?"})
        finally:
            await delete_doc(client, doc_id)

    assert response.status_code == 200
    body = response.json()

    print("\n--- INPUT ---")
    print('  { "question": "How do I get a refund?" }')
    print("\n--- OUTPUT ---")
    for chunk in body["chunks"]:
        print(f"  score={chunk['score']:.4f}  [{chunk['filename']} chunk {chunk['chunk_index']}]")
        print(f"  {chunk['content'][:120]!r}")
        print()

    top = body["chunks"][0]
    assert "refund" in top["content"].lower()
    assert top["score"] > 0.5


async def test_query_returns_pricing_chunk():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        doc_id = await upload_sample(client)
        try:
            response = await client.post("/query/", json={"question": "What does the Pro plan cost?"})
        finally:
            await delete_doc(client, doc_id)

    assert response.status_code == 200
    body = response.json()

    print("\n--- INPUT ---")
    print('  { "question": "What does the Pro plan cost?" }')
    print("\n--- OUTPUT ---")
    for chunk in body["chunks"]:
        print(f"  score={chunk['score']:.4f}  [{chunk['filename']} chunk {chunk['chunk_index']}]")
        print(f"  {chunk['content'][:120]!r}")
        print()

    top = body["chunks"][0]
    assert "pro" in top["content"].lower() or "plan" in top["content"].lower()


async def test_query_returns_security_chunk():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        doc_id = await upload_sample(client)
        try:
            response = await client.post("/query/", json={"question": "Is my data encrypted?"})
        finally:
            await delete_doc(client, doc_id)

    assert response.status_code == 200
    body = response.json()

    print("\n--- INPUT ---")
    print('  { "question": "Is my data encrypted?" }')
    print("\n--- OUTPUT ---")
    for chunk in body["chunks"]:
        print(f"  score={chunk['score']:.4f}  [{chunk['filename']} chunk {chunk['chunk_index']}]")
        print(f"  {chunk['content'][:120]!r}")
        print()

    top = body["chunks"][0]
    assert any(w in top["content"].lower() for w in ("encrypt", "aes", "tls", "security"))


async def test_top_k_override():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        doc_id = await upload_sample(client)
        try:
            response = await client.post("/query/", json={"question": "plans and pricing", "top_k": 2})
        finally:
            await delete_doc(client, doc_id)

    assert response.status_code == 200
    assert len(response.json()["chunks"]) <= 2


async def test_scores_are_descending():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        doc_id = await upload_sample(client)
        try:
            response = await client.post("/query/", json={"question": "refund policy"})
        finally:
            await delete_doc(client, doc_id)

    scores = [c["score"] for c in response.json()["chunks"]]
    assert scores == sorted(scores, reverse=True), "chunks should be ordered highest score first"
