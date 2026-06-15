"""
End-to-end RAG test: upload → embed → retrieve → generate answer with citations.

Requires:
  - PostgreSQL + pgvector running   (docker compose up -d)
  - Ollama running with both models pulled:
      ollama pull nomic-embed-text
      ollama pull qwen2.5:14b

Run:
  pytest tests/test_rag_e2e.py -v -m integration -s
"""
import io

import pytest
from httpx import AsyncClient, ASGITransport

from main import app

pytestmark = pytest.mark.integration

POLICY_DOC = """\
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


async def test_rag_refund_question():
    """Full pipeline: upload policy doc, ask about refunds, get a cited answer."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        up = await client.post(
            "/documents/upload",
            files={"file": ("policy.txt", io.BytesIO(POLICY_DOC.encode()), "text/plain")},
        )
        assert up.status_code == 201
        doc_id = up.json()["id"]

        try:
            response = await client.post(
                "/query/", json={"question": "How do I get a refund?"}
            )
        finally:
            await client.delete(f"/documents/{doc_id}")

    assert response.status_code == 200
    body = response.json()

    print("\n--- INPUT ---")
    print('  { "question": "How do I get a refund?" }')
    print("\n--- ANSWER ---")
    print(f"  {body['answer']}")
    print("\n--- SOURCES ---")
    for s in body["sources"]:
        print(f"  [{s['index']}] {s['filename']} chunk {s['chunk_index']}  score={s['score']:.4f}")
    print("\n--- CHUNKS (raw retrieval) ---")
    for c in body["chunks"]:
        print(f"  score={c['score']:.4f}  {c['content'][:100]!r}")

    assert body["answer"], "answer must not be empty"
    assert body["sources"], "sources must not be empty"
    answer_lower = body["answer"].lower()
    assert any(w in answer_lower for w in ("refund", "30 day", "support@", "business day")), (
        f"answer doesn't mention refund details: {body['answer']!r}"
    )


async def test_rag_response_shape():
    """Verify the response schema: answer str, sources list, chunks list all present."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        up = await client.post(
            "/documents/upload",
            files={"file": ("policy.txt", io.BytesIO(POLICY_DOC.encode()), "text/plain")},
        )
        assert up.status_code == 201
        doc_id = up.json()["id"]

        try:
            response = await client.post(
                "/query/", json={"question": "What subscription plans are available?"}
            )
        finally:
            await client.delete(f"/documents/{doc_id}")

    body = response.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["sources"], list) and body["sources"]
    assert isinstance(body["chunks"], list) and body["chunks"]

    src = body["sources"][0]
    assert {"index", "chunk_id", "filename", "chunk_index", "score"} <= src.keys()
    assert src["index"] == 1

    print("\n--- FULL RESPONSE SCHEMA CHECK PASSED ---")
    print(f"  answer length : {len(body['answer'])} chars")
    print(f"  sources count : {len(body['sources'])}")
    print(f"  chunks  count : {len(body['chunks'])}")
    print(f"\n  answer: {body['answer']}")
