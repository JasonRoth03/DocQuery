from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.embedder import embedder
from app.services.retriever import generate_answer, search_chunks

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    index: int       # 1-based; matches [N] citations in the answer
    chunk_id: str
    filename: str
    chunk_index: int
    score: float


class ChunkMatch(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    chunks: list[ChunkMatch]


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    vector = await embedder.embed([request.question])
    results = await search_chunks(vector[0], db, top_k=request.top_k)
    answer = await generate_answer(request.question, results)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            Source(
                index=i + 1,
                chunk_id=r.chunk_id,
                filename=r.filename,
                chunk_index=r.chunk_index,
                score=r.score,
            )
            for i, r in enumerate(results)
        ],
        chunks=[
            ChunkMatch(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                chunk_index=r.chunk_index,
                content=r.content,
                score=r.score,
            )
            for r in results
        ],
    )
