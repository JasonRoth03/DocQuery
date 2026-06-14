from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    # TODO: embed question, vector search top-k chunks, call LLM with context, return answer + citations
    raise NotImplementedError
