from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.document import Chunk, Document
from app.services.chunker import chunk_text
from app.services.extractor import extract_text
from app.services.file_store import file_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=201)
async def upload_document(file: UploadFile, db: AsyncSession = Depends(get_db)):
    data = await file.read()
    mime = file.content_type or "application/octet-stream"

    text = extract_text(data, mime, file.filename)

    key = await file_store.save(data, file.filename)

    doc = Document(
        filename=file.filename,
        file_key=key,
        mime_type=mime,
        size_bytes=len(data),
    )
    db.add(doc)
    await db.flush()

    raw_chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        model=settings.embedding_model,
    )

    db.add_all(
        Chunk(
            document_id=doc.id,
            chunk_index=c.index,
            content=c.content,
            # embedding filled in by a background job or the embedder service
        )
        for c in raw_chunks
    )

    await db.commit()
    await db.refresh(doc)

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "size_bytes": doc.size_bytes,
        "chunk_count": len(raw_chunks),
    }


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    # TODO: return paginated list of documents
    raise NotImplementedError


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await file_store.delete(doc.file_key)
    await db.delete(doc)
    await db.commit()
