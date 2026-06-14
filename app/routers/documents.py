from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.document import Document
from app.services.file_store import file_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=201)
async def upload_document(file: UploadFile, db: AsyncSession = Depends(get_db)):
    data = await file.read()

    # 1. Persist raw file; get back a stable key
    key = await file_store.save(data, file.filename)

    # 2. Create document row (no content column — text lives in chunks)
    doc = Document(
        filename=file.filename,
        file_key=key,
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
    )
    db.add(doc)
    await db.flush()  # get doc.id before chunking

    # TODO: extract text from `data` (PDF → text, DOCX → text, plain text as-is)
    # TODO: chunk_text(extracted_text) → chunks
    # TODO: embed_texts([c.content for c in chunks]) → embeddings
    # TODO: bulk-insert Chunk rows with embeddings

    await db.commit()
    await db.refresh(doc)
    return {"id": str(doc.id), "filename": doc.filename, "size_bytes": doc.size_bytes}


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    # TODO: return paginated list of documents (no content — just metadata)
    raise NotImplementedError


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await file_store.delete(doc.file_key)
    await db.delete(doc)
    await db.commit()
