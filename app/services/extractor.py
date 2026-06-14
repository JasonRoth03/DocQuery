import io
from pathlib import Path


def extract_text(data: bytes, mime_type: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if mime_type.startswith("text/") or ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")

    if mime_type == "application/pdf" or ext == ".pdf":
        return _from_pdf(data)

    if (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or ext == ".docx"
    ):
        return _from_docx(data)

    raise ValueError(f"Unsupported file type: {mime_type} / {ext}")


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
