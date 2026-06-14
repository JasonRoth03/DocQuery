from fastapi import FastAPI

from app.routers import documents, query

app = FastAPI(title="DocQuery", version="0.1.0")

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}
