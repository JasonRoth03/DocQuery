import uuid
from pathlib import Path
from typing import Protocol

from app.config import settings


class FileStore(Protocol):
    async def save(self, data: bytes, filename: str) -> str:
        """Persist `data` and return a key that can be passed back to load/delete."""
        ...

    async def load(self, key: str) -> bytes:
        """Return the raw bytes for a previously saved file."""
        ...

    async def delete(self, key: str) -> None:
        """Remove the file; no-op if it no longer exists."""
        ...


class LocalFileStore:
    """Stores files on the local filesystem under `base_dir`.

    Key format: ``<uuid>/<original-filename>`` — stable and human-readable.
    Swap this class for an S3FileStore without changing call sites.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.file_store_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, data: bytes, filename: str) -> str:
        key = f"{uuid.uuid4()}/{filename}"
        dest = self.base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return key

    async def load(self, key: str) -> bytes:
        return (self.base_dir / key).read_bytes()

    async def delete(self, key: str) -> None:
        path = self.base_dir / key
        if path.exists():
            path.unlink()
            # remove the now-empty UUID directory
            try:
                path.parent.rmdir()
            except OSError:
                pass


# Module-level singleton — swap for S3FileStore(bucket=...) in production.
file_store: FileStore = LocalFileStore()
