from dataclasses import dataclass


@dataclass
class TextChunk:
    index: int
    content: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[TextChunk]:
    # TODO: implement sliding-window character chunking (or swap for token-aware chunking)
    raise NotImplementedError
