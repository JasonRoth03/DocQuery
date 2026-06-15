from dataclasses import dataclass

import tiktoken

# Split priority: paragraphs → lines → sentences → words → characters (last resort)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class TextChunk:
    index: int
    content: str


def _encoder(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _split(text: str, separator: str) -> list[str]:
    if separator == "":
        # character-level: only reached if all other separators failed
        return list(text)
    parts = [p for p in text.split(separator) if p.strip()]
    # Re-attach the separator to all but the last piece so it survives the merge.
    # Without this, splitting on " " turns "a b c" into ["a","b","c"] and spaces
    # are lost when pieces are re-encoded individually during _merge_with_overlap.
    if len(parts) > 1:
        return [p + separator for p in parts[:-1]] + [parts[-1]]
    return parts


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Return a flat list of pieces, each ≤ chunk_size tokens."""
    if _token_count(text, enc) <= chunk_size:
        return [text]

    if not separators:
        # Force-split purely by token boundary
        tokens = enc.encode(text)
        return [enc.decode(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    sep, *rest = separators
    result: list[str] = []
    for piece in _split(text, sep):
        if _token_count(piece, enc) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, rest, chunk_size, enc))
    return result


def _merge_with_overlap(
    splits: list[str],
    chunk_size: int,
    overlap: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Pack splits into chunks of ≤ chunk_size tokens, prepending overlap tokens from the previous chunk."""
    chunks: list[str] = []
    current: list[int] = []

    for split in splits:
        incoming = enc.encode(split)

        if current and len(current) + len(incoming) > chunk_size:
            chunks.append(enc.decode(current))
            # Carry forward the tail of the current chunk as overlap
            current = current[-overlap:] if overlap else []

        current.extend(incoming)

    if current:
        chunks.append(enc.decode(current))

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "text-embedding-3-small",
) -> list[TextChunk]:
    enc = _encoder(model)
    splits = _recursive_split(text, _SEPARATORS, chunk_size, enc)
    merged = _merge_with_overlap(splits, chunk_size, overlap, enc)
    return [TextChunk(index=i, content=c) for i, c in enumerate(merged)]
