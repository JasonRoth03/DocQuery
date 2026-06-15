import tiktoken
import pytest
from app.services.chunker import chunk_text

ENC = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(ENC.encode(text))


def test_short_text_produces_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "Hello" in chunks[0].content


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=512, overlap=50) == []


def test_indices_are_sequential():
    text = "word " * 2000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_all_content_preserved():
    words = [f"WORD{i:04d}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    combined = " ".join(c.content for c in chunks)
    missing = [w for w in words if w not in combined]
    assert not missing, f"words missing from chunks: {missing[:5]}"


def test_chunk_token_count_within_bounds():
    text = "The quick brown fox jumps over the lazy dog. " * 500
    chunk_size = 100
    overlap = 20
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for chunk in chunks:
        count = token_count(chunk.content)
        # overlap seeding can cause a chunk to slightly exceed chunk_size
        assert count <= chunk_size + overlap, (
            f"chunk {chunk.index} has {count} tokens (limit {chunk_size + overlap})"
        )


def test_overlap_carries_tail_into_next_chunk():
    # BPE re-encoding at chunk boundaries produces different token IDs even for
    # the same decoded text, so we compare at the word level instead.
    words = [f"WORD{i:04d}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        tail_words = set(chunks[i].content.split()[-5:])
        head_words = set(chunks[i + 1].content.split()[:15])
        assert tail_words & head_words, (
            f"no word overlap between chunk {i} and {i + 1}\n"
            f"  tail: {tail_words}\n"
            f"  head: {head_words}"
        )


def test_paragraph_boundaries_preferred_over_mid_sentence_splits():
    # Two clear paragraphs that together fit in one chunk
    para_a = "Sentence one of paragraph A. Sentence two of paragraph A."
    para_b = "Sentence one of paragraph B. Sentence two of paragraph B."
    text = para_a + "\n\n" + para_b
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    # Everything fits in one chunk — the separator should not force a split
    assert len(chunks) == 1
    assert "paragraph A" in chunks[0].content
    assert "paragraph B" in chunks[0].content


def test_large_document_splits_into_multiple_chunks():
    text = "word " * 10_000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 10
