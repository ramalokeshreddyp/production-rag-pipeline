from __future__ import annotations

import pytest
from src.core.exceptions import ChunkingError
from src.core.types import Document
from src.ingestion.chunkers import (
    FixedSizeChunker,
    ParagraphAwareChunker,
    SentenceAwareChunker,
    SlidingWindowChunker,
    count_tokens,
    get_chunker,
)


def test_token_counter():
    text = "This is a simple sentence testing token counter utility."
    tokens = count_tokens(text)
    assert tokens > 0


def test_fixed_size_chunker():
    content = "word " * 500
    doc = Document(content=content, filename="test.txt")
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.document_id == doc.id
        assert chunk.metadata["chunk_strategy"] == "fixed"


def test_sentence_aware_chunker():
    sentences = [
        "First sentence discussing topic A.",
        "Second sentence elaborating on topic A.",
        "Third sentence introducing topic B.",
        "Fourth sentence detailing topic B.",
    ]
    doc = Document(content=" ".join(sentences), filename="sentences.txt")
    chunker = SentenceAwareChunker(chunk_size=25, chunk_overlap=5)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 1
    # Check that sentences aren't chopped in half
    for chunk in chunks:
        assert any(s in chunk.content for s in sentences)


def test_paragraph_aware_chunker():
    paragraphs = [
        "Paragraph 1 contains some information about architecture.",
        "Paragraph 2 discusses scaling and performance optimizations.",
        "Paragraph 3 covers incident management and response.",
    ]
    doc = Document(content="\n\n".join(paragraphs), filename="paragraphs.txt")
    chunker = ParagraphAwareChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2


def test_sliding_window_chunker():
    content = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    doc = Document(content=content, filename="sliding.txt")
    chunker = SlidingWindowChunker(chunk_size=4, chunk_overlap=2)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_invalid_overlap_raises_chunking_error():
    with pytest.raises(ChunkingError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=120)


def test_chunker_factory():
    c1 = get_chunker("fixed")
    assert isinstance(c1, FixedSizeChunker)

    c2 = get_chunker("sentence_aware")
    assert isinstance(c2, SentenceAwareChunker)

    c3 = get_chunker("paragraph_aware")
    assert isinstance(c3, ParagraphAwareChunker)

    c4 = get_chunker("sliding_window")
    assert isinstance(c4, SlidingWindowChunker)

    with pytest.raises(ChunkingError):
        get_chunker("nonexistent_strategy")
