from __future__ import annotations

import pytest
from src.core.types import Chunk, SearchResult
from src.retrieval.reranker import PassThroughReranker


def test_passthrough_reranker():
    reranker = PassThroughReranker()
    c1 = Chunk(document_id="1", content="Chunk 1", chunk_index=0)
    c2 = Chunk(document_id="2", content="Chunk 2", chunk_index=1)
    results = [
        SearchResult(chunk=c1, score=0.9),
        SearchResult(chunk=c2, score=0.8),
    ]

    reranked = reranker.rerank(query="test", results=results, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk.content == "Chunk 1"
