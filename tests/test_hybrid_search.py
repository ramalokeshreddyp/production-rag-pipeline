from __future__ import annotations

import tempfile
import pytest
from src.core.types import Chunk
from src.embeddings.mock_embed import MockEmbeddingModel
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.vector_store.faiss_store import FAISSVectorStore


def test_bm25_retriever():
    chunks = [
        Chunk(document_id="1", content="The quick brown fox jumps over the lazy dog", chunk_index=0),
        Chunk(document_id="2", content="Artificial intelligence and neural network architectures", chunk_index=0),
        Chunk(document_id="3", content="Kubernetes cluster orchestration and autoscaling", chunk_index=0),
    ]
    bm25 = BM25Retriever(chunks=chunks)
    results = bm25.retrieve("brown fox", top_k=1)
    assert len(results) == 1
    assert "fox" in results[0].chunk.content


def test_hybrid_search_rrf():
    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = MockEmbeddingModel(dimension=32)
        store = FAISSVectorStore(index_dir=temp_dir, dimension=32)

        chunks = [
            Chunk(document_id="1", content="Database sharding and consistent hashing algorithm", chunk_index=0),
            Chunk(document_id="2", content="Zero trust architecture and FIDO2 MFA tokens", chunk_index=0),
            Chunk(document_id="3", content="Medallion lakehouse architecture with Bronze Silver Gold", chunk_index=0),
        ]
        embeddings = embedder.embed_documents([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)

        dense = DenseRetriever(store, embedder)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, dense_weight=0.5, bm25_weight=0.5)

        results = hybrid.retrieve("Bronze Silver Gold lakehouse", top_k=1)
        assert len(results) == 1
        assert "Medallion" in results[0].chunk.content
        assert results[0].retrieval_method == "hybrid_rrf"
