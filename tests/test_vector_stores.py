from __future__ import annotations

import tempfile
import pytest
from src.core.types import Chunk
from src.embeddings.mock_embed import MockEmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore


def test_faiss_vector_store_crud():
    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = MockEmbeddingModel(dimension=32)
        store = FAISSVectorStore(index_dir=temp_dir, dimension=32)

        assert store.count() == 0

        chunks = [
            Chunk(
                document_id="doc1",
                content="Cloud multi-region failover occurs within 15 seconds.",
                chunk_index=0,
                metadata={"filename": "cloud.txt"},
            ),
            Chunk(
                document_id="doc1",
                content="Kubernetes HPA scales pods dynamically.",
                chunk_index=1,
                metadata={"filename": "cloud.txt"},
            ),
        ]
        embeddings = embedder.embed_documents([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)

        assert store.count() == 2
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 2

        # Search
        q_vec = embedder.embed_query("multi-region failover")
        results = store.search(q_vec, top_k=1)
        assert len(results) == 1
        assert "failover" in results[0].chunk.content
        assert results[0].score > 0.0

        # Delete
        store.delete()
        assert store.count() == 0
