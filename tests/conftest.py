from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.core.types import Document
from src.embeddings.mock_embed import MockEmbeddingModel
from src.generation.llm_client import MockLLMClient
from src.generation.rag_engine import RAGEngine
from src.ingestion.chunkers import SentenceAwareChunker
from src.vector_store.faiss_store import FAISSVectorStore


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel(dimension=64)


@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def sample_document():
    content = (
        "Enterprise cloud architecture requires resilient topologies to achieve 99.999% availability. "
        "Active-Active multi-region deployment routes traffic dynamically using DNS-based routing. "
        "Automated health checks trigger instant traffic failover within 15 seconds. "
        "Microservices communicate via service mesh providing mutual TLS (mTLS) for zero-trust encryption."
    )
    return Document(
        content=content,
        metadata={"filename": "cloud_architecture_handbook.txt", "domain": "cloud"},
        filename="cloud_architecture_handbook.txt",
    )


@pytest.fixture
def temp_faiss_store(mock_embedding_model):
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FAISSVectorStore(index_dir=temp_dir, dimension=mock_embedding_model.dimension)
        yield store


@pytest.fixture
def populated_rag_engine(mock_embedding_model, mock_llm_client, sample_document):
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FAISSVectorStore(index_dir=temp_dir, dimension=mock_embedding_model.dimension)
        chunker = SentenceAwareChunker(chunk_size=100, chunk_overlap=20)
        engine = RAGEngine(
            vector_store=store,
            embedding_model=mock_embedding_model,
            llm_client=mock_llm_client,
            chunker=chunker,
            retrieval_mode="hybrid",
            top_k=3,
        )
        engine.ingest_documents([sample_document])
        yield engine
