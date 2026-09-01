from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routes import get_rag_engine
from src.embeddings.mock_embed import MockEmbeddingModel
from src.generation.llm_client import MockLLMClient
from src.generation.rag_engine import RAGEngine
from src.vector_store.faiss_store import FAISSVectorStore


@pytest.fixture
def client(tmp_path):
    store = FAISSVectorStore(index_dir=str(tmp_path), dimension=32)
    mock_engine = RAGEngine(
        vector_store=store,
        embedding_model=MockEmbeddingModel(dimension=32),
        llm_client=MockLLMClient(),
    )
    # Ingest sample doc
    from src.core.types import Document
    mock_engine.ingest_documents([
        Document(
            content="Active-Active multi-region deployment achieves 99.999% availability with 15 seconds failover.",
            filename="cloud.txt",
        )
    ])

    import src.api.routes as routes
    routes._rag_engine = mock_engine

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_indexed_chunks"] > 0


def test_query_endpoint(client):
    payload = {
        "query": "How fast does failover occur?",
        "top_k": 2,
        "retrieval_mode": "hybrid",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0


def test_documents_endpoint(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] > 0
    assert data["total_documents"] > 0


def test_evaluate_endpoint(client):
    response = client.get("/api/v1/evaluate?top_k=2")
    assert response.status_code == 200
    data = response.json()
    assert "mean_precision_at_k" in data
    assert "total_queries" in data
