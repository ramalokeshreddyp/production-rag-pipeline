from __future__ import annotations

import pytest
from src.generation.rag_engine import RAGEngine


def test_rag_pipeline_answer_and_citations(populated_rag_engine: RAGEngine):
    query = "How fast does automated failover occur in multi-region deployment?"
    response = populated_rag_engine.query(query, top_k=2)

    assert response.query == query
    assert len(response.answer) > 0
    assert "15 seconds" in response.answer or "failover" in response.answer.lower()
    assert len(response.citations) > 0
    assert response.citations[0].source_document == "cloud_architecture_handbook.txt"
    assert response.citations[0].score >= 0.0
    assert response.latency_seconds >= 0.0


def test_rag_pipeline_out_of_context_refusal(populated_rag_engine: RAGEngine):
    # Query completely unrelated to the cloud document
    query = "What is the secret recipe for Martian chocolate chip cookies?"
    response = populated_rag_engine.query(query, top_k=2)

    assert response.answer == "I don't know."
    assert not response.is_grounded or "I don't know" in response.answer
