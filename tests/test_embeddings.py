from __future__ import annotations

import pytest
from src.embeddings.factory import get_embedding_model
from src.embeddings.mock_embed import MockEmbeddingModel


def test_mock_embedding_model_shape_and_norm():
    model = MockEmbeddingModel(dimension=64)
    assert model.dimension == 64
    assert model.model_name == "mock-embedding-v1"

    docs = ["First test document", "Second test document for embeddings"]
    embeddings = model.embed_documents(docs)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 64
    assert len(embeddings[1]) == 64

    # Query embedding
    q_vec = model.embed_query("test query")
    assert len(q_vec) == 64


def test_embedding_factory():
    m = get_embedding_model("mock")
    assert isinstance(m, MockEmbeddingModel)
