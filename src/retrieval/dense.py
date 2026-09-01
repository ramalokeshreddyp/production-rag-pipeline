from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.core.types import SearchResult
from src.embeddings.base import BaseEmbeddingModel
from src.vector_store.base import BaseVectorStore


class DenseRetriever:
    """Performs semantic dense vector retrieval against a BaseVectorStore."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []

        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        for r in results:
            r.retrieval_method = "dense"
        return results
