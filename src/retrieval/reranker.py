from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.types import SearchResult


class BaseReranker(ABC):
    """Abstract base class for chunk re-ranking models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 4,
    ) -> List[SearchResult]:
        """Re-scores and re-orders candidate search results."""
        pass


class PassThroughReranker(BaseReranker):
    """No-op reranker that preserves existing candidate ordering."""

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 4,
    ) -> List[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Deep cross-encoder neural reranker (e.g. ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"[Warning] CrossEncoder load failed ({e}), falling back to pass-through.")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 4,
    ) -> List[SearchResult]:
        if not results or self.model is None:
            return results[:top_k]

        try:
            pairs = [[query, r.chunk.content] for r in results]
            scores = self.model.predict(pairs)

            # Combine and sort
            scored_results = list(zip(scores, results))
            scored_results.sort(key=lambda x: x[0], reverse=True)

            reranked: List[SearchResult] = []
            for rank, (score, res) in enumerate(scored_results[:top_k], start=1):
                reranked.append(
                    SearchResult(
                        chunk=res.chunk,
                        score=float(score),
                        retrieval_method="cross_encoder_rerank",
                        rank=rank,
                    )
                )
            return reranked
        except Exception as e:
            print(f"[Warning] CrossEncoder prediction failed: {e}")
            return results[:top_k]
