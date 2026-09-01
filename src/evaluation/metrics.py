from __future__ import annotations

from typing import List, Optional, Set
from src.core.types import EvaluationResult, SearchResult


class RetrievalMetrics:
    """Calculates standard Information Retrieval (IR) and RAG evaluation metrics."""

    @staticmethod
    def precision_at_k(
        retrieved: List[SearchResult],
        ground_truth_doc: str,
        k: int = 4,
    ) -> float:
        """Precision@K: Fraction of top-K retrieved chunks originating from relevant source."""
        if not retrieved or k <= 0:
            return 0.0
        top_k_chunks = retrieved[:k]
        relevant_count = sum(
            1
            for r in top_k_chunks
            if ground_truth_doc.lower() in str(r.chunk.metadata.get("filename", "")).lower()
            or ground_truth_doc.lower() in str(r.chunk.metadata.get("source", "")).lower()
        )
        return relevant_count / len(top_k_chunks)

    @staticmethod
    def recall_at_k(
        retrieved: List[SearchResult],
        ground_truth_doc: str,
        k: int = 4,
    ) -> float:
        """Recall@K: 1.0 if at least one relevant chunk found in top-K, else 0.0 (for single gold doc)."""
        if not retrieved or k <= 0:
            return 0.0
        top_k_chunks = retrieved[:k]
        found = any(
            ground_truth_doc.lower() in str(r.chunk.metadata.get("filename", "")).lower()
            or ground_truth_doc.lower() in str(r.chunk.metadata.get("source", "")).lower()
            for r in top_k_chunks
        )
        return 1.0 if found else 0.0

    @staticmethod
    def reciprocal_rank(
        retrieved: List[SearchResult],
        ground_truth_doc: str,
    ) -> float:
        """MRR Rank Score: 1 / rank of the first relevant document retrieved."""
        for rank, r in enumerate(retrieved, start=1):
            if (
                ground_truth_doc.lower() in str(r.chunk.metadata.get("filename", "")).lower()
                or ground_truth_doc.lower() in str(r.chunk.metadata.get("source", "")).lower()
            ):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved: List[SearchResult],
        ground_truth_doc: str,
        k: int = 4,
    ) -> float:
        """Hit Rate@K: Returns 1.0 if gold document is within Top-K, else 0.0."""
        return RetrievalMetrics.recall_at_k(retrieved, ground_truth_doc, k=k)

    @staticmethod
    def keyword_overlap_score(
        retrieved: List[SearchResult],
        target_keywords: List[str],
    ) -> float:
        """Measures keyword recall across retrieved context."""
        if not target_keywords or not retrieved:
            return 0.0
        combined_text = " ".join(r.chunk.content.lower() for r in retrieved)
        matched = sum(1 for kw in target_keywords if kw.lower() in combined_text)
        return matched / len(target_keywords)
