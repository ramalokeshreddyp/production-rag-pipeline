from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.core.types import Chunk, SearchResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Combines Dense Vector and BM25 Sparse Search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        candidate_multiplier: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []

        # Retrieve a wider pool of candidate documents from both retrievers
        candidate_k = max(top_k * candidate_multiplier, 20)

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # Merge using Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum_{m} [ w_m / (k + rank_m(d)) ]
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Process dense results
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

        # Process BM25 results
        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

        # Sort aggregated scores
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        max_rrf = sorted_chunks[0][1] if sorted_chunks else 1.0

        final_results: List[SearchResult] = []
        for rank, (cid, score) in enumerate(sorted_chunks, start=1):
            # Normalize to [0, 1]
            norm_score = (score / max_rrf) if max_rrf > 0 else 0.0
            final_results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=norm_score,
                    retrieval_method="hybrid_rrf",
                    rank=rank,
                )
            )

        return final_results
