from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional
from src.core.types import Chunk, SearchResult


class BM25Retriever:
    """Okapi BM25 sparse keyword retriever implemented from scratch."""

    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.corpus_size: int = 0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []

        if chunks:
            self.index_chunks(chunks)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases and extracts alphanumeric words."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Builds inverted index and term frequency tables for BM25."""
        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        if self.corpus_size == 0:
            self.avg_doc_length = 0.0
            return

        total_length = 0
        for chunk in self.chunks:
            tokens = self.tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for token in tf.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

    def _compute_idf(self, term: str) -> float:
        """Computes Robertson-Spärck Jones IDF with smoothing."""
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        if not query.strip() or self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self._compute_idf(term)

            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom

        # Sort top scores
        scored_indices = sorted(
            [(scores[i], i) for i in range(self.corpus_size) if scores[i] > 0],
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        max_score = scored_indices[0][0] if scored_indices else 1.0

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(scored_indices):
            # Normalize BM25 score to [0, 1] relative to max score
            norm_score = (score / max_score) if max_score > 0 else 0.0
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=norm_score,
                    retrieval_method="bm25",
                    rank=rank + 1,
                )
            )

        return results
