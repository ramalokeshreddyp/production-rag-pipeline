from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from src.core.types import Chunk, SearchResult
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.metrics import RetrievalMetrics
from src.generation.rag_engine import RAGEngine


def test_retrieval_metrics():
    c1 = Chunk(document_id="1", content="Text", chunk_index=0, metadata={"filename": "docA.txt"})
    c2 = Chunk(document_id="2", content="Text", chunk_index=1, metadata={"filename": "docB.txt"})
    retrieved = [SearchResult(chunk=c1, score=0.9), SearchResult(chunk=c2, score=0.8)]

    p = RetrievalMetrics.precision_at_k(retrieved, "docA.txt", k=2)
    assert p == 0.5

    r = RetrievalMetrics.recall_at_k(retrieved, "docA.txt", k=2)
    assert r == 1.0

    mrr = RetrievalMetrics.reciprocal_rank(retrieved, "docB.txt")
    assert mrr == 0.5


def test_benchmark_runner(populated_rag_engine: RAGEngine):
    runner = BenchmarkRunner(populated_rag_engine)
    report = runner.run_benchmark(
        dataset_path="./data/evaluation/golden_qa_dataset.json",
        top_k=2,
    )
    assert report.total_queries > 0
    assert report.mean_hit_rate_at_k >= 0.0
    assert len(report.detailed_results) == report.total_queries
