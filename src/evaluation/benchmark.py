from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.core.types import EvaluationResult
from src.evaluation.metrics import RetrievalMetrics
from src.generation.rag_engine import RAGEngine


class BenchmarkReport(BaseModel):
    """Aggregate benchmark report summary."""

    total_queries: int
    in_domain_queries: int
    out_of_domain_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_hit_rate_at_k: float
    mean_keyword_overlap: float
    out_of_domain_refusal_rate: float
    retrieval_mode: str
    top_k: int
    detailed_results: List[EvaluationResult]


class BenchmarkRunner:
    """Executes automated benchmark evaluation of the RAG pipeline against a golden QA dataset."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def run_benchmark(
        self,
        dataset_path: str | Path = "./data/evaluation/golden_qa_dataset.json",
        top_k: int = 4,
        retrieval_mode: str = "hybrid",
    ) -> BenchmarkReport:
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {dataset_file}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            qa_items: List[Dict[str, Any]] = json.load(f)

        detailed_results: List[EvaluationResult] = []
        in_domain_precisions: List[float] = []
        in_domain_recalls: List[float] = []
        in_domain_mrrs: List[float] = []
        in_domain_hit_rates: List[float] = []
        keyword_overlaps: List[float] = []
        ood_refusals = 0
        ood_total = 0
        in_domain_total = 0

        for item in qa_items:
            qid = item.get("id", "unknown")
            query = item["question"]
            gold_doc = item.get("source_document", "")
            is_in_domain = item.get("is_in_domain", True)
            keywords = item.get("keywords", [])

            # Run RAG query
            response = self.rag_engine.query(
                query_text=query,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )

            retrieved = response.retrieved_chunks

            if is_in_domain:
                in_domain_total += 1
                p_at_k = RetrievalMetrics.precision_at_k(retrieved, gold_doc, k=top_k)
                r_at_k = RetrievalMetrics.recall_at_k(retrieved, gold_doc, k=top_k)
                mrr = RetrievalMetrics.reciprocal_rank(retrieved, gold_doc)
                hit_rate = RetrievalMetrics.hit_rate_at_k(retrieved, gold_doc, k=top_k)
                kw_score = RetrievalMetrics.keyword_overlap_score(retrieved, keywords)

                in_domain_precisions.append(p_at_k)
                in_domain_recalls.append(r_at_k)
                in_domain_mrrs.append(mrr)
                in_domain_hit_rates.append(hit_rate)
                keyword_overlaps.append(kw_score)

                detailed_results.append(
                    EvaluationResult(
                        query_id=qid,
                        query=query,
                        precision_at_k=round(p_at_k, 3),
                        recall_at_k=round(r_at_k, 3),
                        mrr=round(mrr, 3),
                        hit_rate=round(hit_rate, 3),
                        retrieved_count=len(retrieved),
                        ground_truth_found=(r_at_k > 0),
                        notes=f"Answer: {response.answer[:80]}...",
                    )
                )
            else:
                ood_total += 1
                refused = "I don't know" in response.answer or not response.is_grounded
                if refused:
                    ood_refusals += 1

                detailed_results.append(
                    EvaluationResult(
                        query_id=qid,
                        query=query,
                        precision_at_k=1.0 if refused else 0.0,
                        recall_at_k=1.0 if refused else 0.0,
                        mrr=1.0 if refused else 0.0,
                        hit_rate=1.0 if refused else 0.0,
                        retrieved_count=len(retrieved),
                        ground_truth_found=refused,
                        notes=f"[OOD Refusal: {'YES' if refused else 'NO'}] Answer: {response.answer}",
                    )
                )

        mean_p = sum(in_domain_precisions) / len(in_domain_precisions) if in_domain_precisions else 0.0
        mean_r = sum(in_domain_recalls) / len(in_domain_recalls) if in_domain_recalls else 0.0
        mean_mrr = sum(in_domain_mrrs) / len(in_domain_mrrs) if in_domain_mrrs else 0.0
        mean_hit = sum(in_domain_hit_rates) / len(in_domain_hit_rates) if in_domain_hit_rates else 0.0
        mean_kw = sum(keyword_overlaps) / len(keyword_overlaps) if keyword_overlaps else 0.0
        ood_rate = (ood_refusals / ood_total) if ood_total > 0 else 1.0

        return BenchmarkReport(
            total_queries=len(qa_items),
            in_domain_queries=in_domain_total,
            out_of_domain_queries=ood_total,
            mean_precision_at_k=round(mean_p, 4),
            mean_recall_at_k=round(mean_r, 4),
            mean_reciprocal_rank=round(mean_mrr, 4),
            mean_hit_rate_at_k=round(mean_hit, 4),
            mean_keyword_overlap=round(mean_kw, 4),
            out_of_domain_refusal_rate=round(ood_rate, 4),
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            detailed_results=detailed_results,
        )
