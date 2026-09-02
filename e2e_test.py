from __future__ import annotations

import sys
from pathlib import Path
from src.generation.rag_engine import RAGEngine
from src.evaluation.benchmark import BenchmarkRunner
from src.api.app import app
from fastapi.testclient import TestClient

def run_e2e():
    print("=" * 70)
    print("PRODUCTION RAG PIPELINE END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Pipeline Initialization
    print("\n[STEP 1/6] Initializing RAG Pipeline Engine...")
    engine = RAGEngine(retrieval_mode="hybrid", top_k=4)
    print(f"  [+] Vector Store : {engine.vector_store.__class__.__name__}")
    print(f"  [+] Embedding    : {engine.embedding_model.model_name} (dim: {engine.embedding_model.dimension})")
    print(f"  [+] LLM Client   : {engine.llm_client.model_name}")
    print(f"  [+] Chunker      : {engine.chunker.__class__.__name__}")

    # 2. Ingestion Phase
    print("\n[STEP 2/6] Ingesting Document Corpus from ./data/sample_docs ...")
    ingest_res = engine.ingest_directory("./data/sample_docs")
    print(f"  [+] Documents Processed : {ingest_res.documents_processed}")
    print(f"  [+] Total Chunks        : {ingest_res.total_chunks_created}")
    print(f"  [+] Ingestion Duration  : {ingest_res.duration_seconds:.3f}s")
    print(f"  [+] Ingested Files      : {', '.join(ingest_res.files_ingested)}")

    # 3. Retrieval Comparison (Dense vs BM25 vs Hybrid)
    print("\n[STEP 3/6] Testing Query Phase with Multiple Retrieval Modes...")
    test_query = "What are the two major limitations of Large Language Models described in the AI overview?"
    print(f"  Query: \"{test_query}\"")

    for mode in ["dense", "bm25", "hybrid"]:
        res = engine.query(test_query, retrieval_mode=mode, top_k=3)
        print(f"\n  >> Mode [{mode.upper()}]:")
        print(f"     Answer: {res.answer}")
        print(f"     Latency: {res.latency_seconds:.4f}s")
        print(f"     Citations ({len(res.citations)} chunks):")
        for i, c in enumerate(res.citations, 1):
            print(f"       [{i}] {c.source_document} (Chunk #{c.chunk_index}, Score: {c.score:.3f})")

    # 4. Strict Grounding and Out-of-Domain Refusal Check
    print("\n[STEP 4/6] Verifying Anti-Hallucination Guardrails & Refusal Behavior...")
    ood_query = "What is the secret recipe for baking Martian chocolate cookies?"
    print(f"  OOD Query: \"{ood_query}\"")
    ood_res = engine.query(ood_query, retrieval_mode="hybrid")
    print(f"  Model Response : \"{ood_res.answer}\"")
    assert ood_res.answer == "I don't know.", f"Refusal check failed, received: {ood_res.answer}"
    print("  [SUCCESS] Out-of-domain query refused strictly with 'I don\\'t know.'")

    # 5. Golden QA Benchmark Suite
    print("\n[STEP 5/6] Executing Automated IR Benchmark Evaluation...")
    runner = BenchmarkRunner(engine)
    report = runner.run_benchmark(
        dataset_path="./data/evaluation/golden_qa_dataset.json",
        top_k=4,
        retrieval_mode="hybrid",
    )
    print(f"  [+] Total Benchmark Queries : {report.total_queries}")
    print(f"  [+] In-Domain Queries       : {report.in_domain_queries}")
    print(f"  [+] Out-of-Domain Queries   : {report.out_of_domain_queries}")
    print(f"  [+] Retrieval Precision @ K : {report.mean_precision_at_k * 100:.2f}%")
    print(f"  [+] Retrieval Recall @ K    : {report.mean_recall_at_k * 100:.2f}%")
    print(f"  [+] Hit Rate @ K            : {report.mean_hit_rate_at_k * 100:.2f}%")
    print(f"  [+] Mean Reciprocal Rank    : {report.mean_reciprocal_rank:.4f}")
    print(f"  [+] Mean Keyword Overlap    : {report.mean_keyword_overlap * 100:.2f}%")
    print(f"  [+] OOD Refusal Rate        : {report.out_of_domain_refusal_rate * 100:.2f}%")

    # 6. REST API Verification via TestClient
    print("\n[STEP 6/6] Testing FastAPI REST Endpoints...")
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        print(f"  [+] GET /health -> status: {health_resp.json()['status']}")

        query_resp = client.post(
            "/api/v1/query",
            json={"query": "How fast does automated failover occur in multi-region deployment?", "top_k": 2},
        )
        assert query_resp.status_code == 200
        data = query_resp.json()
        print(f"  [+] POST /api/v1/query -> status: {query_resp.status_code}, answer: \"{data['answer'][:70]}...\"")
        print(f"      Citations: {len(data['citations'])} source chunks attached")

        docs_resp = client.get("/api/v1/documents")
        assert docs_resp.status_code == 200
        print(f"  [+] GET /api/v1/documents -> indexed docs: {docs_resp.json()['total_documents']}, chunks: {docs_resp.json()['total_chunks']}")

    print("\n" + "=" * 70)
    print("[ALL PASS] ALL END-TO-END RAG PIPELINE CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_e2e()
