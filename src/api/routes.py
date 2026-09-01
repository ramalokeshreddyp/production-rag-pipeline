from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.core.types import IngestResponse, RAGResponse
from src.evaluation.benchmark import BenchmarkReport, BenchmarkRunner
from src.generation.rag_engine import RAGEngine

router = APIRouter()

# Global RAG engine instance managed by lifespan / dependency
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question string", json_schema_extra={"example": "What are the key limitations of LLMs?"})
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve")
    retrieval_mode: Optional[str] = Field(default="hybrid", description="dense, bm25, or hybrid")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class DirectoryIngestRequest(BaseModel):
    directory_path: str = Field(default="./data/sample_docs", description="Directory path containing documents to ingest")
    recursive: bool = Field(default=True, description="Whether to scan subdirectories recursively")


class HealthResponse(BaseModel):
    status: str
    total_indexed_chunks: int
    embedding_model: str
    llm_model: str
    retrieval_mode: str


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns the operational health and configuration state of the RAG pipeline."""
    engine = get_rag_engine()
    return HealthResponse(
        status="healthy",
        total_indexed_chunks=engine.vector_store.count(),
        embedding_model=engine.embedding_model.model_name,
        llm_model=engine.llm_client.model_name,
        retrieval_mode=engine.retrieval_mode,
    )


@router.post("/api/v1/query", response_model=RAGResponse, tags=["Query"])
async def query_pipeline(request: QueryRequest):
    """Executes full RAG query with dense/sparse/hybrid retrieval, guarded prompt, and citations."""
    engine = get_rag_engine()
    try:
        response = engine.query(
            query_text=request.query,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            temperature=request.temperature or 0.0,
            filters=request.filters,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/api/v1/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_directory(request: DirectoryIngestRequest):
    """Ingests all supported documents from a local directory."""
    engine = get_rag_engine()
    try:
        return engine.ingest_directory(request.directory_path, recursive=request.recursive)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory ingestion failed: {str(e)}")


@router.post("/api/v1/ingest/files", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_files(files: List[UploadFile] = File(...)):
    """Uploads and ingests multiple document files (PDF, TXT, MD, DOCX, HTML)."""
    engine = get_rag_engine()
    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    saved_paths: List[str] = []

    try:
        for file in files:
            file_path = Path(temp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(file_path))

        response = engine.ingest_files(saved_paths)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload ingestion failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/api/v1/documents", tags=["Knowledge Base"])
async def list_documents():
    """Lists all chunks and documents currently indexed in the vector database."""
    engine = get_rag_engine()
    chunks = engine.vector_store.get_all_chunks()
    
    # Aggregate by document
    docs_summary: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        doc_name = chunk.metadata.get("filename") or chunk.metadata.get("source") or chunk.document_id
        if doc_name not in docs_summary:
            docs_summary[doc_name] = {
                "document_name": doc_name,
                "document_id": chunk.document_id,
                "chunk_count": 0,
                "total_tokens": 0,
                "file_type": chunk.metadata.get("file_type", "unknown"),
            }
        docs_summary[doc_name]["chunk_count"] += 1
        docs_summary[doc_name]["total_tokens"] += chunk.token_count

    return {
        "total_documents": len(docs_summary),
        "total_chunks": len(chunks),
        "documents": list(docs_summary.values()),
    }


@router.get("/api/v1/evaluate", response_model=BenchmarkReport, tags=["Evaluation"])
async def evaluate_pipeline(
    dataset_path: str = Query(default="./data/evaluation/golden_qa_dataset.json"),
    top_k: int = Query(default=4, ge=1, le=10),
    retrieval_mode: str = Query(default="hybrid"),
):
    """Runs automated benchmark evaluation on the golden evaluation dataset."""
    engine = get_rag_engine()
    try:
        runner = BenchmarkRunner(engine)
        return runner.run_benchmark(dataset_path=dataset_path, top_k=top_k, retrieval_mode=retrieval_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark evaluation failed: {str(e)}")


@router.post("/api/v1/clear", tags=["Knowledge Base"])
async def clear_knowledge_base():
    """Clears all indexed documents and resets vector databases."""
    engine = get_rag_engine()
    engine.clear()
    return {"message": "Knowledge base cleared successfully.", "total_chunks": 0}
