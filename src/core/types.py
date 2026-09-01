from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a raw or parsed source document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_path: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None


class Chunk(BaseModel):
    """Represents a segmented portion of a document ready for embedding."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    token_count: int = 0
    char_count: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Represents a retrieved chunk with similarity score and metadata."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"
    rank: Optional[int] = None


class Citation(BaseModel):
    """Represents a source citation returned with an answer."""

    source_document: str
    chunk_index: int
    score: float
    text_snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Full response from the RAG query pipeline."""

    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[SearchResult]
    retrieval_mode: str
    model_used: str
    is_grounded: bool = True
    latency_seconds: float = 0.0


class EvaluationResult(BaseModel):
    """Benchmark evaluation metrics for a query or dataset."""

    query_id: Optional[str] = None
    query: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    retrieved_count: int
    ground_truth_found: bool
    notes: Optional[str] = None


class IngestResponse(BaseModel):
    """Response returned after an ingestion operation."""

    documents_processed: int
    total_chunks_created: int
    duration_seconds: float
    files_ingested: List[str]
    vector_store_count: int
