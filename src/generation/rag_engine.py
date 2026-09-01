from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.core.exceptions import RAGException
from src.core.types import (
    Chunk,
    Citation,
    Document,
    IngestResponse,
    RAGResponse,
    SearchResult,
)
from src.embeddings.base import BaseEmbeddingModel
from src.embeddings.factory import get_embedding_model
from src.generation.llm_client import BaseLLMClient, get_llm_client
from src.generation.prompt_templates import build_rag_prompt
from src.ingestion.chunkers import BaseChunker, get_chunker
from src.ingestion.loaders import DocumentLoader
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    PassThroughReranker,
)
from src.vector_store.base import BaseVectorStore
from src.vector_store.factory import get_vector_store


class RAGEngine:
    """Production-grade RAG Pipeline orchestrating ingestion, indexing, retrieval, and generation."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        llm_client: Optional[BaseLLMClient] = None,
        chunker: Optional[BaseChunker] = None,
        reranker: Optional[BaseReranker] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store(
            dimension=self.embedding_model.dimension
        )
        self.llm_client = llm_client or get_llm_client()
        self.chunker = chunker or get_chunker(
            strategy=settings.CHUNKING_STRATEGY,
            chunk_size=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        )
        self.document_loader = DocumentLoader()

        # Retrievers
        self.dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
        )
        self.bm25_retriever = BM25Retriever()
        self._sync_bm25_from_vector_store()

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            dense_weight=settings.HYBRID_DENSE_WEIGHT,
            bm25_weight=settings.HYBRID_BM25_WEIGHT,
        )

        if reranker:
            self.reranker = reranker
        elif settings.ENABLE_RERANKING:
            self.reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
        else:
            self.reranker = PassThroughReranker()

        self.retrieval_mode = retrieval_mode or settings.RETRIEVAL_MODE
        self.top_k = top_k or settings.TOP_K

    def _sync_bm25_from_vector_store(self) -> None:
        """Populates BM25 index with all chunks currently stored in vector store."""
        try:
            chunks = self.vector_store.get_all_chunks()
            if chunks:
                self.bm25_retriever.index_chunks(chunks)
        except Exception as e:
            print(f"[Warning] BM25 sync from vector store encountered: {e}")

    def ingest_files(self, file_paths: List[str | Path]) -> IngestResponse:
        """Ingests a list of document file paths into the pipeline."""
        start_time = time.time()
        docs: List[Document] = []
        for path in file_paths:
            doc = self.document_loader.load_file(path)
            docs.append(doc)

        return self._process_and_store_documents(docs, start_time)

    def ingest_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
    ) -> IngestResponse:
        """Ingests all supported documents found in a directory."""
        start_time = time.time()
        docs = self.document_loader.load_directory(directory_path, recursive=recursive)
        return self._process_and_store_documents(docs, start_time)

    def ingest_documents(self, documents: List[Document]) -> IngestResponse:
        """Directly ingests a list of pre-constructed Document objects."""
        start_time = time.time()
        return self._process_and_store_documents(documents, start_time)

    def _process_and_store_documents(
        self, documents: List[Document], start_time: float
    ) -> IngestResponse:
        if not documents:
            return IngestResponse(
                documents_processed=0,
                total_chunks_created=0,
                duration_seconds=time.time() - start_time,
                files_ingested=[],
                vector_store_count=self.vector_store.count(),
            )

        # 1. Chunk documents
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return IngestResponse(
                documents_processed=len(documents),
                total_chunks_created=0,
                duration_seconds=time.time() - start_time,
                files_ingested=[d.filename or d.id for d in documents],
                vector_store_count=self.vector_store.count(),
            )

        # 2. Embed chunks
        texts = [c.content for c in all_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        # 3. Store in vector database
        self.vector_store.add_chunks(all_chunks, embeddings)

        # 4. Update BM25 index
        self._sync_bm25_from_vector_store()

        duration = time.time() - start_time
        return IngestResponse(
            documents_processed=len(documents),
            total_chunks_created=len(all_chunks),
            duration_seconds=round(duration, 3),
            files_ingested=[d.filename or d.id for d in documents],
            vector_store_count=self.vector_store.count(),
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieves and re-ranks top chunks matching the user query."""
        k = top_k or self.top_k
        mode = (retrieval_mode or self.retrieval_mode).lower().strip()

        if mode == "dense":
            candidates = self.dense_retriever.retrieve(query=query, top_k=k * 2, filters=filters)
        elif mode == "bm25":
            candidates = self.bm25_retriever.retrieve(query=query, top_k=k * 2)
        elif mode == "hybrid":
            candidates = self.hybrid_retriever.retrieve(query=query, top_k=k * 2, filters=filters)
        else:
            candidates = self.hybrid_retriever.retrieve(query=query, top_k=k * 2, filters=filters)

        # Re-ranking step
        reranked = self.reranker.rerank(query=query, results=candidates, top_k=k)
        return reranked

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        temperature: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Executes full query-retrieval-generation pipeline returning answer + source citations."""
        start_time = time.time()
        retrieval_mode = retrieval_mode or self.retrieval_mode

        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retrieve(
            query=query_text,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            filters=filters,
        )

        # 2. Format guarded prompt with citations
        system_prompt, user_prompt = build_rag_prompt(
            query=query_text,
            retrieved_chunks=retrieved_chunks,
        )

        # 3. Generate answer via LLM
        answer = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        # 4. Build source citations
        citations: List[Citation] = []
        for res in retrieved_chunks:
            chunk = res.chunk
            source_doc = chunk.metadata.get("filename") or chunk.metadata.get("source") or "unknown"
            citations.append(
                Citation(
                    source_document=str(source_doc),
                    chunk_index=chunk.chunk_index,
                    score=round(res.score, 4),
                    text_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    metadata=chunk.metadata,
                )
            )

        is_grounded = "I don't know" not in answer

        latency = time.time() - start_time
        return RAGResponse(
            query=query_text,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            retrieval_mode=retrieval_mode,
            model_used=self.llm_client.model_name,
            is_grounded=is_grounded,
            latency_seconds=round(latency, 3),
        )

    def clear(self) -> None:
        """Clears all indexed chunks and data stores."""
        self.vector_store.delete()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever.bm25_retriever = self.bm25_retriever
