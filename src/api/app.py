from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import get_rag_engine, router
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize RAG engine and preload index."""
    engine = get_rag_engine()
    print(
        f"[INFO] Initialized RAGEngine | Store: {settings.VECTOR_STORE_TYPE} ({engine.vector_store.count()} chunks) | Embed: {engine.embedding_model.model_name} | LLM: {engine.llm_client.model_name}"
    )
    yield
    print("[INFO] Shutting down RAG API service.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="Production-Ready RAG Pipeline API",
        description=(
            "An enterprise-grade Retrieval-Augmented Generation (RAG) system built from scratch "
            "featuring multi-format ingestion, sentence-aware chunking, hybrid search (BM25 + Dense), "
            "cross-encoder reranking, anti-hallucination prompts, and exact source citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
