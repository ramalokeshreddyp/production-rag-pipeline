from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    UI_PORT: int = 8501

    # Embedding Settings
    EMBEDDING_PROVIDER: Literal["openai", "huggingface", "mock"] = "openai"
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 1536

    # Vector Store Settings
    VECTOR_STORE_TYPE: Literal["chroma", "faiss"] = "chroma"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_knowledge_base"
    FAISS_INDEX_DIR: str = "./data/faiss_index"

    # Chunking Strategy
    CHUNKING_STRATEGY: Literal[
        "fixed", "sentence_aware", "paragraph_aware", "sliding_window"
    ] = "sentence_aware"
    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 80

    # Retrieval & Search Settings
    TOP_K: int = 4
    RETRIEVAL_MODE: Literal["dense", "bm25", "hybrid"] = "hybrid"
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_BM25_WEIGHT: float = 0.4
    ENABLE_RERANKING: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 15

    # LLM Settings
    LLM_PROVIDER: Literal["openai", "huggingface", "mock"] = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024


settings = Settings()
