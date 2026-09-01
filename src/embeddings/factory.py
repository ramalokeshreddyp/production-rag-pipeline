from __future__ import annotations

import os
from typing import Optional
from src.config import settings
from src.core.exceptions import ConfigurationError
from src.embeddings.base import BaseEmbeddingModel
from src.embeddings.mock_embed import MockEmbeddingModel
from src.embeddings.openai_embed import OpenAIEmbeddingModel


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseEmbeddingModel:
    """Factory function returning the configured embedding model instance."""
    selected_provider = (provider or settings.EMBEDDING_PROVIDER).lower()

    if selected_provider == "openai":
        key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not key or key == "your-openai-api-key-here":
            # Fallback to mock embedding if API key is not configured
            print(
                "[Warning] OpenAI API key not provided or invalid. Falling back to MockEmbeddingModel for zero-dependency execution."
            )
            return MockEmbeddingModel()
        return OpenAIEmbeddingModel(
            model_name=model_name or settings.EMBEDDING_MODEL,
            api_key=key,
        )

    elif selected_provider in {"huggingface", "hf", "sentence-transformers"}:
        try:
            from src.embeddings.hf_embed import SentenceTransformerEmbeddingModel
            return SentenceTransformerEmbeddingModel(
                model_name=model_name or settings.HF_EMBEDDING_MODEL
            )
        except Exception as e:
            print(
                f"[Warning] HuggingFace embedding initialization failed ({e}). Falling back to MockEmbeddingModel."
            )
            return MockEmbeddingModel()

    elif selected_provider == "mock":
        return MockEmbeddingModel()

    else:
        raise ConfigurationError(
            f"Unsupported embedding provider: '{selected_provider}'. Allowed: openai, huggingface, mock"
        )
