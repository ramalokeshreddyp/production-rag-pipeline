from __future__ import annotations

import os
from typing import List, Optional
from openai import OpenAI

from src.core.exceptions import EmbeddingError
from src.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Generates embeddings using OpenAI's API (e.g. text-embedding-3-small)."""

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is required to initialize OpenAIEmbeddingModel."
            )
        self.client = OpenAI(api_key=self.api_key)
        self._dimensions = dimensions or self.MODEL_DIMENSIONS.get(model_name, 1536)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            # Batch in sizes of 100 to avoid request size limits
            batch_size = 100
            embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Replace empty strings with a single whitespace to avoid API errors
                batch = [t if t.strip() else " " for t in batch]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self._model_name,
                )
                embeddings.extend([item.embedding for item in response.data])
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding generation failed: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            text = " "
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self._model_name,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI query embedding generation failed: {str(e)}") from e

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name
