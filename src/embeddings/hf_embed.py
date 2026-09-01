from __future__ import annotations

from typing import List, Optional
from src.core.exceptions import EmbeddingError
from src.embeddings.base import BaseEmbeddingModel


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """Generates embeddings locally using SentenceTransformers (Hugging Face)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is required. Install via `pip install sentence-transformers`."
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load SentenceTransformer model '{model_name}': {str(e)}"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"HuggingFace document embedding failed: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"HuggingFace query embedding failed: {str(e)}") from e

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
