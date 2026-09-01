from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates an embedding vector for a single query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated embedding vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name or identifier of the model."""
        pass
