from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.core.types import Chunk, SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for persistent vector databases."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """Inserts text chunks and their corresponding embedding vectors."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Performs vector similarity search and returns Top-K scored search results."""
        pass

    @abstractmethod
    def delete(self, filter_criteria: Optional[Dict[str, Any]] = None) -> None:
        """Deletes chunks matching criteria or clears the store if filter_criteria is None."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of chunks currently stored in index."""
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Retrieves all indexed chunks."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Ensures in-memory state is flushed to persistent storage."""
        pass
