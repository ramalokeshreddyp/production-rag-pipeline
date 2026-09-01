from __future__ import annotations

import hashlib
import math
from typing import List
from src.embeddings.base import BaseEmbeddingModel


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, zero-dependency embedding model for offline testing and development.
    Uses n-gram character hashing with L2-normalization to create high-dimensional dense vectors."""

    def __init__(self, dimension: int = 128, model_name: str = "mock-embedding-v1"):
        self._dimension = dimension
        self._model_name = model_name

    def _hash_text_to_vector(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dimension

        vector = [0.0] * self._dimension
        words = text.lower().split()
        
        # Word hashing
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            pos = h % self._dimension
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vector[pos] += sign * (1.0 / math.sqrt(i + 1))

        # Character bigram hashing for sub-word features
        cleaned = "".join(text.lower().split())
        for i in range(len(cleaned) - 1):
            gram = cleaned[i : i + 2]
            h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16)
            pos = h % self._dimension
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vector[pos] += 0.3 * sign

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        else:
            vector[0] = 1.0

        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text_to_vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
