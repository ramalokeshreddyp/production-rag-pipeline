from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.exceptions import VectorStoreError
from src.core.types import Chunk, SearchResult
from src.vector_store.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """Vector database implementation using FAISS index with disk serialization."""

    def __init__(
        self,
        index_dir: str = "./data/faiss_index",
        dimension: int = 1536,
    ):
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "chunks_metadata.json"

        self.chunks_map: Dict[int, Dict[str, Any]] = {}
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: Dict[int, str] = {}
        self.faiss_available = False
        self.index = None
        self.numpy_embeddings: List[np.ndarray] = []

        self._init_backend()
        self._load()

    def _init_backend(self) -> None:
        try:
            import faiss
            self.faiss = faiss
            self.faiss_available = True
            # Use IndexFlatIP for normalized cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            self.faiss_available = False
            self.index = None

    def _load(self) -> None:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks_map = {int(k): v for k, v in data.get("chunks", {}).items()}
                    self.id_to_int = data.get("id_to_int", {})
                    self.int_to_id = {int(k): v for k, v in data.get("int_to_id", {}).items()}
            except Exception as e:
                print(f"[Warning] Failed to load FAISS metadata: {e}")

        if self.faiss_available and self.index_path.exists():
            try:
                self.index = self.faiss.read_index(str(self.index_path))
            except Exception as e:
                print(f"[Warning] Failed to read FAISS index: {e}")

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Mismatch between chunk count and embedding count.")

        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity (Inner Product on L2 normalized vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        if self.faiss_available:
            if self.index is None or self.index.d != vectors.shape[1]:
                self.dimension = vectors.shape[1]
                self.index = self.faiss.IndexFlatIP(self.dimension)

        start_idx = len(self.chunks_map)
        for i, chunk in enumerate(chunks):
            current_int_id = start_idx + i
            self.chunks_map[current_int_id] = {
                "chunk": chunk.model_dump(),
                "embedding": vectors[i].tolist(),
            }
            self.id_to_int[chunk.id] = current_int_id
            self.int_to_id[current_int_id] = chunk.id

        if self.faiss_available:
            self.index.add(vectors)
        else:
            self.numpy_embeddings.extend([v for v in vectors])

        self.persist()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self.count() == 0:
            return []

        q_vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        actual_k = min(top_k, self.count())
        search_results: List[SearchResult] = []

        if self.faiss_available and self.index is not None:
            distances, indices = self.index.search(q_vec, actual_k)
            for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx not in self.chunks_map:
                    continue
                chunk_data = self.chunks_map[idx]["chunk"]
                chunk = Chunk(**chunk_data)
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=float(score),
                        retrieval_method="dense_faiss",
                        rank=rank + 1,
                    )
                )
        else:
            # Fallback pure NumPy cosine similarity
            all_indices = list(self.chunks_map.keys())
            matrix = np.array(
                [self.chunks_map[idx]["embedding"] for idx in all_indices],
                dtype=np.float32,
            )
            scores = np.dot(matrix, q_vec[0])
            sorted_order = np.argsort(-scores)[:actual_k]
            for rank, order_idx in enumerate(sorted_order):
                int_id = all_indices[order_idx]
                chunk_data = self.chunks_map[int_id]["chunk"]
                chunk = Chunk(**chunk_data)
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=float(scores[order_idx]),
                        retrieval_method="dense_faiss_numpy",
                        rank=rank + 1,
                    )
                )

        return search_results

    def delete(self, filter_criteria: Optional[Dict[str, Any]] = None) -> None:
        self.chunks_map.clear()
        self.id_to_int.clear()
        self.int_to_id.clear()
        self.numpy_embeddings.clear()
        self._init_backend()
        self.persist()

    def count(self) -> int:
        return len(self.chunks_map)

    def get_all_chunks(self) -> List[Chunk]:
        return [Chunk(**data["chunk"]) for data in self.chunks_map.values()]

    def persist(self) -> None:
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chunks": self.chunks_map,
                        "id_to_int": self.id_to_int,
                        "int_to_id": self.int_to_id,
                    },
                    f,
                    indent=2,
                )
            if self.faiss_available and self.index is not None:
                self.faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            raise VectorStoreError(f"Failed to persist FAISS index: {str(e)}") from e
