from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.exceptions import VectorStoreError
from src.core.types import Chunk, SearchResult
from src.vector_store.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """Persistent vector store implementation using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_knowledge_base",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # Create or get collection using cosine distance
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise VectorStoreError(
                "chromadb is required. Install via `pip install chromadb`."
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB at '{persist_directory}': {str(e)}"
            ) from e

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})."
            )

        ids: List[str] = [chunk.id for chunk in chunks]
        documents: List[str] = [chunk.content for chunk in chunks]
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            # Flatten metadata values for Chroma (convert nested dicts/lists to JSON strings or primitives)
            flat_meta: Dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                else:
                    flat_meta[k] = json.dumps(v)
            metadatas.append(flat_meta)

        try:
            # Batch upsert to ChromaDB
            batch_size = 250
            for i in range(0, len(ids), batch_size):
                end_i = i + batch_size
                self.collection.upsert(
                    ids=ids[i:end_i],
                    embeddings=embeddings[i:end_i],
                    documents=documents[i:end_i],
                    metadatas=metadatas[i:end_i],
                )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks to ChromaDB: {str(e)}") from e

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self.count() == 0:
            return []

        try:
            actual_k = min(top_k, self.count())
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": actual_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                query_kwargs["where"] = filters

            results = self.collection.query(**query_kwargs)

            search_results: List[SearchResult] = []
            if not results or not results["ids"] or not results["ids"][0]:
                return []

            ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for rank, (cid, doc_text, meta, dist) in enumerate(
                zip(ids, docs, metas, distances)
            ):
                # Convert cosine distance to similarity: similarity = 1 - distance
                similarity_score = max(0.0, 1.0 - float(dist))

                # Reconstruct chunk metadata
                recovered_meta = dict(meta or {})
                doc_id = str(recovered_meta.pop("document_id", "unknown"))
                chunk_index = int(recovered_meta.pop("chunk_index", 0))
                token_count = int(recovered_meta.pop("token_count", 0))
                char_count = int(recovered_meta.pop("char_count", len(doc_text)))

                chunk = Chunk(
                    id=cid,
                    document_id=doc_id,
                    content=doc_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    char_count=char_count,
                    metadata=recovered_meta,
                )

                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=similarity_score,
                        retrieval_method="dense_chroma",
                        rank=rank + 1,
                    )
                )

            return search_results
        except Exception as e:
            raise VectorStoreError(f"ChromaDB search query failed: {str(e)}") from e

    def delete(self, filter_criteria: Optional[Dict[str, Any]] = None) -> None:
        try:
            if filter_criteria is None:
                # Delete all
                all_ids = self.collection.get()["ids"]
                if all_ids:
                    self.collection.delete(ids=all_ids)
            else:
                self.collection.delete(where=filter_criteria)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB deletion failed: {str(e)}") from e

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            raise VectorStoreError(f"ChromaDB count failed: {str(e)}") from e

    def get_all_chunks(self) -> List[Chunk]:
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            chunks: List[Chunk] = []
            if not data or not data["ids"]:
                return []

            for cid, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                recovered_meta = dict(meta or {})
                doc_id = str(recovered_meta.pop("document_id", "unknown"))
                chunk_index = int(recovered_meta.pop("chunk_index", 0))
                token_count = int(recovered_meta.pop("token_count", 0))
                char_count = int(recovered_meta.pop("char_count", len(doc_text)))

                chunk = Chunk(
                    id=cid,
                    document_id=doc_id,
                    content=doc_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    char_count=char_count,
                    metadata=recovered_meta,
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            raise VectorStoreError(f"ChromaDB get_all_chunks failed: {str(e)}") from e

    def persist(self) -> None:
        # Chroma PersistentClient automatically commits changes
        pass
