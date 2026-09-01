from __future__ import annotations

from typing import Optional
from src.config import settings
from src.core.exceptions import ConfigurationError
from src.vector_store.base import BaseVectorStore
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.faiss_store import FAISSVectorStore


def get_vector_store(
    store_type: Optional[str] = None,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> BaseVectorStore:
    """Factory to retrieve configured vector store instance."""
    selected_type = (store_type or settings.VECTOR_STORE_TYPE).lower().strip()

    if selected_type == "chroma":
        try:
            return ChromaVectorStore(
                persist_directory=persist_dir or settings.CHROMA_PERSIST_DIR,
                collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
            )
        except Exception as e:
            print(f"[Warning] Chroma store init failed ({e}), falling back to FAISS.")
            return FAISSVectorStore(
                index_dir=persist_dir or settings.FAISS_INDEX_DIR,
                dimension=dimension or settings.EMBEDDING_DIMENSION,
            )

    elif selected_type == "faiss":
        return FAISSVectorStore(
            index_dir=persist_dir or settings.FAISS_INDEX_DIR,
            dimension=dimension or settings.EMBEDDING_DIMENSION,
        )

    else:
        raise ConfigurationError(
            f"Unsupported vector store type: '{selected_type}'. Allowed: chroma, faiss"
        )
