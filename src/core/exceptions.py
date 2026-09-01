"""Domain-specific exceptions for the RAG pipeline."""


class RAGException(Exception):
    """Base exception for all RAG pipeline errors."""

    pass


class DocumentLoadError(RAGException):
    """Raised when loading or reading a document fails."""

    pass


class ChunkingError(RAGException):
    """Raised when chunking document text fails."""

    pass


class EmbeddingError(RAGException):
    """Raised when vector embedding generation fails."""

    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""

    pass


class RetrievalError(RAGException):
    """Raised when retrieval operations fail."""

    pass


class GenerationError(RAGException):
    """Raised when LLM text generation fails."""

    pass


class ConfigurationError(RAGException):
    """Raised when configuration parameters are invalid or missing."""

    pass
