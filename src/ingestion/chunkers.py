from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.exceptions import ChunkingError
from src.core.types import Chunk, Document

# Optional tiktoken integration for accurate token measurement
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken if available, else estimates based on whitespace/words."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Heuristic fallback: ~0.75 words per token (1 token ~= 4 characters)
    return max(1, len(text.split()) * 4 // 3)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Splits a document into a list of Chunk objects."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Convenience method to chunk a collection of documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """Splits text into chunks of fixed character or token length with exact sliding overlap."""

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        # We operate on words to avoid splitting inside single words
        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        index = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if not chunk_text:
                continue

            chunk = Chunk(
                document_id=document.id,
                content=chunk_text,
                chunk_index=index,
                token_count=count_tokens(chunk_text),
                char_count=len(chunk_text),
                metadata={
                    **document.metadata,
                    "chunk_strategy": "fixed",
                    "source_doc_id": document.id,
                    "filename": document.filename or "unknown",
                },
            )
            chunks.append(chunk)
            index += 1

            if i + self.chunk_size >= len(words):
                break

        return chunks


class SentenceAwareChunker(BaseChunker):
    """Splits text along grammatical sentence boundaries, packing sentences up to chunk_size tokens."""

    SENTENCE_SPLIT_REGEX = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=\n\n)"
    )

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        # Split into raw sentences
        raw_sentences = [
            s.strip()
            for s in self.SENTENCE_SPLIT_REGEX.split(text)
            if s.strip()
        ]
        if not raw_sentences:
            raw_sentences = [text]

        chunks: List[Chunk] = []
        current_chunk_sentences: List[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in raw_sentences:
            sent_tokens = count_tokens(sentence)

            # If a single sentence is larger than chunk_size, split it using fixed size
            if sent_tokens > self.chunk_size:
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(
                        self._create_chunk(document, chunk_text, chunk_index)
                    )
                    chunk_index += 1
                    current_chunk_sentences = []
                    current_tokens = 0

                # Split the large sentence
                sub_chunker = FixedSizeChunker(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                temp_doc = Document(
                    content=sentence,
                    metadata=document.metadata,
                    filename=document.filename,
                )
                for sub_chunk in sub_chunker.chunk(temp_doc):
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.document_id = document.id
                    chunks.append(sub_chunk)
                    chunk_index += 1
                continue

            if current_tokens + sent_tokens > self.chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    self._create_chunk(document, chunk_text, chunk_index)
                )
                chunk_index += 1

                # Calculate overlap sentences
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk_sentences):
                    s_tok = count_tokens(s)
                    if overlap_tokens + s_tok <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tok
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(count_tokens(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sent_tokens

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk(document, chunk_text, chunk_index))

        return chunks

    def _create_chunk(self, doc: Document, text: str, index: int) -> Chunk:
        return Chunk(
            document_id=doc.id,
            content=text,
            chunk_index=index,
            token_count=count_tokens(text),
            char_count=len(text),
            metadata={
                **doc.metadata,
                "chunk_strategy": "sentence_aware",
                "source_doc_id": doc.id,
                "filename": doc.filename or "unknown",
            },
        )


class ParagraphAwareChunker(BaseChunker):
    """Splits text on double newlines / paragraph breaks while merging small paragraphs."""

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            return []

        chunks: List[Chunk] = []
        current_paragraphs: List[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in raw_paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        token_count=count_tokens(chunk_text),
                        char_count=len(chunk_text),
                        metadata={
                            **document.metadata,
                            "chunk_strategy": "paragraph_aware",
                            "source_doc_id": document.id,
                            "filename": document.filename or "unknown",
                        },
                    )
                )
                chunk_index += 1
                current_paragraphs = [para]
                current_tokens = para_tokens
            else:
                current_paragraphs.append(para)
                current_tokens += para_tokens

        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    token_count=count_tokens(chunk_text),
                    char_count=len(chunk_text),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "paragraph_aware",
                        "source_doc_id": document.id,
                        "filename": document.filename or "unknown",
                    },
                )
            )

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Creates a sliding window over tokens/characters with fixed step stride."""

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        words = text.split()
        chunks: List[Chunk] = []
        chunk_index = 0

        for i in range(0, len(words), step):
            window_words = words[i : i + self.chunk_size]
            window_text = " ".join(window_words)
            if not window_text:
                continue

            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=window_text,
                    chunk_index=chunk_index,
                    token_count=count_tokens(window_text),
                    char_count=len(window_text),
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "sliding_window",
                        "source_doc_id": document.id,
                        "filename": document.filename or "unknown",
                    },
                )
            )
            chunk_index += 1
            if i + self.chunk_size >= len(words):
                break

        return chunks


def get_chunker(
    strategy: str = "sentence_aware",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> BaseChunker:
    """Factory to retrieve a chunker by strategy name."""
    strategy = strategy.lower().strip()
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy in {"sentence", "sentence_aware"}:
        return SentenceAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy in {"paragraph", "paragraph_aware"}:
        return ParagraphAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy in {"sliding", "sliding_window"}:
        return SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ChunkingError(
            f"Unknown chunking strategy '{strategy}'. Supported: fixed, sentence_aware, paragraph_aware, sliding_window"
        )
