from __future__ import annotations

from typing import List, Tuple
from src.core.types import SearchResult


SYSTEM_PROMPT = """You are a helpful and precise assistant designed to answer questions accurately based ONLY on the provided context.

Guidelines:
1. Use ONLY the information provided in the 'Context' section below.
2. Do not extrapolate, speculate, or use any external knowledge you might have.
3. If the answer to the question cannot be found within the provided context, you must respond with the exact phrase: "I don't know."
4. When answering, formulate a clear, concise, and direct response grounded strictly in the facts from the context.
"""


def build_rag_prompt(query: str, retrieved_chunks: List[SearchResult]) -> Tuple[str, str]:
    """Formats the system prompt and user prompt including context chunks with source citations."""
    if not retrieved_chunks:
        context_str = "No relevant context found."
    else:
        context_blocks = []
        for i, res in enumerate(retrieved_chunks, start=1):
            chunk = res.chunk
            source_file = chunk.metadata.get("filename") or chunk.metadata.get("source") or "unknown_source"
            chunk_idx = chunk.chunk_index
            source_info = f"{source_file} (Chunk #{chunk_idx}, Similarity: {res.score:.3f})"
            
            block = (
                f"[CHUNK {i}]:\n"
                f"{chunk.content}\n"
                f"Source: {source_info}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

    user_prompt = f"""Context:
---
{context_str}
---

Question: {query}

Answer:"""

    return SYSTEM_PROMPT, user_prompt
