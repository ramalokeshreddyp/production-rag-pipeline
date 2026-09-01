# Artificial Intelligence & Retrieval-Augmented Generation (RAG) Architecture Overview

## 1. Fundamentals of Large Language Models (LLMs)
Large Language Models (LLMs) are deep learning models based on the transformer architecture, trained on vast corpora of textual data. While LLMs excel at language synthesis, text generation, and reasoning, they suffer from two major limitations:
- **Knowledge Cutoff**: Their weights encode static information present only up to the time of pre-training.
- **Hallucinations**: When asked about specialized, confidential, or unfamiliar topics, LLMs may generate factually incorrect yet grammatically fluent assertions.

## 2. Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) solves these limitations by separating the model's parametric memory (neural network weights) from its non-parametric memory (an external, dynamic knowledge base). 

### 2.1 The Two Phases of RAG
1. **Ingestion Phase**:
   - Documents in various formats (PDF, Markdown, TXT, HTML) are extracted, cleaned, and parsed.
   - Text is split into semantic chunks (typically 300 to 500 tokens) with overlap (15-20%) to preserve contextual continuity across boundaries.
   - Chunks are passed through an embedding model (e.g., `text-embedding-3-small` or `all-MiniLM-L6-v2`) to produce dense vector representations.
   - The embeddings and associated metadata are indexed in a vector store (such as ChromaDB or FAISS).

2. **Query Phase**:
   - The user query is converted into an embedding using the identical embedding model.
   - A vector similarity search (often using cosine similarity) retrieves the Top-$K$ most relevant document chunks.
   - Chunks are formatted into a structured prompt containing strict grounding instructions.
   - The LLM synthesizes an answer referencing only the retrieved context and emits source citations. If the required information is absent from the context, the model explicitly responds with: "I don't know."

## 3. Hybrid Search & Re-ranking
Standard dense vector retrieval can occasionally miss exact keyword matches (e.g., specific error codes, SKU identifiers, or technical acronyms). Hybrid search combines:
- **Dense Vector Search**: Captures high-level semantic meaning and conceptual similarity.
- **Sparse BM25 Search**: Matches exact keywords and rare lexical tokens using term frequency-inverse document frequency formulas.

The candidate rankings are fused using Reciprocal Rank Fusion (RRF):
$$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
where $k$ is a smoothing constant (commonly $k=60$).

Subsequently, a Cross-Encoder Re-ranker (such as `ms-marco-MiniLM-L-6-v2`) scores the combined candidates against the query simultaneously, improving retrieval precision before prompt assembly.
