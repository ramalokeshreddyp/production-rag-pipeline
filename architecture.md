# 🏛️ System Architecture & Engineering Design

## 📌 1. Executive Summary & Objective

The **Production-Ready Retrieval-Augmented Generation (RAG) Pipeline** is an enterprise-grade AI architecture designed to solve the two foundational shortcomings of static Large Language Models (LLMs):
1. **Knowledge Latency**: Inability to reflect proprietary or real-time data beyond the model's pre-training cutoff.
2. **Hallucination Risk**: Generation of plausible yet factually incorrect statements when answering specialized domain queries.

By decoupling the language model's parametric memory from non-parametric external vector storage, the system ensures verifiable, deterministic, and context-grounded responses with verbatim source citations.

---

## 🏗️ 2. High-Level System Architecture

```mermaid
flowchart TB
    subgraph INGESTION["📥 Phase 1: Ingestion & Indexing Pipeline (Batch / Offline)"]
        direction TB
        RAW["📄 Raw Corpus (PDF, TXT, MD, HTML, DOCX)"]
        LOADER["🧹 Document Loader & Text Sanitizer\n(Unicode NFKC, Tag Stripper, Whitespace Normalizer)"]
        CHUNKER{"✂️ Chunking Engine\n(Sentence-Aware / Fixed / Paragraph / Sliding)"}
        
        subgraph INDEXING["Dual Indexing Tier"]
            BM25_IDX["🔤 Inverted BM25 Index\n(Term Frequencies, Doc Frequencies, Avg Length)"]
            EMBED_MOD["🧠 Vector Embedding Engine\n(OpenAI / SentenceTransformers / Local)"]
            VEC_DB[("💾 Persistent Vector Database\n(ChromaDB / FAISS HNSW)")]
        end
        
        RAW --> LOADER
        LOADER --> CHUNKER
        CHUNKER -->|Token Chunks| BM25_IDX
        CHUNKER -->|Text Chunks| EMBED_MOD
        EMBED_MOD -->|Dense Embeddings| VEC_DB
    end

    subgraph QUERY["🔍 Phase 2: Real-Time Query & Generation Pipeline (Online)"]
        direction TB
        USER_Q["👤 User Query"]
        Q_EMBED["⚡ Query Embedding Model"]
        
        subgraph RETRIEVAL["Hybrid Retrieval & Re-ranking"]
            DENSE_SEARCH["🔎 Dense Vector Search\n(Cosine Similarity / Inner Product)"]
            SPARSE_SEARCH["🔤 Okapi BM25 Search\n(Exact Keyword Match)"]
            RRF_FUSION["⚖️ Reciprocal Rank Fusion (RRF)\nRRF(d) = Σ wm / (k + rank)"]
            RERANKER["🔬 Cross-Encoder Neural Re-ranker\n(ms-marco-MiniLM-L-6-v2)"]
        end
        
        PROMPT_BUILDER["📝 Context & Guarded Prompt Builder\n(Strict Anti-Hallucination Template)"]
        LLM_GEN["🤖 Large Language Model Engine\n(GPT-4o-mini / Local / Fallback)"]
        OUTPUT["📋 Final Grounded Response + Source Citations"]
        
        USER_Q --> Q_EMBED
        USER_Q --> SPARSE_SEARCH
        Q_EMBED --> DENSE_SEARCH
        DENSE_SEARCH -->|Top-N Dense| RRF_FUSION
        SPARSE_SEARCH -->|Top-N Sparse| RRF_FUSION
        RRF_FUSION -->|Fused Candidates| RERANKER
        RERANKER -->|Calibrated Top-K| PROMPT_BUILDER
        USER_Q --> PROMPT_BUILDER
        PROMPT_BUILDER --> LLM_GEN
        LLM_GEN --> OUTPUT
    end

    INGESTION -.->|Indexed Chunks & Embeddings| RETRIEVAL
```

---

## ⚙️ 3. Detailed Data Flow & Execution Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / UI / API
    participant API as FastAPI Router
    participant Engine as RAG Orchestrator
    participant BM25 as BM25 Indexer
    participant Store as Vector Store (Chroma/FAISS)
    participant Rerank as Cross-Encoder Reranker
    participant Prompt as Prompt Builder
    participant LLM as LLM Provider (OpenAI/Local)

    Client->>API: POST /api/v1/query { query, top_k, mode: "hybrid" }
    API->>Engine: query(query_text, top_k=4, mode="hybrid")
    
    par Parallel Candidate Retrieval
        Engine->>Store: Dense Vector Cosine Search (Query Vector)
        Store-->>Engine: Top-15 Dense Candidates
    and
        Engine->>BM25: Sparse Lexical Match (Query Terms)
        BM25-->>Engine: Top-15 Lexical Candidates
    end

    Engine->>Engine: Reciprocal Rank Fusion (RRF) Ranking
    Engine->>Rerank: Re-rank Top-15 Fused Candidates
    Rerank-->>Engine: Calibrated Top-4 Chunks

    Engine->>Prompt: build_rag_prompt(query, top_chunks)
    Prompt-->>Engine: (System Prompt, User Prompt + Citations)
    
    Engine->>LLM: generate(system_prompt, user_prompt)
    LLM-->>Engine: Grounded Answer (or "I don't know.")
    
    Engine->>Engine: Extract citations & compute latency
    Engine-->>API: RAGResponse (Answer, Citations, Latency, Metadata)
    API-->>Client: 200 OK JSON Response
```

---

## 🧩 4. Core Subsystems & Modular Responsibilities

### 4.1 Ingestion Subsystem (`src/ingestion/`)
- **`DocumentLoader`**: Factory that loads files based on mime/extension (`.pdf`, `.md`, `.txt`, `.html`, `.docx`).
- **`TextCleaner`**: Normalizes Unicode to NFKC format, strips HTML boilerplates and zero-width artifacts, and standardizes multi-newline spacing.
- **`BaseChunker` Hierarchy**:
  - `SentenceAwareChunker`: Grammatical sentence boundary detection using regular expression lookaheads `(?<=[.!?])\s+(?=[A-Z0-9])`.
  - `FixedSizeChunker`: Exact token/character windowing with sliding stride.
  - `ParagraphAwareChunker`: Structural newline boundary preservation with overflow splitting.
  - `SlidingWindowChunker`: Continuous overlapping token strides.

### 4.2 Embedding & Vector Database Subsystem (`src/embeddings/`, `src/vector_store/`)
- **`BaseEmbeddingModel` Protocol**: Provides unified document batching (`embed_documents`) and query encoding (`embed_query`).
  - `OpenAIEmbeddingModel`: Utilizes `text-embedding-3-small` (1536 dims) or `text-embedding-3-large` (3072 dims).
  - `SentenceTransformerEmbeddingModel`: Utilizes `all-MiniLM-L6-v2` for localized inference.
  - `MockEmbeddingModel`: Deterministic, high-dimensional hashing vectorizer for zero-dependency test pipelines.
- **`BaseVectorStore` Protocol**:
  - `ChromaVectorStore`: Persistent SQLite + HNSW cosine index.
  - `FAISSVectorStore`: High-speed vector index (`IndexFlatIP`) with JSON metadata persistence.

### 4.3 Retrieval & Ranking Subsystem (`src/retrieval/`)
- **`BM25Retriever`**: Clean Python implementation of Okapi BM25 ($k_1=1.5, b=0.75$) with Robertson-Spärck Jones IDF.
- **`DenseRetriever`**: High-dimensional cosine distance similarity.
- **`HybridRetriever`**: Fuses lexical keyword matches with semantic vectors using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
- **`CrossEncoderReranker`**: Re-ranks the top 15-20 candidates using cross-attention models (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### 4.4 Generation & Guardrails Subsystem (`src/generation/`)
- **`build_rag_prompt`**: Formats candidate chunks into `[CHUNK 1]`, `[CHUNK 2]` structures with source paths and similarity scores.
- **Strict Grounding Contract**: Prompt directives force the LLM to refuse unsupported questions using the exact fallback string: `"I don't know."`

---

## 📊 5. Architectural Trade-offs & Design Decisions

| Decision | Alternative Considered | Selected Rationale |
| :--- | :--- | :--- |
| **Hybrid Search (Dense + BM25)** | Dense Only | Dense vectors struggle with exact acronyms, serial codes, and technical keys. Hybrid search provides the highest recall across conceptual and keyword queries. |
| **Sentence-Aware Chunking** | Fixed Token Slicing | Fixed slicing splits thoughts across boundaries. Sentence-aware chunking maintains semantic integrity of facts and clauses. |
| **Dual Vector Store Adapters** | Single DB Binding | Interface abstraction (`BaseVectorStore`) enables drop-in switching between ChromaDB (cloud/persistent) and FAISS (low-latency high-throughput). |
| **Pydantic v2 Domain Models** | Plain Python Dictionaries | Type-safe runtime validation, zero-cost schema serialization, and automatic FastAPI OpenAPI specification generation. |

---

## 🔒 6. Security & Production Hardening

1. **Zero-Trust Input Sanitization**: Strips dangerous script tags and malicious control sequences before indexing.
2. **Environment Variable Secret Management**: API keys are isolated via Pydantic `BaseSettings` and `.env` files.
3. **Container Isolation**: Multi-stage Docker container running with non-root security principles and explicit volume mounts for index persistence.
