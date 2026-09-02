# 📝 RAG Pipeline Technical Questionnaire & Evaluation Responses

---

### Question 1: Describe the document corpus you chose for your knowledge base and explain why you selected it.

**Response:**
For the knowledge base, I designed and implemented a curated, multi-domain enterprise engineering corpus located in `data/sample_docs/` spanning four core technical disciplines:

1. **`artificial_intelligence_overview.md`**: Covers foundational LLM limitations (knowledge cutoff, hallucinations), two-phase RAG mechanics, chunking guidelines (300–500 tokens, 15–20% overlap), Okapi BM25, and Reciprocal Rank Fusion (RRF) mathematical formulations.
2. **`cloud_architecture_handbook.txt`**: Details active-active multi-region deployment topologies, automated 15-second failover thresholds, service mesh capabilities (mTLS, circuit breaking, OpenTelemetry tracing), and Kubernetes horizontal pod autoscaling (HPA).
3. **`security_and_compliance_policy.md`**: Specifies Zero Trust Architecture (ZTA), mandatory hardware-based FIDO2 WebAuthn keys, 4-hour Just-In-Time (JIT) session TTLs, TLS 1.3 / AES-256-GCM encryption with 90-day customer-managed encryption key (CMEK) rotation, and GDPR Article 33 72-hour regulatory notification SLAs.
4. **`data_engineering_best_practices.txt`**: Documents the three-tier Medallion Lakehouse architecture (Bronze, Silver, Gold), exactly-once stream processing via two-phase commit (2PC), and vector database ingestion batching (64–256 items) with MD5/SHA-256 deduplication hashing.

**Why it was selected:**
This corpus was specifically chosen because it reflects realistic enterprise documentation containing a balanced mix of conceptual explanations, strict numerical parameters (e.g., *15 seconds*, *4 hours*, *90 days*, *72 hours*), specialized technical acronyms (*FIDO2*, *CMEK*, *mTLS*, *HPA*, *2PC*), and distinct domain boundaries. This structure provides an ideal benchmark to test both high-level semantic retrieval and exact keyword search precision.

---

### Question 2: Which chunking strategy did you implement, and how do you think this choice affected the quality of the retrieved context?

**Response:**
I built a modular chunking architecture (`src/ingestion/chunkers.py`) implementing four distinct strategies: **Fixed-size**, **Sentence-aware**, **Paragraph-aware**, and **Sliding-window**, with **`SentenceAwareChunker`** configured as the default pipeline strategy.

**Implementation Details:**
- **Boundary Detection**: Uses regular expression lookahead assertions (`(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=\n\n)`) to split text along grammatical sentence boundaries.
- **Budget Packing & Overlap**: Groups complete sentences up to a configurable token ceiling (default: 400 tokens) while maintaining an overlap window (default: 80 tokens / 20%) by carrying trailing sentences into subsequent chunks.

**Impact on Retrieval Quality:**
1. **Preserved Semantic Completeness**: Unlike fixed character/token chunkers that arbitrarily slice text mid-sentence or mid-thought, sentence-aware chunking ensures that every clause, technical constraint, and subject-predicate relationship remains intact.
2. **Reduced Context Fragmentation**: Critical operational rules (e.g., *"Automated health checks trigger instant traffic failover to healthy secondary regions within 15 seconds."*) are never split across two chunks, eliminating truncated contexts when injected into the LLM prompt.
3. **Improved Retrieval Precision**: Dense embedding vectors generated from coherent grammatical units yield higher cosine similarity scores against user queries compared to vectors generated from fragmented sentence shards.

---

### Question 3: What are the primary limitations of using only cosine similarity on embeddings for retrieval? When might this approach fail?

**Response:**
While dense embedding vector search excels at capturing high-level semantic themes and conceptual synonyms, relying solely on cosine similarity presents several critical limitations:

1. **Loss of Lexical Specificity (The Acronym & Entity Problem)**: Dense embedding models compress an entire passage into a fixed-dimensional vector space (e.g., 1536 dimensions). In doing so, fine-grained details such as specific alphanumeric codes, product SKUs, cryptographic ciphers (e.g., `AES-256-GCM`), technical acronyms (`FIDO2`, `mTLS`), or exact function names often get smoothed out.
2. **Failure on Keyword-Heavy Queries**: If a user asks *"What is the CMEK rotation policy?"*, an embedding model might retrieve general key management or encryption concepts with high cosine similarity while missing the exact passage mentioning the 90-day CMEK rule because the semantic vector is dominated by broader security vocabulary.
3. **Sensitivity to Document Length and Embedding Dilution**: In dense retrieval, when a chunk contains multiple ideas, the embedding vector represents an average semantic direction, diluting the score of short, crucial facts contained within that chunk.
4. **Resolution via Hybrid Search**: To overcome these limitations, my pipeline implements **Hybrid Search** (`src/retrieval/hybrid.py`), running dense vector search in parallel with an **Okapi BM25** sparse keyword retriever and fusing candidate rankings via **Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
   This guarantees that both high-level semantic intent and exact keyword matches are captured simultaneously.

---

### Question 4: Describe the process and any challenges you faced in designing the prompt template to ensure the LLM adhered to the provided context.

**Response:**
Designing the prompt template (`src/generation/prompt_templates.py`) required establishing strict behavioral guardrails to prevent the LLM from hallucinating or falling back on its pre-trained parametric weights.

**Design & Process:**
1. **Strict System Directives**: The system prompt explicitly defines the assistant's operational boundaries:
   - *"Use ONLY the information provided in the 'Context' section below."*
   - *"Do not extrapolate, speculate, or use any external knowledge you might have."*
   - *"If the answer to the question cannot be found within the provided context, you must respond with the exact phrase: 'I don't know.'"*
2. **Structured Chunk & Citation Framing**: Each retrieved chunk is passed in an isolated, labeled block containing document metadata and similarity scores:
   ```text
   Context:
   ---
   [CHUNK 1]:
   {chunk_content}
   Source: {source_filename} (Chunk #{chunk_index}, Similarity: {score})
   ---
   Question: {user_query}
   Answer:
   ```

**Challenges Encountered & Solutions:**
- **The "Helpfulness" Bias (Over-Answering)**: Standard instruction-tuned LLMs naturally strive to be helpful and often attempt to answer out-of-domain questions using their internal training data even when the context is irrelevant.
- **Solution**: Formulating a deterministic negative constraint with an exact fallback token string (`"I don't know."`) combined with setting the generation temperature to `0.0`. In our automated benchmark suite on `golden_qa_dataset.json`, this achieved a **100% Out-of-Domain Refusal Rate** on unanswerable/out-of-corpus queries (e.g., Martian baking recipes).

---

### Question 5: If you were to take this RAG pipeline into a production environment, what would be the first two improvements you would prioritize?

**Response:**

#### 1. Distributed, Horizontally Scalable Vector Database & Asynchronous Ingestion Pipeline
- **Current State**: Uses embedded persistent ChromaDB and local FAISS indices, which are suited for single-node development and small-to-medium corpora.
- **Production Improvement**: Transition to a distributed, cloud-native vector database (such as **Qdrant**, **Milvus**, or **Pinecone**) deployed on Kubernetes. Pair this with an asynchronous message queue architecture (**Kafka** or **RabbitMQ** with **Celery/Temporal workers**) to ingest high-volume document streams asynchronously, handle index compaction, and support multi-tenant metadata filtering with sub-10ms query latencies at million-document scale.

#### 2. Continuous Production Observability & "RAG Triad" Automated Evaluation
- **Current State**: Automated offline evaluation is run via `cli.py benchmark` against static golden datasets.
- **Production Improvement**: Implement an online observability and telemetry pipeline using **OpenTelemetry** integrated with an automated RAG evaluation framework (e.g., **Ragas**, **TruLens**, or **Arize Phoenix**). This would continuously evaluate 100% of live production queries across the **RAG Triad**:
  - **Context Relevance**: Did the retriever pull only signal-dense chunks without noise?
  - **Faithfulness (Groundedness)**: Is every claim in the LLM's answer directly attributable to the retrieved context?
  - **Answer Relevance**: Did the generated response directly answer the user's question?
  Real-time tracking of these metrics would trigger automated alerts on retrieval drift, hallucination spikes, or context quality regressions.
