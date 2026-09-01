from __future__ import annotations

import json
import time
from pathlib import Path
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Initializes and caches the RAG Engine instance."""
    from src.generation.rag_engine import RAGEngine
    engine = RAGEngine()
    # Auto-ingest sample docs if store is empty
    if engine.vector_store.count() == 0:
        sample_dir = Path("./data/sample_docs")
        if sample_dir.exists():
            engine.ingest_directory(sample_dir)
    return engine


engine = get_engine()

# Sidebar configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649595.png", width=64)
    st.title("RAG Controls")
    st.markdown("---")

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Hybrid combines dense vectors with BM25 keyword matching using Reciprocal Rank Fusion (RRF).",
    )

    top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=4)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown("---")
    st.subheader("System Status")
    total_chunks = engine.vector_store.count()
    st.metric(label="Total Indexed Chunks", value=total_chunks)
    st.info(
        f"**Embedding:** `{engine.embedding_model.model_name}`\n\n"
        f"**Vector Store:** `{engine.vector_store.__class__.__name__}`\n\n"
        f"**LLM:** `{engine.llm_client.model_name}`"
    )

# Main Navigation Tabs
tab_query, tab_docs, tab_eval = st.tabs([
    "💬 Query Playground",
    "📚 Document Ingestion",
    "📊 Benchmark Evaluation",
])

# -----------------------------------------------------------------------------
# TAB 1: Query Playground
# -----------------------------------------------------------------------------
with tab_query:
    st.markdown('<div class="main-header">Knowledge Base Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions grounded strictly in your proprietary documentation.</div>', unsafe_allow_html=True)

    # Sample preset queries
    st.markdown("**Try a sample question:**")
    col1, col2, col3 = st.columns(3)
    preset_query = None
    if col1.button("🤖 Limitations of LLMs & RAG"):
        preset_query = "What are the two major limitations of Large Language Models described in the AI overview?"
    if col2.button("☁️ Cloud Multi-Region Failover"):
        preset_query = "How fast does automated failover occur during a multi-region cloud outage?"
    if col3.button("🔒 Security MFA & Session TTL"):
        preset_query = "What is the mandatory MFA requirement and session TTL specified in the security policy?"

    user_query = st.text_input(
        "Enter your question:",
        value=preset_query or "",
        placeholder="e.g. What cryptographic standards are required for data in transit and at rest?",
    )

    if st.button("Search & Generate Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant context and generating grounded answer..."):
                response = engine.query(
                    query_text=user_query,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    temperature=temperature,
                )

            st.markdown("### 📝 Generated Answer")
            if "I don't know" in response.answer:
                st.warning(f"**{response.answer}** (The required information was not found in the ingested knowledge base).")
            else:
                st.success(response.answer)

            st.caption(f"⏱️ Latency: {response.latency_seconds:.3f}s | Retrieval Mode: `{response.retrieval_mode}` | Model: `{response.model_used}`")

            # Display Citations
            st.markdown("### 📖 Source Citations & Context Chunks")
            if not response.citations:
                st.write("No source chunks retrieved.")
            else:
                for i, citation in enumerate(response.citations, start=1):
                    with st.expander(f"Chunk #{i}: {citation.source_document} (Score: {citation.score:.4f}, Index #{citation.chunk_index})", expanded=(i == 1)):
                        st.markdown(f"**Source Document:** `{citation.source_document}`")
                        st.markdown(f"**Relevance Score:** `{citation.score:.4f}`")
                        st.markdown("**Chunk Content:**")
                        st.code(citation.text_snippet, language="markdown")

# -----------------------------------------------------------------------------
# TAB 2: Document Ingestion
# -----------------------------------------------------------------------------
with tab_docs:
    st.markdown('<div class="main-header">Document & Knowledge Base Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload files or ingest local directories into the vector database.</div>', unsafe_allow_html=True)

    col_up, col_dir = st.columns(2)

    with col_up:
        st.subheader("Upload Document Files")
        uploaded_files = st.file_uploader(
            "Select documents (.txt, .md, .pdf, .html, .docx)",
            accept_multiple_files=True,
            type=["txt", "md", "pdf", "html", "htm", "docx"],
        )
        if st.button("Ingest Uploaded Files", use_container_width=True) and uploaded_files:
            import tempfile, shutil
            temp_dir = tempfile.mkdtemp()
            saved_paths = []
            for f in uploaded_files:
                p = Path(temp_dir) / f.name
                p.write_bytes(f.getbuffer())
                saved_paths.append(str(p))

            with st.spinner("Processing and indexing documents..."):
                res = engine.ingest_files(saved_paths)
            shutil.rmtree(temp_dir, ignore_errors=True)
            st.success(f"Successfully processed {res.documents_processed} document(s), created {res.total_chunks_created} chunks in {res.duration_seconds}s!")
            st.rerun()

    with col_dir:
        st.subheader("Ingest Local Directory")
        dir_input = st.text_input("Directory Path:", value="./data/sample_docs")
        if st.button("Ingest Directory", use_container_width=True):
            if Path(dir_input).exists():
                with st.spinner(f"Ingesting documents from '{dir_input}'..."):
                    res = engine.ingest_directory(dir_input)
                st.success(f"Ingested {res.documents_processed} files, generated {res.total_chunks_created} chunks!")
                st.rerun()
            else:
                st.error(f"Directory '{dir_input}' does not exist.")

    st.markdown("---")
    st.subheader("Indexed Documents Summary")
    chunks = engine.vector_store.get_all_chunks()
    if not chunks:
        st.info("No documents currently indexed.")
    else:
        summary: dict = {}
        for c in chunks:
            fname = c.metadata.get("filename", "unknown")
            if fname not in summary:
                summary[fname] = {"filename": fname, "chunks": 0, "tokens": 0, "strategy": c.metadata.get("chunk_strategy", "n/a")}
            summary[fname]["chunks"] += 1
            summary[fname]["tokens"] += c.token_count

        st.dataframe(list(summary.values()), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Benchmark Evaluation
# -----------------------------------------------------------------------------
with tab_eval:
    st.markdown('<div class="main-header">RAG Benchmark Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Measure Retrieval Precision@K, Recall@K, MRR, and Out-of-Domain Refusal Rate.</div>', unsafe_allow_html=True)

    if st.button("Run Full Evaluation Benchmark", type="primary", use_container_width=True):
        from src.evaluation.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(engine)
        with st.spinner("Running automated evaluation against golden QA dataset..."):
            report = runner.run_benchmark(
                dataset_path="./data/evaluation/golden_qa_dataset.json",
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision @ K", f"{report.mean_precision_at_k * 100:.1f}%")
        m2.metric("Recall @ K", f"{report.mean_recall_at_k * 100:.1f}%")
        m3.metric("Hit Rate @ K", f"{report.mean_hit_rate_at_k * 100:.1f}%")
        m4.metric("Mean Reciprocal Rank", f"{report.mean_reciprocal_rank:.3f}")
        m5.metric("OOD Refusal Rate", f"{report.out_of_domain_refusal_rate * 100:.1f}%")

        st.markdown("### Detailed Query Results")
        results_table = [
            {
                "Query ID": r.query_id,
                "Query": r.query,
                "Precision@K": r.precision_at_k,
                "Recall@K": r.recall_at_k,
                "MRR": r.mrr,
                "Ground Truth Found": "✅" if r.ground_truth_found else "❌",
                "Notes": r.notes,
            }
            for r in report.detailed_results
        ]
        st.dataframe(results_table, use_container_width=True)
