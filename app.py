"""
app.py
------
Streamlit frontend for the Advanced PDF RAG Assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Advanced PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.rag_pipeline import create_rag_pipeline, ask_question  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Sidebar — purple-indigo gradient ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #4c1d95 0%, #312e81 40%, #1e1b4b 100%) !important;
        border-right: 1px solid #6d28d9;
    }
    [data-testid="stSidebar"] * { color: #e9d5ff !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f0abfc !important;
        letter-spacing: 0.3px;
    }
    [data-testid="stSidebar"] .stDivider { border-color: #6d28d9 !important; opacity: 0.5; }
    [data-testid="stSidebar"] code {
        background: rgba(0,0,0,0.35) !important;
        color: #a5f3fc !important;
        border: 1px solid #6d28d9 !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stButton button {
        background: rgba(109,40,217,0.4) !important;
        border: 1px solid #7c3aed !important;
        color: #f0abfc !important;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(124,58,237,0.6) !important;
    }
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric div { color: #c4b5fd !important; }

    /* ── Source chip ── */
    .source-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px 4px 4px 0;
        font-size: 13px;
        color: #93c5fd;
        font-weight: 500;
    }

    /* ── Welcome card ── */
    .welcome-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #4c1d95;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .welcome-card h2 { color: #c084fc; margin: 0 0 8px 0; }
    .welcome-card p  { color: #94a3b8; margin: 0; font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 RAG Assistant")
    st.markdown("Ask intelligent questions about your PDF using Gemini AI.")
    st.divider()

    st.markdown("### ⚙️ System Configuration")

    st.markdown("**🔤 Embedding Model**")
    st.code("all-MiniLM-L6-v2", language=None)

    st.markdown("**🗄️ Vector Database**")
    st.code("ChromaDB (local)", language=None)

    st.markdown("**🤖 Language Model**")
    st.code("Gemini 2.5 Flash", language=None)

    st.markdown("**🔗 Framework**")
    st.code("LangChain", language=None)

    st.divider()

    st.markdown("### 📄 Document")
    st.code("generative_ai_course_notes.pdf", language=None)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.metric("Turns", len(st.session_state.get("messages", [])) // 2)

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Gemini · Streamlit")


# ── Load RAG pipeline (cached across sessions) ────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading RAG pipeline — this may take a moment...")
def load_pipeline():
    """Load and cache the retriever and LLM for the session lifetime."""
    try:
        return create_rag_pipeline()
    except FileNotFoundError as exc:
        return exc
    except Exception as exc:
        return exc


pipeline = load_pipeline()

# ── Handle pipeline load errors ───────────────────────────────────────────────
if isinstance(pipeline, FileNotFoundError):
    st.error(
        f"**📄 PDF not found.**\n\n"
        f"{pipeline}\n\n"
        "Place your PDF in `data/pdfs/` and update `PDF_PATH` in `src/document_loader.py`."
    )
    st.stop()

if isinstance(pipeline, Exception):
    st.error(
        f"**⚠️ Failed to initialise RAG pipeline.**\n\n"
        f"`{pipeline}`\n\n"
        "Check that GEMINI_API_KEY is set in your secrets."
    )
    st.stop()

retriever, llm = pipeline


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="welcome-card">
        <h2>📚 Advanced PDF RAG Assistant</h2>
        <p>
            Ask any question about your document. The assistant retrieves the most
            relevant passages and generates a grounded, accurate answer using a
            Gemini-powered AI stack.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Chat history init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Render previous messages ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask something about your PDF…")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching the document…"):
            answer, documents, sources = ask_question(
                question=question,
                retriever=retriever,
                llm=llm,
                chat_history=st.session_state.messages,
            )

        st.markdown(answer)

        # ── Sources ───────────────────────────────────────────────────────────
        if sources:
            st.markdown("---")
            st.markdown("**📌 Sources**")
            chips = " ".join(
                f'<span class="source-chip">📄 {s}</span>' for s in sources
            )
            st.markdown(chips, unsafe_allow_html=True)

        # ── Retrieved context (expandable) ────────────────────────────────────
        if documents:
            with st.expander(f"🔍 View Retrieved Context ({len(documents)} chunk(s))", expanded=False):
                for i, doc in enumerate(documents, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} — Page {int(page) + 1 if page != '?' else '?'}**")
                    st.markdown(
                        f"<div style='background:#1e293b;padding:12px;border-radius:8px;"
                        f"font-size:13px;color:#cbd5e1'>{doc.page_content}</div>",
                        unsafe_allow_html=True,
                    )
                    if i < len(documents):
                        st.markdown("---")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
