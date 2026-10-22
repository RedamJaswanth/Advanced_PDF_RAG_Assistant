"""
app.py
------
Streamlit frontend for the Advanced PDF RAG Assistant.
Powered by Google Gemini 3.8 Flash, HuggingFace embeddings, and ChromaDB.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Advanced PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.rag_pipeline import create_rag_pipeline, ask_question  # noqa: E402
from src.llm import get_api_key  # noqa: E402

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
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .welcome-card h2 { color: #c084fc; margin: 0 0 8px 0; font-size: 24px; }
    .welcome-card p  { color: #94a3b8; margin: 0; font-size: 15px; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 RAG Assistant")
    st.markdown("Context-aware PDF question answering powered by Google Gemini and ChromaDB.")
    st.divider()

    st.markdown("### 📄 Document Source")
    uploaded_file = st.file_uploader(
        "Upload a PDF (or use default)",
        type=["pdf"],
        help="Upload your own PDF to ask questions against it, or use the default course notes.",
    )

    default_pdf_path = Path("data/pdfs/generative_ai_course_notes.pdf")

    if uploaded_file is not None:
        active_doc_name = uploaded_file.name
        # Persist uploaded file to a temporary location
        temp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        active_pdf_path = temp_dir / uploaded_file.name
        with open(active_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_pdf_path_str = str(active_pdf_path)
        st.success(f"Loaded: `{active_doc_name}`")
    else:
        active_doc_name = "generative_ai_course_notes.pdf (Default)"
        active_pdf_path_str = str(default_pdf_path)
        st.info("Using default course notes.")

    st.divider()

    st.markdown("### ⚙️ System Configuration")
    st.markdown("**🔤 Embedding Model**")
    st.code("all-MiniLM-L6-v2 (HuggingFace)", language=None)

    st.markdown("**🗄️ Vector Database**")
    st.code("ChromaDB", language=None)

    st.markdown("**🤖 Language Model**")
    st.code("gemini-3.8-flash (Google GenAI)", language=None)

    st.markdown("**🔗 Framework**")
    st.code("LangChain", language=None)

    st.divider()

    # Optional custom API key override
    env_key = get_api_key()
    custom_key = None
    if not env_key:
        custom_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key if not configured in secrets.",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.metric("Turns", len(st.session_state.get("messages", [])) // 2)

    st.divider()
    st.caption("Built with LangChain · Google Gemini · ChromaDB · Streamlit")


# ── Load RAG pipeline (cached by document path and file modification time) ────
@st.cache_resource(show_spinner="⚙️ Indexing PDF and initialising Gemini pipeline...")
def load_pipeline(doc_path: str, doc_name: str, api_key: str | None = None):
    """Load and cache the retriever and LLM for the active document."""
    try:
        return create_rag_pipeline(pdf_path=doc_path, api_key=api_key)
    except FileNotFoundError as exc:
        return exc
    except Exception as exc:
        return exc


active_key = custom_key or env_key

if not active_key:
    st.warning(
        "🔑 **Gemini API Key Required**\n\n"
        "Please provide a Gemini API key to query the document. You can:\n"
        "1. Enter your key in the sidebar text box, or\n"
        "2. Add `GEMINI_API_KEY = \"your_key\"` to your Streamlit secrets."
    )
    pipeline = None
else:
    pipeline = load_pipeline(active_pdf_path_str, active_doc_name, active_key)

# ── Handle pipeline load errors ───────────────────────────────────────────────
if isinstance(pipeline, FileNotFoundError):
    st.error(
        f"**📄 Document not found.**\n\n"
        f"`{pipeline}`\n\n"
        "Please upload a PDF from the sidebar to begin."
    )
    st.stop()

if isinstance(pipeline, Exception):
    st.error(
        f"**⚠️ Failed to initialise RAG pipeline.**\n\n"
        f"`{pipeline}`\n\n"
        "Please check your `GEMINI_API_KEY` configuration."
    )
    st.stop()

if pipeline is not None:
    retriever, llm = pipeline
else:
    retriever, llm = None, None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="welcome-card">
        <h2>📚 Advanced PDF RAG Assistant</h2>
        <p>
            Ask intelligent questions about <strong>{active_doc_name}</strong>.
            The assistant searches the indexed document using semantic embeddings
            and synthesises accurate, grounded responses with <strong>Google Gemini 3.8 Flash</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Chat history init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset chat history if user switches document
if "last_doc" not in st.session_state:
    st.session_state.last_doc = active_doc_name
elif st.session_state.last_doc != active_doc_name:
    st.session_state.messages = []
    st.session_state.last_doc = active_doc_name


# ── Render previous messages ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask something about your PDF…")

if question:
    if retriever is None or llm is None:
        st.error("Please configure a valid `GEMINI_API_KEY` before asking questions.")
        st.stop()

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
