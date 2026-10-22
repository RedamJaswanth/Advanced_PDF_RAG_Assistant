"""
rag_pipeline.py
---------------
Orchestrates the full Retrieval-Augmented Generation pipeline:
  1. Load & chunk the PDF
  2. Embed chunks into ChromaDB
  3. Retrieve relevant chunks for a user query
  4. Generate a grounded answer via Google Gemini 3.5 Flash Lite
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from src.document_loader import PDF_PATH, load_pdf, split_documents
from src.embeddings import create_embeddings
from src.llm import create_llm, get_api_key
from src.vector_store import create_vector_store, get_retriever

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise and helpful AI assistant specialising in answering questions \
from PDF documents.

Guidelines:
- Answer ONLY from the provided PDF context below.
- If the answer is not present in the context, respond exactly with:
  "I could not find the answer in the provided document."
- Do NOT hallucinate or use outside knowledge.
- Be concise, clear, and cite the page number(s) when relevant.
- Use the conversation history to handle follow-up questions correctly.\
"""


# ── Pipeline factory ───────────────────────────────────────────────────────────

def create_rag_pipeline(
    pdf_path: str = PDF_PATH,
    api_key: str | None = None,
    persist_directory: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> tuple[VectorStoreRetriever, BaseChatModel]:
    """
    Build the end-to-end RAG pipeline and return the retriever and LLM.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to index.
    api_key : str, optional
        Gemini API key. If None, read from environment or Streamlit secrets.
    persist_directory : str, optional
        Chroma persistence path. If None, runs in-memory.
    model : str
        Gemini model name (default: gemini-3.5-flash-lite).

    Returns
    -------
    tuple[VectorStoreRetriever, BaseChatModel]
        (retriever, llm) — both ready for inference.
    """
    logger.info("Initialising RAG pipeline for: %s with model %s", pdf_path, model)

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings, persist_directory=persist_directory)
    retriever = get_retriever(vector_store)
    llm = create_llm(model=model, api_key=api_key)

    logger.info("RAG pipeline ready.")
    return retriever, llm


# ── Question answering ─────────────────────────────────────────────────────────

def ask_question(
    question: str,
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
    chat_history: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
) -> tuple[str, list[Document], list[str]]:
    """
    Retrieve relevant context and generate a grounded answer.
    """
    if not question or not question.strip():
        return "Please enter a valid question.", [], []

    chat_history = chat_history or []

    # ── Format conversation history ────────────────────────────────────────────
    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in chat_history[-6:]
    )

    # ── Retrieve relevant chunks ───────────────────────────────────────────────
    logger.info("Retrieving context for question: %s", question[:80])
    documents = retriever.invoke(question)

    if not documents:
        logger.warning("No relevant documents retrieved for the query.")
        return (
            "I could not find relevant content in the document to answer your question.",
            [],
            [],
        )

    context = "\n\n---\n\n".join(doc.page_content for doc in documents)

    # ── Build prompt ───────────────────────────────────────────────────────────
    history_section = (
        f"Conversation History:\n{history_text}\n\n" if history_text else ""
    )

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{history_section}"
        f"PDF Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # ── Generate answer with robust model fallback ─────────────────────────────
    logger.info("Generating answer...")
    active_key = api_key or get_api_key()
    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

    response = None
    last_err = None

    # First attempt with the provided llm instance
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.warning("Initial LLM invoke failed: %s. Attempting fallback models...", exc)
        last_err = exc
        for candidate in candidate_models:
            try:
                fallback_llm = ChatGoogleGenerativeAI(
                    model=candidate,
                    temperature=0.2,
                    google_api_key=active_key,
                )
                response = fallback_llm.invoke(prompt)
                logger.info("Fallback succeeded with model: %s", candidate)
                break
            except Exception as e:
                logger.warning("Fallback candidate %s failed: %s", candidate, e)
                last_err = e

    if response is None:
        raise last_err or RuntimeError("Failed to generate response from Gemini.")

    raw_content = response.content
    if isinstance(raw_content, str):
        answer = raw_content
    elif isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        answer = "".join(parts) if parts else str(raw_content)
    else:
        answer = str(raw_content)

    logger.info("Answer generated (%d chars).", len(answer))

    # ── Extract & deduplicate source page numbers ──────────────────────────────
    sources: list[str] = list(
        dict.fromkeys(
            f"Page {doc.metadata['page'] + 1}"
            for doc in documents
            if doc.metadata.get("page") is not None
        )
    )

    return answer, documents, sources
