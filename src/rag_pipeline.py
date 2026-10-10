"""
rag_pipeline.py
---------------
Orchestrates the full Retrieval-Augmented Generation pipeline:
  1. Load & chunk the PDF
  2. Embed chunks into ChromaDB
  3. Retrieve relevant chunks for a user query
  4. Generate a grounded answer via the local LLM
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from src.document_loader import load_pdf, split_documents
from src.embeddings import create_embeddings
from src.llm import create_llm
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

def create_rag_pipeline() -> tuple[VectorStoreRetriever, BaseChatModel]:
    """
    Build the end-to-end RAG pipeline and return the retriever and LLM.

    Steps
    -----
    1. Load the PDF document.
    2. Split it into overlapping chunks.
    3. Embed chunks and store them in ChromaDB.
    4. Create a similarity retriever over the vector store.
    5. Load the local Ollama LLM.

    Returns
    -------
    tuple[VectorStoreRetriever, BaseChatModel]
        (retriever, llm) — both ready for inference.
    """
    logger.info("Initialising RAG pipeline...")

    documents = load_pdf()
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vector_store)
    llm = create_llm()

    logger.info("RAG pipeline ready.")
    return retriever, llm


# ── Question answering ─────────────────────────────────────────────────────────

def ask_question(
    question: str,
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
    chat_history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[Document], list[str]]:
    """
    Retrieve relevant context and generate a grounded answer.

    Parameters
    ----------
    question : str
        The user's current question.
    retriever : VectorStoreRetriever
        ChromaDB retriever loaded by create_rag_pipeline().
    llm : BaseChatModel
        Chat model loaded by create_rag_pipeline().
    chat_history : list[dict], optional
        Prior conversation turns as [{"role": ..., "content": ...}, ...].

    Returns
    -------
    tuple[str, list[Document], list[str]]
        (answer_text, retrieved_documents, deduplicated_page_sources)
    """
    if not question or not question.strip():
        return "Please enter a valid question.", [], []

    chat_history = chat_history or []

    # ── Format conversation history ────────────────────────────────────────────
    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in chat_history[-6:]   # limit to last 6 turns
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

    # ── Generate answer ────────────────────────────────────────────────────────
    logger.info("Generating answer...")
    response = llm.invoke(prompt)
    answer: str = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
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