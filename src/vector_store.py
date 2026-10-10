"""
vector_store.py
---------------
Creates and manages the ChromaDB vector store used for
semantic document retrieval.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_PATH: str = "chroma_db"
RETRIEVER_K: int = 4          # Number of chunks to retrieve per query
SEARCH_TYPE: str = "similarity"


def create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str = CHROMA_PATH,
) -> Chroma:
    """
    Build a ChromaDB vector store from document chunks and persist it to disk.

    Parameters
    ----------
    chunks : list[Document]
        Chunked documents produced by split_documents().
    embeddings : Embeddings
        Embedding model used to encode each chunk.
    persist_directory : str
        Directory path where ChromaDB data will be saved.

    Returns
    -------
    Chroma
        A populated, persisted Chroma vector store instance.
    """
    logger.info(
        "Creating ChromaDB vector store with %d chunk(s) at '%s'.",
        len(chunks),
        persist_directory,
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    logger.info("ChromaDB vector store created and persisted successfully.")
    return vector_store


def get_retriever(
    vector_store: Chroma,
    k: int = RETRIEVER_K,
    search_type: str = SEARCH_TYPE,
) -> VectorStoreRetriever:
    """
    Wrap the vector store in a LangChain retriever for query-time use.

    Parameters
    ----------
    vector_store : Chroma
        The populated Chroma vector store.
    k : int
        Number of top-similar chunks to return per query.
    search_type : str
        Retrieval strategy — "similarity" or "mmr" (max marginal relevance).

    Returns
    -------
    VectorStoreRetriever
        A retriever ready to be called with retriever.invoke(query).
    """
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    logger.info(
        "Retriever configured: search_type='%s', k=%d.", search_type, k
    )
    return retriever