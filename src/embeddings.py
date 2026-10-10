"""
embeddings.py
-------------
Creates and returns the sentence-transformer embedding model
used for encoding document chunks and user queries.
"""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Instantiate a HuggingFace sentence-transformer embedding model.

    The model runs locally — no API key required.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to all-MiniLM-L6-v2
        which offers a good balance of speed and quality for RAG tasks.

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain-compatible embeddings object.
    """
    logger.info("Loading embedding model: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully.")
    return embeddings