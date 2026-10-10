"""
document_loader.py
------------------
Handles PDF loading and document chunking for the RAG pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PDF_PATH: str = "data/pdfs/generative_ai_course_notes.pdf"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


def load_pdf(pdf_path: str = PDF_PATH) -> list[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file, relative to the project root.

    Returns
    -------
    list[Document]
        One Document per page, with page metadata attached.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist at the given path.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. "
            "Please place your PDF in data/pdfs/ and update PDF_PATH in document_loader.py."
        )

    logger.info("Loading PDF from: %s", pdf_path)
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    logger.info("Loaded %d page(s) from '%s'.", len(documents), path.name)
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for retrieval.

    Parameters
    ----------
    documents : list[Document]
        Raw documents returned by load_pdf().
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    list[Document]
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d page(s) into %d chunk(s) (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks