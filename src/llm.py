"""
llm.py
------
Instantiates the local Ollama language model used for answer generation.
Ollama must be running locally with the target model pulled.
"""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_MODEL: str = "qwen2.5:3b"
OLLAMA_TEMPERATURE: float = 0.2   # Lower = more factual, less creative
OLLAMA_BASE_URL: str = "http://localhost:11434"


def create_llm(
    model: str = OLLAMA_MODEL,
    temperature: float = OLLAMA_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
) -> ChatOllama:
    """
    Create and return a ChatOllama LLM instance.

    Prerequisites
    -------------
    1. Ollama must be installed: https://ollama.com/download
    2. The target model must be pulled:
           ollama pull qwen2.5:3b
    3. Ollama must be running:
           ollama serve

    Parameters
    ----------
    model : str
        Name of the Ollama model to use.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    base_url : str
        Ollama server URL.

    Returns
    -------
    ChatOllama
        A LangChain-compatible chat model instance.
    """
    logger.info("Connecting to Ollama model '%s' at %s", model, base_url)
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )
    logger.info("Ollama LLM ready.")
    return llm