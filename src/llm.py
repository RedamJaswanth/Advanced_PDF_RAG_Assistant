"""
llm.py
------
Instantiates the Google Gemini language model used for answer generation.
Reads GEMINI_API_KEY from environment variables or Streamlit secrets.
"""

from __future__ import annotations

import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# gemini-3.5-flash-lite provides high speed and standard free-tier quota limits
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TEMPERATURE: float = 0.2  # Lower = more factual, grounded answers


def get_api_key() -> str | None:
    """Retrieve GEMINI_API_KEY from environment variables or Streamlit secrets."""
    # 1. Direct environment variable
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    # 2. Streamlit secrets (for Streamlit Community Cloud)
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"])
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass

    return None


def create_llm(
    model: str | None = None,
    temperature: float = TEMPERATURE,
    api_key: str | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Create and return a ChatGoogleGenerativeAI LLM instance.

    Parameters
    ----------
    model : str, optional
        Name of the Gemini model to use (default: gemini-3.5-flash-lite).
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    api_key : str, optional
        Explicit API key if provided.

    Returns
    -------
    ChatGoogleGenerativeAI
        A LangChain-compatible chat model instance.
    """
    resolved_key = api_key or get_api_key()
    if not resolved_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set GEMINI_API_KEY in your "
            "environment variables (.env) or Streamlit Secrets."
        )

    chosen_model = model or os.getenv("GEMINI_MODEL") or GEMINI_MODEL
    logger.info("Connecting to Google Gemini model '%s'", chosen_model)
    llm = ChatGoogleGenerativeAI(
        model=chosen_model,
        temperature=temperature,
        google_api_key=resolved_key,
    )
    logger.info("Google Gemini LLM ready.")
    return llm
