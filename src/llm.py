"""
llm.py
------
Creates the Gemini language model used for answer generation.
The API key is read from the GEMINI_API_KEY environment variable
(set as a secret on Streamlit Cloud, or in your local .env file).
"""

from __future__ import annotations

import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.2   # Lower = more factual, less creative


def create_llm(
    model: str = GEMINI_MODEL,
    temperature: float = GEMINI_TEMPERATURE,
) -> ChatGoogleGenerativeAI:
    """Create and return a Gemini chat model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "or to the Streamlit Cloud secrets."
        )

    logger.info("Connecting to Gemini model '%s'", model)
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
    logger.info("Gemini LLM ready.")
    return llm
