"""
Application configuration for the RAG PDF Chatbot.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file when running locally
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()

    try:
        return int(value)
    except ValueError:
        return default


class Settings:
    # ============================================================
    # Gemini
    # ============================================================

    GEMINI_API_KEY = os.environ.get(
        "GEMINI_API_KEY",
        ""
    ).strip()

    # Gemini 3.1 Flash-Lite
    GEMINI_MODEL = os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.1-flash-lite"
    ).strip()

    # Gemini embedding model
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL",
        "gemini-embedding-001"
    ).strip()

    EMBEDDING_DIMENSIONS = _get_int(
        "EMBEDDING_DIMENSIONS",
        768
    )

    # ============================================================
    # Qdrant
    # ============================================================

    QDRANT_URL = os.environ.get(
        "QDRANT_URL",
        ""
    ).strip()

    QDRANT_API_KEY = os.environ.get(
        "QDRANT_API_KEY",
        ""
    ).strip()

    QDRANT_COLLECTION = os.environ.get(
        "QDRANT_COLLECTION",
        "rag_documents"
    ).strip()

    # ============================================================
    # Application
    # ============================================================

    MAX_FILE_SIZE_MB = _get_int(
        "MAX_FILE_SIZE_MB",
        20
    )

    TOP_K = _get_int(
        "TOP_K",
        5
    )

    CHUNK_SIZE = _get_int(
        "CHUNK_SIZE",
        1000
    )

    CHUNK_OVERLAP = _get_int(
        "CHUNK_OVERLAP",
        150
    )

    # ============================================================
    # Paths
    # ============================================================

    UPLOAD_DIR = BASE_DIR / "uploads"

    RAG_PROMPT_PATH = BASE_DIR / "config" / "rag_prompt.txt"

    # ============================================================
    # Environment validation
    # ============================================================

    @classmethod
    def validate(cls):
        """
        Validate required environment variables.
        """

        problems = []

        if not cls.GEMINI_API_KEY:
            problems.append(
                "GEMINI_API_KEY is not set."
            )

        if not cls.QDRANT_URL:
            problems.append(
                "QDRANT_URL is not set."
            )

        if not cls.QDRANT_API_KEY:
            problems.append(
                "QDRANT_API_KEY is not set."
            )

        if problems:
            raise RuntimeError(
                "Configuration errors:\n- "
                + "\n- ".join(problems)
            )

        cls.UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


# Create settings object
settings = Settings()
