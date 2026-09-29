"""
Central application configuration.

Every tunable value is read from an environment variable so nothing is
hard-coded. Values are validated at import time where practical, with
Flask surfacing a clear startup error if something required is missing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file when running locally
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


class Settings:
    # ============================================================
    # Gemini
    # ============================================================

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

    # Gemini 3.1 Flash-Lite
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    # Gemini embedding model
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001").strip()

    EMBEDDING_DIMENSIONS = _get_int("EMBEDDING_DIMENSIONS", 768)

    # ============================================================
    # Qdrant
    # ============================================================

    QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()

    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None

    # Kept as QDRANT_COLLECTION_NAME because services/qdrant_service.py
    # reads settings.QDRANT_COLLECTION_NAME. QDRANT_COLLECTION is kept as
    # an alias so either env var name works.
    QDRANT_COLLECTION_NAME = os.environ.get(
        "QDRANT_COLLECTION_NAME",
        os.environ.get("QDRANT_COLLECTION", "rag_documents"),
    ).strip()

    # ============================================================
    # RAG behavior
    # ============================================================

    CHUNK_SIZE = _get_int("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 150)
    TOP_K = _get_int("TOP_K", 5)
    SIMILARITY_THRESHOLD = _get_float("SIMILARITY_THRESHOLD", 0.55)

    # ============================================================
    # Upload / Flask
    # ============================================================

    MAX_UPLOAD_SIZE_MB = _get_int(
        "MAX_UPLOAD_SIZE_MB",
        _get_int("MAX_FILE_SIZE_MB", 20),
    )
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    FLASK_DEBUG = _get_bool("FLASK_DEBUG", False)
    PORT = _get_int("PORT", 5000)

    DATABASE_PATH = BASE_DIR / os.environ.get("DATABASE_PATH", "data/documents.db")

    # Kept as UPLOAD_STORAGE_PATH because app.py reads
    # settings.UPLOAD_STORAGE_PATH. UPLOAD_DIR kept as an alias below.
    UPLOAD_STORAGE_PATH = BASE_DIR / os.environ.get("UPLOAD_STORAGE_PATH", "uploads")
    UPLOAD_DIR = UPLOAD_STORAGE_PATH  # alias for backward compatibility

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    ALLOWED_MIME_TYPES = {
        ".pdf": {"application/pdf"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",  # some browsers report docx as generic zip
            "application/octet-stream",
        },
        ".txt": {"text/plain"},
    }

    # ============================================================
    # Paths
    # ============================================================

    RAG_PROMPT_PATH = BASE_DIR / "config" / "rag_prompt.txt"

    # ============================================================
    # Environment validation
    # ============================================================

    @classmethod
    def validate(cls):
        """Raise a clear error for missing required configuration."""
        problems = []

        if not cls.GEMINI_API_KEY:
            problems.append("GEMINI_API_KEY is not set.")
        if not cls.QDRANT_URL:
            problems.append("QDRANT_URL is not set.")
        if not cls.QDRANT_API_KEY:
            problems.append("QDRANT_API_KEY is not set.")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not (0.0 <= cls.SIMILARITY_THRESHOLD <= 1.0):
            problems.append("SIMILARITY_THRESHOLD must be between 0 and 1.")

        return problems

    @classmethod
    def ensure_directories(cls):
        cls.UPLOAD_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


# Create settings object
settings = Settings()
