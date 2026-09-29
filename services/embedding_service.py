"""
Embedding generation via the Gemini API (gemini-embedding-001 by default).

Uses the current unified `google-genai` SDK. Document chunks are embedded
with task_type="RETRIEVAL_DOCUMENT" and user queries with
task_type="RETRIEVAL_QUERY", per Google's documented best practice for
asymmetric retrieval quality. Output dimensionality is explicitly requested
so it always matches the configured Qdrant collection size.
"""

import logging
from typing import List

from google import genai
from google.genai.errors import APIError
from google.genai.types import EmbedContentConfig

from config.settings import settings

logger = logging.getLogger(__name__)

_client = None


class EmbeddingError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise EmbeddingError("MISSING_API_KEY", "GEMINI_API_KEY is not configured.")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _embed(texts: List[str], task_type: str) -> List[List[float]]:
    if not texts:
        return []
    try:
        client = _get_client()
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.EMBEDDING_DIMENSIONS,
            ),
        )
        return [list(e.values) for e in response.embeddings]
    except APIError as exc:
        logger.error("Gemini embedding API error: %s", exc)
        raise EmbeddingError("EMBEDDING_API_ERROR", "The embedding service returned an error.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected embedding failure")
        raise EmbeddingError("EMBEDDING_FAILED", "Failed to generate embeddings.") from exc


def embed_documents(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a list of document chunk texts, batching to stay under API limits."""
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(_embed(batch, task_type="RETRIEVAL_DOCUMENT"))
    return vectors


def embed_query(text: str) -> List[float]:
    """Embed a single user question."""
    results = _embed([text], task_type="RETRIEVAL_QUERY")
    if not results:
        raise EmbeddingError("EMBEDDING_FAILED", "Failed to generate a query embedding.")
    return results[0]
