"""
Qdrant vector database service.

Handles collection creation/health-check, upserting document chunks, and
similarity search with a configurable score threshold. All configuration
comes from environment variables — no hard-coded credentials or collection
names.
"""

import logging
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import settings

logger = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


class QdrantServiceError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        try:
            _client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            raise QdrantServiceError("QDRANT_UNAVAILABLE", f"Could not connect to Qdrant: {exc}") from exc
    return _client


def ensure_collection() -> None:
    """Create the configured collection if it doesn't already exist."""
    client = get_client()
    try:
        exists = client.collection_exists(settings.QDRANT_COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise QdrantServiceError("QDRANT_UNAVAILABLE", f"Could not reach Qdrant: {exc}") from exc

    if exists:
        return

    try:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )
        # Payload index for fast per-document deletion.
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise QdrantServiceError("COLLECTION_CREATE_FAILED", f"Could not create Qdrant collection: {exc}") from exc


def health_check() -> bool:
    """Return True only if Qdrant is reachable and the collection is usable."""
    try:
        client = get_client()
        client.get_collections()
        ensure_collection()
        return True
    except Exception:  # noqa: BLE001
        return False


def upsert_chunks(points: List[dict]) -> None:
    """
    points: list of {"id": str, "vector": List[float], "payload": dict}
    """
    if not points:
        return
    client = get_client()
    try:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )
    except UnexpectedResponse as exc:
        raise QdrantServiceError("QDRANT_INSERT_FAILED", f"Qdrant rejected the insert: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise QdrantServiceError("QDRANT_INSERT_FAILED", f"Failed to store vectors in Qdrant: {exc}") from exc


def search(query_vector: List[float], top_k: int, score_threshold: float) -> List[dict]:
    """Return the top_k chunks above score_threshold, each with payload + score."""
    client = get_client()
    try:
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in result.points
        ]
    except UnexpectedResponse as exc:
        raise QdrantServiceError("QDRANT_SEARCH_FAILED", f"Qdrant search failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise QdrantServiceError("QDRANT_SEARCH_FAILED", f"Failed to search Qdrant: {exc}") from exc


def delete_document_vectors(document_id: str) -> None:
    """Delete all points belonging to a document (called on document deletion)."""
    client = get_client()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise QdrantServiceError("QDRANT_DELETE_FAILED", f"Failed to delete vectors from Qdrant: {exc}") from exc
