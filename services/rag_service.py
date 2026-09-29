"""
RAG orchestration.

Two independent pipelines, as required:

INGESTION:  file -> validate -> extract -> clean -> chunk -> embed -> store
ANSWERING:  question -> embed -> search -> filter by threshold -> build
            context -> Gemini -> grounded answer + sources
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

from config.settings import settings
from services import document_service, embedding_service, gemini_service, qdrant_service
from services.document_service import (
    STATUS_CHUNKING,
    STATUS_COMPLETED,
    STATUS_EMBEDDING,
    STATUS_EXTRACTING,
    STATUS_FAILED,
    STATUS_INDEXING,
    ExtractionError,
)
from services.embedding_service import EmbeddingError
from services.qdrant_service import QdrantServiceError
from utils.chunking import chunk_pages
from utils.helpers import new_id

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_document(doc_id: str, file_path: Path, extension: str, original_filename: str) -> int:
    """
    Run the full ingestion pipeline for an already-validated, already-saved
    file. Updates the document's status at each stage. Returns the number
    of chunks indexed. Raises IngestionError on any unrecoverable failure
    (the document status will be set to 'failed' with a reason).
    """
    try:
        document_service.update_status(doc_id, STATUS_EXTRACTING)
        extracted = document_service.extract_text(file_path, extension)
        cleaned_pages = [_clean_text(p) for p in extracted.pages]

        document_service.update_status(doc_id, STATUS_CHUNKING)
        chunks = chunk_pages(cleaned_pages)
        if not chunks:
            raise IngestionError("NO_CHUNKS", "No usable text chunks could be produced from this document.")

        document_service.update_status(doc_id, STATUS_EMBEDDING)
        vectors = embedding_service.embed_documents([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise IngestionError("EMBEDDING_MISMATCH", "Embedding count did not match chunk count.")

        document_service.update_status(doc_id, STATUS_INDEXING)
        qdrant_service.ensure_collection()

        points = [
            {
                "id": new_id(),
                "vector": vector,
                "payload": {
                    "document_id": doc_id,
                    "filename": original_filename,
                    "chunk_id": i,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                },
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        qdrant_service.upsert_chunks(points)

        document_service.set_chunk_count(doc_id, len(chunks))
        document_service.update_status(doc_id, STATUS_COMPLETED)
        return len(chunks)

    except (ExtractionError, EmbeddingError, QdrantServiceError, IngestionError) as exc:
        document_service.update_status(doc_id, STATUS_FAILED, error_message=exc.message)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected ingestion failure for document %s", doc_id)
        document_service.update_status(doc_id, STATUS_FAILED, error_message="Unexpected server error during ingestion.")
        raise IngestionError("INGESTION_FAILED", "Unexpected server error during ingestion.") from exc


def answer_question(question: str, history: List[Tuple[str, str]]) -> dict:
    """
    Run the full question-answering pipeline.

    Returns {"answer": str, "sources": [...]}. If no chunks clear the
    similarity threshold, the model is still asked to answer but the
    context block will say so, per the RAG prompt's instructions, and
    sources will be an empty list.
    """
    question = question.strip()
    if not question:
        raise IngestionError("EMPTY_QUESTION", "Please enter a question.")

    query_vector = embedding_service.embed_query(question)
    qdrant_service.ensure_collection()
    raw_matches = qdrant_service.search(
        query_vector=query_vector,
        top_k=settings.TOP_K,
        score_threshold=settings.SIMILARITY_THRESHOLD,
    )

    answer_text = gemini_service.generate_answer(question, raw_matches, history)

    sources = [
        {
            "filename": m["payload"].get("filename"),
            "page_number": m["payload"].get("page_number"),
            "relevance": round(m["score"], 4),
            "snippet": (m["payload"].get("text") or "")[:280],
        }
        for m in raw_matches
    ]

    return {"answer": answer_text, "sources": sources}
