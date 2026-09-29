"""
Gemini text generation service.

Produces the final grounded answer from retrieved context.
Generation and embedding are kept as separate responsibilities.
"""

import logging
from typing import List, Tuple

from google import genai
from google.genai.errors import APIError

from config.settings import settings


logger = logging.getLogger(__name__)

_client = None


class GenerationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _get_client() -> genai.Client:
    """
    Create and reuse the Gemini client.
    """

    global _client

    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError(
                "MISSING_API_KEY",
                "GEMINI_API_KEY is not configured."
            )

        _client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

    return _client


def _load_prompt_template() -> str:
    """
    Load the RAG prompt template.
    """

    try:
        return settings.RAG_PROMPT_PATH.read_text(
            encoding="utf-8"
        )

    except FileNotFoundError as exc:
        raise GenerationError(
            "PROMPT_MISSING",
            "RAG prompt configuration file is missing."
        ) from exc


def _format_history(
    history: List[Tuple[str, str]]
) -> str:
    """
    Format previous conversation turns.
    """

    if not history:
        return "(no prior turns)"

    lines = []

    for role, text in history[-6:]:
        speaker = (
            "User"
            if role == "user"
            else "Assistant"
        )

        lines.append(
            f"{speaker}: {text}"
        )

    return "\n".join(lines)


def _format_context(
    chunks: List[dict]
) -> str:
    """
    Format retrieved Qdrant chunks for Gemini.
    """

    if not chunks:
        return (
            "(no relevant context was found "
            "in the knowledge base)"
        )

    blocks = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        payload = chunk.get("payload", {})

        source = payload.get(
            "filename",
            "unknown source"
        )

        page = payload.get(
            "page_number"
        )

        if page:
            location = (
                f"{source}, page {page}"
            )
        else:
            location = source

        text = payload.get(
            "text",
            ""
        )

        blocks.append(
            f"[Source {i} — {location}]\n{text}"
        )

    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    chunks: List[dict],
    history: List[Tuple[str, str]]
) -> str:
    """
    Generate a grounded answer using the retrieved PDF context.
    """

    template = _load_prompt_template()

    prompt = template.format(
        context=_format_context(chunks),
        history=_format_history(history),
        question=question,
    )

    try:
        client = _get_client()

        logger.info(
            "Generating answer using Gemini model: %s",
            settings.GEMINI_MODEL
        )

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None
        )

        if not text:
            logger.error(
                "Gemini returned an empty response."
            )

            raise GenerationError(
                "EMPTY_RESPONSE",
                "The model returned an empty response."
            )

        return text.strip()

    except APIError as exc:
        # Log the complete Gemini API error
        # so it can be diagnosed from Render logs.
        logger.exception(
            "Gemini generation API error: %s",
            exc
        )

        raise GenerationError(
            "GENERATION_API_ERROR",
            f"Gemini API error: {exc}"
        ) from exc

    except GenerationError:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected generation failure: %s",
            exc
        )

        raise GenerationError(
            "GENERATION_FAILED",
            f"Failed to generate an answer: {exc}"
        ) from exc


def health_check() -> bool:
    """
    Lightweight Gemini health check.

    This verifies that the client can be constructed
    and that the API key is available.
    """

    try:
        _get_client()

        return bool(
            settings.GEMINI_API_KEY
        )

    except Exception as exc:
        logger.exception(
            "Gemini health check failed: %s",
            exc
        )

        return False
