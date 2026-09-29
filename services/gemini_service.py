"""
Gemini text generation service — produces the final grounded answer from
retrieved context. Kept separate from embedding_service.py since generation
and embedding are distinct responsibilities that may evolve independently.
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
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError("MISSING_API_KEY", "GEMINI_API_KEY is not configured.")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _load_prompt_template() -> str:
    try:
        return settings.RAG_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GenerationError("PROMPT_MISSING", "RAG prompt configuration file is missing.") from exc


def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    lines = []
    for role, text in history[-6:]:  # keep the last few turns only
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "(no relevant context was found in the knowledge base)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        payload = c["payload"]
        source = payload.get("filename", "unknown source")
        page = payload.get("page_number")
        location = f"{source}, page {page}" if page else source
        blocks.append(f"[Source {i} — {location}]\n{payload.get('text', '')}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: List[dict], history: List[Tuple[str, str]]) -> str:
    template = _load_prompt_template()
    prompt = template.format(
        context=_format_context(chunks),
        history=_format_history(history),
        question=question,
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not text:
            raise GenerationError("EMPTY_RESPONSE", "The model returned an empty response.")
        return text.strip()
    except APIError as exc:
        logger.error("Gemini generation API error: %s", exc)
        raise GenerationError("GENERATION_API_ERROR", "The AI service returned an error. Please try again.") from exc
    except GenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected generation failure")
        raise GenerationError("GENERATION_FAILED", "Failed to generate an answer.") from exc


def health_check() -> bool:
    """Lightweight check that the Gemini client can be constructed (API key present)."""
    try:
        _get_client()
        return bool(settings.GEMINI_API_KEY)
    except Exception:  # noqa: BLE001
        return False
