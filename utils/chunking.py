"""
Configurable text chunking.

Splits text on paragraph/sentence boundaries where possible instead of
cutting at arbitrary character positions, while respecting a target
chunk size and overlap (both configurable via environment variables).
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from config.settings import settings

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: Optional[int] = None


def _split_into_sentences(paragraph: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s for s in sentences if s]


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    page_number: Optional[int] = None,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Chunk boundaries prefer paragraph breaks, then sentence breaks, only
    falling back to a hard character cut for a single sentence/paragraph
    that itself exceeds `chunk_size`.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Flatten into sentence-level units so we can pack them into chunks.
    units: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_into_sentences(para))

    chunks: List[str] = []
    current = ""

    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # Build overlap from the tail of the previous chunk.
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap_text} {unit}".strip()
        else:
            current = unit

        # A single unit longer than chunk_size: hard-split it.
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap:] if chunk_overlap > 0 else current[chunk_size:]

    if current:
        chunks.append(current)

    return [
        Chunk(text=c, chunk_index=i, page_number=page_number)
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def chunk_pages(pages: List[str]) -> List[Chunk]:
    """Chunk a list of page texts, preserving page numbers (1-indexed) and
    producing a single continuous chunk_index sequence across the document."""
    all_chunks: List[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, page_number=page_num)
        for c in page_chunks:
            c.chunk_index = len(all_chunks)
            all_chunks.append(c)
    return all_chunks
